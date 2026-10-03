import pandas as pd
import numpy as np
import glob
import gc
import optuna
from dateutil.relativedelta import relativedelta
from pandas.api.types import CategoricalDtype
from functools import partial
from optuna.samplers import RandomSampler
import argparse
import itertools
from tqdm.auto import tqdm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import polars as pl
from scipy.stats import norm
from constants import LOW_ATM_MONEYNESS, HIGH_ATM_MONEYNESS, N_TRIALS, N_JOBS, DAYS_CONVERTER, VOL_REGIMES, \
                       TITLE_FONT_SIZE, LABEL_AXIS_FONT_SIZE, WIDTH, HEIGHT, D_HORIZONS
DERIBIT_FEE_PREMIUM_PCT = 0.125
BINANCE_TF = 0.001
MIN_FEE_BPS_UNDERLYING = 0.0006
# Share of the quoted bid-ask spread paid on each trade. P&L is measured mid to mid, so buying at the ask or selling at
# the bid costs half the spread; a round trip costs one full spread.
SPREAD_COST_SHARE = 0.5
# Days to expiry [min, max) of the straddles traded for each forecasting horizon D: the option maturity is matched to
# the horizon, and options in their last two days are never traded.
EXPIRY_WINDOWS = {'1D': (2, 7), '1W': (7, 21), '1M': (30, 60)}
# True: the position is set once per day from the first bar's signal, held until the day's last bar and closed there.
# False: the position can change at every bar (original behaviour).
HOLD_FULL_DAY = True
# Bar of the day (0 = first) whose signal sets the held position. Bar 0 cannot be used for the RV strategy: its filter
# needs the previous bar's realised value, which does not exist for the day's first bar.
HOLD_SIGNAL_BAR = 1
# True: every day a new position is opened per horizon D and held for HOLDING_DAYS[D] days (entry and exit at the same
# bar of the day). The portfolio of a horizon is the equal-weighted average of its open positions (overlapping
# tranches). False: positions are closed within the day (HOLD_FULL_DAY) or can change every bar.
HOLD_OVER_HORIZON = True
HOLDING_DAYS = {'1D': 1, '1W': 7, '1M': 30}
# An option is only bought into a tranche if it still has at least this many days to expiry when the tranche is closed.
MIN_DAYS_TO_EXPIRY_AT_EXIT = 1


def _reindex_cols(df, cols):
    out = df.reindex(columns=cols)
    out.columns.names = df.columns.names
    return out


non_expired = {'15m': pd.to_timedelta('15min') / pd.to_timedelta('360D'),
               '30m': pd.to_timedelta('30min') / pd.to_timedelta('360D'),
               '1h': pd.to_timedelta('1h') / pd.to_timedelta('360D')}


def callback(study, trial) -> None:
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    if study.best_trial.number == trial.number:
        if trial.user_attrs['target'] == 'RV':
            study.set_user_attr(key='lower_bound_atm', value=trial.user_attrs['lower_bound_atm'])
            study.set_user_attr(key='upper_bound_atm', value=trial.user_attrs['upper_bound_atm'])
        study.set_user_attr(key='threshold_filter', value=trial.user_attrs['threshold_filter'])
        study.set_user_attr(key='sharpe', value=trial.user_attrs['sharpe'])


def objective_daily(trial: optuna.trial.Trial, **kwargs) -> float:
    param = dict([('threshold_filter', trial.suggest_float(name='threshold_filter', low=0, high=1, step=.1))])
    data = kwargs['data']
    sharpe = 0
    if kwargs['target'] == 'RV':
        param.update(
            dict([('lower_bound_atm', trial.suggest_float(name='lower_bound_atm', low=np.log(-.1+LOW_ATM_MONEYNESS),
                                                          high=np.log(.1+LOW_ATM_MONEYNESS))),
                  ('upper_bound_atm', trial.suggest_float(name='upper_bound_atm', low=np.log(-.1+HIGH_ATM_MONEYNESS),
                                                          high=np.log(.1+HIGH_ATM_MONEYNESS)))]))
        option_chain_data = kwargs['option_chain_data']
        filter_signal = kwargs['filter_signal']
        svix = kwargs['svix']
        forecast_flat = data[['forecast']].droplevel(axis=1, level=0)
        if not forecast_flat.index.is_unique:
            forecast_flat = forecast_flat[~forecast_flat.index.duplicated(keep='last')]
        svix_dedup = svix
        if not svix.index.get_level_values('timestamp').is_unique:
            print(f"[objective_daily] WARNING: duplicate timestamp(s) in svix data "
                  f"for target={kwargs['target']!r} -- dropping duplicates (keep last).")
            svix_dedup = svix[~svix.index.duplicated(keep='last')]
        forecast_broadcast = forecast_flat.reindex(svix_dedup.index.get_level_values('timestamp'))
        forecast_broadcast.index = svix_dedup.index
        vrp = svix_dedup.sub(forecast_broadcast)
        markets = vrp.columns.get_level_values('market').unique().tolist()
        signals = (-1*np.sign(vrp)).groupby('aggregated', group_keys=False).apply(
            lambda x, y: lagged_signal(x.mul(y)), y=filter_signal.abs().gt(param['threshold_filter']).astype(int))
        atm_filter = \
            pl.col('moneyness').is_between(lower_bound=param['lower_bound_atm'], upper_bound=param['upper_bound_atm'])
        option_chain_data = option_chain_data.filter(atm_filter)
        option_chain_data = option_chain_data.filter(pl.len().over(['symbol', 'timestamp']) == 2)
        try:
            number_of_unique_straddles = option_chain_data.select(pl.col('symbol').n_unique()).collect().item()
        except pl.exceptions.ComputeError:
            raise optuna.TrialPruned()
        if number_of_unique_straddles == 0:
            raise optuna.TrialPruned()
        if number_of_unique_straddles > 0:
            active_count = signals.ne(0).sum(axis=1)
            straddles_per_horizon = \
                eligible_straddle_counts(option_chain_data, signals.columns.get_level_values('D').unique().tolist())
            signals = signals.div(signals.columns.get_level_values('D').map(straddles_per_horizon).to_numpy(), axis=1)
            signals = signals.div(active_count.replace(0, np.nan), axis=0).fillna(0)
            delta_weights = signals.groupby('aggregated', group_keys=False).apply(weight_changes)
            signals = \
                pl.from_pandas(signals.stack('market').reset_index()).with_columns(
                    pl.col('timestamp').cast(pl.Datetime('ms')),
                    pl.col('aggregated').cast(pl.Int64), pl.col('market').cast(pl.Int64)
                ).lazy()
            available_ds = signals.collect_schema().names()[3:]
            delta_weights = \
                pl.from_pandas(delta_weights.stack('market').reset_index()).with_columns(
                    pl.col('timestamp').cast(pl.Datetime('ms')),
                    pl.col('aggregated').cast(pl.Int64), pl.col('market').cast(pl.Int64)
                ).lazy()
            option_chain_data = option_chain_data.join(signals, on='timestamp', how='left')
            option_chain_data = option_chain_data.sort(['symbol', 'aggregated', 'market', 'timestamp'])
            option_chain_data = \
                option_chain_data.group_by(['symbol', 'aggregated', 'market', 'timestamp'], maintain_order=True)
            option_chain_data = option_chain_data.agg([pl.col('mid_price').sum(), pl.col('spread').sum(),
                                                       pl.col('transaction_fees').sum()]+
                                                      [pl.col(d).last() for d in available_ds] +
                                                      [pl.col(f'eligible_{d}').last() for d in available_ds])
            option_chain_data = \
                option_chain_data.with_columns(
                    [(pl.col('mid_price').log().diff().over(['symbol', 'aggregated', 'market'])).alias('returns'),
                                                (SPREAD_COST_SHARE * pl.col('spread') / pl.col('mid_price')).alias('basc_fees')])
            option_chain_data = option_chain_data.join(delta_weights, on=['aggregated', 'market', 'timestamp'],
                                                       how='left', suffix='_delta_weights')
            exprs = []
            for D in filter_signal.columns.get_level_values('D').unique().tolist():
                eligible = pl.col(f'eligible_{D}')
                exprs += [pl.when(eligible).then((pl.col('returns') * pl.col(D) - (pl.col('transaction_fees') *
                          pl.col(f'{D}_delta_weights')).abs()).fill_null(0)).alias(f'returns_tc_{D}'),
                          pl.when(eligible).then((pl.col('returns') * pl.col(D) - (pl.col('basc_fees') *
                          pl.col(f'{D}_delta_weights')).abs()).fill_null(0)).alias(f'returns_basc_{D}'),
                          pl.when(eligible).then(pl.col('returns') * pl.col(D)).alias(f'returns_{D}')]
            option_chain_data = option_chain_data.sort(['symbol', 'aggregated', 'market', 'timestamp'])
            option_chain_data = option_chain_data.with_columns(exprs)
            option_chain_data = \
                option_chain_data.drop(['returns', 'spread', 'transaction_fees', 'mid_price'] +
                                       filter_signal.columns.get_level_values('D').unique().tolist() +
                                       ['basc_fees'] +
                                       [f'{d}_delta_weights' for d in
                                        filter_signal.columns.get_level_values('D').unique().tolist()])
            option_chain_data = option_chain_data.sort(['symbol', 'aggregated', 'market', 'timestamp'])
            returns_cols = \
                [f'{prefix}_{D}' for D in filter_signal.columns.get_level_values('D').unique().tolist() for prefix in
                 ['returns', 'returns_tc', 'returns_basc']]
            option_chain_data = option_chain_data.group_by(['aggregated', 'market', 'timestamp'])
            option_chain_data = option_chain_data.agg([pl.col(c).mean().fill_null(0).alias(c) for c in returns_cols])
            option_chain_data = \
                option_chain_data.collect().to_pandas().set_index(['aggregated', 'market', 'timestamp'])
            option_chain_data = option_chain_data.unstack('market')
            option_chain_data = option_chain_data.dropna(how='all', axis=1)
            returns_cols = \
                pd.MultiIndex.from_product(
                    [[f'returns_{D}' for D in filter_signal.columns.get_level_values('D').unique().tolist()],
                     markets])
            returns = _reindex_cols(option_chain_data, returns_cols)
            returns_tc_cols = \
                pd.MultiIndex.from_product(
                    [[f'returns_tc_{D}' for D in filter_signal.columns.get_level_values('D').unique().tolist()],
                     markets])
            returns_tc = _reindex_cols(option_chain_data, returns_tc_cols)
            returns_basc_cols = \
                pd.MultiIndex.from_product(
                    [[f'returns_basc_{D}' for D in filter_signal.columns.get_level_values('D').unique().tolist()],
                     markets])
            returns_basc = _reindex_cols(option_chain_data, returns_basc_cols)
            returns = returns_table(returns, fees=None, underlying_name=underlying_name)
            returns_tc = returns_table(returns_tc, fees='transaction_fees', underlying_name=underlying_name)
            returns_basc = returns_table(returns_basc, fees='basc_fees', underlying_name=underlying_name)
            pnl = pd.concat([returns, returns_tc, returns_basc]).groupby(['aggregated', 'transaction_fees', 'spread'])
            sharpe = pnl.apply(lambda x: x.mean().div(x.std())).mean().mean()
        trial.set_user_attr(key='lower_bound_atm', value=(param['lower_bound_atm']))
        trial.set_user_attr(key='upper_bound_atm', value=(param['upper_bound_atm']))
    elif kwargs['target'] == 'returns':
        spot_data = kwargs['spot_data']
        forecast = data
        signals = np.sign(forecast)
        quantiles = forecast.abs().quantile(param['threshold_filter'])
        signal_filter = (forecast.abs().gt(quantiles, axis=1)).astype(int)
        signals = lagged_signal(signals.mul(signal_filter)).fillna(0)
        active_count = signals.ne(0).sum(axis=1)
        signals = signals.div(active_count.replace(0, np.nan), axis=0).fillna(0)
        delta_weights = weight_changes(signals)
        signals = signals.stack('market').reset_index().assign(aggregated='')
        signals = pl.from_pandas(signals).with_columns(
            pl.col('timestamp').cast(pl.Datetime('ms')), pl.col('aggregated'),
            pl.col('market').cast(pl.Int64)).lazy()
        delta_weights = delta_weights.stack('market').reset_index().assign(aggregated='')
        delta_weights = pl.from_pandas(delta_weights).with_columns(
            pl.col('timestamp').cast(pl.Datetime('ms')), pl.col('aggregated'),
            pl.col('market').cast(pl.Int64)).lazy()
        spot_data = spot_data.join(signals, on='timestamp', how='left')
        spot_data = spot_data.sort(['underlying_name', 'aggregated', 'market', 'timestamp'])
        spot_data = spot_data.group_by(['underlying_name', 'aggregated', 'market', 'timestamp'], maintain_order=True)
        spot_data = spot_data.agg([pl.col('returns').sum(), pl.col('basc_fees').last(),
                                   pl.col('transaction_fees').last()]+
                                  [pl.col(d).last() for d in forecast.columns.get_level_values('D').unique().tolist()])
        spot_data = spot_data.join(delta_weights, on=['aggregated', 'market', 'timestamp'], how='left',
                                   suffix='_delta_weights')
        exprs = []
        for D in data.columns.get_level_values('D').unique().tolist():
            exprs += pnl_expressions(
                return_col='returns',
                weight_col=D,
                delta_weight_col=f'{D}_delta_weights',
                transaction_fee_col='transaction_fees',
                basc_fee_col='basc_fees',
                D=D,
            )
        spot_data = spot_data.sort(['underlying_name', 'aggregated', 'market', 'timestamp'])
        spot_data = spot_data.with_columns(exprs)
        spot_data = spot_data.drop(['returns', 'transaction_fees']+data.columns.get_level_values('D').tolist() +
                                   ['basc_fees'] +
                                   [f'{d}_delta_weights' for d in data.columns.get_level_values('D').tolist()])
        spot_data = spot_data.sort(['underlying_name', 'aggregated', 'market', 'timestamp'])
        returns_cols = \
            [f'{prefix}_{D}' for D in data.columns.get_level_values('D').unique().tolist() for prefix in
             ['returns', 'returns_tc', 'returns_basc']]
        spot_data = spot_data.group_by(['aggregated', 'market', 'timestamp'])
        spot_data = spot_data.agg([pl.col(c).mean().alias(c) for c in returns_cols])
        spot_data = spot_data.collect().to_pandas().set_index(['aggregated', 'market', 'timestamp'])
        spot_data = spot_data.unstack('market')
        spot_data = spot_data.dropna(how='all', axis=1)
        markets = spot_data.columns.get_level_values('market').unique().tolist()
        returns_cols = \
            pd.MultiIndex.from_product([[f'returns_{D}' for D in data.columns.get_level_values('D').unique().tolist()],
                                        markets])
        returns = _reindex_cols(spot_data, returns_cols)
        returns_tc_cols = \
            pd.MultiIndex.from_product([[f'returns_tc_{D}' for D in
                                         data.columns.get_level_values('D').unique().tolist()], markets])
        returns_tc = _reindex_cols(spot_data, returns_tc_cols)
        returns_basc_cols = \
            pd.MultiIndex.from_product([[f'returns_basc_{D}' for D in
                                         data.columns.get_level_values('D').unique().tolist()], markets])
        returns_basc = _reindex_cols(spot_data, returns_basc_cols)
        returns = returns_table(returns, fees=None, underlying_name=underlying_name)
        returns_tc = returns_table(returns_tc, fees='transaction_fees', underlying_name=underlying_name)
        returns_basc = returns_table(returns_basc, fees='basc_fees', underlying_name=underlying_name)
        pnl = pd.concat([returns, returns_tc, returns_basc]).groupby(['aggregated', 'transaction_fees', 'spread'])
        sharpe = pnl.apply(lambda x: x.mean().div(x.std())).mean().mean()
    if not np.isfinite(sharpe):
        raise optuna.TrialPruned()
    trial.set_user_attr(key='sharpe', value=sharpe)
    trial.set_user_attr(key='threshold_filter', value=param['threshold_filter'])
    trial.set_user_attr(key='target', value=kwargs['target'])
    return trial.user_attrs['sharpe']


def lagged_signal(x: pd.DataFrame) -> pd.DataFrame:
    """Signal usable at each bar (one-bar lag). With HOLD_FULL_DAY or HOLD_OVER_HORIZON the position is set from the
    signal of bar HOLD_SIGNAL_BAR and carried unchanged to the day's last bar."""
    x = x.shift()
    entry = HOLD_SIGNAL_BAR + 1
    if (HOLD_FULL_DAY or HOLD_OVER_HORIZON) and len(x) > entry:
        x.iloc[entry:] = np.tile(x.iloc[[entry]].to_numpy(), (len(x) - entry, 1))
        x.iloc[:entry] = 0
    return x


def weight_changes(x: pd.DataFrame) -> pd.DataFrame:
    """Traded amount at each bar. With HOLD_FULL_DAY (and no multi-day holding) the position is closed on the day's
    last bar, so both the entry and the exit are charged."""
    delta = x.diff()
    if HOLD_FULL_DAY and not HOLD_OVER_HORIZON and len(x) > 1:
        delta.iloc[-1] = -x.iloc[-1].to_numpy()
    return delta


def horizon_weights(day_signals: pd.DataFrame, sign_history: list) -> pd.DataFrame:
    """First-moment strategy with overlapping tranches. `day_signals` holds today's held signal (columns (D, market)).
    The weight of a horizon is the average of the signs of the last HOLDING_DAYS[D] days; it changes once per day, at
    the entry bar, so the entering and the expiring tranche are netted before costs are charged."""
    entry = HOLD_SIGNAL_BAR + 1
    today_sign = day_signals.iloc[entry] if len(day_signals) > entry else day_signals.iloc[-1] * 0
    holding = day_signals.columns.get_level_values('D').map(HOLDING_DAYS).to_numpy()
    def average(history):
        out = pd.Series(0., index=day_signals.columns)
        for position, column in enumerate(day_signals.columns):
            recent = [s.get(column, 0.) for s in history[-int(holding[position]):]]
            out.iloc[position] = np.nansum(recent) / holding[position]
        return out
    weight_before = average(sign_history)
    sign_history.append(today_sign.fillna(0))
    del sign_history[:-max(HOLDING_DAYS.values())]
    weight_after = average(sign_history)
    weights = pd.DataFrame(np.tile(weight_before.to_numpy(), (len(day_signals), 1)), index=day_signals.index,
                           columns=day_signals.columns)
    weights.iloc[entry:] = weight_after.to_numpy()
    return weights


def paired_straddles(straddle_data: pl.LazyFrame) -> pd.DataFrame:
    """Today's straddles (both legs quoted at the bar), one row per straddle and bar, before any moneyness filter."""
    horizons = list(EXPIRY_WINDOWS)
    out = straddle_data.filter(pl.len().over(['symbol', 'timestamp']) == 2).group_by(['symbol', 'timestamp'])
    out = out.agg([pl.col('mid_price').sum(), pl.col('spread').sum(), pl.col('transaction_fees').sum(),
                   pl.col('moneyness').first()] + [pl.col(f'eligible_{D}').all() for D in horizons])
    return out.sort(['symbol', 'timestamp']).collect().to_pandas()


def horizon_hold_pnl(straddles: pd.DataFrame, signals: pd.DataFrame, tranches: list, lower_bound: float,
                     upper_bound: float, target: str) -> pd.DataFrame:
    """Second-moment strategy with positions held over the forecasting horizon.
    A tranche is opened every day for each horizon D: the straddles inside the calibrated moneyness band and inside
    the expiry window of D at bar HOLD_SIGNAL_BAR, equally weighted, long or short according to the day's signal. It
    is closed HOLDING_DAYS[D] days later at the same bar. Open tranches are marked on every bar, including the move
    from the previous day's last quote to today's first one. The P&L of a horizon is the sum of its open tranches,
    each with weight 1 / HOLDING_DAYS[D]. Entry and exit each pay the transaction fee / the bid-ask spread.
    `tranches` is the list of open tranches; it is updated in place.
    Returns the bar-level P&L in the layout the rest of the script expects."""
    horizons = signals.columns.get_level_values('D').unique().tolist()
    markets = signals.columns.get_level_values('market').unique().tolist()
    aggregations = signals.index.get_level_values('aggregated').unique().tolist()
    bars = signals.index.get_level_values('timestamp').unique().sort_values()
    if getattr(bars, 'tz', None) is not None:
        bars = bars.tz_localize(None)
    bars = pd.DatetimeIndex(bars).astype('datetime64[ms]')
    signal_bar, entry = HOLD_SIGNAL_BAR, HOLD_SIGNAL_BAR + 1
    straddles = straddles.assign(timestamp=pd.to_datetime(straddles['timestamp']).astype('datetime64[ms]'))
    def wide(column):
        return straddles.pivot(index='timestamp', columns='symbol', values=column).reindex(bars)
    mid = wide('mid_price') if len(straddles) else pd.DataFrame(index=bars)
    mid = mid.where(mid > 0)
    fee = {'tc': wide('transaction_fees') if len(straddles) else pd.DataFrame(index=bars),
           'basc': (SPREAD_COST_SHARE * wide('spread') / mid) if len(straddles) else pd.DataFrame(index=bars)}
    columns = pd.MultiIndex.from_product([[f'{prefix}_{D}' for D in horizons
                                           for prefix in ['returns', 'returns_tc', 'returns_basc']], markets],
                                         names=[None, 'market'])
    pnl_today = dict([(a, pd.DataFrame(0., index=bars, columns=columns)) for a in aggregations])
    # ---- today's new tranches
    if len(bars) > entry and len(straddles):
        at_signal_bar = straddles[straddles['timestamp'] == bars[signal_bar]].set_index('symbol')
        in_band = at_signal_bar['moneyness'].between(lower_bound, upper_bound) & (at_signal_bar['mid_price'] > 0)
        for D in horizons:
            symbols = at_signal_bar.index[in_band & at_signal_bar[f'eligible_{D}'].astype(bool)].tolist()
            signs = dict([((a, m), float(signals.loc[a].iloc[entry][(D, m)])) for a in aggregations for m in markets])
            signs = dict([(key, 0. if np.isnan(value) else value) for key, value in signs.items()])
            if len(symbols) == 0 or not any(signs.values()):
                continue
            tranches.append(dict(D=D, symbols=symbols, signs=signs, last_mid=None, new=True,
                                 exit_time=bars[signal_bar] + pd.Timedelta(days=HOLDING_DAYS[D]),
                                 entry_fee=dict([(name, fee[name].reindex(columns=symbols).iloc[entry])
                                                 for name in fee])))
    # ---- mark every open tranche on today's bars
    for tranche in list(tranches):
        D, symbols, weight = tranche['D'], tranche['symbols'], 1. / HOLDING_DAYS[tranche['D']]
        prices = mid.reindex(columns=symbols)
        cost = dict([(name, pd.Series(0., index=bars)) for name in fee])
        closed = False
        if tranche['new']:
            filled = prices.ffill()
            returns = np.log(filled).diff()
            returns.iloc[:entry] = np.nan                       # the position starts at the signal bar's price
            for name in fee:
                cost[name].iloc[entry] = np.nanmean(tranche['entry_fee'][name].to_numpy(dtype=float)) \
                    if tranche['entry_fee'][name].notna().any() else 0.
            tranche['new'] = False
        else:
            previous = pd.DataFrame([tranche['last_mid'].reindex(symbols)],
                                    index=[bars[0] - pd.Timedelta(milliseconds=1)])
            filled = pd.concat([previous, prices]).ffill()
            returns = np.log(filled).diff().iloc[1:]
            exit_position = int(bars.searchsorted(tranche['exit_time']))
            if exit_position < len(bars):                       # the tranche is closed today
                exit_position = max(exit_position, 0)
                returns.iloc[exit_position + 1:] = np.nan
                for name in fee:
                    fees_now = fee[name].reindex(columns=symbols).ffill().iloc[exit_position]
                    fees_now = fees_now.fillna(tranche['entry_fee'][name])
                    cost[name].iloc[exit_position] = np.nanmean(fees_now.to_numpy(dtype=float)) \
                        if fees_now.notna().any() else 0.
                closed = True
        tranche_return = returns.mean(axis=1).fillna(0.)        # equal-weighted across the tranche's straddles
        for (a, m), sign in tranche['signs'].items():
            if sign == 0:
                continue
            gross = sign * weight * tranche_return
            pnl_today[a][(f'returns_{D}', m)] += gross
            pnl_today[a][(f'returns_tc_{D}', m)] += gross - abs(sign) * weight * cost['tc']
            pnl_today[a][(f'returns_basc_{D}', m)] += gross - abs(sign) * weight * cost['basc']
        tranche['last_mid'] = filled.iloc[-1]
        if closed:
            tranches.remove(tranche)
    out = pd.concat(pnl_today, names=['aggregated', 'timestamp'])
    out['target'] = target
    return out.set_index('target', append=True).reorder_levels(['aggregated', 'target', 'timestamp'])


def expiry_eligibility_expressions() -> list:
    """One boolean column per horizon D: True when the option's days to expiry fall in EXPIRY_WINDOWS[D]. When
    positions are held over the horizon, the option must also outlive the holding period."""
    days_to_expiry = pl.col('tau').mul(360).round(6)
    out = []
    for D, (low, high) in EXPIRY_WINDOWS.items():
        if HOLD_OVER_HORIZON:
            low = max(low, HOLDING_DAYS[D] + MIN_DAYS_TO_EXPIRY_AT_EXIT)
        out.append(days_to_expiry.is_between(lower_bound=low, upper_bound=high, closed='left').alias(f'eligible_{D}'))
    return out


def eligible_straddle_counts(straddles: pl.LazyFrame, horizons: list) -> dict:
    """Number of distinct straddles eligible for each horizon D (NaN when there is none)."""
    counts = straddles.group_by('symbol').agg([pl.col(f'eligible_{D}').any() for D in horizons])
    counts = counts.select([pl.col(f'eligible_{D}').sum() for D in horizons]).collect()
    if counts.height == 0:
        return dict([(D, np.nan) for D in horizons])
    counts = counts.row(0, named=True)
    return dict([(D, float(counts[f'eligible_{D}']) if counts[f'eligible_{D}'] else np.nan) for D in horizons])


def pnl_expressions(return_col: str, weight_col: str, delta_weight_col: str,
                    transaction_fee_col: str, basc_fee_col: str, D: str):
    gross = pl.col(return_col) * pl.col(weight_col)
    # a bar with no trade (or no fee quote) costs nothing; it must not blank the P&L of a position carried into it
    tf_cost = (pl.col(transaction_fee_col) * pl.col(delta_weight_col)).abs().fill_null(0)
    basc_cost = (pl.col(basc_fee_col) * pl.col(delta_weight_col)).abs().fill_null(0)
    return [(gross - tf_cost).fill_null(0).alias(f'returns_tc_{D}'),
            (gross - basc_cost).fill_null(0).alias(f'returns_basc_{D}'), gross.alias(f'returns_{D}')]


def return_straddle_table(data: pl.DataFrame, freq: str, underlying_price: pl.DataFrame, **kwargs) -> pl.DataFrame:
    non_expired_filter = pl.col('tau').ge(non_expired[freq])
    straddle_data = data.with_columns(pl.col('symbol').str.replace(r"-(C|P)$", "").alias('symbol'))
    straddle_data = straddle_data.sort(['symbol', 'type', 'timestamp'])
    straddle_data = \
        straddle_data.group_by_dynamic(index_column='timestamp', group_by=['symbol', 'type'], every=freq).last()
    straddle_data = straddle_data.sort(['symbol', 'type', 'timestamp'])
    straddle_data = (
        straddle_data.with_columns((((pl.col('spread').shift().add(pl.col('spread'))).mul(.5))
                                    .over(['symbol', 'type']))))
    straddle_data = straddle_data.filter(non_expired_filter)
    straddle_data = straddle_data.with_columns(expiry_eligibility_expressions())
    straddle_data = straddle_data.drop(['tau'])
    straddle_data = straddle_data.join(underlying_price.lazy(), on='timestamp', how='left')
    straddle_data = straddle_data.with_columns([
        pl.min_horizontal(
            pl.lit(DERIBIT_FEE_PREMIUM_PCT),
            (pl.lit(MIN_FEE_BPS_UNDERLYING) * pl.col('underlying_price') / pl.col('mid_price'))
        ).alias('transaction_fees'),
        (.5*(pl.col('spread')+pl.col('spread').shift().over(['symbol', 'type'])) /
         pl.col('mid_price')).alias('basc_fees')])
    return straddle_data


def return_spot_table(data: pl.DataFrame) -> pl.DataFrame:
    spot_data = data
    spot_data = spot_data.with_columns(pl.col('timestamp').cast(pl.Datetime(time_unit='ms')))
    spot_data = spot_data.select([pl.col('symbol').alias('underlying_name'), pl.col('spreads[0]'), pl.col('timestamp'),
                                  pl.col('mid_price[0]')])
    spot_data = spot_data.with_columns(pl.col('spreads[0]').alias('spread'),
                                       pl.col('mid_price[0]').alias('mid_price'),
                                       pl.col('mid_price[0]').log().diff().alias('returns'))
    spot_data = spot_data.sort('timestamp')
    spot_data = \
        spot_data.group_by_dynamic(index_column='timestamp', every=best_model_tags['returns'][-1].replace('in', ''),
                                   group_by='underlying_name').agg(pl.col('spread').last(),
                                                                   pl.col('mid_price').last(),
                                                                   pl.col('returns').sum())
    spot_data = spot_data.sort('timestamp')
    spot_data = spot_data.with_columns([pl.lit(2 * BINANCE_TF).alias('transaction_fees'),
                                        (SPREAD_COST_SHARE * .5*(pl.col('spread')+pl.col('spread').shift()) /
                                         pl.col('mid_price')).alias('basc_fees')])
    return spot_data


def return_underlying_price_table(data: pl.DataFrame) -> pl.DataFrame:
    out = data.with_columns(pl.col('timestamp').cast(pl.Datetime(time_unit='ms')))
    out = out.select([pl.col('timestamp'), pl.col('mid_price[0]').alias('underlying_price')])
    return out.sort('timestamp')


def make_filter(data, **kwargs) -> pd.DataFrame:
    if target == 'RV':
        mean_bar = kwargs['mean_bar'].droplevel(level=0, axis=1)
        filter = data[['forecast']].subtract(data[['values']].rename(columns={'values': 'forecast'}, level=0).shift())
        filter = filter.droplevel(level=0, axis=1)
        filter = filter.div(mean_bar)
    return filter


def returns_table(data, fees, underlying_name):
    markets = data.columns.get_level_values('market').unique().tolist()
    D = data.columns.get_level_values(0).unique().str.split('_').str[-1].tolist()
    cols = pd.MultiIndex.from_product([[underlying_name], D, markets], names=['underlying_name', 'D', 'market'])
    data.columns = cols
    data = data.dropna()
    if fees == 'transaction_fees':
        data = data.assign(transaction_fees=1, spread=0).set_index(['transaction_fees', 'spread'], append=True)
    elif fees == 'basc_fees':
        data = data.assign(transaction_fees=0, spread=1).set_index(['transaction_fees', 'spread'], append=True)
    else:
        data = data.assign(transaction_fees=0, spread=0).set_index(['transaction_fees', 'spread'], append=True)
    return data


def probabilistic_sharpe_ratio(pnl, benchmark=0.0, periods_per_year=360, vol_regime= False):
    if not vol_regime:
        pnl = pnl.droplevel(0, 1).groupby(['aggregated', 'target', 'transaction_fees', 'spread'])
    else:
        pnl = pnl.set_index(
            ['aggregated', 'target', 'transaction_fees', 'spread', 'vol_regime', 'timestamp', 'D', 'underlying_name',
             'market'])['returns'].unstack(['underlying_name', 'market']).droplevel(-2, 0)
        pnl = pnl.groupby(['aggregated', 'target', 'transaction_fees', 'spread', 'vol_regime', 'D'])
    sr = pnl.mean() / pnl.std(ddof=1)
    n = pnl.count()
    n = n.where(n >= 3, np.nan)
    benchmark_periodic = benchmark / np.sqrt(periods_per_year)
    skew = pnl.skew()
    kurt = pnl.agg(lambda x: x.kurtosis())
    denominator = np.sqrt(1 - skew * sr + ((kurt + 2) / 4) * sr ** 2)
    denominator = denominator.where((denominator > 0) & np.isfinite(denominator), np.nan)
    z = (sr - benchmark_periodic).mul(np.sqrt(n - 1)) / denominator
    return z.apply(norm.cdf)


def regime_annualisation(n_days: pd.DataFrame, periods_per_year: int = 360) -> pd.DataFrame:
    """Annualisation factor of a regime-conditional Sharpe ratio: sqrt(periods_per_year * p), where p is the share
    of the strategy's days that fall in the regime (days in the cell / days of that strategy over all regimes)."""
    other_levels = [name for name in n_days.index.names if name != 'vol_regime']
    share = n_days / n_days.groupby(level=other_levels, dropna=False).transform('sum')
    return (periods_per_year * share) ** .5


def _trim_leading_nan(series: pd.Series) -> pd.Series:
    """Drop rows before the series' first non-NaN value, so a trace's x-axis
    starts at that series' own first valid date instead of the shared index start."""
    first_valid = series.first_valid_index()
    if first_valid is None:
        return series.iloc[0:0]
    return series.loc[first_valid:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all P&L tables.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2025-07-01')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='./data_centre')
    parser.add_argument('--cpu_count', type=str, help='Number of CPUs to use to perform the analyses', default=1)
    args = parser.parse_args()
    print(args)
    ####################################################################################################################
    ## Order
    ####################################################################################################################
    pairs_order = CategoricalDtype(categories=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'], ordered=True)
    d_order = CategoricalDtype(categories=['1d', '1w', '1m'], ordered=True)
    volatility_order = CategoricalDtype(categories=['low', 'normal', 'high'], ordered=True)
    market_order = CategoricalDtype(categories=['$X^S$', '$X^{(S,O)}$'], ordered=True)
    date_range = pd.date_range(start=args.start_date, end=args.end_date, freq='D', inclusive='left', tz='UTC')
    mase = pd.read_csv(f'{args.target_path}/crypto_option_spot/metric.csv')
    mase = mase.set_index('D')
    model_ranking = mase.groupby(by=[pd.Grouper(key='target'),
                                     pd.Grouper(key='regression'),
                                     pd.Grouper(key='freq')])[['MASE']].mean()
    best_model_tags = model_ranking.groupby('target').idxmin().to_dict()['MASE']
    mase_comparison = \
        model_ranking.groupby(['target', 'regression', 'freq'])['MASE'].mean().groupby(level='target').idxmin()
    start_ts = pd.Timestamp(args.start_date, tz='UTC')
    end_ts = pd.Timestamp(args.end_date, tz='UTC')
    data_per_target = dict()
    forecasts_parquet = f'{args.target_path}/crypto_option_spot/forecasts.parquet'
    forecasts_csv = f'{args.target_path}/crypto_option_spot/forecasts.csv'
    for target, values in mase_comparison.items():
        regression = best_model_tags[target][1]
        freq = best_model_tags[target][-1]
        lazy = pl.scan_parquet(forecasts_parquet)
        lazy = \
            lazy.filter((pl.col('regression') == regression.lower()) & (pl.col('freq') == freq) &
                        (pl.col('target') == target))
        lazy = lazy.filter((pl.col('timestamp') >= pl.lit(start_ts)) & (pl.col('timestamp') < pl.lit(end_ts)))
        tmp = lazy.collect().to_pandas()
        tmp = tmp.rename(columns={'y_hat': 'forecast', 'y': 'values'})
        tmp = tmp.set_index('timestamp')
        tmp.index = pd.to_datetime(tmp.index, utc=True)
        tmp = pd.pivot_table(data=tmp, columns=['underlying_name', 'target', 'D', 'market'],
                             values=['values', 'forecast'], index='timestamp')
        data_per_target[target] = tmp
    d_values = data_per_target['RV'].columns.get_level_values('D').str.lower()
    annualisation = pd.Series(360 / d_values.map(D_HORIZONS), index=data_per_target['RV'].columns)
    data_per_target['RV'] = data_per_target['RV'].mul(annualisation, axis=1)
    svix_parquet = f'{args.target_path}/crypto_option_spot/svix.parquet'
    lazy = pl.scan_parquet(svix_parquet)
    lazy = lazy.filter((pl.col('timestamp') >= pl.lit(start_ts)) & (pl.col('timestamp') < pl.lit(end_ts)))
    svix = lazy.collect().to_pandas()
    svix = svix.set_index(['aggregated', 'underlying_name', 'D', 'timestamp'])
    svix_agg = svix.xs(1, axis=0, level=0, drop_level=False)
    svix_non_agg = svix.xs(0, axis=0, level=0, drop_level=False)
    svix = pd.concat([svix_agg, svix_non_agg])
    svix = svix.loc[svix.index.get_level_values('timestamp').normalize().isin(date_range), :]
    svix = svix.groupby(
        ['aggregated', 'underlying_name', 'D',
         pd.Grouper(level='timestamp', freq=best_model_tags['RV'][-1])]).last()
    rv = data_per_target['RV'].xs('forecast', axis=1, level=0).xs('RV', axis=1, level=1)
    rv = rv.resample(best_model_tags['RV'][-1]).sum().replace(0, np.nan)
    rv.columns = rv.columns.set_levels(rv.columns.levels[-2].str.lower(), level=-2).dropna()
    mean_bar = data_per_target['RV'].xs('RV', axis=1, level=2)['values'].apply(
        lambda x: x.resample(f'{DAYS_CONVERTER[x.name[1]]}D').mean().shift()).ffill()
    mean_bar = mean_bar.reindex(data_per_target['RV'].index).ffill()
    mean_bar = mean_bar.loc[mean_bar.index.normalize().isin(date_range), :]
    target_underlying_names = list(itertools.product(['RV', 'returns'], list(pairs_order.categories)))
    file_cutoff = pd.Timestamp(args.start_date)
    pnl = \
        dict([(target, dict([(underlying_name, list()) for underlying_name in pairs_order.categories]))
              for target in ['RV', 'returns']])
    forecast_underlying_name = pd.IndexSlice
    number_freq_period_in_day = pd.to_timedelta('1D')//pd.to_timedelta(best_model_tags['RV'][-1])
    for target_underlying_name in tqdm(target_underlying_names):
        target, underlying_name = tuple(target_underlying_name)
        underlying_name_mean_bar = mean_bar.loc[:, mean_bar.columns.get_level_values(0) == underlying_name]
        if target == 'RV':
            files = glob.glob(f'{args.target_path}/crypto_option_spot/option_helper_files/*/{underlying_name}/*')
            files.sort()
            underlying_name_dates = list(map(lambda x: x.split('/')[-1], files))
            underlying_name_dates = \
                [date for date in underlying_name_dates if
                 (pd.to_datetime(date) + relativedelta(days=1) > file_cutoff) &
                 (pd.to_datetime(date) < pd.Timestamp(args.end_date))]
            open_tranches = list()
            for idx_file in range(1, len(underlying_name_dates)):
                yesterday_date = pd.to_datetime(underlying_name_dates[idx_file-1])
                yesterday_file = pl.scan_csv(f'{args.target_path}/crypto_option_spot/option_helper_files/'
                                             f'{underlying_name_dates[idx_file-1].split("-")[0]}/{underlying_name}/'
                                             f'{underlying_name_dates[idx_file-1]}',
                                             schema_overrides={'timestamp': pl.Datetime(time_unit='ms'),
                                                               'mid_price': pl.Float64,
                                                               'spread': pl.Float64, 'moneyness': pl.Float64,
                                                               'tau': pl.Float64})
                yesterday_file = (
                    yesterday_file.select(['symbol', 'timestamp', 'mid_price', 'spread', 'moneyness', 'tau', 'type']))
                yesterday_spot_underlying_file = \
                    pl.scan_parquet(f'{args.target_path}/{underlying_name}/*/spot/{underlying_name_dates[idx_file-1]}')
                yesterday_underlying_price = \
                    return_underlying_price_table(yesterday_spot_underlying_file).collect()
                yesterday_straddle_data = \
                    return_straddle_table(data=yesterday_file,
                                          freq=best_model_tags['RV'][-1].replace('in', ''),
                                          underlying_price=yesterday_underlying_price)
                try:
                    yesterday_straddle_data = yesterday_straddle_data.collect().lazy()
                except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as exc:
                    print(f"WARNING: RV {underlying_name} day {idx_file} "
                          f"({underlying_name_dates[idx_file]}) failed reading/parsing yesterday's option file "
                          f"({exc.__class__.__name__}: {exc}) -- skipping this day.", flush=True)
                    continue
                yesterday_mean_bar = \
                    underlying_name_mean_bar[underlying_name_mean_bar.index.date == yesterday_date.date()]
                if yesterday_mean_bar.dropna().empty:
                    continue
                yesterday_data = data_per_target['RV'][data_per_target['RV'].index.date == yesterday_date.date()]
                tradable_assets = yesterday_data.isnull().all().groupby('underlying_name').sum()
                tradable_assets = tradable_assets[tradable_assets == 0].index
                if underlying_name not in tradable_assets:
                    continue
                yesterday_data = \
                    yesterday_data.loc[:, forecast_underlying_name[:, underlying_name, target, :]]
                yesterday_data = yesterday_data.droplevel(level=[1, 2], axis=1)
                yesterday_filter = make_filter(data=yesterday_data, mean_bar=yesterday_mean_bar)
                yesterday_svix_underlying_name = \
                    svix.loc[(svix.index.get_level_values(3).date == yesterday_date.date()) &
                              (svix.index.get_level_values(1) == underlying_name), ['svix']]
                yesterday_svix_underlying_name = yesterday_svix_underlying_name.unstack('D').droplevel(axis=0, level=1)
                yesterday_svix_underlying_name = yesterday_svix_underlying_name.droplevel(axis=1, level=0)
                yesterday_svix_underlying_name.columns = yesterday_svix_underlying_name.columns.str.upper()
                study = \
                    optuna.create_study(direction='maximize',
                                        study_name=f'Parameter optimisation RV: Process '
                                                   f'{underlying_name}_{underlying_name_dates[idx_file]} has started.',
                                        sampler=RandomSampler(123))
                objective = partial(objective_daily, option_chain_data=yesterday_straddle_data,
                                    data=yesterday_data.loc[:, forecast_underlying_name['forecast', :, :]],
                                    filter_signal=yesterday_filter, svix=yesterday_svix_underlying_name,
                                    target=target, tradable_assets=tradable_assets)
                try:
                    study.optimize(objective, n_trials=N_TRIALS // 20, callbacks=[callback], show_progress_bar=True,
                                   n_jobs=N_JOBS, gc_after_trial=True)
                    lower_bound_moneyness, upper_bound_moneyness, threshold_filter = \
                        study.user_attrs['lower_bound_atm'], \
                        study.user_attrs['upper_bound_atm'], study.user_attrs['threshold_filter']
                    file = pl.scan_csv(f'{args.target_path}/crypto_option_spot/option_helper_files/'
                                       f'{underlying_name_dates[idx_file].split("-")[0]}/{underlying_name}/'
                                       f'{underlying_name_dates[idx_file]}',
                                       schema_overrides={'timestamp': pl.Datetime(time_unit='ms'), 'mid_price': pl.Float64,
                                                         'spread': pl.Float64, 'moneyness': pl.Float64,
                                                         'tau': pl.Float64})
                    file = file.select(['symbol', 'timestamp', 'mid_price', 'spread', 'moneyness', 'tau', 'type'])
                    today_spot_underlying_file = \
                        pl.scan_parquet(f'{args.target_path}/{underlying_name}/*/spot/{underlying_name_dates[idx_file]}')
                    today_underlying_price = return_underlying_price_table(today_spot_underlying_file).collect()
                    straddle_data = return_straddle_table(data=file, freq=best_model_tags['RV'][1:][-1].replace('in', ''),
                                                          underlying_price=today_underlying_price)
                    straddle_data = straddle_data.sort(['symbol', 'type', 'timestamp'])
                    if HOLD_OVER_HORIZON:
                        today_straddles = paired_straddles(straddle_data)
                    straddle_data = straddle_data.filter(
                        pl.col('moneyness').is_between(lower_bound=lower_bound_moneyness,
                                                       upper_bound=upper_bound_moneyness))
                    straddle_data = straddle_data.filter(pl.len().over(['symbol', pl.col('timestamp')]) == 2)
                    today_date = pd.to_datetime(underlying_name_dates[idx_file])
                    today_mean_bar = underlying_name_mean_bar[mean_bar.index.date == today_date.date()]
                    today_data = \
                        data_per_target['RV'].loc[data_per_target['RV'].index.date == today_date.date()]
                    today_data = today_data.loc[:, forecast_underlying_name[:, underlying_name, target, :]]
                    today_data = today_data.droplevel(level=[1, 2], axis=1)
                    today_filter = make_filter(data=today_data, mean_bar=today_mean_bar)
                    today_filter = today_filter.abs().gt(threshold_filter).astype(int)
                    markets = today_data.columns.get_level_values('market').unique().tolist()
                    today_svix_underlying_name = \
                        svix.loc[(svix.index.get_level_values(3).date == today_date.date()) &
                                (svix.index.get_level_values(1) == underlying_name), ['svix']]
                    today_svix_underlying_name = today_svix_underlying_name.unstack('D').droplevel(axis=0, level=1)
                    today_svix_underlying_name = today_svix_underlying_name.droplevel(axis=1, level=0)
                    today_svix_underlying_name.columns = today_svix_underlying_name.columns.str.upper()
                    today_forecast_flat = today_data[['forecast']].droplevel(axis=1, level=0)
                    if not today_forecast_flat.index.is_unique:
                        print(f"WARNING: duplicate timestamp(s) in today's forecast data "
                              f"for {underlying_name} -- dropping duplicates (keep last).")
                        today_forecast_flat = today_forecast_flat[~today_forecast_flat.index.duplicated(keep='last')]
                    today_svix_dedup = today_svix_underlying_name
                    if not today_svix_underlying_name.index.get_level_values('timestamp').is_unique:
                        print(f"WARNING: duplicate timestamp(s) in today's svix data "
                              f"for {underlying_name} -- dropping duplicates (keep last).")
                        today_svix_dedup = \
                            today_svix_underlying_name[~today_svix_underlying_name.index.duplicated(keep='last')]
                    today_forecast_broadcast = \
                        today_forecast_flat.reindex(today_svix_dedup.index.get_level_values('timestamp'))
                    today_forecast_broadcast.index = today_svix_dedup.index
                    today_vrp_underlying_name = today_svix_dedup.sub(today_forecast_broadcast)
                    today_vrp_underlying_name = -1*np.sign(today_vrp_underlying_name)
                    today_vrp_underlying_name = today_vrp_underlying_name.mul(today_filter, level='timestamp')
                    today_vrp_underlying_name = today_vrp_underlying_name.groupby('aggregated', group_keys=False)
                    today_signals_underlying_name = today_vrp_underlying_name.apply(lagged_signal)
                    if HOLD_OVER_HORIZON:
                        straddle_data = horizon_hold_pnl(
                            straddles=today_straddles, signals=today_signals_underlying_name,
                            tranches=open_tranches, lower_bound=lower_bound_moneyness,
                            upper_bound=upper_bound_moneyness, target=target)
                    else:
                        today_number_of_unique_straddles = straddle_data.select(pl.col('symbol').n_unique()).collect().item()
                        if today_number_of_unique_straddles == 0:
                            print(f"WARNING: RV {underlying_name} day {idx_file} ({underlying_name_dates[idx_file]}) "
                                  f"has no straddle inside the calibrated moneyness band -- skipping this day.",
                                  flush=True)
                            continue
                        today_active_count = today_signals_underlying_name.ne(0).sum(axis=1)
                        today_straddles_per_horizon = eligible_straddle_counts(
                            straddle_data, today_signals_underlying_name.columns.get_level_values('D').unique().tolist())
                        today_signals_underlying_name = (
                            today_signals_underlying_name
                            .div(today_signals_underlying_name.columns.get_level_values('D').map(
                                today_straddles_per_horizon).to_numpy(), axis=1)
                            .div(today_active_count.replace(0, np.nan), axis=0)
                            .fillna(0)
                        )
                        today_delta_weights = \
                            today_signals_underlying_name.groupby('aggregated', group_keys=False).apply(weight_changes)
                        today_delta_weights = \
                            pl.from_pandas(today_delta_weights.stack('market').reset_index()).with_columns(
                                pl.col('timestamp').cast(pl.Datetime('ms')),
                                pl.col('aggregated').cast(pl.Int64), pl.col('market').cast(pl.Int64)
                            ).lazy()
                        today_signals_underlying_name = \
                            pl.from_pandas(today_signals_underlying_name.stack('market').reset_index()).with_columns(
                                pl.col('timestamp').cast(pl.Datetime('ms')),
                                pl.col('aggregated').cast(pl.Int64), pl.col('market').cast(pl.Int64)
                            ).lazy()
                        straddle_data = straddle_data.join(today_signals_underlying_name, on='timestamp', how='left')
                        straddle_data = straddle_data.sort(['symbol', 'aggregated', 'market', 'timestamp'])
                        straddle_data = \
                            straddle_data.group_by(['symbol', 'aggregated', 'market', 'timestamp'], maintain_order=True)
                        straddle_data = straddle_data.agg([pl.col('mid_price').sum(), pl.col('spread').sum(),
                                                           pl.col('transaction_fees').sum()] +
                                                          [pl.col(d).last() for d in
                                                           today_filter.columns.get_level_values('D').unique().tolist()] +
                                                          [pl.col(f'eligible_{d}').last() for d in
                                                           today_filter.columns.get_level_values('D').unique().tolist()])
                        straddle_data = \
                            straddle_data.with_columns([(pl.col('mid_price').log().diff().over(['symbol', 'aggregated',
                                                                                                'market'])).alias('returns'),
                                                        (SPREAD_COST_SHARE * pl.col('spread') / pl.col('mid_price')).alias('basc_fees')])
                        straddle_data = straddle_data.join(today_delta_weights, how='left', suffix='_delta_weights',
                                                           on=['aggregated', 'market', 'timestamp'])
                        exprs = []
                        for D in today_filter.columns.get_level_values('D').unique().tolist():
                            eligible = pl.col(f'eligible_{D}')
                            exprs += [pl.when(eligible).then((pl.col('returns') * pl.col(D) - (pl.col('transaction_fees') *
                                      pl.col(f'{D}_delta_weights')).abs()).fill_null(0)).alias(f'returns_tc_{D}'),
                                      pl.when(eligible).then((pl.col('returns') * pl.col(D) - (pl.col('basc_fees') *
                                      pl.col(f'{D}_delta_weights')).abs()).fill_null(0)).alias(f'returns_basc_{D}'),
                                      pl.when(eligible).then(pl.col('returns') * pl.col(D)).alias(f'returns_{D}')]
                        straddle_data = straddle_data.sort(['symbol', 'aggregated', 'market', 'timestamp'])
                        straddle_data = straddle_data.with_columns(exprs)
                        straddle_data = \
                            straddle_data.drop(['returns', 'spread', 'transaction_fees', 'mid_price'] +
                                               today_filter.columns.get_level_values('D').unique().tolist() +
                                               ['basc_fees'] +
                                               [f'{d}_delta_weights' for d in
                                                today_filter.columns.get_level_values('D').unique().tolist()])
                        straddle_data = straddle_data.sort(['symbol', 'aggregated', 'market', 'timestamp'])
                        returns_cols = \
                            [f'{prefix}_{D}' for D in today_filter.columns.get_level_values('D').unique().tolist()
                             for prefix in ['returns', 'returns_tc', 'returns_basc']]
                        straddle_data = straddle_data.group_by(['aggregated', 'market', 'timestamp'])
                        straddle_data = \
                            (straddle_data.agg([pl.col(c).mean().fill_null(0).alias(c) for c in returns_cols])
                             .with_columns(pl.lit(target).alias('target')))
                        straddle_data = \
                            straddle_data.collect().to_pandas().set_index(['aggregated', 'market', 'target', 'timestamp'])
                        straddle_data = straddle_data.unstack('market')
                        straddle_data = straddle_data.dropna(how='all', axis=1)
                    returns_cols = pd.MultiIndex.from_product(
                        [[f'returns_{D}' for D in today_filter.columns.get_level_values('D').unique().tolist()], markets])
                    returns = _reindex_cols(straddle_data, returns_cols)
                    returns_tc_cols = pd.MultiIndex.from_product(
                        [[f'returns_tc_{D}' for D in today_filter.columns.get_level_values('D').unique().tolist()],
                         markets])
                    returns_tc = _reindex_cols(straddle_data, returns_tc_cols)
                    returns_basc_cols = \
                        pd.MultiIndex.from_product(
                            [[f'returns_basc_{D}' for D in today_filter.columns.get_level_values('D').unique().tolist()],
                             markets])
                    returns_basc = _reindex_cols(straddle_data, returns_basc_cols)
                    returns = returns_table(returns, fees=None, underlying_name=underlying_name)
                    returns_tc = returns_table(returns_tc, fees='transaction_fees', underlying_name=underlying_name)
                    returns_basc = returns_table(returns_basc, fees='basc_fees', underlying_name=underlying_name)
                    returns_tmp = pd.concat([returns, returns_tc, returns_basc])
                    pnl[target][underlying_name].append(returns_tmp)
                    del returns, returns_tc, returns_basc, straddle_data, study
                    if idx_file % 20 == 0:
                        gc.collect()
                except (ValueError, KeyError, pl.exceptions.ColumnNotFoundError, pl.exceptions.ComputeError,
                       pl.exceptions.InvalidOperationError) as exc:
                    print(f"WARNING: RV {underlying_name} day {idx_file} "
                          f"({underlying_name_dates[idx_file]}) optuna optimization produced no usable "
                          f"result ({exc.__class__.__name__}: {exc}) -- skipping this day.", flush=True)
                    continue
        elif target == 'returns':
            files = glob.glob(f'{args.target_path}/{underlying_name}/*/spot/*')
            files.sort()
            underlying_name_dates = list(map(lambda x: x.split('/')[-1], files))
            underlying_name_dates = [f for f in underlying_name_dates if
                                     pd.to_datetime(f.split('/')[-1]) + relativedelta(days=1) >= file_cutoff][:-1]
            sign_history = list()
            for idx_file in range(1, len(underlying_name_dates)-1):
                yesterday_date = pd.to_datetime(underlying_name_dates[idx_file - 1])
                yesterday_file = pl.scan_parquet(f'{args.target_path}/{underlying_name}/*/spot/'
                                                 f'{underlying_name_dates[idx_file - 1]}')
                yesterday_spot = return_spot_table(yesterday_file)
                try:
                    yesterday_spot = yesterday_spot.collect().lazy()
                except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as exc:
                    print(f"WARNING: returns {underlying_name} day {idx_file} "
                          f"({underlying_name_dates[idx_file]}) failed reading/parsing yesterday's spot file "
                          f"({exc.__class__.__name__}: {exc}) -- skipping this day.", flush=True)
                    continue
                yesterday_data = \
                    data_per_target['returns'][data_per_target['returns'].index.date == yesterday_date.date()]
                yesterday_data = yesterday_data.loc[:, forecast_underlying_name['forecast']].dropna(how='all', axis=1)
                tradable_assets = yesterday_data.columns.get_level_values('underlying_name').unique().shape[0]
                try:
                    yesterday_data = \
                        yesterday_data.loc[:, forecast_underlying_name[underlying_name, target, :, :]]
                except KeyError:
                    continue
                yesterday_data = yesterday_data.droplevel(level=[0, 1], axis=1)
                study = \
                    optuna.create_study(direction='maximize',
                                        study_name=f'Parameter optimisation returns: Process '
                                                   f'{underlying_name}_{underlying_name_dates[idx_file]} has started.',
                                        sampler=RandomSampler(123))
                objective = partial(objective_daily, spot_data=yesterday_spot, data=yesterday_data, target=target,
                                    tradable_assets=tradable_assets)
                try:
                    study.optimize(objective, n_trials=N_TRIALS // 20, callbacks=[callback], show_progress_bar=True,
                                   n_jobs=N_JOBS, gc_after_trial=True)
                    threshold_filter = study.user_attrs['threshold_filter']
                except (ValueError, KeyError, pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as exc:
                    print(f"WARNING: returns {underlying_name} day {idx_file} "
                          f"({underlying_name_dates[idx_file]}) optuna optimization produced no usable "
                          f"result ({exc.__class__.__name__}: {exc}) -- skipping this day.", flush=True)
                    continue
                today_date = pd.to_datetime(underlying_name_dates[idx_file])
                today_file = \
                    pl.scan_parquet(f'{args.target_path}/{underlying_name}/{today_date.year}/spot/'
                                    f'{underlying_name_dates[idx_file]}')
                today_spot = return_spot_table(today_file)
                today_data = \
                    data_per_target['returns'].loc[data_per_target['returns'].index.date == today_date.date(),
                    forecast_underlying_name[:, :, target, :]]
                today_data = today_data.dropna(axis=1, how='all')
                tradable_assets = today_data.columns.get_level_values('underlying_name').unique().shape[0]
                try:
                    today_forecast = \
                        today_data.loc[:, forecast_underlying_name['forecast', underlying_name, target, :]]
                except KeyError:
                    continue
                today_forecast = today_forecast.droplevel(level=[0, 1, 2], axis=1)
                markets = today_forecast.columns.get_level_values('market').unique().tolist()
                signals = np.sign(today_forecast)
                quantiles = yesterday_data.abs().quantile(threshold_filter).reindex(today_forecast.columns)
                signal_filter = (today_forecast.abs().gt(quantiles, axis=1)).astype(int)
                signals = lagged_signal(signals.mul(signal_filter)).fillna(0)
                if HOLD_OVER_HORIZON:
                    signals = horizon_weights(signals, sign_history)
                else:
                    active_count = signals.ne(0).sum(axis=1)
                    signals = signals.div(active_count.replace(0, np.nan), axis=0).fillna(0)
                today_delta_weights = weight_changes(signals)
                signals = signals.stack('market').reset_index().assign(aggregated='')
                today_delta_weights = today_delta_weights.stack('market').reset_index().assign(aggregated='')
                signals = pl.from_pandas(signals).with_columns(
                    pl.col('timestamp').cast(pl.Datetime('ms')), pl.col('aggregated'),
                    pl.col('market').cast(pl.Int64)).lazy()
                today_delta_weights = pl.from_pandas(today_delta_weights).with_columns(
                    pl.col('timestamp').cast(pl.Datetime('ms')), pl.col('aggregated'),
                    pl.col('market').cast(pl.Int64)).lazy()
                today_spot = today_spot.join(signals, on='timestamp', how='left')
                spot_data = today_spot.sort(['underlying_name', 'aggregated', 'market', 'timestamp'])
                today_spot = today_spot.group_by(['underlying_name', 'aggregated', 'market', 'timestamp'],
                                                 maintain_order=True)
                today_spot = today_spot.agg([pl.col('returns').sum(), pl.col('basc_fees').last(),
                                             pl.col('transaction_fees').last()]+
                                            [pl.col(d).last() for d in
                                             today_forecast.columns.get_level_values('D').unique().tolist()])
                today_spot = today_spot.join(today_delta_weights, on=['aggregated', 'market', 'timestamp'],
                                             how='left', suffix='_delta_weights')
                exprs = []
                for D in today_forecast.columns.get_level_values('D').unique().tolist():
                    exprs += pnl_expressions(return_col='returns', weight_col=D, delta_weight_col=f'{D}_delta_weights',
                                             transaction_fee_col='transaction_fees', basc_fee_col='basc_fees', D=D)
                today_spot = today_spot.sort(['underlying_name', 'aggregated', 'market', 'timestamp'])
                today_spot = today_spot.with_columns(exprs)
                today_spot = today_spot.drop(
                    ['returns', 'transaction_fees']+
                    today_forecast.columns.get_level_values('D').unique().tolist()+
                    ['basc_fees']+
                    [f'{d}_delta_weights' for d in today_forecast.columns.get_level_values('D').unique().tolist()])
                today_spot = today_spot.sort(['underlying_name', 'aggregated', 'market', 'timestamp'])
                returns_cols = \
                    [f'{prefix}_{D}' for D in today_forecast.columns.get_level_values('D').unique().tolist() for prefix in
                     ['returns', 'returns_tc', 'returns_basc']]
                today_spot = today_spot.group_by(['aggregated', 'market', 'timestamp'])
                today_spot = \
                    today_spot.agg([pl.col(c).mean().alias(c) for c in returns_cols]).with_columns(
                        pl.lit(target).alias('target'))
                today_spot = \
                    today_spot.collect().to_pandas().set_index(['aggregated', 'target', 'market', 'timestamp'])
                today_spot = today_spot.unstack('market')
                returns_cols = \
                    pd.MultiIndex.from_product(
                        [[f'returns_{D}' for D in today_forecast.columns.get_level_values('D').unique().tolist()],
                         markets])
                returns = _reindex_cols(today_spot, returns_cols)
                returns_tc_cols = pd.MultiIndex.from_product(
                    [[f'returns_tc_{D}' for D in today_forecast.columns.get_level_values('D').unique().tolist()],
                     markets])
                returns_tc = _reindex_cols(today_spot, returns_tc_cols)
                returns_basc_cols = pd.MultiIndex.from_product(
                    [[f'returns_basc_{D}' for D in today_forecast.columns.get_level_values('D').unique().tolist()],
                     markets])
                returns_basc = _reindex_cols(today_spot, returns_basc_cols)
                returns = returns_table(returns, fees=None, underlying_name=underlying_name)
                returns_tc = returns_table(returns_tc, fees='transaction_fees', underlying_name=underlying_name)
                returns_basc = returns_table(returns_basc, fees='basc_fees', underlying_name=underlying_name)
                pnl[target][underlying_name].append(pd.concat([returns, returns_tc, returns_basc]))
                del returns, returns_tc, returns_basc, today_spot, study
                if idx_file % 20 == 0:
                    gc.collect()
    daily_grouper_level = [pd.Grouper(level='aggregated'), pd.Grouper(level='target'),
                           pd.Grouper(level='transaction_fees'), pd.Grouper(level='spread'),
                           pd.Grouper(level='timestamp', freq='D')]
    for underlying_name, target in list(itertools.product(list(pairs_order.categories), ['RV', 'returns'])):
        try:
            index_names = pnl[target][underlying_name][0].index.names
            combined = \
                pd.concat(
                    [df.reset_index() for df in pnl[target][underlying_name]], ignore_index=True).set_index(index_names)
            pnl[target][underlying_name] = combined.groupby(daily_grouper_level).sum()
            gc.collect()
        except IndexError:
            continue
    for target in ['RV', 'returns']:
        pnl[target] = pd.concat(pnl[target], axis=1)
    grouper_level = [pd.Grouper(level='aggregated'), pd.Grouper(level='target'), pd.Grouper(level='transaction_fees'),
                     pd.Grouper(level='spread')]
    ####################################################################################################################
    ## Plot difference in cumulative P&L
    ####################################################################################################################
    title_dd = dict([(False, 'First Moment-based Strategy'), (True, 'Second Moment-based Strategy')])
    fee_keys = [(0, 0), (1, 0), (0, 1)]
    market_styles = [(0, 'solid', 'X<sup>S</sup>'), (1, 'dash', 'X<sup>(S,O)</sup>')]
    pnl_figures = [('RV', 1, '', 'pnl_second_moment.pdf'),
                   ('RV', 0, ' (non-aggregated Crypto SVIX)', 'pnl_second_moment_non_aggregated.pdf'),
                   ('returns', '', '', 'pnl_first_moment.pdf')]
    for target, aggregated_flag, title_suffix, file_name in pnl_figures:
        tmp_cum_pnl = pnl[target].groupby(grouper_level, group_keys=False).apply(lambda x: x.cumsum())
        tmp_cum_pnl = tmp_cum_pnl.droplevel('target')
        fig = make_subplots(rows=3, cols=len(d_order.categories), shared_xaxes=False,
                            vertical_spacing=0.14, horizontal_spacing=0.08,
                            row_titles=['P&L<sup><span>&#8212;</span>0</sup>',
                                        'P&L<sup><span>&#8212;</span>TF</sup>',
                                        'P&L<sup><span>&#8212;</span>BASC</sup>'],
                            column_titles=[f'<i>{D}</i>' for D in d_order.categories])
        for idx, pair in enumerate(list(pairs_order.categories)):
            pair_slice = tmp_cum_pnl.xs(pair, axis=1, level='underlying_name', drop_level=True)
            pair_slice = pair_slice.groupby(['aggregated', 'transaction_fees', 'spread']).transform('ffill')
            pair_slice = pair_slice.xs(aggregated_flag, axis=0, level='aggregated', drop_level=True)
            pair_slice = pair_slice.droplevel(0, 1)
            for row, fee_key in enumerate(fee_keys):
                fee_slice = pair_slice.xs(fee_key, axis=0, level=['transaction_fees', 'spread'])
                for col, D in enumerate(d_order.categories):
                    for market, dash, _ in market_styles:
                        if (D.upper(), market) not in fee_slice.columns:
                            continue
                        s = _trim_leading_nan(fee_slice[(D.upper(), market)])
                        fig.add_trace(trace=go.Scatter(x=s.index.astype(str), y=s.values, mode='lines',
                                                       line=dict(color=px.colors.qualitative.Plotly[idx], width=30,
                                                                 dash=dash),
                                                       name=f'{pair.replace("USDT", "/USDT")}', showlegend=False),
                                      col=col + 1, row=row + 1)
        # legend: first row = X^S (solid) for every pair, second row = X^(S,O) (dashed) under the same pair
        for _, dash, market_label in market_styles:
            for idx, pair in enumerate(list(pairs_order.categories)):
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                         line=dict(color=px.colors.qualitative.Plotly[idx], width=30, dash=dash),
                                         name=f'\u00a0{pair.replace("USDT", "/USDT")} - {market_label}\u00a0\u00a0\u00a0',
                                         showlegend=True), row=1, col=1)
        fig.update_layout(title=dict(text=f'<b>P&L Curves - {title_dd[target == "RV"]}{title_suffix}</b>',
                                     x=0.5, xanchor='center', y=0.98, yanchor='top', pad=dict(b=250),
                                     font=dict(size=10 * TITLE_FONT_SIZE), automargin=True,
                                     yref='container'), font=dict(size=5 * LABEL_AXIS_FONT_SIZE), width=WIDTH * 15,
                          height=5 * len(d_order.categories) * HEIGHT, paper_bgcolor='white', plot_bgcolor='white',
                          margin=dict(t=30 * TITLE_FONT_SIZE),
                          legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5,
                                      entrywidth=0.25, entrywidthmode='fraction',
                                      font=dict(size=12 * LABEL_AXIS_FONT_SIZE), itemwidth=30 * LABEL_AXIS_FONT_SIZE))
        fig.update_annotations(font=dict(size=13 * LABEL_AXIS_FONT_SIZE))
        fig.update_xaxes(tickangle=45, showline=True, linewidth=2, linecolor='black', mirror=False, showgrid=False,
                         title='', nticks=6, tickfont=dict(size=10 * LABEL_AXIS_FONT_SIZE))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False, showgrid=False,
                         tickfont=dict(size=10 * LABEL_AXIS_FONT_SIZE))
        fig.write_image(f'./figures/{file_name}', format='pdf')
    proba_sharpe_rv = probabilistic_sharpe_ratio(pnl=pnl['RV'])
    proba_sharpe_returns = probabilistic_sharpe_ratio(pnl=pnl['returns'])
    proba_sharpe = pd.concat([proba_sharpe_returns, proba_sharpe_rv])
    proba_sharpe = proba_sharpe.rename(columns=str.lower, level='D')
    sharpe_mean = pd.concat([pnl['RV'].droplevel(0, 1).groupby(grouper_level).mean(),
                             pnl['returns'].droplevel(0, 1).groupby(grouper_level).mean()])
    sharpe_std = pd.concat([pnl['RV'].droplevel(0, 1).groupby(grouper_level).std(),
                            pnl['returns'].droplevel(0, 1).groupby(grouper_level).std()])
    no_fees = pd.IndexSlice
    sharpe = sharpe_mean.div(sharpe_std).mul(360 ** .5).round(2)
    sharpe = sharpe.rename(columns=str.lower, level='D')
    fee_variants = [(0, 0, ''), (1, 0, '_tf'), (0, 1, '_basc')]
    proba_sharpe = proba_sharpe.round(2)
    d_label_map = {d: rf'\textit{{{d}}}' for d in d_order.categories}
    market_label_map = {0: '$X^S$', 1: '$X^{(S,O)}$'}
    sharpe = sharpe.rename(columns=market_label_map, level='market')
    sharpe = sharpe.rename(columns=d_label_map, level='D')
    proba_sharpe = proba_sharpe.rename(columns=market_label_map, level='market')
    proba_sharpe = proba_sharpe.rename(columns=d_label_map, level='D')
    sharpe.index = sharpe.index.set_levels(['0.0', '1.0', ''], level='aggregated')
    proba_sharpe.index = proba_sharpe.index.set_levels(['', '0.0', '1.0'], level='aggregated')
    sharpe_columns_order = \
        pd.MultiIndex.from_product([pairs_order.categories.tolist(), market_order.categories.tolist()],
                                   names=['underlying_name', 'market'])
    sharpe_index_order_aggregated = pd.MultiIndex.from_tuples(
        [(agg, target, rf'\textit{{{d}}}') for agg in ['', str(float(1))] for target in ['returns', 'RV']
         for d in d_order.categories.tolist()
         if not ((target == 'returns' and agg in [str(float(1))]) or (target == 'RV' and agg == ''))],
        names=['aggregated', 'target', 'D']
    )
    sharpe_index_order_non_aggregated = \
        pd.MultiIndex.from_tuples([(agg, target, rf'\textit{{{d}}}') for agg in [str(float(0))] for target in ['RV']
                                   for d in d_order.categories.tolist()], names=['aggregated', 'target', 'D'])
    for aggregated, (tf_flag, spread_flag, fee_suffix) in itertools.product([False, True], fee_variants):
        tmp_sharpe = sharpe.xs((tf_flag, spread_flag), axis=0, level=[-2, -1]).stack('D')
        tmp_proba_sharpe = \
            proba_sharpe.xs((tf_flag, spread_flag), axis=0, level=[-2, -1]).stack('D')
        tmp_proba_sharpe.index = tmp_proba_sharpe.index.set_levels(tmp_proba_sharpe.index.levels[0].astype(str),
                                                                   level='aggregated')
        if not aggregated:
            tmp_sharpe = tmp_sharpe.reindex(sharpe_index_order_non_aggregated).reindex(sharpe_columns_order, axis=1)
            tmp_proba_sharpe = \
                tmp_proba_sharpe.reindex(sharpe_index_order_non_aggregated).reindex(sharpe_columns_order, axis=1)
        else:
            tmp_sharpe = tmp_sharpe.reindex(sharpe_index_order_aggregated).reindex(sharpe_columns_order, axis=1)
            tmp_proba_sharpe = \
                tmp_proba_sharpe.reindex(sharpe_index_order_aggregated).reindex(sharpe_columns_order, axis=1)
        tmp_sharpe = tmp_sharpe.astype(object)
        sharpe_xs = tmp_sharpe.xs('$X^S$', level=1, axis=1, drop_level=False)
        sharpe_x_so = tmp_sharpe.xs('$X^{(S,O)}$', level=1, axis=1, drop_level=False)
        xs_wins = pd.DataFrame(sharpe_xs.droplevel('market', axis=1).to_numpy(dtype=float)
                               >= sharpe_x_so.droplevel('market', axis=1).to_numpy(dtype=float),
                               index=tmp_sharpe.index, columns=sharpe_xs.columns)
        tmp_proba_sharpe = tmp_proba_sharpe.map(lambda x: rf'$\mathbf{{{x:.2f}}}$' if x > .95 else x)
        tmp_proba_sharpe = tmp_proba_sharpe.astype(object)
        def _bold_if(value, is_winner):
            if pd.isna(value):
                return value
            return rf'$\mathbf{{{value:.2f}}}$' if is_winner else f'${value:.2f}$'
        tmp_sharpe.loc[:, pd.IndexSlice[:, '$X^S$']] = np.vectorize(_bold_if, otypes=[object])(
            sharpe_xs.to_numpy(), xs_wins.to_numpy())
        tmp_sharpe.loc[:, pd.IndexSlice[:, '$X^{(S,O)}$']] = np.vectorize(_bold_if, otypes=[object])(
            sharpe_x_so.to_numpy(), ~xs_wins.to_numpy())
        with open(f'{args.target_path}/crypto_option_spot/'
                  f'sharpe{dict([(False, "_non_aggregated"), (True, "_aggregated")])[aggregated]}{fee_suffix}.txt',
                  'w') as sharpe_file:
            sharpe_file.writelines(tmp_sharpe.droplevel('aggregated', axis=0).to_latex(escape=False, float_format='%.2f'))
        with open(f'{args.target_path}/crypto_option_spot/'
                  f'proba_sharpe{dict([(False, "_non_aggregated"), (True, "_aggregated")])[aggregated]}{fee_suffix}.txt',
                  'w') as proba_sharpe_file:
            proba_sharpe_file.writelines(
                tmp_proba_sharpe.droplevel('aggregated', axis=0).to_latex(escape=False, float_format='%.2f'))
    start = datetime.fromisoformat(str(args.start_date))
    end = datetime.fromisoformat(str(args.end_date))
    rv = pl.scan_parquet(f'{args.target_path}/*/*/spot/*').select(['symbol', 'timestamp', 'mid_price[0]'])
    rv = rv.rename({'symbol': 'underlying_name',
                    'mid_price[0]': 'mid_price'}).with_columns([pl.col('timestamp').cast(pl.Datetime(time_unit='ms'))])
    rv = rv.filter(pl.col('timestamp').is_between(pl.lit(start), pl.lit(end), closed='left'))
    rv = rv.sort(['underlying_name', 'timestamp']).group_by_dynamic(index_column='timestamp',
                                                                    every='5m',
                                                                    group_by='underlying_name')
    rv = rv.agg((pl.col('mid_price').log().diff().pow(2).sum()).alias('RV'))
    rv = rv.sort('timestamp')
    rv = rv.group_by('timestamp', maintain_order=True).agg(pl.col('RV').mean().alias('RV'))
    rv = pl.concat([rv.group_by_dynamic(index_column='timestamp', every=f'{DAYS_CONVERTER[D]}d').agg(pl.col('RV').sum())
                   .with_columns(pl.lit(D).alias('D')) for D in ['1D', '1W', '1M']])
    full_ts = (
        pl.DataFrame({'timestamp': pl.datetime_range(start=start, end=end, interval='1d', eager=True, closed='left',
                                                     time_unit='ms')}))
    rv = \
        full_ts.lazy().join(rv.select('D').unique(), how='cross').join(rv, on=['timestamp', 'D'],
                                                                       how='left').sort(['D', 'timestamp'])
    rv = rv.with_columns(pl.col('RV').cast(pl.Float64).fill_null(strategy='forward').over('D'))
    rv = rv.with_columns([pl.col('RV').quantile(q).over('D').alias(f'q{int(q * 100)}') for q in [0.33, 0.66]])
    rv = rv.with_columns((pl.when(pl.col('RV') <= pl.col('q33'))
                         .then(pl.lit(rf'\textit{{{VOL_REGIMES[0].lower()}}}'))
                         .when(pl.col('RV') <= pl.col('q66'))
                         .then(pl.lit(rf'\textit{{{VOL_REGIMES[1].lower()}}}'))
                         .otherwise(pl.lit(rf'\textit{{{VOL_REGIMES[2].lower()}}}')).alias('vol_regime')))
    rv = rv.drop(['q33', 'q66', 'RV'])
    pnl_returns = \
        pnl['returns'].droplevel(0, 1).stack('D', future_stack=True).stack('market', future_stack=True).stack()
    pnl_returns = pd.DataFrame(pnl_returns).rename(columns={0: 'returns'}).reset_index()
    pnl_returns = pnl_returns.merge(rv.collect().to_pandas(), on=['D', 'timestamp'], how='inner')
    pnl_returns_proba_sharpe = probabilistic_sharpe_ratio(pnl_returns, vol_regime=True)
    pnl_rv = \
        pnl['RV'].droplevel(0, 1).stack('D', future_stack=True).stack('market', future_stack=True).stack()
    pnl_rv = pd.DataFrame(pnl_rv).rename(columns={0: 'returns'}).reset_index()
    pnl_rv = pnl_rv.merge(rv.collect().to_pandas(), on=['D', 'timestamp'], how='inner')
    pnl_rv_proba_sharpe = probabilistic_sharpe_ratio(pnl_rv, vol_regime=True)
    pnl_proba_sharpe_vol_regime = pd.concat([pnl_returns_proba_sharpe, pnl_rv_proba_sharpe])
    pnl_proba_sharpe_vol_regime = \
        pnl_proba_sharpe_vol_regime.rename(index=lambda x: rf'\textit{{{x.lower()}}}', level='D')
    pnl_proba_sharpe_vol_regime = pnl_proba_sharpe_vol_regime.rename(columns=market_label_map, level='market')
    pnl_returns = pnl_returns.groupby(
        ['aggregated', 'target', 'transaction_fees', 'spread', 'vol_regime', 'D', 'underlying_name', 'market'],
        dropna=False)[['returns']]
    pnl_returns_mean = pnl_returns.mean()
    pnl_returns_std = pnl_returns.std()
    sharpe_pnl_per_vol_regime_returns = \
        (pnl_returns_mean / pnl_returns_std).mul(regime_annualisation(pnl_returns.count()))
    pnl_rv = \
        pnl_rv.groupby(['aggregated', 'target', 'transaction_fees', 'spread', 'vol_regime', 'D', 'underlying_name',
                        'market'])[['returns']]
    pnl_rv_mean = pnl_rv.mean()
    pnl_rv_std = pnl_rv.std()
    sharpe_pnl_per_vol_regime_rv = (pnl_rv_mean / pnl_rv_std).mul(regime_annualisation(pnl_rv.count()))
    sharpe_pnl_per_vol_regime = pd.concat([sharpe_pnl_per_vol_regime_returns, sharpe_pnl_per_vol_regime_rv])
    sharpe_pnl_per_vol_regime = sharpe_pnl_per_vol_regime.rename(columns={'returns': 'sharpe'})
    sharpe_pnl_per_vol_regime.index = \
        sharpe_pnl_per_vol_regime.index.set_levels(sharpe_pnl_per_vol_regime.index.levels[0].fillna(''),
                                                   level='aggregated')
    sharpe_pnl_per_vol_regime = \
        pd.pivot(sharpe_pnl_per_vol_regime.reset_index(),
                 index=['aggregated', 'target', 'vol_regime', 'D', 'transaction_fees', 'spread'],
                 columns=['underlying_name', 'market'], values='sharpe').round(2)
    sharpe_pnl_per_vol_regime = sharpe_pnl_per_vol_regime.rename(columns=market_label_map, level='market')
    sharpe_pnl_per_vol_regime = \
        sharpe_pnl_per_vol_regime.rename(index=lambda x: rf'\textit{{{x.lower()}}}', level='D')
    sharpe_pnl_per_vol_regime_aggregated = pd.MultiIndex.from_tuples(
        [(agg, target, rf'\textit{{{vol_regime}}}', rf'\textit{{{d}}}') for agg in ['', 1]
         for target in ['returns', 'RV']
         for vol_regime in volatility_order.categories.tolist()
         for d in d_order.categories.tolist()
         if not ((target == 'returns' and agg in [1]) or (target == 'RV' and agg == ''))],
        names=['aggregated', 'target', 'vol_regime', 'D']
    )
    sharpe_pnl_per_vol_regime_non_aggregated = pd.MultiIndex.from_tuples(
        [(agg, target, rf'\textit{{{vol_regime}}}', rf'\textit{{{d}}}')
         for agg in [0]
         for target in ['RV']
         for vol_regime in volatility_order.categories.tolist()
         for d in d_order.categories.tolist()], names=['aggregated', 'target', 'vol_regime', 'D']
    )
    for aggregated, (tf_flag, spread_flag, fee_suffix) in itertools.product([False, True], fee_variants):
        fee_sharpe = sharpe_pnl_per_vol_regime.xs((tf_flag, spread_flag), axis=0, level=['transaction_fees', 'spread'])
        fee_proba_sharpe = pnl_proba_sharpe_vol_regime.xs((tf_flag, spread_flag), axis=0, level=['transaction_fees', 'spread'])
        if not aggregated:
            tmp_sharpe = fee_sharpe.reindex(sharpe_pnl_per_vol_regime_non_aggregated)
            tmp_sharpe = tmp_sharpe.reindex(sharpe_columns_order, axis=1)
            tmp_proba_sharpe = fee_proba_sharpe.reindex(sharpe_pnl_per_vol_regime_non_aggregated).round(2)
            tmp_proba_sharpe = tmp_proba_sharpe.reindex(sharpe_columns_order, axis=1)
        else:
            tmp_sharpe = fee_sharpe.reindex(sharpe_pnl_per_vol_regime_aggregated)
            tmp_sharpe = tmp_sharpe.reindex(sharpe_columns_order, axis=1)
            tmp_proba_sharpe = fee_proba_sharpe.reindex(sharpe_pnl_per_vol_regime_aggregated).round(2)
            tmp_proba_sharpe = tmp_proba_sharpe.reindex(sharpe_columns_order, axis=1)
        tmp_sharpe = tmp_sharpe.round(2)
        tmp_sharpe = tmp_sharpe.astype(object)
        sharpe_per_volatility_regime_xs = tmp_sharpe.xs('$X^S$', level=1, axis=1, drop_level=False)
        sharpe_per_volatility_regime_x_so = tmp_sharpe.xs('$X^{(S,O)}$', level=1, axis=1, drop_level=False)
        xs_wins = pd.DataFrame(
            sharpe_per_volatility_regime_xs.droplevel('market', axis=1).to_numpy(dtype=float)
            >= sharpe_per_volatility_regime_x_so.droplevel('market', axis=1).to_numpy(dtype=float),
            index=tmp_sharpe.index, columns=sharpe_per_volatility_regime_xs.columns,
        )
        tmp_proba_sharpe = tmp_proba_sharpe.map(lambda x: rf'$\mathbf{{{x:.2f}}}$' if x > .95 else x)
        tmp_proba_sharpe = tmp_proba_sharpe.astype(object)
        def _bold_if(value, is_winner):
            if pd.isna(value):
                return value
            return rf'$\mathbf{{{value:.2f}}}$' if is_winner else f'${value:.2f}$'
        tmp_sharpe.loc[:, pd.IndexSlice[:, '$X^S$']] = np.vectorize(_bold_if, otypes=[object])(
            sharpe_per_volatility_regime_xs.to_numpy(), xs_wins.to_numpy())
        tmp_sharpe.loc[:, pd.IndexSlice[:, '$X^{(S,O)}$']] = np.vectorize(_bold_if, otypes=[object])(
            sharpe_per_volatility_regime_x_so.to_numpy(), ~xs_wins.to_numpy())
        with open(f'{args.target_path}/crypto_option_spot/'
                  f'sharpe_per_volatility{dict([(False, "_non_aggregated"), (True, "_aggregated")])[aggregated]}{fee_suffix}.txt',
                  'w') as sharpe_file:
            sharpe_file.writelines(
                tmp_sharpe.droplevel('aggregated', axis=0).to_latex(escape=False, float_format='%.2f'))
        with open(f'{args.target_path}/crypto_option_spot/'
                  f'proba_sharpe_per_volatility'
                  f'{dict([(False, "_non_aggregated"),(True, "_aggregated")])[aggregated]}{fee_suffix}.txt',
                  'w') as proba_sharpe_file:
            proba_sharpe_file.writelines(
                tmp_proba_sharpe.droplevel('aggregated', axis=0).to_latex(escape=False, float_format='%.2f'))