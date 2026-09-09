import pdb
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import gc
import resource
import platform
import optuna
from dateutil.relativedelta import relativedelta
from pandas.api.types import CategoricalDtype
from functools import partial
from optuna.samplers import RandomSampler
import argparse
import itertools
from functools import reduce
from tqdm.auto import tqdm
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import polars as pl
from constants import LOW_ATM_MONEYNESS, HIGH_ATM_MONEYNESS, N_TRIALS, N_JOBS, SYMBOLS, DAYS_CONVERTER, VOL_REGIMES, \
                       TITLE_FONT_SIZE, LABEL_AXIS_FONT_SIZE, WIDTH, HEIGHT
DERIBIT_FEE_PREMIUM_PCT = 0.125
BINANCE_TF = 0.001
MIN_FEE_BPS_UNDERLYING = 0.0006


def _reindex_cols(df, cols):
    out = df.reindex(columns=cols)
    out.columns.names = df.columns.names
    return out


non_expired = {'15m': pd.to_timedelta('15min') / pd.to_timedelta('360D'),
               '30m': pd.to_timedelta('30min') / pd.to_timedelta('360D'),
               '1h': pd.to_timedelta('1h') / pd.to_timedelta('360D')}


def callback(study, trial) -> None:
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
            lambda x, y: (x.mul(y)).shift(), y=filter_signal.abs().gt(param['threshold_filter']).astype(int))
        atm_filter = \
            pl.col('moneyness').is_between(lower_bound=param['lower_bound_atm'], upper_bound=param['upper_bound_atm'])
        option_chain_data = option_chain_data.filter(atm_filter)
        option_chain_data = option_chain_data.filter(pl.len().over(['symbol', 'timestamp']) == 2)
        number_of_unique_straddles = option_chain_data.select(pl.col('symbol').n_unique()).collect().item()
        if number_of_unique_straddles > 0:
            active_count = signals.ne(0).sum(axis=1)
            signals = signals.div(number_of_unique_straddles).div(active_count.replace(0, np.nan), axis=0).fillna(0)
            delta_weights = signals.groupby('aggregated', group_keys=False).apply(lambda x: x.diff())
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
                                                       pl.col('transaction_fees').last()]+
                                                      [pl.col(d).last() for d in available_ds])
            option_chain_data = \
                option_chain_data.with_columns(
                    [(pl.col('mid_price').log().diff().over(['symbol', 'aggregated', 'market'])).alias('returns'),
                                                (pl.col('spread') / pl.col('mid_price')).alias('basc_fees')])
            option_chain_data = option_chain_data.join(delta_weights, on=['aggregated', 'market', 'timestamp'],
                                                       how='left', suffix='_delta_weights')
            exprs = []
            for D in filter_signal.columns.get_level_values('D').unique().tolist():
                exprs += [((pl.col('returns') * pl.col(D) - (pl.col('transaction_fees') *
                           pl.col(f'{D}_delta_weights')).abs()).fill_null(0)).alias(f'returns_tc_{D}'),
                          ((pl.col('returns') * pl.col(D) - (pl.col('basc_fees') *
                           pl.col(f'{D}_delta_weights')).abs()).fill_null(0)).alias(f'returns_basc_{D}'),
                          (pl.col('returns') * pl.col(D)).alias(f'returns_{D}')]
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
            option_chain_data = option_chain_data.agg([pl.col(c).mean().alias(c) for c in returns_cols])
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
        quantiles = forecast.apply(lambda x, y: x.quantile(y), y=param['threshold_filter'])
        signal_filter = (forecast.abs().gt(quantiles, axis=1)).astype(int)
        signals = (signals.mul(signal_filter)).shift().fillna(0)
        active_count = signals.ne(0).sum(axis=1)
        signals = signals.div(active_count.replace(0, np.nan), axis=0).fillna(0)
        delta_weights = signals.diff()
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
            exprs += [((pl.col('returns') * pl.col(D) - (pl.col('transaction_fees') *
                                                         pl.col(f'{D}_delta_weights')).abs()).fill_null(0)).alias(
                f'returns_tc_{D}'),
                      ((pl.col('returns') * pl.col(D) - (pl.col('basc_fees') *
                                                         pl.col(f'{D}_delta_weights')).abs()).fill_null(0)).alias(
                          f'returns_basc_{D}'),
                      (pl.col('returns') * pl.col(D)).alias(f'returns_{D}')]
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
    trial.set_user_attr(key='sharpe', value=sharpe)
    trial.set_user_attr(key='threshold_filter', value=param['threshold_filter'])
    trial.set_user_attr(key='target', value=kwargs['target'])
    return trial.user_attrs['sharpe']


def return_straddle_table(data: pl.DataFrame, freq: str, underlying_price: pl.DataFrame, **kwargs) -> pl.DataFrame:
    non_expired_filter = pl.col('tau').ge(non_expired[freq])
    straddle_data = data.with_columns(pl.col('symbol').str.replace(r"-(C|P)$", "").alias('symbol'))
    straddle_data = straddle_data.sort(['symbol', 'type', 'timestamp'])
    straddle_data = \
        straddle_data.group_by_dynamic(index_column='timestamp', group_by=['symbol', 'type'], every=freq).last()
    straddle_data = straddle_data.sort(['symbol', 'type', 'timestamp'])
    straddle_data = (
        straddle_data.with_columns((((pl.col('spread').shift().add(pl.col('spread'))).mul(.5)).over('symbol'))))
    straddle_data = straddle_data.filter(non_expired_filter)
    straddle_data = straddle_data.drop(['tau'])
    straddle_data = straddle_data.join(underlying_price.lazy(), on='timestamp', how='left')
    straddle_data = straddle_data.with_columns([
        pl.min_horizontal(
            pl.lit(DERIBIT_FEE_PREMIUM_PCT),
            (pl.lit(MIN_FEE_BPS_UNDERLYING) * pl.col('underlying_price') / pl.col('mid_price'))
        ).alias('transaction_fees'),
        (.5*(pl.col('spread')+pl.col('spread').shift()) /
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
    spot_data.sort('timestamp')
    spot_data = \
        spot_data.group_by_dynamic(index_column='timestamp', every=best_model_tags['returns'][-1].replace('in', ''),
                                   group_by='underlying_name').agg(pl.col('spread').last(),
                                                                   pl.col('mid_price').last(),
                                                                   pl.col('returns').sum())
    spot_data = spot_data.sort('timestamp')
    spot_data = spot_data.with_columns([pl.lit(2 * BINANCE_TF).alias('transaction_fees'),
                                        (.5*(pl.col('spread')+pl.col('spread').shift()) /
                                         pl.col('mid_price')).alias('basc_fees')])
    return spot_data


def return_underlying_price_table(data: pl.DataFrame) -> pl.DataFrame:
    out = data.with_columns(pl.col('timestamp').cast(pl.Datetime(time_unit='ms')))
    out = out.select([pl.col('timestamp'), pl.col('mid_price[0]').alias('underlying_price')])
    return out.sort('timestamp')


def make_filter(data, **kwargs) -> pd.DataFrame:
    if target == 'RV':
        mean_bar = kwargs['mean_bar']
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
    pairs_order = CategoricalDtype(categories=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT'], ordered=True)
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
        if Path(forecasts_parquet).exists():
            lazy = pl.scan_parquet(forecasts_parquet)
            lazy = \
                lazy.filter((pl.col('regression') == regression.lower()) & (pl.col('freq') == freq) &
                            (pl.col('target') == target))
            lazy = lazy.filter((pl.col('timestamp') >= pl.lit(start_ts)) & (pl.col('timestamp') < pl.lit(end_ts)))
            tmp = lazy.collect().to_pandas()
        else:
            chunks = pd.read_csv(f'{args.target_path}/crypto_option_spot/forecasts.csv', chunksize=1_000_000)
            tmp = pd.concat(chunk.query('target == @values[0] and regression == @values[1] and freq == @values[-1]')
                            for chunk in chunks)
        tmp = tmp.rename(columns={'y_hat': 'forecast', 'y': 'values'})
        tmp = tmp.set_index('timestamp')
        tmp.index = pd.to_datetime(tmp.index, utc=True)
        tmp = pd.pivot_table(data=tmp, columns=['underlying_name', 'target', 'D', 'market'],
                             values=['values', 'forecast'], index='timestamp')
        data_per_target[target] = tmp
    scaling_factor = pd.to_timedelta('360D') // pd.to_timedelta(best_model_tags['RV'][-1])
    rv_idx = pd.IndexSlice
    data_per_target['RV'].loc[:, rv_idx[:, :, 'RV', :, :]] = \
        data_per_target['RV'].loc[:, rv_idx[:, :, 'RV', :, :]].pow(.5).mul(scaling_factor ** .5)
    svix_parquet = f'{args.target_path}/crypto_option_spot/svix.parquet'
    svix_csv = f'{args.target_path}/crypto_option_spot/svix.csv'
    if Path(svix_parquet).exists():
        lazy = pl.scan_parquet(svix_parquet)
        lazy = lazy.filter((pl.col('timestamp') >= pl.lit(start_ts)) & (pl.col('timestamp') < pl.lit(end_ts)))
        svix = lazy.collect().to_pandas()
        if 'index' in svix.columns:
            svix = svix.drop(columns=['index'])
    else:
        svix = pd.read_csv(svix_csv, index_col='index')
        svix['timestamp'] = pd.to_datetime(svix['timestamp'], utc=True)
        svix = svix.loc[svix['timestamp'].dt.normalize().isin(date_range), :]
    svix['svix'] = svix['svix']/100
    svix = svix.set_index(['aggregated', 'underlying_name', 'D', 'timestamp'])
    svix_agg = \
        pd.concat([svix.xs(1, axis=0, level=0,
                           drop_level=False).droplevel(axis=0, level=1).assign(underlying_name=underlying_name)
                   for underlying_name in SYMBOLS]).set_index('underlying_name', append=True)
    svix_agg = svix_agg.swaplevel(-1, 1).swaplevel(-1, 2)
    svix_non_agg = svix.xs(0, axis=0, level=0, drop_level=False)
    svix = pd.concat([svix_agg, svix_non_agg])
    svix = svix.loc[svix.index.get_level_values('timestamp').normalize().isin(date_range), :]
    rv = data_per_target['RV'].xs('forecast', axis=1, level=0).xs('RV', axis=1, level=1)
    rv = rv.resample(best_model_tags['RV'][-1]).sum().replace(0, np.nan)
    rv.columns = rv.columns.set_levels(rv.columns.levels[-2].str.lower(), level=-2).dropna()
    mean_bar = data_per_target['RV'].xs('RV', axis=1, level=2)['values'].apply(
        lambda x: x.resample(f'{DAYS_CONVERTER[x.name[1]]}D').mean().shift()).ffill()
    mean_bar = mean_bar.reindex(data_per_target['RV'].index).ffill()
    mean_bar = mean_bar.loc[mean_bar.index.normalize().isin(date_range), :]
    target_underlying_names = list(itertools.product(['RV', 'returns'], SYMBOLS))
    file_cutoff = pd.Timestamp(args.start_date) # + relativedelta(days=DAYS_CONVERTER['3M']+1)
    pnl = \
        dict([(target, dict([(underlying_name, list()) for underlying_name in SYMBOLS]))
              for target in ['RV', 'returns']])
    forecast_underlying_name = pd.IndexSlice
    number_freq_period_in_day = pd.to_timedelta('1D')//pd.to_timedelta(best_model_tags['RV'][-1])
    for target_underlying_name in tqdm(target_underlying_names):
        target, underlying_name = tuple(target_underlying_name)
        if target == 'RV':
            files = glob.glob(f'{args.target_path}/crypto_option_spot/option_helper_files/*/{underlying_name}/*')
            files.sort()
            underlying_name_dates = list(map(lambda x: x.split('/')[-1], files))
            underlying_name_dates = \
                [date for date in underlying_name_dates if
                 (pd.to_datetime(date) + relativedelta(days=1) >= file_cutoff) &
                 (pd.to_datetime(date) < pd.Timestamp(args.end_date))]
            for idx_file in range(1, len(underlying_name_dates)):
                yesterday_date = pd.to_datetime(underlying_name_dates[idx_file-1])
                yesterday_file = pl.scan_csv(f'{args.target_path}/crypto_option_spot/option_helper_files/'
                                             f'{underlying_name_dates[idx_file-1].split("-")[0]}/{underlying_name}/'
                                             f'{underlying_name_dates[idx_file-1]}',
                                             schema_overrides={'timestamp': pl.Datetime(time_unit='ms')})
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
                yesterday_mean_bar = mean_bar[mean_bar.index.date == yesterday_date.date()][underlying_name]
                yesterday_data = data_per_target['RV'][data_per_target['RV'].index.date == yesterday_date.date()]
                tradable_assets = yesterday_data.columns.get_level_values('underlying_name').unique().shape[0]
                if yesterday_data.isnull().all().sum() > 0:
                    continue
                yesterday_data = \
                    yesterday_data.loc[:, forecast_underlying_name[:, underlying_name, target, :]]
                yesterday_data = yesterday_data.droplevel(level=[1, 2], axis=1)
                yesterday_filter = make_filter(data=yesterday_data, mean_bar=yesterday_mean_bar)
                yesterday_svix_underlying_name = \
                    svix.loc[(svix.index.get_level_values(3).date == yesterday_date.date())&
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
                                    target=target, scaling_factor=scaling_factor, tradable_assets=tradable_assets)
                try:
                    study.optimize(objective, n_trials=N_TRIALS // 20, callbacks=[callback], show_progress_bar=True,
                                   n_jobs=N_JOBS, gc_after_trial=True)
                    lower_bound_moneyness, upper_bound_moneyness, threshold_filter = \
                        study.user_attrs['lower_bound_atm'], \
                        study.user_attrs['upper_bound_atm'], study.user_attrs['threshold_filter']
                except (ValueError, KeyError, pl.exceptions.ColumnNotFoundError) as exc:
                    print(f"WARNING: RV {underlying_name} day {idx_file} "
                          f"({underlying_name_dates[idx_file]}) optuna optimization produced no usable "
                          f"result ({exc.__class__.__name__}: {exc}) -- skipping this day.", flush=True)
                    continue
                file = pl.scan_csv(f'{args.target_path}/crypto_option_spot/option_helper_files/'
                                   f'{underlying_name_dates[idx_file].split("-")[0]}/{underlying_name}/'
                                   f'{underlying_name_dates[idx_file]}',
                                   schema_overrides={'timestamp': pl.Datetime(time_unit='ms')})
                file = file.select(['symbol', 'timestamp', 'mid_price', 'spread', 'moneyness', 'tau', 'type'])
                today_spot_underlying_file = \
                    pl.scan_parquet(f'{args.target_path}/{underlying_name}/*/spot/{underlying_name_dates[idx_file]}')
                today_underlying_price = return_underlying_price_table(today_spot_underlying_file).collect()
                straddle_data = return_straddle_table(data=file, freq=best_model_tags['RV'][1:][-1].replace('in', ''),
                                                      underlying_price=today_underlying_price)
                straddle_data = straddle_data.sort(['symbol', 'type', 'timestamp'])
                straddle_data = \
                    straddle_data.filter(pl.len().over(
                        ['symbol', 'type', pl.col('timestamp').dt.date()]) == number_freq_period_in_day)
                straddle_data = straddle_data.filter(pl.len().over(['symbol', pl.col('timestamp')]) == 2)
                today_date = pd.to_datetime(underlying_name_dates[idx_file])
                today_mean_bar = mean_bar[mean_bar.index.date == today_date.date()][underlying_name]
                today_data = \
                    data_per_target['RV'].loc[data_per_target['RV'].index.date == today_date.date()]
                today_data = today_data.loc[:, forecast_underlying_name[:, underlying_name, target, :]]
                today_data = today_data.droplevel(level=[1, 2], axis=1)
                today_filter = make_filter(data=today_data, mean_bar=today_mean_bar)
                today_filter = today_filter.gt(threshold_filter).astype(int)
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
                today_signals_underlying_name = today_vrp_underlying_name.apply(lambda x: x.shift())
                today_number_of_unique_straddles = straddle_data.select(pl.col('symbol').n_unique()).collect().item()
                today_signals_underlying_name = today_signals_underlying_name.div(today_number_of_unique_straddles)
                today_delta_weights = \
                    today_signals_underlying_name.groupby('aggregated', group_keys=False).apply(lambda x: x.diff())
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
                                                   pl.col('transaction_fees').last()] +
                                                  [pl.col(d).last() for d in
                                                   today_filter.columns.get_level_values('D').unique().tolist()])
                straddle_data = \
                    straddle_data.with_columns([(pl.col('mid_price').log().diff().over(['symbol', 'aggregated',
                                                                                        'market'])).alias('returns'),
                                                (pl.col('spread')/pl.col('mid_price')).alias('basc_fees')])
                straddle_data = straddle_data.join(today_delta_weights, how='left', suffix='_delta_weights',
                                                   on=['aggregated', 'market', 'timestamp'])
                exprs = []
                for D in today_filter.columns.get_level_values('D').unique().tolist():
                    exprs += [((pl.col('returns') * pl.col(D) - (pl.col('transaction_fees') *
                                                                 pl.col(f'{D}_delta_weights')).abs()).fill_null(
                        0)).alias(f'returns_tc_{D}'),
                              ((pl.col('returns') * pl.col(D) - (pl.col('basc_fees') *
                                                                 pl.col(f'{D}_delta_weights')).abs()).fill_null(
                                  0)).alias(f'returns_basc_{D}'),
                              (pl.col('returns') * pl.col(D)).alias(f'returns_{D}')]
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
                    (straddle_data.agg([pl.col(c).mean().alias(c) for c in returns_cols])
                     .with_columns(pl.lit(target).alias('target')))
                straddle_data = \
                    straddle_data.collect().to_pandas().set_index(['aggregated', 'market', 'target', 'timestamp'])
                straddle_data = straddle_data.unstack('market')
                straddle_data = straddle_data.dropna(how='all', axis=1)
                returns_cols = pd.MultiIndex.from_product(
                    [[f'returns_{D}' for D in today_filter.columns.get_level_values('D').unique().tolist()], markets])
                try:
                    returns = _reindex_cols(straddle_data, returns_cols)
                except KeyError:
                    continue
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
        elif target == 'returns':
            files = glob.glob(f'{args.target_path}/{underlying_name}/*/spot/*')
            files.sort()
            underlying_name_dates = list(map(lambda x: x.split('/')[-1], files))
            underlying_name_dates = [f for f in underlying_name_dates if
                                     pd.to_datetime(f.split('/')[-1]) + relativedelta(days=1) >= file_cutoff][:-1]
            for idx_file in range(1, len(underlying_name_dates)-1):
                yesterday_date = pd.to_datetime(underlying_name_dates[idx_file - 1])
                yesterday_file = pl.scan_parquet(f'{args.target_path}/{underlying_name}/*/spot/'
                                                 f'{underlying_name_dates[idx_file - 1]}')
                yesterday_spot = return_spot_table(yesterday_file)
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
                                    scaling_factor=scaling_factor, tradable_assets=tradable_assets)
                try:
                    study.optimize(objective, n_trials=N_TRIALS // 20, callbacks=[callback], show_progress_bar=True,
                                   n_jobs=N_JOBS, gc_after_trial=True)
                    threshold_filter = study.user_attrs['threshold_filter']
                except (ValueError, KeyError) as exc:
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
                quantiles = today_forecast.apply(lambda x, y: x.quantile(y), y=threshold_filter)
                signal_filter = (today_forecast.abs().gt(quantiles)).astype(int)
                signals = (signals.mul(signal_filter)).shift().fillna(0)
                active_count = signals.ne(0).sum(axis=1)
                signals = signals.div(active_count.replace(0, np.nan), axis=0).fillna(0)
                today_delta_weights = signals.diff()
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
                    exprs += [((pl.col('returns') * pl.col(D) - (pl.col('transaction_fees') *
                                                                 pl.col(f'{D}_delta_weights')).abs()).fill_null(
                        0)).alias(f'returns_tc_{D}'),
                              ((pl.col('returns') * pl.col(D) - (pl.col('basc_fees') *
                                                                 pl.col(f'{D}_delta_weights')).abs()).fill_null(
                                  0)).alias(f'returns_basc_{D}'),
                              (pl.col('returns') * pl.col(D)).alias(f'returns_{D}')]
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
    for underlying_name, target in list(itertools.product(SYMBOLS, ['RV', 'returns'])):
        index_names = pnl[target][underlying_name][0].index.names
        combined = \
            pd.concat(
                [df.reset_index() for df in pnl[target][underlying_name]], ignore_index=True).set_index(index_names)
        pnl[target][underlying_name] = combined.groupby(daily_grouper_level).sum()
        gc.collect()
    pnl = pd.concat(pd.concat([reduce(lambda l, r: l.join(r, how='outer'), pnl[target].values())])
                    for target in pnl.keys())
    grouper_level = [pd.Grouper(level='aggregated'), pd.Grouper(level='target'), pd.Grouper(level='transaction_fees'),
                     pd.Grouper(level='spread')]
    pnl = pnl.groupby(grouper_level + [pd.Grouper(level='timestamp', freq='D')]).sum()
    ####################################################################################################################
    ## Plot difference in cumulative P&L
    ####################################################################################################################
    title_dd = dict([(False, 'First Moment-based Strategy'), (True, 'Second Moment-based Strategy')])
    diff_cum_pnl = pnl.groupby(grouper_level, group_keys=False).apply(lambda x: x.cumsum())
    diff_cum_pnl = diff_cum_pnl.xs(1, level='market', axis=1).sub(diff_cum_pnl.xs(0, level='market', axis=1))
    all_diff_tmp_cum_pnl = \
        {'returns': diff_cum_pnl.xs('returns', level='target', drop_level=True).xs('', level='aggregated',
                                                                                   drop_level=True),
         'RV': diff_cum_pnl.loc[pd.IndexSlice[:, 'RV', :], :].xs('RV', level='target', drop_level=True)}
    for target in ['RV', 'returns']:
        tmp_diff_cum_pnl = all_diff_tmp_cum_pnl[target]
        fig = make_subplots(rows=3, cols=len(d_order.categories), shared_xaxes=False,
                            row_titles=['P&L<sup><span>&#8212;</span>0</sup>',
                                        'P&L<sup><span>&#8212;</span>TF</sup>',
                                        'P&L<sup><span>&#8212;</span>BASC</sup>'],
                            column_titles=[f'<i>{D}</i>' for D in d_order.categories])
        if target == 'RV':
            fig2 = make_subplots(rows=3, cols=len(d_order.categories), shared_xaxes=False,
                                 row_titles=['P&L<sup><span>&#8212;</span>0</sup>',
                                             'P&L<sup><span>&#8212;</span>TF</sup>',
                                             'P&L<sup><span>&#8212;</span>BASC</sup>'],
                                 column_titles=[f'<i>{D}</i>' for D in d_order.categories])
        for idx, pair in enumerate(SYMBOLS):
            tmp_diff_cum_pnl_pair = tmp_diff_cum_pnl.xs(pair, axis=1)
            if target == 'RV':
                tmp_diff_cum_pnl_pair = \
                    tmp_diff_cum_pnl.xs(pair, axis=1).xs(1, axis=0, level='aggregated', drop_level=True)
                tmp_diff_cum_pnl_pair_non_agg = \
                    tmp_diff_cum_pnl.xs(pair, axis=1).xs(0, axis=0, level='aggregated', drop_level=True)
            tmp_diff_cum_pnl_pair_no_fees = \
                tmp_diff_cum_pnl_pair.xs(0, axis=0, level='transaction_fees').xs(0, axis=0, level='spread')
            tmp_diff_cum_pnl_pair_tf = \
                tmp_diff_cum_pnl_pair.xs(1, axis=0, level='transaction_fees').xs(0, axis=0, level='spread')
            tmp_diff_cum_pnl_pair_basc = \
                tmp_diff_cum_pnl_pair.xs(0, axis=0, level='transaction_fees').xs(1, axis=0, level='spread')
            for col, D in enumerate(d_order.categories):
                fig.add_trace(trace=go.Scatter(x=tmp_diff_cum_pnl_pair_no_fees.index,
                                               y=tmp_diff_cum_pnl_pair_no_fees[D.upper()].values,
                                               marker=dict(color=px.colors.qualitative.Plotly[idx]),
                                               name=f'{pair.replace("USDT", "/USDT")}', showlegend=False),
                              col=col + 1, row=1)
                fig.add_trace(trace=go.Scatter(x=tmp_diff_cum_pnl_pair_tf.index,
                                               y=tmp_diff_cum_pnl_pair_tf[D.upper()].values,
                                               marker=dict(color=px.colors.qualitative.Plotly[idx]),
                                               name=f'{pair.replace("USDT", "/USDT")}', showlegend=False),
                              col=col + 1, row=2)
                fig.add_trace(trace=go.Scatter(x=tmp_diff_cum_pnl_pair_basc.index,
                                               y=tmp_diff_cum_pnl_pair_basc[D.upper()].values,
                                               marker=dict(color=px.colors.qualitative.Plotly[idx]),
                                               name=f'{pair.replace("USDT", "/USDT")}', showlegend=False),
                              col=col + 1, row=3)
                if target == 'RV':
                    tmp_diff_cum_pnl_pair = \
                        tmp_diff_cum_pnl.xs(pair, axis=1).xs(0, axis=0, level='aggregated', drop_level=True)
                    tmp_diff_cum_pnl_pair_no_fees = \
                        tmp_diff_cum_pnl_pair.xs(0, axis=0, level='transaction_fees').xs(0, axis=0, level='spread')
                    tmp_diff_cum_pnl_pair_tf = \
                        tmp_diff_cum_pnl_pair.xs(1, axis=0, level='transaction_fees').xs(0, axis=0, level='spread')
                    tmp_diff_cum_pnl_pair_basc = \
                        tmp_diff_cum_pnl_pair.xs(0, axis=0, level='transaction_fees').xs(1, axis=0, level='spread')
                    fig2.add_trace(trace=go.Scatter(x=tmp_diff_cum_pnl_pair_no_fees.index,
                                                    y=tmp_diff_cum_pnl_pair_no_fees[D.upper()].values,
                                                    marker=dict(color=px.colors.qualitative.Plotly[idx]),
                                                    name=f'{pair.replace("USDT", "/USDT")}', showlegend=False),
                                   col=col + 1, row=1)
                    fig2.add_trace(trace=go.Scatter(x=tmp_diff_cum_pnl_pair_tf.index,
                                                    y=tmp_diff_cum_pnl_pair_tf[D.upper()].values,
                                                    marker=dict(color=px.colors.qualitative.Plotly[idx]),
                                                    name=f'{pair.replace("USDT", "/USDT")}', showlegend=False),
                                   col=col + 1, row=2)
                    fig2.add_trace(trace=go.Scatter(x=tmp_diff_cum_pnl_pair_basc.index,
                                                    y=tmp_diff_cum_pnl_pair_basc[D.upper()].values,
                                                    marker=dict(color=px.colors.qualitative.Plotly[idx]),
                                                    name=f'{pair.replace("USDT", "/USDT")}', showlegend=False),
                                   col=col + 1, row=3)
        fig.update_layout(title=dict(text=f'<b>&#916; P&L Curves - {title_dd[target == "RV"]}</b>',
                                     x=0.5, xanchor='center', y=0.98, yanchor='top', pad=dict(b=250),
                                     font=dict(size=10 * TITLE_FONT_SIZE), automargin=True,
                                     yref='container'), font=dict(size=5 * LABEL_AXIS_FONT_SIZE), width=WIDTH * 15,
                          height=5 * len(d_order.categories) * HEIGHT,
                          paper_bgcolor='white', plot_bgcolor='white')
        fig.update_traces(line=dict(width=30))
        fig.update_xaxes(tickangle=45, showline=True, linewidth=2, linecolor='black', mirror=False, showgrid=False,
                         title='', tickfont=dict(size=7.5 * LABEL_AXIS_FONT_SIZE))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False, showgrid=False,
                         tickfont=dict(size=7.5 * LABEL_AXIS_FONT_SIZE))
        fig.write_image(f'./figures/pnl_{dict([("returns", "first"), ("RV", "second")])[target]}_moment.pdf',
                        format='pdf')
        if target == 'RV':
            fig2.update_layout(title=dict(text=f'<b>&#916; P&L Curves - {title_dd[target == "RV"]}'
                                               f' (non-aggregated Crypto SVIX)</b>',
                                          font=dict(size=10 * TITLE_FONT_SIZE), automargin=True,
                                          yref='container'), font=dict(size=5 * LABEL_AXIS_FONT_SIZE),
                               width=WIDTH * 15, height=5 * len(d_order.categories) * HEIGHT,
                               legend=dict(font=dict(size=10 * LABEL_AXIS_FONT_SIZE), itemwidth=100),
                               paper_bgcolor='white', plot_bgcolor='white')
            fig2.update_traces(line=dict(width=30))
            fig2.update_layout(title=dict(x=0.5, xanchor='center', y=0.98, yanchor='top', pad=dict(b=250)))
            fig2.update_annotations(font=dict(size=10 * LABEL_AXIS_FONT_SIZE))
            fig2.update_xaxes(tickangle=45, showline=True, linewidth=2, linecolor='black', mirror=False,
                              showgrid=False, title='', tickfont=dict(size=7.5 * LABEL_AXIS_FONT_SIZE))
            fig2.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False, showgrid=False,
                              tickfont=dict(size=7.5 * LABEL_AXIS_FONT_SIZE))
            fig2.write_image(f'./figures/pnl_second_moment_non_aggregated.pdf', format='pdf')
    sharpe_mean = pnl.groupby(grouper_level).mean()
    sharpe_std = pnl.groupby(grouper_level).std()
    no_fees = pd.IndexSlice
    sharpe = sharpe_mean.div(sharpe_std).mul(360 ** .5).loc[no_fees[:, :, 0, 0], :].round(2)
    sharpe.columns = sharpe.columns.set_levels(['$X^S$', '$X^{(S,O)}$'], level='market')
    sharpe.columns = sharpe.columns.set_levels([r'\textit{1d}', r'\textit{1m}', r'\textit{1w}'], level='D')
    sharpe.index = sharpe.index.set_levels(['0.0', '1.0', ''], level='aggregated')
    sharpe_columns_order = pd.MultiIndex.from_product([pairs_order.categories.tolist(),
                                                       market_order.categories.tolist()],
                                                      names=['underlying_name', 'market'])
    sharpe_index_order_aggregated = pd.MultiIndex.from_tuples(
        [(agg, target, rf'\textit{{{d}}}') for agg in ['', str(float(1))]
         for target in ['returns', 'RV']
         for d in d_order.categories.tolist()
         if not ((target == 'returns' and agg in [str(float(1))]) or (target == 'RV' and agg == ''))],
        names=['aggregated', 'target', 'D']
    )
    sharpe_index_order_non_aggregated = pd.MultiIndex.from_tuples(
        [(agg, target, rf'\textit{{{d}}}') for agg in [str(float(0))]
         for target in ['RV']
         for d in d_order.categories.tolist()],
        names=['aggregated', 'target', 'D']
    )
    for aggregated in [False, True]:
        if not aggregated:
            tmp_sharpe = sharpe.droplevel(axis=0, level=[-2, -1]).stack('D')
            tmp_sharpe = tmp_sharpe.reindex(sharpe_index_order_non_aggregated).reindex(sharpe_columns_order, axis=1)
        else:
            tmp_sharpe = \
                sharpe.droplevel(axis=0, level=[-2, -1]).stack('D').reindex(sharpe_index_order_aggregated).reindex(
                sharpe_columns_order, axis=1)
        tmp_sharpe = tmp_sharpe.astype(object)
        sharpe_xs = tmp_sharpe.xs('$X^S$', level=1, axis=1, drop_level=False)
        sharpe_x_so = tmp_sharpe.xs('$X^{(S,O)}$', level=1, axis=1, drop_level=False)
        xs_wins = pd.DataFrame(
            sharpe_xs.droplevel('market', axis=1).to_numpy(dtype=float)
            >= sharpe_x_so.droplevel('market', axis=1).to_numpy(dtype=float),
            index=tmp_sharpe.index, columns=sharpe_xs.columns,
        )

        def _bold_if(value, is_winner):
            if pd.isna(value):
                return value
            return rf'$\mathbf{{{value:.2f}}}$' if is_winner else f'${value:.2f}$'
        tmp_sharpe.loc[:, pd.IndexSlice[:, '$X^S$']] = np.vectorize(_bold_if, otypes=[object])(
            sharpe_xs.to_numpy(), xs_wins.to_numpy())
        tmp_sharpe.loc[:, pd.IndexSlice[:, '$X^{(S,O)}$']] = np.vectorize(_bold_if, otypes=[object])(
            sharpe_x_so.to_numpy(), ~xs_wins.to_numpy())
        with open(f'{args.target_path}/crypto_option_spot/'
                  f'sharpe{dict([(False, "_non_aggregated"), (True, "_aggregated")])[aggregated]}.txt',
                  'w') as sharpe_file:
            sharpe_file.writelines(tmp_sharpe.to_latex(escape=False, float_format='%.2f'))
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
    pnl = pl.from_pandas(pnl.stack('D', future_stack=True).stack('market', future_stack=True).stack(
        'underlying_name', future_stack=True).reset_index().replace('', np.nan))
    pnl = pnl.join(rv.collect(), on=['D', 'timestamp'], how='inner').rename({'0': 'returns'})
    pnl = \
        pnl.sort(['aggregated', 'target', 'transaction_fees', 'spread', 'vol_regime', 'D', 'underlying_name', 'market'])
    pnl = \
        pnl.group_by(['aggregated', 'target', 'transaction_fees', 'spread', 'vol_regime', 'D', 'underlying_name',
                      'market'])
    sharpe_pnl_per_vol_regime = pnl.agg([pl.col('returns').mean().alias('mean'),
                                         pl.col('returns').std().alias('std')])
    sharpe_pnl_per_vol_regime = \
        sharpe_pnl_per_vol_regime.with_columns((pl.col('mean')/pl.col('std')).mul(360 ** .5).alias('sharpe'))
    sharpe_pnl_per_vol_regime = sharpe_pnl_per_vol_regime.select(['aggregated', 'target', 'transaction_fees', 'spread',
                                                                  'vol_regime', 'D', 'underlying_name', 'market',
                                                                  'sharpe'])
    sharpe_pnl_per_vol_regime = sharpe_pnl_per_vol_regime.sort(['aggregated', 'target', 'transaction_fees', 'spread',
                                                                'vol_regime', 'D', 'underlying_name', 'market'])
    sharpe_pnl_per_vol_regime = sharpe_pnl_per_vol_regime.with_columns(pl.col('aggregated').fill_null('')).to_pandas()
    sharpe_pnl_per_vol_regime = sharpe_pnl_per_vol_regime.query('transaction_fees == 0 and spread == 0')
    sharpe_pnl_per_vol_regime = sharpe_pnl_per_vol_regime.drop(['transaction_fees', 'spread'], axis=1)
    sharpe_pnl_per_vol_regime = \
        pd.pivot(sharpe_pnl_per_vol_regime, index=['aggregated', 'target', 'vol_regime', 'D'],
                 columns=['underlying_name', 'market'], values='sharpe').round(2)
    sharpe_pnl_per_vol_regime.columns = \
        sharpe_pnl_per_vol_regime.columns.set_levels(['$X^S$', '$X^{(S,O)}$'], level='market')
    sharpe_pnl_per_vol_regime.index = \
        sharpe_pnl_per_vol_regime.index.set_levels([r'\textit{1d}', r'\textit{1w}', r'\textit{1m}'], level='D')
    sharpe_pnl_per_vol_regime_aggregated = pd.MultiIndex.from_tuples(
        [(agg, target, f'\textit{{{vol_regime}}}', f'\textit{{{d}}}') for agg in ['', str(float(1))]
         for target in ['returns', 'RV']
         for vol_regime in volatility_order.categories.tolist()
         for d in d_order.categories.tolist()
         if not ((target == 'returns' and agg in [str(float(1))]) or (target == 'RV' and agg == ''))],
        names=['aggregated', 'target', 'vol_regime', 'D']
    )
    sharpe_pnl_per_vol_regime_non_aggregated = pd.MultiIndex.from_tuples(
        [(agg, target, rf'\textit{{{vol_regime}}}', rf'\textit{{{d}}}')
         for agg in [(str(float(0)))]
         for target in ['RV']
         for vol_regime in volatility_order.categories.tolist()
         for d in d_order.categories.tolist()], names=['aggregated', 'target', 'vol_regime', 'D']
    )
    for aggregated in [False, True]:
        if not aggregated:
            tmp_sharpe = sharpe_pnl_per_vol_regime.reindex(sharpe_pnl_per_vol_regime_non_aggregated)
            tmp_sharpe = tmp_sharpe.reindex(sharpe_columns_order, axis=1)
        else:
            tmp_sharpe = sharpe_pnl_per_vol_regime.reindex(sharpe_pnl_per_vol_regime_aggregated)
            tmp_sharpe = tmp_sharpe.reindex(sharpe_columns_order, axis=1)
        tmp_sharpe = tmp_sharpe.droplevel(axis=0, level=0).round(2)
        tmp_sharpe = tmp_sharpe.astype(object)
        sharpe_per_volatility_regime_xs = tmp_sharpe.xs('$X^S$', level=1, axis=1, drop_level=False)
        sharpe_per_volatility_regime_x_so = tmp_sharpe.xs('$X^{(S,O)}$', level=1, axis=1, drop_level=False)
        xs_wins = pd.DataFrame(
            sharpe_per_volatility_regime_xs.droplevel('market', axis=1).to_numpy(dtype=float)
            >= sharpe_per_volatility_regime_x_so.droplevel('market', axis=1).to_numpy(dtype=float),
            index=tmp_sharpe.index, columns=sharpe_per_volatility_regime_xs.columns,
        )

        def _bold_if(value, is_winner):
            if pd.isna(value):
                return value
            return rf'$\mathbf{{{value:.2f}}}$' if is_winner else f'${value:.2f}$'
        tmp_sharpe.loc[:, pd.IndexSlice[:, '$X^S$']] = np.vectorize(_bold_if, otypes=[object])(
            sharpe_per_volatility_regime_xs.to_numpy(), xs_wins.to_numpy())
        tmp_sharpe.loc[:, pd.IndexSlice[:, '$X^{(S,O)}$']] = np.vectorize(_bold_if, otypes=[object])(
            sharpe_per_volatility_regime_x_so.to_numpy(), ~xs_wins.to_numpy())
        with open(f'{args.target_path}/crypto_option_spot/'
                  f'sharpe_per_volatility{dict([(False, "_non_aggregated"), (True, "_aggregated")])[aggregated]}.txt',
                  'w') as sharpe_file:
            sharpe_file.writelines(tmp_sharpe.to_latex(escape=False, float_format='%.2f'))
