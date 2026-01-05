import concurrent.futures
import datetime
import itertools
import pdb
import sqlite3
import os
import numpy as np
import pandas as pd
import optuna
import pytz
from model.lab import PnLExplained, shuffle_single_feature
from optuna.samplers import RandomSampler
from dateutil.relativedelta import relativedelta
from constants import DATE_FORMAT, N_JOBS, N_TRIALS, FEES_SPOT, FEES_OPTIONS, DAYS_CONVERTER, SYMBOLS,\
    LOW_ATM_MONEYNESS, HIGH_ATM_MONEYNESS, VOL_REGIMES, ATM_MONEYNESSES, ENGINE
from functools import partial
from typing import Dict, List, Tuple, Union
from scipy.optimize import minimize
from scipy.stats.distributions import norm
import statsmodels.api as sm
from pandas.api.types import CategoricalDtype
import polars as pl
from itertools import product
from sklearn.linear_model import LinearRegression
import argparse


########################################################################################################################
## Functions used to parallelise FrameworkApplication methods
########################################################################################################################


def process_daily_option_chain_date(date: datetime.datetime, directory: str,
                                    freq: str) -> Tuple[datetime, pl.LazyFrame]:
    print(f'Option chain data for {date.strftime(DATE_FORMAT)} is being generated.........')
    option_chain_data = pl.scan_parquet(f'{directory}/*/{date.year}/options/{date.strftime(DATE_FORMAT)}')
    option_chain_data = option_chain_data.sort(['underlying_name', 'timestamp'])
    option_chain_data = (option_chain_data.group_by_dynamic(index_column='timestamp', every=freq.replace('in', ''),
                                                            group_by='underlying_name', include_boundaries=False,
                                                            period=None).agg([pl.all().last()]))
    print(f'Option chain data for {date.strftime(DATE_FORMAT)} has been generated.')
    return date, option_chain_data


def daily_date_sym_option_type(date_sym_option_type: List[str]):
    return date_sym_option_type


def iv_regression_per_option_type(option_chain_data: pd.DataFrame, option_type: str, freq: str) -> Tuple:
    sym = option_chain_data.underlying_name.unique()[0]
    date = option_chain_data.timestamp.iloc[0].date()
    print(f'[IV interpolation]: '
          f'IV interpolation for {sym}_{option_type}_{date.strftime(DATE_FORMAT)} '
          f'IV surface has started...................')
    option_chain_data_type = \
        option_chain_data.query(f'type == "{option_type}"')[['mark_iv', 'moneyness', 'tau', 'timestamp']]
    option_chain_data_type_group = option_chain_data_type.groupby(by=pd.Grouper(key='timestamp', freq=freq, sort=True))
    unique_date = np.unique([group.date() for group in option_chain_data_type_group.groups]).tolist()[-1]
    betas = dict()
    for group in option_chain_data_type_group.groups:
        if group.date() == unique_date:
            try:
                tmp = option_chain_data_type_group.get_group(group)[['mark_iv', 'moneyness', 'tau']]
            except KeyError:
                continue
            mark_iv = tmp.pop('mark_iv')
            tmp = tmp.assign(const=1, moneyness2=tmp.moneyness ** 2, tau2=tmp.tau ** 2,
                             moneyness_tau=tmp.moneyness * tmp.tau)
            tmp = tmp[['const', 'moneyness', 'tau', 'moneyness2', 'tau2', 'moneyness_tau']]
            xtx = tmp.values.T @ tmp.values
            try:
                inv_xtx = np.linalg.inv(xtx)
            except np.linalg.LinAlgError:
                inv_xtx = np.linalg.pinv(xtx)
            xty = tmp.values.T @ mark_iv.values
            betas[group] = inv_xtx @ xty
    betas = pd.DataFrame(betas, index=['const', 'moneyness', 'tau', 'moneyness2', 'tau2', 'moneyness_tau']).T
    print(f'[IV interpolation]: IV interpolation for '
          f'{sym}_{option_type}_{date.strftime(DATE_FORMAT)} IV surface is now complete.')
    return sym, option_type, betas


def BS_pricer(K: float, underlying_price: float, tau: float, call_iv: float, put_iv: float) -> float:
    moneyness = np.log(underlying_price / K)
    call_d1 = moneyness + call_iv ** 2 * tau / 2
    call_d1 /= (call_iv * tau ** .5)
    call_d2 = call_d1 - call_iv * tau ** .5
    put_d1 = moneyness + put_iv ** 2 * tau / 2
    put_d1 /= (put_iv * tau ** .5)
    put_d2 = put_d1 - put_iv * tau ** .5
    call = norm.cdf(call_d1) * underlying_price - norm.cdf(call_d2) * K
    put = K * norm.cdf(-put_d2) - norm.cdf(-put_d1) * underlying_price
    return (call - put) ** 2


def find_forward_price(iv: Dict[(str, Dict[(str, np.array)])], sym: str, s0: np.array, tau: float) -> List[float]:
    fwd_price = list()
    call_iv, put_iv = iv[sym]['call'], iv[sym]['put']
    for idx in range(s0.shape[0]):
        rres = minimize(BS_pricer, x0=s0[idx], args=(s0[idx], tau, call_iv[idx], put_iv[idx]))
        fwd_price.append(rres.x[0])
    return sym, fwd_price


def return_symbol_option_type(symbol_option_type: Tuple) -> Tuple:
    return symbol_option_type


def preprocess_strikes(strikes) -> Tuple[pd.DataFrame]:
    strikes = pd.DataFrame(index=strikes.timestamp.unique().tolist(), columns=strikes.strike_price.unique().tolist(),
                           data=1)
    strikes = strikes.mul(strikes.columns)
    strikes.sort_index(axis=1, inplace=True)
    discretized = ((strikes.T.shift(-1) - strikes.T.shift()) / 2)
    return strikes, discretized.T


def svix_daily(date: datetime, option_chain: pd.DataFrame, iv_interpolation: pd.DataFrame, D: str,
               freq: str, aggregated: bool = True) -> pd.DataFrame:
    T = pd.to_timedelta(DAYS_CONVERTER[D.upper()], unit='D') / pd.to_timedelta('360D')
    time_bucket = pd.to_timedelta('1D') // pd.to_timedelta(freq)
    iv_interpolation.dropna(how='all', axis=1, inplace=True)
    symbol = iv_interpolation.columns.get_level_values(0).unique().tolist()
    fwd_price = dict([(sym, list()) for sym in symbol])
    svix = dict([(sym, dict([(option_type, dict()) for option_type in ['call', 'put']])) for sym in symbol])
    symbol_option_type = list(itertools.product(symbol, ['call', 'put']))
    print(f'[SVIX Computation]: SVIX_{D.lower()} {date.strftime(DATE_FORMAT)} is being computed.........\n')
    underlying_price = option_chain[['timestamp', 'underlying_price', 'underlying_name']].set_index('timestamp')
    underlying_price = underlying_price.loc[underlying_price.index.date == date]
    underlying_price = underlying_price.drop_duplicates(keep='last')
    underlying_price = pd.pivot_table(underlying_price.reset_index(), columns='underlying_name',
                                      values='underlying_price', index='timestamp')
    underlying_price = \
        underlying_price.reindex(index=pd.date_range(start=date, freq=freq, end=date + relativedelta(days=1),
                                                     inclusive='left', tz=pytz.UTC)).ffill()
    iv_option_type = dict([(sym, dict()) for sym in symbol])
    X = np.ones(shape=(len(ATM_MONEYNESSES), time_bucket, 5))
    X[:, :, 0] = X[:, :, 0] * ATM_MONEYNESSES.reshape(-1, 1)
    X[:, :, 1] = X[:, :, 1] * T
    X[:, :, 2] = X[:, :, 2] * ATM_MONEYNESSES.reshape(-1, 1) ** 2
    X[:, :, 3] = T ** 2
    X[:, :, 4] = X[:, :, 4] * ATM_MONEYNESSES.reshape(-1, 1) * T
    for sym_option_type in symbol_option_type:
        iv_interpolation_coef = iv_interpolation.xs((sym_option_type[0], sym_option_type[1]), axis=1)
        iv_interpolation_coef = iv_interpolation_coef.reindex(pd.date_range(start=date, end=date+relativedelta(days=1),
                                                                            freq=freq, inclusive='left', tz=pytz.UTC))
        iv_interpolation_coef.ffill(inplace=True)
        iv_interpolation_coef = iv_interpolation_coef.values
        iv = np.sum(iv_interpolation_coef[:, 1:] * X, axis=2).T + iv_interpolation_coef[:, 0].reshape(-1, 1)
        iv = np.mean(iv, axis=1)
        iv = np.where(iv >= .01, iv, .01)
        iv_option_type[sym_option_type[0]][sym_option_type[1]] = iv
    for sym in symbol:
        sym, fwd_prices = \
            find_forward_price(iv_option_type, sym,
                               underlying_price.filter(regex=sym).rename(columns=
                                                                         {sym: 'underlying_price'}).underlying_price, T)
        fwd_price[sym] = fwd_prices
    fwd_price = pd.DataFrame(fwd_price, index=pd.date_range(start=date, freq=freq, end=date + relativedelta(days=1),
                                                            inclusive='left'))
    ####################################################################################################################
    ## Strikes + BS prices
    ####################################################################################################################
    for sym_option_type in symbol_option_type:
        sym, option_type = sym_option_type[0], sym_option_type[1]
        option_chain_sym_type = option_chain.query(f'(underlying_name == "{sym}") & (type == "{option_type}")')
        strikes = option_chain_sym_type[['strike_price', 'timestamp']].set_index('timestamp')
        strikes = strikes.loc[strikes.index.date == date].reset_index()
        strikes, strikes_grid = preprocess_strikes(strikes)
        strikes = \
            strikes.reindex(pd.date_range(start=date, freq=freq, end=date + relativedelta(days=1), inclusive='left',
                            tz=pytz.UTC))
        strikes_grid = strikes_grid.reindex(pd.date_range(start=date, freq=freq, end=date + relativedelta(days=1),
                                                          inclusive='left', tz=pytz.UTC))
        fwd_price_sym = fwd_price[sym].values.reshape(-1, 1)
        moneyness = np.log(strikes.values / fwd_price_sym.reshape(-1, 1))
        iv = iv_option_type[sym][option_type].reshape(-1, 1)
        d1 = - moneyness + iv ** 2 * T / 2
        d1 /= (iv * T ** .5)
        d2 = d1 - iv * T ** .5
        if option_type == 'call':
            otm = strikes.le(fwd_price_sym).values
            prices = - (strikes.values * norm.cdf(d2)) + norm.cdf(d1) * fwd_price_sym
        else:
            otm = strikes.gt(fwd_price_sym).values
            prices = strikes.values * norm.cdf(-d2) - norm.cdf(-d1) * fwd_price_sym
        svix[sym][option_type] = 2 * (np.nan_to_num((prices * otm * strikes_grid.values)).sum(axis=1) /
                                      fwd_price_sym.reshape(fwd_price_sym.shape[0]) ** 2)
        if aggregated:
            volume = option_chain_sym_type.groupby(by=pd.Grouper(key='timestamp', freq=freq))['volume'].sum().values
            svix[sym][option_type] = pd.concat([pd.Series(svix[sym][option_type]).dropna(how='all', axis=0),
                                                pd.Series(volume)], axis=1)
        else:
            svix[sym][option_type] = pd.concat([pd.Series(svix[sym][option_type]).dropna(how='all', axis=0)], axis=1)
    for sym in svix.keys():
        svix[sym] = pd.concat(svix[sym], axis=1)
    svix = pd.concat(svix, axis=1).reindex(list(range(0, pd.to_timedelta('1D')//pd.to_timedelta(freq))))
    if aggregated:
        svix = svix.stack(level=[0, 1], future_stack=True)
        svix[2] = svix[0] * svix[1]
        svix = svix.unstack([1, 2])
        svix = svix.swaplevel(axis=1, i=1, j=0).swaplevel(axis=1, i=2, j=1)
        svix.index = \
            pd.date_range(start=date, end=date + relativedelta(days=1), inclusive='left', freq=freq, tz=pytz.UTC)
        tmp = pd.Series()
        for option_type in ['call', 'put']:
            volume = svix.xs(key=(option_type, 1), axis=1, level=[1, 2]).sum(axis=1)
            iv_volume = svix.xs(key=(option_type, 2), axis=1, level=[1, 2])
            if tmp.empty:
                tmp = iv_volume.div(volume.values.reshape(-1, 1)).sum(axis=1).div(2)
            else:
                tmp += iv_volume.div(volume.values.reshape(-1, 1)).sum(axis=1).div(2)
    else:
        tmp = \
            .5 * (svix.iloc[:, ::2].droplevel(axis=1, level=1) + svix.iloc[:, 1::2].droplevel(axis=1,
                                                                                              level=1)).droplevel(
                axis=1, level=1)
        tmp.index = \
            pd.date_range(start=date, end=date + relativedelta(days=1), inclusive='left', freq=freq, tz=pytz.UTC)
    print(f'[SVIX Computation]: SVIX_{D.lower()}_{date.strftime(DATE_FORMAT)} has been computed.\n')
    return tmp, D


def objective_daily(trial: optuna.trial.Trial, transaction_cost: bool, spread: bool, mean_bar: pd.DataFrame,
                    **kwargs) -> float:
    param = dict([('threshold_filter', trial.suggest_float(name='threshold_filter', low=0, high=1, step=.1))])
    data = kwargs['data']
    if kwargs['target'] == 'RV':
        signal_filter_long = \
            ((data['forecast'].subtract(data['values'].shift())).div(mean_bar.loc[data.index, :])).gt(
                param['threshold_filter'])
        signal_filter_short = \
            ((data['forecast'].subtract(data['values'].shift())).div(mean_bar.loc[data.index, :])).lt(
                -param['threshold_filter'])
        signal_filter_long = signal_filter_long.astype(int)
        signal_filter_short = signal_filter_short.astype(int)
        option_chain_data = kwargs['option_chain_data']
        vrp = kwargs['vrp']
        param.update(
            dict([('lower_bound_atm', trial.suggest_float(name='lower_bound_atm', low=np.log(LOW_ATM_MONEYNESS),
                                                          high=np.log(.05 + LOW_ATM_MONEYNESS))),
                  ('upper_bound_atm', trial.suggest_float(name='upper_bound_atm', low=np.log(-.05 + HIGH_ATM_MONEYNESS),
                                                          high=np.log(HIGH_ATM_MONEYNESS)))]))
        option_chain_data.loc[:, 'atm'] = option_chain_data['moneyness'].gt(param['lower_bound_atm']) & \
                                   option_chain_data['moneyness'].lt(param['upper_bound_atm'])
        option_chain_data = option_chain_data[option_chain_data.atm].drop(['atm', 'moneyness'], axis=1)
        option_chain_data = \
            option_chain_data.groupby(by=[pd.Grouper(key='option_tag'),
                                          pd.Grouper(level=0)]).agg({'underlying_name': 'last',
                                                                     'underlying_price': 'last',
                                                                     'spread': 'sum', 'mid_price': 'sum'})
        option_chain_data['transaction_fees'] = FEES_SPOT * option_chain_data['underlying_price']
        option_chain_data['transaction_fees'] = option_chain_data['transaction_fees'].where(
            option_chain_data['transaction_fees'].le(FEES_OPTIONS * option_chain_data['mid_price']),
            FEES_OPTIONS * option_chain_data['mid_price'])
        option_chain_data['long_returns'] = option_chain_data.groupby(pd.Grouper(level=0), group_keys=False).apply(
            lambda x: x['mid_price'] - x['mid_price'].shift())
        option_chain_data['short_returns'] = -option_chain_data['long_returns']
        if transaction_cost:
            option_chain_data['long_returns'] = \
                option_chain_data['long_returns'] - option_chain_data.groupby(
                    pd.Grouper(level=0), group_keys=False).apply(
                    lambda x: x['transaction_fees'] + x['transaction_fees'].shift())
            option_chain_data['short_returns'] = \
                option_chain_data['short_returns'] - option_chain_data.groupby(
                    pd.Grouper(level=0), group_keys=False).apply(
                    lambda x: x['transaction_fees'] + x['transaction_fees'].shift())
        if spread:
            option_chain_data['long_returns'] = \
                option_chain_data['long_returns'] - option_chain_data.groupby(
                    pd.Grouper(level=0), group_keys=False).apply(
                    lambda x: .5*(x['spread'] + x['spread'].shift()))
            option_chain_data['short_returns'] = \
                option_chain_data['short_returns'] - option_chain_data.groupby(
                    pd.Grouper(level=0), group_keys=False).apply(
                    lambda x: .5 * (x['spread'] + x['spread'].shift()))
        option_chain_data['long_returns'] = \
            option_chain_data['long_returns'].div(option_chain_data['mid_price'].groupby(
                pd.Grouper(level=0), group_keys=False).apply(lambda x: x.shift()))
        option_chain_data['short_returns'] = \
            option_chain_data['short_returns'].div(option_chain_data['mid_price'].groupby(
                pd.Grouper(level=0), group_keys=False).apply(lambda x: x.shift()))
        long_returns = option_chain_data[['long_returns', 'underlying_name']].groupby(
            [pd.Grouper(key='underlying_name'), pd.Grouper(level=1)])[['long_returns']].mean()
        short_returns = option_chain_data[['short_returns', 'underlying_name']].groupby(
            [pd.Grouper(key='underlying_name'), pd.Grouper(level=1)])[['short_returns']].mean()
        long_returns = \
            pd.concat([long_returns.assign(D='1D'), long_returns.assign(D='1W'),
                       long_returns.assign(D='1M')]).set_index('D', append=True)
        long_returns = long_returns.unstack(level=[0, 2]).replace([-np.inf, np.inf], np.nan).droplevel(axis=1, level=0)
        short_returns = pd.concat(
            [short_returns.assign(D='1D'), short_returns.assign(D='1W'),
             short_returns.assign(D='1M')]).set_index('D', append=True)
        short_returns = \
            short_returns.unstack(level=[0, 2]).replace([-np.inf, np.inf], np.nan).droplevel(axis=1, level=0)
        vrp = np.sign(vrp.reindex(option_chain_data.index.get_level_values(1))).sort_index()
        short_vrp, long_vrp = (vrp == 1).astype(int), (vrp == -1).astype(int)
        signal_filter_long.columns = \
            signal_filter_long.columns.set_levels(signal_filter_long.columns.levels[-1].str.lower(), level=1)
        signal_filter_short.columns = \
            signal_filter_short.columns.set_levels(signal_filter_short.columns.levels[-1].str.lower(), level=1)
        long_returns.columns = \
            long_returns.columns.set_levels(long_returns.columns.levels[-1].str.lower(), level=1)
        short_returns.columns = \
            short_returns.columns.set_levels(short_returns.columns.levels[-1].str.lower(), level=1)
        long_returns = long_returns.mul(long_vrp.loc[long_returns.index]).mul(signal_filter_long.loc[long_returns.index])
        short_returns = \
            short_returns.mul(short_returns.loc[long_returns.index]).mul(signal_filter_short.loc[short_returns.index])
        returns = long_returns.add(short_returns)
        sharpe = (returns.stack(level=1).swaplevel(-1, 0).groupby(level=0).mean()
                  /returns.stack(level=1).swaplevel(-1, 0).groupby(level=0).std()).mean(axis=1).to_dict()
        trial.set_user_attr(key='lower_bound_atm', value=param['lower_bound_atm'])
        trial.set_user_attr(key='upper_bound_atm', value=param['upper_bound_atm'])
    else:
        signal_filter_long = ((data['forecast'].subtract(data['values'].shift())).div(mean_bar)).gt(
            param['threshold_filter'])
        signal_filter_short = \
            ((data['forecast'].subtract(data['values'].shift())).div(mean_bar)).lt(
                -param['threshold_filter'])
        signal_filter_long = signal_filter_long.astype(int)
        signal_filter_short = signal_filter_short.astype(int)
        spot_returns_long, spot_returns_short = kwargs['spot_returns']['values'], -kwargs['spot_returns']['values']
        spot_returns_long = spot_returns_long.loc[:, signal_filter_long.columns.get_level_values(0).unique()]
        spot_returns_short = spot_returns_short.loc[:, signal_filter_short.columns.get_level_values(0).unique()]
        spot_prices = kwargs['spot_prices']
        if transaction_cost:
            fees = .001 * (spot_prices+spot_prices.shift())/spot_prices.shift()
            fees = \
                pd.concat([pd.DataFrame(fees.unstack()).assign(D='1D'),
                           pd.DataFrame((fees.unstack())).assign(D='1W'),
                           pd.DataFrame((fees.unstack())).assign(D='1M')]).reset_index(0).rename(
                    columns={'level_0': 'underlying_name', 0: 'fees'})
            fees = pd.pivot(data=fees, columns=['underlying_name', 'D'], values='fees')
            spot_returns_long, spot_returns_short = spot_returns_long.subtract(fees), spot_returns_short.subtract(fees)
        if spread:
            spread_prices = kwargs['spread_prices']/spot_prices.shift()
            spread_prices = \
                pd.concat([pd.DataFrame(spread_prices.unstack()).assign(D='1D'),
                           pd.DataFrame((spread_prices.unstack())).assign(D='1W'),
                           pd.DataFrame((spread_prices.unstack())).assign(D='1M')]).reset_index(0).rename(
                    columns={'level_0': 'underlying_name', 0: 'fees'})
            spread_prices = pd.pivot(data=spread_prices, columns=['underlying_name', 'D'], values='fees')
            spot_returns_long = spot_returns_long.subtract(.5*(spread_prices.add(spread_prices)))
            spot_returns_short = spot_returns_short.subtract(.5*(spread_prices.add(spread_prices)))
        signals = np.sign(data.xs('forecast', axis=1, level=0)).shift()
        spot_returns_long, spot_returns_short = \
            spot_returns_long.dropna(how='all', axis=1), spot_returns_short.dropna(how='all', axis=1)
        spot_returns = (signals.mul(signal_filter_long).mul(spot_returns_long)).dropna(how='all', axis=1) + \
                       (signals.mul(signal_filter_short).mul(spot_returns_short)).dropna(how='all', axis=1)
        mean_returns, std_returns = spot_returns.mean(), spot_returns.std()
        sharpe = mean_returns.div(std_returns).groupby(by=pd.Grouper(level=1)).mean().to_dict()
    trial.set_user_attr(key='threshold_filter', value=param['threshold_filter'])
    trial.set_user_attr(key='target', value=kwargs['target'])
    if len(sharpe) > 0:
        trial.set_user_attr(key='sharpe',
                            value=0 if np.isnan(sum(sharpe.values())/len(sharpe)) else sum(sharpe.values())/len(sharpe))
    else:
        trial.set_user_attr(key='sharpe', value=0)
    return trial.user_attrs['sharpe']


def callback(study, trial) -> None:
    if study.best_trial.number == trial.number:
        if trial.user_attrs['target'] == 'RV':
            study.set_user_attr(key='lower_bound_atm', value=trial.user_attrs['lower_bound_atm'])
            study.set_user_attr(key='upper_bound_atm', value=trial.user_attrs['upper_bound_atm'])
        study.set_user_attr(key='threshold_filter', value=trial.user_attrs['threshold_filter'])
        study.set_user_attr(key='sharpe', value=trial.user_attrs['sharpe'])


def export_returns_regression_latex(res: sm.OLS, symbol: str, D: str):
    table = []
    rename = {
        'const': r"\textbf{const}",
        f'{symbol}_VRP_{D}': f'\\textbf{{$\\text{{VRP}}_\\textbf{{0 $\\rightarrow$ {{{D}}}$}}',
        f'Crypto_SVIX_{D}': f'\\textbf{{$\\text{{Crypto SVIX}}_\\textbf{{0 $\\rightarrow$ {{{D}}}$}}',
    }
    params = res.params
    bse = res.bse
    tvals = res.tvalues
    pvals = res.pvalues
    conf = res.conf_int()
    for i, name in enumerate(params.index):
        rowname = rename.get(name, name)
        row = (
                ("\\multirow{3}{*}{} & " if i == 0 else "& ")
                + f"{rowname} & "
                + f"{params[name]:10.4f}  & "
                + f"{bse[name]:10.3f}     & "
                + f"{tvals[name]:10.3f}  & "
                + f"{pvals[name]:10.3f}        & "
                + f"{conf.iloc[i, 0]:10.3f}    & "
                + f"{conf.iloc[i, 1]:10.3f}     \\\\"
        )
        table.append(row)

    return "\n".join(table)


def vrp_svix_effect(data: pd.DataFrame) -> pd.DataFrame:
    avg1s = data[['$\\text{returns}_{0 \\rightarrow D}$']].groupby(by=[pd.Grouper(level=0), pd.Grouper(level=1)])
    avg1s = avg1s.apply(lambda x: (x - x.shift()).abs().sum() / (x.shape[0] - 1))
    feature_importance = dict([(feature, 0) for feature in data.columns[:-1]])
    count = 0
    model = LinearRegression()
    while count < 50:
        for feature in data.columns[:-1]:
            tmp = data.copy()
            shuffled_test, feature = \
                shuffle_single_feature(data[['$\\text{VRP}_{0 \\rightarrow D}$',
                                             '$\\text{Crypto SVIX}_{0 \\rightarrow D}$']], feature)
            tmp['returns_hat'] = \
                model.fit(shuffled_test,
                          data['$\\text{returns}_{0 \\rightarrow D}$']).predict(shuffled_test)
            feature_importance[feature] += \
            ((tmp.diff(axis=1).iloc[:, -1] / avg1s.values[0]).abs()).groupby(by=pd.Grouper(level=0)).mean().values[0]
        count += 1
    feature_importance = pd.Series(feature_importance) / 50
    return feature_importance


def return_date(date: datetime.datetime) -> datetime.datetime:
    return date


class FrameworkApplication:
    _option_chain = dict()
    _option_moneyness = None
    _vrp = None
    _iv_interpolation_coef = None

    def __init__(self, target_path: str, data: pd.DataFrame = None):
        self._data = data
        self._db_connect = sqlite3.connect(f'{target_path}/crypto_option_spot.db')
        self._underlying_prices = \
            pl.scan_parquet(f'{target_path}/*/*/spot/*-*-*').select([pl.col('symbol').alias('underlying_name'),
                                                                     pl.col('timestamp'), pl.col('mid_price[0]'),
                                                                     pl.col('spreads[0]')])
        self._underlying_prices = self._underlying_prices.collect(engine=ENGINE)
        self._underlying_spreads = self._underlying_prices.pivot(
            index='timestamp', on='underlying_name', values='spreads[0]').to_pandas().set_index('timestamp')
        self._underlying_prices = self._underlying_prices.pivot(
            index='timestamp', on='underlying_name', values='mid_price[0]')
        self._underlying_rv = self._underlying_prices.select(['timestamp'] + [
            (pl.col(sym) / pl.col(sym).shift(1)).log() ** 2 for sym in SYMBOLS]).to_pandas()
        self._underlying_returns = self._underlying_prices.select(['timestamp'] + [
            (pl.col(sym) / pl.col(sym).shift(1)).log() for sym in SYMBOLS]).to_pandas()

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        self._data = data

    @property
    def underlying_returns(self) -> pd.DataFrame:
        return self._underlying_returns

    @property
    def underlying_rv(self) -> pd.DataFrame:
        return self._underlying_rv

    @staticmethod
    def process_daily_option_chain(freq: str, directory: str, date_range: pd.DatetimeIndex) -> None:
        for date in date_range:
            date, daily_option_chain_date = process_daily_option_chain_date(date, directory, freq)
            daily_option_chain_date = daily_option_chain_date.filter(pl.col('underlying_name').is_in(SYMBOLS))
            daily_option_chain_date = daily_option_chain_date.with_columns(
                ((pl.col('expiration')-pl.col('timestamp')).dt.total_days() / 360).alias('tau'))
            FrameworkApplication._option_chain[date.date()] = daily_option_chain_date.collect().to_pandas()
        return None

    @staticmethod
    def iv_regression(freq: str) -> None:
        symbol = data.columns.get_level_values(1).unique().tolist()
        iv_interpolation_data = \
            dict([(sym, dict([(option_type, None) for option_type in ['call', 'put']])) for sym in symbol])
        date_sym_option_type = \
            list(itertools.product(FrameworkApplication._option_chain.keys(), symbol, ['call', 'put']))
        for daily_date_sym_option_type_ls in date_sym_option_type:
            if not FrameworkApplication._option_chain[daily_date_sym_option_type_ls[0]].empty:
                daily_option_chain = FrameworkApplication._option_chain[daily_date_sym_option_type_ls[0]].query(
                            f'underlying_name == "{daily_date_sym_option_type_ls[1]}"')
                if not daily_option_chain.empty:
                    sym, option_type, betas = \
                        iv_regression_per_option_type(daily_option_chain, daily_date_sym_option_type_ls[2], freq)
                    if iv_interpolation_data[sym][option_type] is None:
                        iv_interpolation_data[sym][option_type] = betas
                    else:
                        iv_interpolation_data[sym][option_type] = \
                            pd.concat([iv_interpolation_data[sym][option_type], betas])
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(symbol), os.cpu_count() - 2)) as executor3:
            futures3 = [executor3.submit(lambda x: x, x=sym) for sym in symbol]
            for future3 in concurrent.futures.as_completed(futures3):
                sym = future3.result()
                try:
                    iv_interpolation_data[sym] = pd.concat(iv_interpolation_data[sym], axis=1)
                except ValueError:
                    iv_interpolation_data.pop(sym)
                    continue
        FrameworkApplication._iv_interpolation_coef = pd.concat(iv_interpolation_data, axis=1)
        FrameworkApplication._iv_interpolation_coef = \
            FrameworkApplication._iv_interpolation_coef.groupby(
                FrameworkApplication._iv_interpolation_coef.index).last()
        return pd.concat(iv_interpolation_data, axis=1)

    @staticmethod
    def svix(freq: str, aggregated: bool = True) -> pd.DataFrame:
        iv_interpolation_coef = FrameworkApplication._iv_interpolation_coef
        svix = dict([(D, list()) for D in ['1d', '1w', '1m']])
        dates = list(FrameworkApplication._option_chain.keys())
        date_range_D = list(itertools.product(dates, ['1d', '1w', '1m']))
        for date_D in date_range_D:
            svix_s, D = svix_daily(date_D[0], FrameworkApplication._option_chain[date_D[0]],
                                   iv_interpolation_coef.loc[iv_interpolation_coef.index.date == date_D[0]],
                                   date_D[1], freq, aggregated)
            svix[D].append(svix_s)
        for D in svix.keys():
            svix[D] = pd.concat(svix[D])
            if aggregated:
                svix[D].name = D
        svix = pd.concat(svix, axis=1)
        svix.index = pd.to_datetime(svix.index, utc=True)
        svix.index.name = 'timestamp'
        return svix

    @staticmethod
    def backtest_per_day(transaction_cost: bool, spread: bool, freq: str, date: datetime.datetime,
                         mean_bar: pd.DataFrame, **kwargs) -> Union[Tuple[pd.Series | pd.DataFrame], None]:
        data = kwargs['data']
        if kwargs['target'] == 'RV':
            vrp = kwargs['vrp']
            vrp_y = kwargs['vrp_y']
            option_chain_data = FrameworkApplication._option_chain[date]
            option_chain_data_y = FrameworkApplication._option_chain[date - relativedelta(days=1)]
            option_chain_data['option_tag'] = option_chain_data.symbol.str.split('-').apply(lambda x: '-'.join(x[:-1]))
            option_chain_data_y['option_tag'] = option_chain_data_y.symbol.str.split('-').apply(
                lambda x: '-'.join(x[:-1]))
            option_chain_data_y = \
                option_chain_data_y[['option_tag', 'underlying_price', 'mid_price', 'spread', 'moneyness', 'timestamp',
                                     'underlying_name']]
            ############################################################################################################
            ## Parameter optimisation
            ############################################################################################################
            try:
                if option_chain_data_y.empty:
                    raise ValueError
                option_chain_data_y.set_index('timestamp', inplace=True)
                study = optuna.create_study(
                    direction='maximize',
                    study_name=f'Parameter optimisation RV: Process {date.strftime(DATE_FORMAT)} has started.',
                    sampler=RandomSampler(123))
                objective = partial(objective_daily, option_chain_data=option_chain_data_y,
                                    data=data.loc[data.index.date != date], vrp=vrp_y,
                                    transaction_cost=transaction_cost, spread=spread, target=kwargs['target'],
                                    mean_bar=mean_bar)
                study.optimize(objective, n_trials=N_TRIALS // 20, callbacks=[callback], show_progress_bar=True,
                               n_jobs=N_JOBS)
            except ValueError:
                return None
            else:
                lower_bound_moneyness, upper_bound_moneyness, threshold_filter = study.user_attrs['lower_bound_atm'], \
                    study.user_attrs['upper_bound_atm'], study.user_attrs['threshold_filter']
            ############################################################################################################
            ## Backtest
            ############################################################################################################
            option_chain_data['atm'] = \
                option_chain_data['moneyness'].ge(lower_bound_moneyness) & \
                option_chain_data['moneyness'].le(upper_bound_moneyness)
            option_chain_data = \
                option_chain_data[option_chain_data.atm]
            option_chain_data = option_chain_data.set_index('timestamp')
            option_chain_data = \
                option_chain_data.groupby(by=[pd.Grouper(key='option_tag'),
                                              pd.Grouper(level=0)]).agg({'underlying_name': 'last', 'atm': 'last',
                                                                         'underlying_price': 'last',
                                                                         'spread': 'sum', 'mid_price': 'sum'})
            option_chain_data['long_returns'] = option_chain_data.groupby(pd.Grouper(level=0), group_keys=False).apply(
                lambda x: x['mid_price'] - x['mid_price'].shift())
            option_chain_data['short_returns'] = -option_chain_data['long_returns']
            if transaction_cost:
                option_chain_data['transaction_fees'] = FEES_SPOT * option_chain_data['underlying_price']
                option_chain_data['transaction_fees'] = option_chain_data['transaction_fees'].where(
                    option_chain_data['transaction_fees'].le(FEES_OPTIONS * option_chain_data['mid_price']),
                    FEES_OPTIONS * option_chain_data['mid_price'])
                option_chain_data['long_returns'] = option_chain_data['long_returns'].subtract(
                    option_chain_data['transaction_fees'].groupby(pd.Grouper(level=0),
                                                                  group_keys=False).apply(lambda x: x+x.shift()))
                option_chain_data['short_returns'] = option_chain_data['short_returns'].subtract(
                    option_chain_data['transaction_fees'].groupby(pd.Grouper(level=0),
                                                                  group_keys=False).apply(lambda x: x + x.shift()))
            if spread:
                option_chain_data['long_returns'] = \
                    option_chain_data['long_returns'] - option_chain_data.groupby(
                        pd.Grouper(level=0), group_keys=False).apply(
                        lambda x: .5 * (x['spread'] + x['spread'].shift()))
                option_chain_data['short_returns'] = \
                    option_chain_data['short_returns'] - option_chain_data.groupby(
                        pd.Grouper(level=0), group_keys=False).apply(
                        lambda x: .5 * (x['spread'] + x['spread'].shift()))
            option_chain_data['long_returns'] = \
                option_chain_data['long_returns'].div(option_chain_data['mid_price'].groupby(
                    pd.Grouper(level=0), group_keys=False).apply(lambda x: x.shift()))
            option_chain_data['short_returns'] = \
                option_chain_data['short_returns'].div(option_chain_data['mid_price'].groupby(
                    pd.Grouper(level=0), group_keys=False).apply(lambda x: x.shift()))
            long_returns = option_chain_data[['long_returns', 'underlying_name']].groupby(
                [pd.Grouper(key='underlying_name'), pd.Grouper(level=1)])[['long_returns']].mean()
            short_returns = option_chain_data[['short_returns', 'underlying_name']].groupby(
                [pd.Grouper(key='underlying_name'), pd.Grouper(level=1)])[['short_returns']].mean()
            long_returns = \
                pd.concat([long_returns.assign(D='1D'), long_returns.assign(D='1W'),
                           long_returns.assign(D='1M')]).set_index('D', append=True)
            long_returns = long_returns.unstack(level=[0, 2]).replace([-np.inf, np.inf], np.nan).droplevel(axis=1,
                                                                                                           level=0)
            short_returns = pd.concat(
                [short_returns.assign(D='1D'), short_returns.assign(D='1W'),
                 short_returns.assign(D='1M')]).set_index('D', append=True)
            short_returns = \
                short_returns.unstack(level=[0, 2]).replace([-np.inf, np.inf], np.nan).droplevel(axis=1, level=0)
            vrp = np.sign(vrp.reindex(option_chain_data.index.get_level_values(1))).sort_index()
            short_vrp, long_vrp = (vrp == 1).astype(int), (vrp == -1).astype(int)
            signal_filter_long = \
                ((data['forecast'].subtract(data['values'].shift())).div(mean_bar.loc[data.index, :])).gt(
                    threshold_filter)
            signal_filter_short = \
                ((data['forecast'].subtract(data['values'].shift())).div(mean_bar.loc[data.index, :])).lt(
                    -threshold_filter)
            signal_filter_long = signal_filter_long.astype(int)
            signal_filter_short = signal_filter_short.astype(int)
            signal_filter_long.columns = \
                signal_filter_long.columns.set_levels(signal_filter_long.columns.levels[-1].str.lower(), level=1)
            signal_filter_short.columns = \
                signal_filter_short.columns.set_levels(signal_filter_short.columns.levels[-1].str.lower(), level=1)
            long_returns.columns = \
                long_returns.columns.set_levels(long_returns.columns.levels[-1].str.lower(), level=1)
            short_returns.columns = \
                short_returns.columns.set_levels(short_returns.columns.levels[-1].str.lower(), level=1)
            long_returns = long_returns.apply(lambda x: x.interpolate(method='time')).mul(
                long_vrp.loc[long_returns.index]).mul(signal_filter_long.loc[long_returns.index])
            long_vrp.loc[long_returns.index].mul(signal_filter_long.loc[long_returns.index])
            short_returns = \
                short_returns.apply(lambda x: x.interpolate(method='time')).mul(
                    short_returns.loc[long_returns.index]).mul(signal_filter_short.loc[short_returns.index])
            returns = long_returns.add(short_returns)
            returns = returns.loc[~returns.index.duplicated(), :]
            atm = option_chain_data[['underlying_name', 'atm']].groupby(['underlying_name',
                                                                         pd.Grouper(level=1)])[['atm']].sum()
            atm = pd.concat([atm.assign(D='1d'), atm.assign(D='1w'), atm.assign(D='1m')])
            atm = pd.pivot(atm.reset_index(0), columns=['underlying_name', 'D'], values='atm')
            n_trades = long_vrp.loc[long_returns.index].mul(signal_filter_long.loc[long_returns.index]).mul(atm).add(
                short_vrp.loc[short_returns.index].mul(signal_filter_short.loc[short_returns.index])).mul(atm)
            n_trades = n_trades.loc[~n_trades.index.duplicated(), :]
            returns = returns.reindex(pd.date_range(start=date, end=date + relativedelta(days=1), freq=freq,
                                                    inclusive='left', tz=pytz.UTC))
            n_trades = n_trades.reindex(returns.index)
            return returns, n_trades
        else:
            spot_returns_long, spot_returns_short = kwargs['spot_returns']['values'], -kwargs['spot_returns']['values']
            spot_prices = kwargs['spot_prices']
            spot_returns_y = kwargs['spot_returns_y']
            spot_prices_y = kwargs['spot_prices_y']
            spread_prices = kwargs['spread_prices']
            study = optuna.create_study(
                direction='maximize', study_name=f'Parameter optimisation: '
                                                 f'process for {date.strftime(DATE_FORMAT)} has started.',
                sampler=RandomSampler(123))
            objective = partial(objective_daily, spot_returns=spot_returns_y, transaction_cost=transaction_cost,
                                spread=spread, spot_prices=spot_prices_y, target=kwargs['target'],
                                data=data.loc[data.index.date < date, :],
                                spread_prices=spread_prices.loc[spread_prices.index.date ==
                                                                (date - relativedelta(days=1)), :],
                                mean_bar=mean_bar.loc[mean_bar.index.date == (date - relativedelta(days=1)), :])
            study.optimize(objective, n_trials=N_TRIALS//20, callbacks=[callback], show_progress_bar=True,
                           n_jobs=os.cpu_count()//2)
            ############################################################################################################
            ## Backtest
            ############################################################################################################
            threshold_filter = study.user_attrs['threshold_filter']
            signal_filter = (data.loc[data.index.date == date, :]['forecast']-
                             data.loc[data.index.date == date, :]['values'].shift())
            signal_filter = signal_filter.div(mean_bar.loc[mean_bar.index.date == date])
            signal_filter_long, signal_filter_short = \
                signal_filter.gt(threshold_filter), signal_filter.lt(-threshold_filter)
            signal = np.sign(data['forecast']).shift()
            signal_long, signal_short = signal.mul(signal_filter_long), signal.mul(signal_filter_short)
            if transaction_cost:
                fees = .001 * (spot_prices + spot_prices.shift()) / spot_prices.shift()
                fees = \
                    pd.concat([pd.DataFrame(fees.unstack()).assign(D='1D'),
                               pd.DataFrame((fees.unstack())).assign(D='1W'),
                               pd.DataFrame((fees.unstack())).assign(D='1M')]).reset_index(0).rename(
                        columns={'level_0': 'underlying_name', 0: 'fees'})
                fees = pd.pivot(data=fees, columns=['underlying_name', 'D'], values='fees')
                spot_returns_long, spot_returns_short = \
                    spot_returns_long.subtract(fees), spot_returns_short.subtract(fees)
            if spread:
                spread_prices = spread_prices / spot_prices.shift()
                spread_prices = \
                    pd.concat([pd.DataFrame(spread_prices.unstack()).assign(D='1D'),
                               pd.DataFrame((spread_prices.unstack())).assign(D='1W'),
                               pd.DataFrame((spread_prices.unstack())).assign(D='1M')]).reset_index(0).rename(
                        columns={'level_0': 'underlying_name', 0: 'fees'})
                spread_prices = pd.pivot(data=spread_prices, columns=['underlying_name', 'D'], values='fees')
                spot_returns_long = spot_returns_long.subtract(.5 * (spread_prices.add(spread_prices.shift())))
                spot_returns_short = spot_returns_short.subtract(.5 * (spread_prices.add(spread_prices.shift())))
            returns = ((spot_returns_long.mul(signal_long)) +
                       (spot_returns_short.mul(signal_short))).dropna(how='all', axis=1).dropna()
            return returns

    def backtest(self, transaction_cost: bool, spread: bool, freq, **kwargs) -> Tuple[pd.DataFrame] | pd.DataFrame:
        pnl = list()
        if kwargs['target'] == 'RV':
            trading_volume = list()
            dates = list(FrameworkApplication._option_chain.keys())
            vrp = kwargs['vrp']
            dates = dates[dates.index(kwargs['vrp'].index.get_level_values(-1)[0].date()):]
        else:
            spot_returns = self._data.xs('returns', level=2, axis=1)
            spot_prices = self._underlying_prices.sort(by='timestamp').group_by_dynamic(index_column='timestamp',
                                                                                        every=freq.replace('in', ''))
            spot_prices = spot_prices.agg(pl.all().last()).to_pandas().set_index('timestamp')
            dates = list(np.unique(spot_returns.index.date))
        if kwargs['target'] == 'RV':
            vrp = pd.pivot(vrp.reset_index([0, 1]), columns=['underlying_name', 'D'],
                           values='$\\text{VRP}_{0 \\rightarrow D}$')
            mean_bar = self._data.xs('RV', axis=1, level=2)['values'].apply(
                lambda x: x.resample(f'{DAYS_CONVERTER[x.name[1]]}D').mean()).shift().ffill()
            mean_bar = mean_bar.reindex(self._data.index).ffill()
            for date in dates[1:]:
                today, yesterday = date, date - relativedelta(days=1)
                try:
                    if vrp.loc[vrp.index.date == today].empty:
                        raise KeyError
                    returns, n_trades = \
                        self.backtest_per_day(option_chain_data=FrameworkApplication._option_chain[today],
                                              option_chain_data_y=FrameworkApplication._option_chain[yesterday],
                                              vrp=vrp.loc[vrp.index.date == today],
                                              vrp_y=vrp.loc[vrp.index.date == yesterday],
                                              data=self._data.xs('RV', axis=1, level=2, drop_level=True).loc[
                                                   (today - relativedelta(days=30)).strftime(DATE_FORMAT):
                                                   today.strftime(DATE_FORMAT)], date=today,
                                              transaction_cost=transaction_cost, spread=spread,  freq=freq,
                                              target=kwargs['target'], mean_bar=mean_bar)
                except KeyError:
                    continue
                else:
                    pnl.append(returns)
                    trading_volume.append(n_trades)
        else:
            mean_bar = spot_returns['values'].apply(lambda x: x.resample(f'{DAYS_CONVERTER[x.name[1]]}D').mean())
            mean_bar = mean_bar.reindex(spot_returns.index).ffill()
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()//2) as executor:
                futures = [executor.submit(return_date, date) for date in dates[1:]]
                for future in concurrent.futures.as_completed(futures):
                    today, yesterday = future.result(), future.result() - relativedelta(days=1)
                    spot_returns_today = spot_returns.loc[spot_returns.index.date == today]
                    spot_returns_yesterday = \
                        spot_returns.loc[spot_returns.index.date == yesterday]
                    spot_prices_today = spot_prices.loc[spot_prices.index.date == today]
                    spot_prices_yesterday = spot_prices.loc[spot_prices.index.date == yesterday]
                    data = spot_returns.loc[(spot_returns.index.date >= today - relativedelta(days=30)) &
                                            (spot_returns.index.date <= today), :]
                    spread_prices = \
                        self._underlying_spreads.loc[
                        (self._underlying_spreads.index.date <= today) &
                        (self._underlying_spreads.index.date >= today - relativedelta(days=30)),:].resample(freq).last()
                    returns = self.backtest_per_day(spot_returns=spot_returns_today,
                                                    spot_returns_y=spot_returns_yesterday,
                                                    transaction_cost=transaction_cost, spread=spread, freq=freq,
                                                    target=kwargs['target'], spot_prices=spot_prices_today,
                                                    spot_prices_y=spot_prices_yesterday, date=today, data=data,
                                                    spread_prices=spread_prices,
                                                    mean_bar=mean_bar.loc[(mean_bar.index.date >= yesterday) &
                                                                          (mean_bar.index.date <= today), :])
                    pnl.append(returns)
        pnl = pd.concat(pnl).sort_index()
        if kwargs['target'] == 'RV':
            trading_volume = pd.concat(trading_volume)
            trading_volume = \
                trading_volume.reindex(pd.date_range(start=pnl.index[0].date(), end=pnl.index[-1].date(), freq=freq,
                                                     tz=pytz.UTC))
        pnl = pnl.reindex(pd.date_range(start=pnl.index[0].date() - relativedelta(days=1),
                                        end=pnl.index[-1].date()+relativedelta(days=1), freq=freq, tz=pytz.UTC,
                                        inclusive='left'))
        if target == 'RV':
            return pnl, trading_volume
        return pnl

    @property
    def db_connect(self):
        return self._db_connect


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all performance tables.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2025-07-01')
    parser.add_argument('--top_features', type=int,
                        help='Number of top features to consider in feature importance analysis', default=5)
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    args = parser.parse_args()
    print(args)
    ####################################################################################################################
    ## Order
    ####################################################################################################################
    pairs_order = CategoricalDtype(categories=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT'], ordered=True)
    d_order = CategoricalDtype(categories=['1d', '1w', '1m'], ordered=True)
    volatility_order = CategoricalDtype(categories=['Low', 'Normal', 'High'], ordered=True)
    market_order = CategoricalDtype(categories=[0, 1], ordered=True)
    ####################################################################################################################
    framework_application_obj = FrameworkApplication(target_path=args.target_path)
    pnl_explained_obj = PnLExplained(underlying_rv=framework_application_obj.underlying_rv,
                                     underlying_returns=framework_application_obj.underlying_returns)
    top_features = args.top_features
    rv_mkt = framework_application_obj.underlying_rv.set_index('timestamp').stack().swaplevel(0, 1)
    df_idx = rv_mkt.index.to_frame()
    df_idx['timestamp'] = df_idx['timestamp'].astype('datetime64[ns, UTC]')
    rv_mkt.index = pd.MultiIndex.from_frame(df_idx)
    rv_mkt = \
        pd.concat([pd.DataFrame(rv_mkt.groupby(
            by=[pd.Grouper(level=0),
                pd.Grouper(level=1, freq=d.upper())]).sum() *
                   ((pd.to_timedelta('360D')//pd.to_timedelta(f'{DAYS_CONVERTER[d.upper()]}D')) ** .5)).assign(
            D=d.upper()) for d in d_order.categories])
    rv_mkt = rv_mkt.groupby([pd.Grouper(key='D'),
                             pd.Grouper(level=1)]).mean().rename(columns={0: 'volatility_regime'})
    volatility_regime = rv_mkt.groupby(pd.Grouper(level='D'), group_keys=False)
    volatility_regime = \
        volatility_regime.apply(lambda x: pd.qcut(x['volatility_regime'], q=[0, .45, .9, 1], labels=VOL_REGIMES))
    ####################################################################################################################
    ## Model selection
    ####################################################################################################################
    if os.path.exists(f'{args.target_path}/metric.csv'):
        mase = pd.read_csv(f'{args.target_path}/metric.csv')
    else:
        mase = \
            pd.read_sql(con=framework_application_obj.db_connect,
                        sql='SELECT "MASE","regression","target","D","market","freq","timestamp", "volatility_regime" '
                            'FROM metric;')
    mase['timestamp'] = pd.to_datetime(mase['timestamp'], utc=True)
    mase = mase.set_index(['D', 'timestamp'])
    model_ranking = mase.groupby(by=[pd.Grouper(key='regression'), pd.Grouper(key='freq')])[['MASE']].mean()
    with open(f'{args.target_path}/crypto_option_spot/mase_per_regression_freq_pair.txt', 'w') as \
            mase_per_regression_freq_pair_file:
        mase_per_regression_freq_pair_file.writelines(model_ranking.unstack(0).droplevel(axis=1, level=0).loc[
            ['15min', '30min', '1h'], ['lasso', 'elastic', 'lightgbm', 'lstm']].to_latex())
    best_model_tags = list(model_ranking.idxmin().iloc[0])
    ####################################################################################################################
    ## MASE Reporting
    ####################################################################################################################
    if os.path.exists(f'{args.target_path}/metric.csv'):
        mase_comparison = pd.read_csv(f'{args.target_path}/metric.csv')
        mase_comparison = \
            mase_comparison.query(f'(regression == "{best_model_tags[0]}") & (freq == "{best_model_tags[-1]}")')
    else:
        mase_comparison = \
            pd.read_sql(con=framework_application_obj.db_connect,
                        sql=f'SELECT "timestamp","MASE","regression","target","D","market","freq","volatility_regime" '
                            f'FROM metric WHERE regression="{best_model_tags[0]}" AND freq="{best_model_tags[-1]}";')
    mase_comparison['timestamp'] = pd.to_datetime(mase_comparison['timestamp'], utc=True)
    mase_comparison = mase_comparison.set_index(['D', 'timestamp'])
    mase_comparison = mase_comparison.groupby(by=pd.Grouper(key='market'))['MASE'].mean()
    best_market = mase_comparison.idxmin()
    best_model = mase.query(f'(regression == "{best_model_tags[0]}") & (freq == "{best_model_tags[-1]}")')
    mase_global = \
        best_model.groupby(by=[pd.Grouper(level='D'), pd.Grouper(key='volatility_regime'),
                               pd.Grouper(key='market')])[['MASE']].mean().reset_index()
    mase_global['D'] = mase_global['D'].transform(lambda x: f'<i>{x.lower()}</i>')
    mase_global['market'] = np.where(mase_global['market'] == 1, '<i>X<sup>(S,O)</sup></i>', '<i>X<sup>S</sup></i>')
    mase_per_volatility = \
        best_model.groupby(by=[pd.Grouper(level='D'),
                               pd.Grouper(key='volatility_regime'), pd.Grouper(key='market')])[['MASE']].mean()
    mase_per_volatility_copy = mase_per_volatility.copy()
    mase_per_volatility_copy = \
        mase_per_volatility.groupby(by=[pd.Grouper(level='D'),
                                        pd.Grouper(level='volatility_regime')],
                                    group_keys=False).apply(lambda x: 100 * x.pct_change()).round(4)
    mase_per_volatility = mase_per_volatility_copy.fillna(mase_per_volatility)
    mase_per_volatility = mase_per_volatility.unstack([1, -1])
    mase_per_volatility = mase_per_volatility.loc[:, [('MASE', 'Low', 0), ('MASE', 'Low', 1),
                                                      ('MASE', 'Normal', 0), ('MASE', 'Normal', 1),
                                                      ('MASE', 'High', 0), ('MASE', 'High', 1)]]
    mase_per_volatility.index = mase_per_volatility.index.str.lower()
    mase_per_volatility = mase_per_volatility.loc[d_order.categories, :]
    mase_per_volatility.loc[:, mase_per_volatility.columns.get_level_values(2) == 1] = \
        mase_per_volatility.loc[:, mase_per_volatility.columns.get_level_values(2) == 1].round(2)
    mase_per_volatility.columns = mase_per_volatility.columns.set_names('Market Regimes', level=0)
    mase_per_volatility = mase_per_volatility.style.format(decimal='.', thousands=',', precision=4)
    os.makedirs(f'{args.target_path}/crypto_option_spot', exist_ok=True)
    with open(f'{args.target_path}/crypto_option_spot/mase_per_volatility.txt', 'w') as mase_per_volatility_file:
        mase_per_volatility_file.writelines(mase_per_volatility.to_latex())
    ####################################################################################################################
    ## Features and target values
    ####################################################################################################################
    if os.path.exists(f'{args.target_path}/spot_features.csv'):
        features = pd.read_csv(f'{args.target_path}/spot_features.csv')
    else:
        features = pd.read_sql(con=framework_application_obj.db_connect, sql='SELECT * FROM spot_features;')
    features['timestamp'] = pd.to_datetime(features['timestamp'], utc=pytz.UTC)
    targets = features[['underlying_name', 'timestamp', 'target', 'target_values']]
    targets = pd.pivot_table(targets, index=['target', 'timestamp'], columns='underlying_name',
                             values='target_values')
    features = features.drop(['target', 'target_values'], axis=1)
    features = pd.pivot(features, index=['underlying_name', 'timestamp'], columns='features',
                        values='features_values')
    ####################################################################################################################
    ## Feature importance
    ####################################################################################################################
    # if best_market == 1:
    if os.path.exists(f'{args.target_path}/option_features.csv'):
        option_features = pd.read_csv(f'{args.target_path}/option_features.csv')
    else:
        option_features = pd.read_sql(con=framework_application_obj.db_connect,
                                      sql='SELECT * FROM option_features;')
    option_features['timestamp'] = pd.to_datetime(option_features['timestamp'], utc=pytz.UTC)
    option_features = pd.pivot_table(option_features, index=['underlying_name', 'timestamp'], columns='features',
                                     values='features_values')
    features = features.join(option_features)
    if os.path.exists(f'{args.target_path}/features_importance.csv'):
        feature_importance = pd.read_csv(f'{args.target_path}/features_importance.csv')
        feature_importance = \
            feature_importance.query(
                f'regression == "{best_model_tags[0].lower()}" & '
                f'(freq == "{best_model_tags[-1]}") & (market == {best_market})').drop('volatility_regime', axis=1)
    else:
        feature_importance = pd.read_sql(con=framework_application_obj.db_connect,
                                         sql=f'SELECT "features","D","freq","market","MASE","target","timestamp" '
                                             f"FROM features_importance "
                                             f"WHERE regression = '{best_model_tags[0].lower()}' AND "
                                             f"freq = '{best_model_tags[-1]}' AND market = '{best_market}'")
    feature_importance = feature_importance.drop(['freq', 'market'], axis=1)
    feature_importance['timestamp'] = pd.to_datetime(feature_importance['timestamp'], utc=pytz.UTC)
    feature_importance = feature_importance.set_index(['D', 'timestamp'])
    feature_importance = \
        pd.concat([feature_importance, volatility_regime.reindex(feature_importance.index).ffill()], axis=1)
    feature_importance_ranking = feature_importance.groupby(by=[pd.Grouper(key='target'),
                                                                pd.Grouper(key='volatility_regime'),
                                                                pd.Grouper(key='features'),
                                                                pd.Grouper(level='D')])[['MASE']].mean().dropna()
    feature_importance_ranking = \
        feature_importance_ranking.groupby(by=[pd.Grouper(level='target'), pd.Grouper(level='volatility_regime'),
                                               pd.Grouper(level='D')],
                                           group_keys=False).apply(
            lambda g: g.nlargest(top_features, 'MASE')).reset_index()
    feature_importance_ranking = \
        feature_importance_ranking.sort_values('MASE', ascending=False).groupby(
            ['target', 'D', 'volatility_regime'],
            group_keys=False).apply(
            lambda g: g.assign(rank=g['MASE'].rank(method='first', ascending=False).astype(int).astype(str)))
    feature_importance_ranking['features'] = \
        feature_importance_ranking.features.str.replace(r'_1([DWM])', lambda m: f"_1{m.group(1).lower()}",
                                                        regex=True)
    feature_importance_ranking['D'] = feature_importance_ranking['D'].str.lower()
    if os.path.exists(f'{args.target_path}/forecasts.csv'):
        chunks = pd.read_csv(f'{args.target_path}/forecasts.csv', chunksize=1_000_000)
        data = pd.concat(chunk.query('regression == @best_model_tags[0].lower() and freq == @best_model_tags[-1] '
                                     'and market == @best_market') for chunk in chunks)
    else:
        data = pd.read_sql(con=framework_application_obj.db_connect,
                           sql=f'SELECT * FROM forecasts '
                               f'WHERE regression = "{best_model_tags[0].lower()}" '
                               f'AND freq = "{best_model_tags[-1]}" '
                               f'AND market = "{best_market}";').set_index('timestamp')
    data = data.set_index('timestamp')
    data.index = pd.to_datetime(data.index, utc=True)
    data = \
        pd.pivot_table(data=data, columns=['underlying_name', 'target', 'D'], values=['values', 'forecast'],
                       index='timestamp')
    data.index = pd.to_datetime(data.index, utc=True)
    framework_application_obj.data = data.resample(best_model_tags[-1]).sum().dropna()
    framework_application_obj.symbol = framework_application_obj.data.columns.get_level_values(1).unique().tolist()
    date_range = pd.date_range(start=args.start_date, end=args.end_date, freq='1D').tolist()
    framework_application_obj.process_daily_option_chain(best_model_tags[-1], f'{args.target_path}', date_range)
    framework_application_obj.iv_regression(best_model_tags[-1])
    ####################################################################################################################
    ## SVIX
    ####################################################################################################################
    for aggregated in [True, False]:
        svix2 = framework_application_obj.svix(best_model_tags[-1], aggregated)
        svix = (svix2 ** .5).unstack()
        svix.index = svix.index.set_names(['D', 'timestamp']) if aggregated else \
            svix.index.set_names(['D', 'underlying_name', 'timestamp'])
        svix.name = 'svix'
        svix = pd.DataFrame(svix.reset_index())
        if aggregated:
            svix = svix.assign(underlying_name=None)
        svix.assign(aggregated={True: 1, False: 0}[aggregated]).to_sql(
            con=framework_application_obj.db_connect, name=f'svix', if_exists='append')
        print(f'[Insertion]: SVIX {dict([(True, "aggregated"), (False, "")])[aggregated]} has been inserted '
              f'into the database.')
    ####################################################################################################################
    ## VRP
    ####################################################################################################################
    returns = data.xs('forecast', axis=1, level=0).xs('returns', axis=1, level=1).dropna()
    returns.columns = returns.columns.set_levels(returns.columns.levels[-1].str.lower(), level=-1)
    rv = data.xs('forecast', axis=1, level=0).xs('RV', axis=1, level=1)
    rv = rv.resample(best_model_tags[-1]).sum().replace(0, np.nan)
    rv.columns = rv.columns.set_levels(rv.columns.levels[-1].str.lower(), level=-1).dropna()
    svix = pd.read_sql(sql='SELECT * FROM svix', con=framework_application_obj.db_connect)
    svix['timestamp'] = pd.to_datetime(svix['timestamp'], utc=pytz.UTC)
    svix = svix.set_index(['aggregated', 'timestamp'])
    vrp = dict()
    scaling_factor = pd.to_timedelta('360D') // pd.to_timedelta(best_model_tags[-1])
    for aggregated, underlying_name in list(product([False, True], list(pairs_order.categories))):
        tmp_svix = svix.xs(int(aggregated), level=0, axis=0)
        if aggregated:
            if 'agg_svix' not in vars():
                agg_svix = tmp_svix.drop('underlying_name', axis=1)
                agg_svix = pd.pivot_table(agg_svix.reset_index(), columns='D', values='svix', index='timestamp')
                agg_svix.index = pd.to_datetime(agg_svix.index, utc=pytz.UTC)
            vrp[(aggregated, underlying_name)] = \
                agg_svix.subtract((rv[underlying_name] ** .5)).mul(scaling_factor ** .5)
        else:
            tmp_svix = pd.pivot_table(tmp_svix.reset_index(), columns='D', values='svix', index='timestamp')
            tmp_svix.index = pd.to_datetime(tmp_svix.index, utc=pytz.UTC)
            vrp[(aggregated, underlying_name)] = \
                tmp_svix.subtract((rv[underlying_name] ** .5)).mul(scaling_factor ** .5)
        vrp[(aggregated, underlying_name)] = vrp[(aggregated, underlying_name)].dropna(axis=1, how='all').dropna()
        vrp[(aggregated, underlying_name)] = pd.DataFrame(vrp[(aggregated, underlying_name)].unstack()).assign(
            underlying_name=underlying_name, aggregated=int(aggregated)).rename(columns={0: 'vrp'})
        pd.concat(vrp.values()).to_sql(name='vrp', con=framework_application_obj.db_connect, if_exists='replace')
        print(f'[Insertion]: VRP {dict([(True, "aggregated"), (False, "")])[aggregated]} has been inserted '
              f'into the database.')
    ####################################################################################################################
    ## VRP vs SVIX
    ####################################################################################################################
    vrp_svix_aggregated = dict()
    vrp = pd.concat(vrp.values())
    vrp = vrp.set_index(['aggregated', 'underlying_name'],
                        append=True).swaplevel(2, 0).swaplevel(2, 1).swaplevel(-1, 1).swaplevel(-1, -2)
    vrp = vrp.rename(columns={'vrp': '$\\text{VRP}_{0 \\rightarrow D}$'})
    vrp.index = vrp.index.set_names(['aggregated', 'underlying_name', 'D', 'timestamp'])
    for aggregated in [True, False]:
        if aggregated:
            tmp_svix = pd.DataFrame(pd.Series(agg_svix.unstack(), name='$\\text{Crypto SVIX}_{0 \\rightarrow D}$'))
            tmp_svix.index = tmp_svix.index.set_names(['D', 'timestamp'])
        else:
            tmp_svix = svix.xs(int(aggregated), axis=0, level=0).dropna()[['D', 'underlying_name', 'svix']]
            tmp_svix = tmp_svix.set_index(['underlying_name', 'D'], append=True).swaplevel(1, 0).swaplevel(-1, 1)
            tmp_svix.index = tmp_svix.index.set_names(['underlying_name', 'D', 'timestamp'])
        tmp_svix = tmp_svix.rename(columns={'svix': '$\\text{Crypto SVIX}_{0 \\rightarrow D}$'})
        vrp_svix = pd.DataFrame(pd.Series(returns.unstack(), name='$\\text{returns}_{0 \\rightarrow D}$').rename_axis(
            ['underlying_name', 'D', 'timestamp'])).join(
            tmp_svix, on=['D', 'timestamp'] if aggregated else ['underlying_name', 'D', 'timestamp']).join(
            vrp.xs(int(aggregated), axis=0, level=0), on=['underlying_name', 'D', 'timestamp']).dropna()
        vrp_svix = vrp_svix.loc[:, ['$\\text{Crypto SVIX}_{0 \\rightarrow D}$', '$\\text{VRP}_{0 \\rightarrow D}$',
                                    '$\\text{returns}_{0 \\rightarrow D}$']]
        vrp_svix = \
            vrp_svix.groupby(by=[pd.Grouper(level=0), pd.Grouper(level=1)],
                             group_keys=False).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        vrp_svix = vrp_svix.groupby(by=[pd.Grouper(level=0), pd.Grouper(level=1)], group_keys=False).apply(
            vrp_svix_effect)
        vrp_svix = vrp_svix.reset_index().assign(aggregated=int(aggregated))
        vrp_svix = pd.melt(vrp_svix, id_vars=['aggregated', 'underlying_name', 'D'], value_name='MASE',
                           var_name='Factors')
        vrp_svix = \
            pd.pivot(vrp_svix, columns=['underlying_name', 'Factors'], index=['aggregated', 'D'], values='MASE')
        vrp_svix = vrp_svix.loc[:, pd.MultiIndex.from_product([list(pairs_order.categories),
                                                               ['$\\text{VRP}_{0 \\rightarrow D}$',
                                                                '$\\text{Crypto SVIX}_{0 \\rightarrow D}$']])]
        tmp_vrp_svix = \
            vrp_svix.loc[:, pd.MultiIndex.from_product([list(pairs_order.categories),
                                                        ['$\\text{Crypto SVIX}_{0 \\rightarrow D}$',
                                                         '$\\text{VRP}_{0 \\rightarrow D}$']])].copy()
        vrp_svix = vrp_svix.round(4)
        vrp_svix = vrp_svix.where(vrp_svix.values <= tmp_vrp_svix.values, vrp_svix.map(lambda x: f"\\textbf{{{x}}}"))
        vrp_svix_aggregated[(aggregated, underlying_name)] = vrp_svix
vrp_svix_aggregated = pd.concat(vrp_svix_aggregated.values())
vrp_svix_aggregated = vrp_svix_aggregated.loc[list(product([0, 1], d_order.categories)), :]
with open(f'{args.target_path}/crypto_option_spot/vrp_svix.txt', 'w') as vrp_svix_file:
    vrp_svix_file.writelines(vrp_svix_aggregated.to_latex(escape=False))
########################################################################################################################
## Backtesting (P&L + Sharpe Ratio) + P&L Attribution analysis
########################################################################################################################
pnl_dd = dict()
trading_volume_dd = dict()
for target in ['RV', 'returns']:
    if target == 'RV':
        for aggregated in [True, False]:
            pnl, trading_volume = \
                framework_application_obj.backtest(vrp=vrp.xs(int(aggregated), 0, 0), transaction_cost=True,
                                                   spread=True, freq=best_model_tags[-1], target=target)
            pnl.index, trading_volume.index = \
                pnl.index.set_names(['timestamp']), trading_volume.index.set_names(['timestamp'])
            pnl_dd[(int(aggregated), target)] = pnl
            trading_volume_dd[(int(aggregated), target)] = trading_volume
    else:
        pnl = framework_application_obj.backtest(transaction_cost=True, spread=True, freq=best_model_tags[-1],
                                                 target=target)
        pnl.columns = pnl.columns.set_levels(pnl.columns.levels[-1].str.lower(), level=1)
        pnl_dd[('', target)] = pnl
pnl = pd.concat(pnl_dd)
trading_volume = pd.concat(trading_volume_dd)
pnl.index = pnl.index.set_names(['aggregated', 'target', 'timestamp'])
trading_volume.index = trading_volume.index.set_names(['aggregated', 'target', 'timestamp'])
pnl.to_sql(name='pnl', con=framework_application_obj.db_connect, if_exists='replace', index=True)
print(f'[Insertion]: P&L has been inserted into the database.')
trading_volume.to_sql(name='trading_volume', con=framework_application_obj.db_connect, if_exists='replace', index=True)
print(f'[Insertion]: Trading volume has been inserted into the database.')

daily_pnl = pnl.groupby(by=[pd.Grouper(level=0), pd.Grouper(level=1), pd.Grouper(level=2, freq='1D')]).mean()
idx = pd.MultiIndex.from_tuples(list(product([0, 1], ['RV'], ['Low', 'Normal', 'High'], ['1d', '1w', '1m']))
                                + list(product([''], ['returns'], ['Low', 'Normal', 'High'], ['1d', '1w', '1m'])),
                                names=['aggregated', 'target', 'volatility_regime', 'D'])
sharpe_mean = pnl.groupby(by=[pd.Grouper(level=0), pd.Grouper(level=1), pd.Grouper(level=2, freq='1D')]).mean().groupby(
    by=[pd.Grouper(level=0), pd.Grouper(level=1)]).mean().stack()
sharpe_std = pnl.groupby(by=[pd.Grouper(level=0), pd.Grouper(level=1), pd.Grouper(level=2, freq='1D')]).mean().groupby(
    by=[pd.Grouper(level=0), pd.Grouper(level=1)]).std().stack()
sharpe = sharpe_mean.div(sharpe_std).mul(np.sqrt(360)).loc[idx, pairs_order.categories]
sharpe_per_volatility_regime = pnl.groupby(by=[pd.Grouper(level=0), pd.Grouper(level=1),
                                               pd.Grouper(level=2, freq='1D')]).mean().stack().stack()
volatility_regime.index = volatility_regime.index.set_levels(volatility_regime.index.levels[-1].str.lower(), level=0)
sharpe_per_volatility_regime = \
    pd.DataFrame(sharpe_per_volatility_regime.rename('P&L')).join(volatility_regime, on=['D', 'timestamp'])
sharpe_per_volatility_regime = \
    sharpe_per_volatility_regime.groupby(by=[pd.Grouper(level=0), pd.Grouper(level=1), pd.Grouper(level=4),
                                             pd.Grouper(level=3), pd.Grouper(level=-1)]).apply(
        lambda x: x['P&L'].mean()/x['P&L'].std()).mul(np.sqrt(360))
sharpe_per_volatility_regime = \
    sharpe_per_volatility_regime.swaplevel(-1, -3).unstack().loc[idx, pairs_order.categories]


factor_analysis = pnl_explained_obj.pnl_analysis(pnl=pnl, features=features, freq=best_model_tags[-1],
                                                 ranking=feature_importance_ranking, targets=data.xs('values',
                                                                                                     level=0, axis=1))
factor_analysis.to_sql(name='factor_analysis', con=framework_application_obj.db_connect, if_exists='replace',
                       index=True)
print(f'[Insertion]: Factor analysis has been inserted into the database.')
# tail_analysis.to_sql(name='tail_analysis', con=framework_application_obj.db_connect, if_exists='replace', index=True)
# print(f'[Insertion]: Tail analysis has been inserted into the database.')