import datetime
import itertools
import os
import pdb

import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from constants import DATE_FORMAT, DAYS_CONVERTER, SYMBOLS, VOL_REGIMES, ATM_MONEYNESSES
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from scipy.stats.distributions import norm
from pandas.api.types import CategoricalDtype
import polars as pl
from itertools import product
import argparse
import concurrent


########################################################################################################################
## Functions used to parallelise FrameworkApplication methods
########################################################################################################################


def process_daily_option_chain_date(date: datetime.datetime, directory: str,
                                    freq: str) -> Tuple[datetime.datetime, pl.LazyFrame]:
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


def find_forward_price(iv: Dict[(str, Dict[(str, np.ndarray)])], sym: str, s0: np.array, tau: float) -> List[float]:
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
        iv_interpolation_coef = iv_interpolation_coef.reindex(
            pd.date_range(start=date, end=date + relativedelta(days=1),
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
            otm = strikes.ge(fwd_price_sym).values
            prices = - (strikes.values * norm.cdf(d2)) + norm.cdf(d1) * fwd_price_sym
        else:
            otm = strikes.lt(fwd_price_sym).values
            prices = strikes.values * norm.cdf(-d2) - norm.cdf(-d1) * fwd_price_sym
        svix[sym][option_type] = np.sqrt((2 / T) * (np.nan_to_num((prices * otm * strikes_grid.values)).sum(axis=1) /
                                                    fwd_price_sym.reshape(fwd_price_sym.shape[0]) ** 2))
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


class FrameworkApplication:
    _option_chain = dict()
    _option_moneyness = None
    _vrp = None
    _iv_interpolation_coef = None

    def __init__(self, target_path: str):
        self._underlying_prices = \
            pl.scan_parquet(f'{target_path}/*/*/spot/*-*-*').select([pl.col('symbol').alias('underlying_name'),
                                                                     pl.col('timestamp'), pl.col('mid_price[0]'),
                                                                     pl.col('spreads[0]')])
        self._underlying_prices = self._underlying_prices.collect()
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
            if not os.path.exists(f'{directory}/{date.year}/option_chain/{date.strftime(DATE_FORMAT)}'):
                date, daily_option_chain_date = process_daily_option_chain_date(date, directory, freq)
                daily_option_chain_date = daily_option_chain_date.filter(pl.col('underlying_name').is_in(SYMBOLS))
                daily_option_chain_date = daily_option_chain_date.with_columns(
                    ((pl.col('expiration')-pl.col('timestamp')).dt.total_days() / 360).alias('tau'))
                FrameworkApplication._option_chain[date.date()] = daily_option_chain_date.collect().to_pandas()
                os.makedirs(f'{directory}/option_chain', exist_ok=True)
                FrameworkApplication._option_chain[date.date()].to_parquet(
                    f'{directory}/option_chain/{date.strftime(DATE_FORMAT)}')
            else:
                try:
                    tmp_df = pd.read_parquet(f"{directory}/{date.year}/options/"
                                             f"{(date-relativedelta(days=1)).strftime(DATE_FORMAT)}")
                except FileNotFoundError:
                    tmp_df = pd.read_parquet(f"{directory}/{date.year-1}/options/"
                                             f"{(date - relativedelta(days=1)).strftime(DATE_FORMAT)}")
            tmp_df['tau'] = (tmp_df['expiration'] - tmp_df['timestamp']).dt.total_seconds() / (60 * 60 * 24 * 360)
            tmp_df = tmp_df.set_index('timestamp').groupby('symbol').resample(freq.replace('in', '')).last()
            sqrt_term = np.sqrt(tmp_df['mark_iv'] * tmp_df['tau'])
            tmp_df['d1'] = (np.log(tmp_df['underlying_price'] / tmp_df['strike_price']) +
                            0.5 * tmp_df['mark_iv'] * tmp_df['tau'])
            tmp_df['d1'] = tmp_df['d1'] / sqrt_term
            tmp_df['d2'] = tmp_df['d1'] - sqrt_term
            tmp_df['bs_price'] = (tmp_df['underlying_price'] * norm.cdf(tmp_df['d1']) -
                                  tmp_df['strike_price'] * norm.cdf(tmp_df['d2']))
            tmp_df['mid_price'] = tmp_df['mid_price'].fillna(tmp_df['bs_price'])
            tmp_df = tmp_df.drop(columns=['bs_price', 'd1', 'd2'])
            FrameworkApplication._option_chain[(date-relativedelta(days=1)).date()] = tmp_df
        return None

    @staticmethod
    def iv_regression(freq: str) -> None:
        symbol = SYMBOLS
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
        return pd.concat(iv_interpolation_data, axis=1).ffill()

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all performance tables.')
    # parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-02')
    # parser.add_argument('--end_date', type=str, help='End date.', default='2025-07-01')
    # parser.add_argument('--top_features', type=int,
    #                     help='Number of top features to consider in feature importance analysis', default=5)
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='./data_centre')
    parser.add_argument('--type', type=str, help='Type of table to generate', default='mase_reporting')
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
    # framework_application_obj = FrameworkApplication(target_path=args.target_path)
    # top_features = args.top_features
    # rv_mkt = framework_application_obj.underlying_rv.set_index('timestamp').stack().swaplevel(0, 1)
    # df_idx = rv_mkt.index.to_frame()
    # df_idx['timestamp'] = df_idx['timestamp'].astype('datetime64[ns, UTC]')
    # rv_mkt.index = pd.MultiIndex.from_frame(df_idx)
    # rv_mkt = \
    #     pd.concat([pd.DataFrame(rv_mkt.groupby(
    #         by=[pd.Grouper(level=0),
    #             pd.Grouper(level=1, freq=f'{d.upper()}E' if d == 'M' else d.upper())]).sum() *
    #                ((pd.to_timedelta('360D')//pd.to_timedelta(f'{DAYS_CONVERTER[d.upper()]}D')) ** .5)).assign(
    #         D=d.upper()) for d in d_order.categories])
    # rv_mkt = rv_mkt.groupby([pd.Grouper(key='D'),
    #                          pd.Grouper(level=1)]).mean().rename(columns={0: 'volatility_regime'})
    # volatility_regime = rv_mkt.groupby(pd.Grouper(level='D'), group_keys=False)
    # volatility_regime = \
    #     volatility_regime.apply(lambda x: pd.qcut(x['volatility_regime'], q=[0, .33, .66, 1], labels=VOL_REGIMES))
    ####################################################################################################################
    ## Model selection
    ####################################################################################################################
    mase = pd.read_csv(f'{args.target_path}/crypto_option_spot/metric.csv')
    model_ranking = mase.groupby(by=[pd.Grouper(key='target'),
                                     pd.Grouper(key='regression'),
                                     pd.Grouper(key='freq')])[['MASE']].mean()
    for target in ['returns', 'RV']:
        model_ranking_tmp = model_ranking.xs(target, axis=0, level=0, drop_level=True)
        with open(f'{args.target_path}/crypto_option_spot/mase_per_regression_freq_pair_{target}.txt', 'w') as \
                mase_per_regression_freq_pair_file_tmp:
                mase_per_regression_freq_pair_file_tmp.writelines(
                    model_ranking_tmp.unstack(0).droplevel(axis=1, level=0).loc[
                    ['15min', '30min', '1h'], ['lasso', 'lightgbm', 'softs']].to_latex())
    best_model_tags = model_ranking.groupby('target').idxmin().to_dict()['MASE']
    mase_comparison = \
        model_ranking.groupby(['target', 'regression', 'freq'])['MASE'].mean().groupby(level='target').idxmin()
    data_per_target = dict()
    for target, values in mase_comparison.items():
        chunks = pd.read_csv(f'{args.target_path}/crypto_option_spot/forecasts.csv', chunksize=1_000_000)
        tmp = pd.concat(chunk.query('target == @values[0] and regression == @values[1] and freq == @values[-1]')
                        for chunk in chunks)
        tmp = tmp.rename(columns={'y_hat': 'forecast', 'y': 'values'})
        tmp = tmp.set_index('timestamp')
        tmp.index = pd.to_datetime(tmp.index, utc=True)
        tmp = pd.pivot_table(data=tmp, columns=['underlying_name', 'target', 'D'], values=['values', 'forecast'],
                             index='timestamp')
        data_per_target[target] = tmp
    # framework_application_obj.data = data_per_target
    # scaling_factor = pd.to_timedelta('360D') // pd.to_timedelta(best_model_tags['RV'][-1])
    # rv_idx = pd.IndexSlice
    # data_per_target['RV'].loc[:, rv_idx[:, :, 'RV', :]] = \
    #     data_per_target['RV'].loc[:, rv_idx[:, :, 'RV', :]].pow(.5).mul(scaling_factor ** .5)
    # framework_application_obj.symbol = SYMBOLS
    # date_range = pd.date_range(start=args.start_date, end=args.end_date, freq='1D').tolist()
    # framework_application_obj.process_daily_option_chain(best_model_tags['RV'][-1],
    #                                                      f'{args.target_path}', date_range)
    ####################################################################################################################
    ## MASE Reporting
    ####################################################################################################################
    if args.type == 'mase_reporting':
        metric_per_target = dict()
        tmp = pd.read_csv(f'{args.target_path}/crypto_option_spot/metric.csv')
        for target, values in mase_comparison.items():
            # chunks = pd.read_csv(f'{args.target_path}/crypto_option_spot/metric.csv', chunksize=1_000_000)
            # tmp = pd.concat(chunk.query('target == @values[0] and regression == @values[1] and freq == @values[-1]')
            #                 for chunk in chunks)
            metric_per_target[target] = \
                tmp.query('target == @values[0] and regression == @values[1] and freq == @values[-1]')
        mase_comparison = pd.concat(metric_per_target.values())
        for target, values in best_model_tags.items():
            regression = best_model_tags[target][1]
            freq = best_model_tags[target][-1]
            best_model = mase_comparison.query('regression == @regression and freq == @freq and target == @target')
            mase_global = \
                best_model.groupby(by=[pd.Grouper(key='D'),#, pd.Grouper(key='volatility_regime'),
                                       pd.Grouper(key='market')])[['MASE']].mean().reset_index()
            mase_global['D'] = mase_global['D'].transform(lambda x: f'<i>{x.lower()}</i>')
            mase_global['market'] = \
                np.where(mase_global['market'] == 1, '<i>X<sup>(S,O)</sup></i>', '<i>X<sup>S</sup></i>')
            mase_conditional = \
                best_model.groupby(by=[pd.Grouper(key='D'),
                                       pd.Grouper(key='volatility_regime'), pd.Grouper(key='market')])[['MASE']].mean()
            mase_conditional_copy = mase_conditional.copy()
            mase_conditional_copy = \
                mase_conditional.groupby(by=[pd.Grouper(level='D'),
                                             pd.Grouper(level='volatility_regime')],
                                         group_keys=False).apply(lambda x: 100 * x.pct_change()).round(4)
            mase_conditional = mase_conditional_copy.fillna(mase_conditional)
            mase_conditional = mase_conditional.unstack([1, -1])
            mase_conditional = mase_conditional.loc[:, [('MASE', 'Low', 0), ('MASE', 'Low', 1),
                                                        ('MASE', 'Normal', 0), ('MASE', 'Normal', 1),
                                                        ('MASE', 'High', 0), ('MASE', 'High', 1)]]
            mase_conditional.index = mase_conditional.index.str.lower()
            mase_conditional = mase_conditional.loc[d_order.categories, :]
            mase_conditional.loc[:, mase_conditional.columns.get_level_values(2) == 1] = \
                mase_conditional.loc[:, mase_conditional.columns.get_level_values(2) == 1].round(2)
            mase_conditional.columns = mase_conditional.columns.set_names('Market Regimes', level=0)
            mase_unconditional = best_model.groupby(by=[pd.Grouper(key='D'), pd.Grouper(key='market')])[['MASE']].mean()
            mase_unconditional_copy = mase_unconditional.copy()
            mase_unconditional = mase_unconditional.groupby(by=pd.Grouper(level='D'), group_keys=False).apply(
                lambda x: 100 * x.pct_change()).round(4)
            mase_unconditional = mase_unconditional.fillna(mase_unconditional_copy)
            mase_unconditional = mase_unconditional.unstack(-1)
            mase_unconditional.index = mase_unconditional.index.str.lower()
            mase_unconditional = mase_unconditional.loc[d_order.categories, :]
            mase_unconditional.columns = \
                pd.MultiIndex.from_product([['MASE'], [''], [0, 1]],
                                           names=['Market Regimes', 'volatility_regime', 'market'])
            mase_performance = pd.concat([mase_unconditional, mase_conditional], axis=1)
            mase_performance = mase_performance.style.format(decimal='.', thousands=',', precision=4)
            with open(f'{args.target_path}/crypto_option_spot/mase_performance_{target}.txt', 'w') as \
                    mase_performance_file:
                mase_performance_file.writelines(mase_performance.to_latex())
    ####################################################################################################################
    ## SVIX
    ####################################################################################################################
    if args.type == 'svix_rv_vrp_effect':
        if os.path.exists(f'{args.target_path}/crypto_option_spot/svix.csv'):
            svix = pd.read_csv(f'{args.target_path}/crypto_option_spot/svix.csv', index_col='index')
            svix['svix'] = svix['svix']/100
            svix['timestamp'] = pd.to_datetime(svix['timestamp'], utc=True)
            svix = svix.set_index(['aggregated', 'underlying_name', 'D', 'timestamp'])
            svix_agg = \
                pd.concat([svix.xs(1, axis=0, level=0,
                                   drop_level=False).droplevel(axis=0, level=1).assign(underlying_name=underlying_name)
                           for underlying_name in SYMBOLS]).set_index('underlying_name', append=True)
            svix_agg = svix_agg.swaplevel(-1, 1).swaplevel(-1, 2)
            svix_non_agg = svix.xs(0, axis=0, level=0, drop_level=False)
            svix = pd.concat([svix_agg, svix_non_agg]).reset_index()
        else:
            framework_application_obj.iv_regression(best_model_tags[-1])
            for aggregated in [True, False]:
                svix2 = framework_application_obj.svix(best_model_tags[-1], aggregated)
                svix = ((scaling_factor * svix2) ** .5).unstack()
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
        ################################################################################################################
        ## VRP
        ################################################################################################################
        returns = data_per_target['returns'].xs('values', axis=1, level=0).xs('returns', axis=1, level=1).dropna()
        returns.columns = returns.columns.set_levels(returns.columns.levels[-1].str.lower(), level=-1)
        rv = data_per_target['RV'].xs('forecast', axis=1, level=0).xs('RV', axis=1, level=1)
        rv = rv.resample(best_model_tags[-1]).sum().replace(0, np.nan)
        rv.columns = rv.columns.set_levels(rv.columns.levels[-1].str.lower(), level=-1).dropna()
        if os.path.exists(f'{args.target_path}/crypto_option_spot/vrp.csv'):
            vrp = \
                pd.merge(svix, rv.unstack().rename('rv').dropna().reset_index(),
                         on=['underlying_name', 'D', 'timestamp'], how='right')
            vrp['vrp'] = vrp['svix'] - vrp['rv']
            vrp = vrp.set_index(['aggregated', 'underlying_name', 'D', 'timestamp'])
        else:
            svix['timestamp'] = pd.to_datetime(svix['timestamp'], utc=pytz.UTC)
            svix = svix.set_index(['aggregated', 'timestamp'])
            vrp = dict()
            for aggregated, underlying_name in list(product([False, True], list(pairs_order.categories))):
                tmp_svix = svix.xs(int(aggregated), level=0, axis=0)
                if aggregated:
                    if 'agg_svix' not in vars():
                        agg_svix = tmp_svix.drop('underlying_name', axis=1)
                        agg_svix = pd.pivot_table(agg_svix.reset_index(), columns='D', values='svix', index='timestamp')
                        agg_svix.index = pd.to_datetime(agg_svix.index, utc=pytz.UTC)
                    vrp[(int(aggregated), underlying_name)] = \
                        agg_svix.reindex(rv.index).subtract(rv[underlying_name].values).dropna()
                else:
                    tmp_svix = pd.pivot_table(tmp_svix.reset_index(), columns=['underlying_name', 'D'], values='svix',
                                              index='timestamp')
                    tmp_svix.index = pd.to_datetime(tmp_svix.index, utc=pytz.UTC)
                    vrp[(int(aggregated), underlying_name)] = \
                        tmp_svix.xs(underlying_name, axis=1,
                                    level=0).reindex(rv.index).subtract(rv[underlying_name].values)
                vrp[(int(aggregated), underlying_name)] = \
                    vrp[(int(aggregated), underlying_name)].dropna(axis=1, how='all').dropna()
                vrp[(int(aggregated), underlying_name)] = \
                    pd.DataFrame(vrp[(int(aggregated), underlying_name)].unstack()).rename(columns={0: 'vrp'})
                vrp[(int(aggregated), underlying_name)] = \
                    vrp[(int(aggregated), underlying_name)].assign(underlying_name=underlying_name,
                                                                   aggregated=int(aggregated))
            vrp_0 = pd.concat([vrp[(0, symbol)] for symbol in pairs_order.categories])
            vrp_1 = pd.concat([vrp[(1, symbol)] for symbol in pairs_order.categories])
            vrp = pd.concat([vrp_0, vrp_1])
            vrp.to_csv(f'{args.target_path}/crypto_option_spot/vrp.csv')
