import concurrent.futures
import glob
import itertools
import pdb
import re
import sqlite3
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import RandomSampler
from data_centre.data import Reader
from dateutil.relativedelta import relativedelta
from constants import DATE_FORMAT, N_JOBS, N_TRIALS, FEES_SPOT, FEES_OPTIONS, FEES_STRADDLE_CAP, MARKETS,\
    OPTION_CHAIN_DATA_COLUMNS, YEARS
from functools import partial
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import plotly.express as px
import statsmodels.formula.api as sm


class FrameworkApplication:
    _reader_obj = Reader()
    _option_prices = None
    _option_moneyness = None
    _vrp = None
    _db_connect = sqlite3.connect('../data_centre/database.db')

    def __init__(self, application: str = None, data: pd.DataFrame = None, mase: pd.DataFrame = None,
                 best_models: pd.Series = None):
        self._data = data
        self._mase = mase
        self._best_models = best_models
        if application not in ['trading', 'vrp', None]:
            raise KeyError('Provide a valid application of this framework: trading or vrp.')
        self._application = application

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        self._data = data

    @property
    def application(self) -> str:
        return self._application

    @application.setter
    def application(self, application: str) -> None:
        self._application = application

    @property
    def mase(self) -> pd.DataFrame:
        return self._mase

    @mase.setter
    def mase(self, mase: pd.DataFrame) -> None:
        self._mase = mase

    @property
    def best_models(self) -> pd.Series:
        return self._best_models

    @data.setter
    def best_models(self, best_models: pd.Series) -> None:
        self._best_models = best_models

    @staticmethod
    def process_daily_option_chain_date(freq: str, data: Dict[str, pd.DataFrame],
                                        application: str = 'trading', **kwargs) -> Dict[str, pd.DataFrame]:
        processed_data = dict()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x, y: (x, y), x=name, y=crypto) for name, crypto in data.items()]
            for future in concurrent.futures.as_completed(futures):
                name, tmp = future.result()
                if 'option_tag' not in tmp.columns:
                    tmp.reset_index('option_tag', inplace=True)
                tmp.loc[:, 'moneyness'] = np.log(tmp.loc[:, 'moneyness'])
                if application == 'trading':
                    underlying_prices = kwargs['underlying_prices']
                    tmp = \
                        tmp.groupby(by=[pd.Grouper(key='option_tag'),
                                        pd.Grouper(level='timestamp', freq=freq, sort=True)]).agg(
                            {'bid_price': 'sum', 'ask_price': 'sum', 'moneyness': 'mean'})
                    tmp.loc[:, 'mid_price'] = tmp.filter(regex='_price').mean(axis=1).values
                    try:
                        exchange_rate = \
                            underlying_prices.loc[tmp.index.get_level_values(1).unique().sort_values(), name]
                    except KeyError:
                        exchange_rate = \
                            underlying_prices.loc[tmp.index.get_level_values(1).unique().sort_values()[1:], name]
                    exchange_rate = exchange_rate.reindex(tmp.index.get_level_values(1))
                    tmp.loc[:, 'mid_price'] = tmp.loc[:, 'mid_price'].mul(exchange_rate.values)
                    tmp.drop(['bid_price', 'ask_price'], axis=1, inplace=True)
                elif application == 'vrp':
                    tmp = \
                        tmp[['moneyness', 'mark_iv', 'expiration', 'type']].query('type == "call"').drop('type', axis=1)
                    tmp.loc[:, 'maturity'] = ((tmp.expiration - tmp.index) / pd.to_timedelta('360D')).values
                    tmp.drop('expiration', axis=1, inplace=True)
                tmp.dropna(inplace=True)
                processed_data[name.replace('USDT', '').lower()] = tmp
        return processed_data

    @staticmethod
    def create_option_chain_data(freq: str) -> None:
        option_chain_data = dict([('prices', dict()), ('moneyness', dict()), ('vrp', dict())])
        underlying_prices = FrameworkApplication._reader_obj.prices_read().resample(freq).last()
        for year in tqdm([2021]): #YEARS
            btc_files, eth_files = glob.glob(
                f'../data_centre/BTC/{year}/*'), glob.glob(f'../data_centre/ETH/{year}/*')
            btc_files.sort()
            eth_files.sort()
            for btc_filename, eth_filename in zip(btc_files, eth_files):
                date = btc_filename.split("/")[-1]
                print(f'Option chain data for {date} is generated................................')
                crypto = \
                    FrameworkApplication.process_daily_option_chain_date(
                        freq=freq, data=dict([('BTCUSDT', pd.read_parquet(btc_filename,
                                                                          columns=OPTION_CHAIN_DATA_COLUMNS)),
                                              ('ETHUSDT', pd.read_parquet(eth_filename,
                                                                          columns=OPTION_CHAIN_DATA_COLUMNS))]),
                        underlying_prices=underlying_prices,
                        application='trading')
                btc, eth = crypto['btc'], crypto['eth']
                prices = pd.concat({'BTCUSDT': pd.pivot(btc[['mid_price']].reset_index(level=0),
                                                        columns='option_tag', values='mid_price'),
                                    'ETHUSDT': pd.pivot(eth[['mid_price']].reset_index(level=0),
                                                        columns='option_tag', values='mid_price')}, axis=1)
                moneyeness = pd.concat({'BTCUSDT': pd.pivot(btc[['moneyness']].reset_index(level=0),
                                                            columns='option_tag', values='moneyness'),
                                        'ETHUSDT': pd.pivot(eth[['moneyness']].reset_index(level=0),
                                                            columns='option_tag', values='moneyness')}, axis=1)
                option_chain_data['prices'][date] = prices
                option_chain_data['moneyness'][date] = moneyeness
                crypto = \
                    FrameworkApplication.process_daily_option_chain_date(
                        freq=freq, data=dict([('BTCUSDT', pd.read_parquet(btc_filename,
                                                                          columns=OPTION_CHAIN_DATA_COLUMNS)),
                                              ('ETHUSDT', pd.read_parquet(eth_filename,
                                                                          columns=OPTION_CHAIN_DATA_COLUMNS))]),
                        application='vrp')
                btc, eth = crypto['btc'].assign(symbol='BTCUSDT'), crypto['eth'].assign(symbol='ETHUSDT')
                option_chain_data['vrp'][date] = pd.concat([btc, eth])
        FrameworkApplication._option_prices = option_chain_data['prices']
        FrameworkApplication._option_moneyness = option_chain_data['moneyness']
        FrameworkApplication._vrp = option_chain_data['vrp']
        print(f'Option chain data for {date} has been generated.')
        return

    @staticmethod
    def objective(trial: optuna.trial.Trial, data: pd.DataFrame, market: str, freq: str, D: str) -> float:
        param = {'filter': trial.suggest_float('filter', 1e-1, 9e-1, log=False)}
        if market == 'options':
            param.update({'atm_interval_width': trial.suggest_float('atm_interval_width', 1e-2, 1.5e-1, log=False)})
        long_filter = data.loc[:, data.columns.get_level_values(2) == 'signal'].gt(100*param['filter'])
        short_filter = data.loc[:, data.columns.get_level_values(2) == 'signal'].lt(-100*param['filter'])
        long = \
            data.loc[:, data.columns.get_level_values(2) == 'signal'].mul(long_filter).droplevel(axis=1, level=[1, 2])
        short = \
            data.loc[:, data.columns.get_level_values(2) == 'signal'].mul(short_filter).droplevel(axis=1, level=[1, 2])
        if market == 'spot':
            prices = FrameworkApplication._reader_obj.prices_read().resample(freq).last()
        elif market == 'options':
            prices, moneyness = list(), list()
            end_date = pd.to_datetime(trial.study.study_name.split('for ')[-1].split(' has')[0], format=DATE_FORMAT,
                                      utc=True)
            date_range = pd.date_range(start=end_date-relativedelta(days=pd.to_timedelta(D)//pd.to_timedelta('1D')),
                                       end=end_date, inclusive='left', freq='1D')
            date_range = np.intersect1d(np.unique(long_filter.index.date), date_range.date)
            for date in date_range:
                try:
                    prices.append(FrameworkApplication._option_prices[date.strftime(DATE_FORMAT)])
                    moneyness.append(FrameworkApplication._option_moneyness[date.strftime(DATE_FORMAT)])
                except KeyError:
                    continue
                except TypeError:
                    pdb.set_trace()
            prices, moneyness = pd.concat(prices), pd.concat(moneyness)
            atm = moneyness.where((moneyness.gt(np.log(1-param['atm_interval_width'])) &
                                   moneyness.lt(np.log(1+param['atm_interval_width']))), 0)
            atm = atm.where(atm == 0, 1)
        prices = prices.loc[long_filter.index, :] if market == 'spot' else prices.mul(atm)
        ################################################################################################################
        ## Long
        ################################################################################################################
        if market == 'spot':
            long_returns = prices.subtract(prices.shift())
            long_fees = (prices.add(prices.shift())).mul(FEES_SPOT)
            long_returns = (long_returns.subtract(long_fees)).div(prices.shift())
            long_returns = (np.sign(long).mul(long_returns)).mean(axis=1)
        else:
            long_returns = prices.subtract(prices.shift())
            long_fees = (prices.add(prices.shift())).mul(FEES_OPTIONS)
            underlying_prices = \
                FrameworkApplication._reader_obj.prices_read().resample(freq).last().loc[long_fees.index, :]
            for pair in ['BTCUSDT', 'ETHUSDT']:
                long_fees.loc[:, long_fees.columns.get_level_values(0) == pair] = \
                    long_fees.loc[:, long_fees.columns.get_level_values(0) == pair].apply(
                    lambda x, y: x.where(x.lt(FEES_STRADDLE_CAP*(y[pair].add(y[pair].shift()))),
                                         FEES_STRADDLE_CAP*(y[pair].add(y[pair].shift()))), y=underlying_prices)
                long = long.reindex(long_returns.index)
                long_returns.loc[:, long_returns.columns.get_level_values(0) == pair] = \
                    long_returns.loc[:, long_returns.columns.get_level_values(0) == pair].apply(
                        lambda x, y: x.mul(np.sign(y)), y=long[pair])
            long_returns = \
                ((long_returns.subtract(long_fees)).div(prices.shift())).replace(np.inf, np.nan).replace(-np.inf,
                                                                                                         np.nan)
            long_returns = pd.concat(
                [long_returns.loc[:, long_returns.columns.get_level_values(0) == pair].mean(axis=1) for
                 pair in ['BTCUSDT', 'ETHUSDT']], axis=1).rename(columns={idx: pair for idx, pair in
                                                                          enumerate(['BTCUSDT', 'ETHUSDT'])})
            long_returns = (long_returns.mul(np.sign(long))).mean(axis=1)
        ################################################################################################################
        ## Short
        ################################################################################################################
        if market == 'spot':
            short_returns = prices.shift().subtract(prices)
            short_fees = (prices.shift().add(prices)).mul(FEES_SPOT)
            short_returns = (short_returns.subtract(short_fees)).div(prices)
            short_returns = (np.sign(short).mul(short_returns)).mean(axis=1)
        else:
            short_returns = prices.subtract(prices.shift())
            short_fees = (prices.add(prices.shift())).mul(FEES_OPTIONS)
            underlying_prices = \
                FrameworkApplication._reader_obj.prices_read().resample(freq).last().loc[long_fees.index, :]
            for pair in ['BTCUSDT', 'ETHUSDT']:
                short_fees.loc[:, short_fees.columns.get_level_values(0) == pair] = \
                    short_fees.loc[:, short_fees.columns.get_level_values(0) == pair].apply(
                        lambda x, y: x.where(x.lt(FEES_STRADDLE_CAP * (y[pair].add(y[pair].shift()))),
                                             FEES_STRADDLE_CAP * (y[pair].add(y[pair].shift()))), y=underlying_prices)
                short = short.reindex(short_returns.index)
                short_returns.loc[:, short_returns.columns.get_level_values(0) == pair] = \
                    short_returns.loc[:, short_returns.columns.get_level_values(0) == pair].apply(
                        lambda x, y: x.mul(np.sign(y)), y=long[pair])
            short_returns = \
                ((short_returns.subtract(short_fees)).div(prices.shift())).replace(np.inf, np.nan).replace(-np.inf,
                                                                                                           np.nan)
            short_returns = pd.concat(
                [short_returns.loc[:, short_returns.columns.get_level_values(0) == pair].mean(axis=1) for
                 pair in ['BTCUSDT', 'ETHUSDT']], axis=1).rename(columns={idx: pair for idx, pair in
                                                                          enumerate(['BTCUSDT', 'ETHUSDT'])})
            short_returns = (short_returns.mul(np.sign(short))).mean(axis=1)
        returns = long_returns.add(short_returns).div(2)
        trial.set_user_attr(key='best_filter', value=param['filter'])
        if market == 'options':
            trial.set_user_attr(key='best_atm_interval_width', value=param['atm_interval_width'])
        return returns.add(1).cumprod().ffill().iloc[-1] # to be debbuged

    @staticmethod
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key='best_filter', value=trial.user_attrs['best_filter'])
            if 'best_atm_interval_width' in trial.user_attrs.keys():
                study.set_user_attr(key='best_atm_interval_width', value=trial.user_attrs['best_atm_interval_width'])

    def backtest(self, D: str, freq: str) -> None:
        dates = list(np.unique(self._data.index.date))
        dates = dates[dates.index(dates[0]+relativedelta(days=pd.to_timedelta(D)//pd.to_timedelta('1D'))):]
        pnl = dict()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x: x, x=date) for date in dates]
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                date = future.result()
                data = self._data.loc[(self._data.index.date >=
                                       date-relativedelta(days=2*(pd.to_timedelta(D)//pd.to_timedelta('1D')))) &
                                      (self._data.index.date <= date), :]
                ########################################################################################################
                ## Market selection
                ########################################################################################################
                market_to_trade = \
                    self._mase.loc[self._mase.index.date == date].mean().idxmin().split('(')[-1].split(',')[0]
                data = data.loc[:, data.columns.get_level_values(1) == market_to_trade]
                data.loc[:, data.columns.get_level_values(2) == 'values'] = \
                    data.loc[:, data.columns.get_level_values(2) == 'values'].shift()
                ahead_d_data = data.loc[data.index.date == date, :]
                data.drop(ahead_d_data.index, inplace=True)
                insample_data = data.iloc[(data.shape[0]//2)-1:, :].copy()
                rolling_mean_ins = data.loc[~data.index.isin(insample_data.index), :].loc[:,
                                   (data.columns.get_level_values(1) == market_to_trade) &
                                   (data.columns.get_level_values(2) == 'values')].mean().droplevel(axis=0, level=[1, 2])
                rolling_mean_oos = data.loc[data.index.isin(insample_data.index), :].loc[:,
                                   (data.columns.get_level_values(1) == market_to_trade) &
                                   (data.columns.get_level_values(2) == 'values')].mean().droplevel(axis=0, level=[1, 2])
                for pair in ['BTCUSDT', 'ETHUSDT']:
                    insample_data[pair, market_to_trade, 'signal'] = \
                        insample_data[pair, market_to_trade, 'forecast'].subtract(
                            insample_data[pair, market_to_trade, 'values']).div(rolling_mean_ins[pair])
                    insample_data[pair, market_to_trade, 'signal'] = \
                        insample_data[pair, market_to_trade, 'forecast'].subtract(
                            insample_data[pair, market_to_trade, 'values']).div(rolling_mean_ins[pair])
                    ahead_d_data[pair, market_to_trade, 'signal'] = \
                        ahead_d_data[pair, market_to_trade, 'forecast'].subtract(
                            ahead_d_data[pair, market_to_trade, 'values']).div(rolling_mean_oos[pair])
                    ahead_d_data[pair, market_to_trade, 'signal'] = \
                        ahead_d_data[pair, market_to_trade, 'forecast'].subtract(
                            ahead_d_data[pair, market_to_trade, 'values']).div(rolling_mean_oos[pair])
                ahead_d_data = \
                    ahead_d_data.droplevel(axis=1, level=1).drop(axis=1, level=1,
                                                                 columns=['values', 'forecast']).droplevel(axis=1, level=1)
                ########################################################################################################
                ## Parameter optimisation
                ########################################################################################################
                study = \
                    optuna.create_study(direction='maximize',
                                        study_name=f'Parameter optimisation: '
                                                   f'process for {date.strftime(DATE_FORMAT)} has started.',
                                        sampler=RandomSampler(123))
                objective = partial(FrameworkApplication.objective, data=insample_data,
                                    market=MARKETS[market_to_trade], freq=freq, D=D)
                study.optimize(objective, n_trials=N_TRIALS // 20, callbacks=[FrameworkApplication.callback],
                               show_progress_bar=True,
                               n_jobs=N_JOBS)
                ########################################################################################################
                ## Backtesting out-of-sample
                ########################################################################################################
                long_filter = ahead_d_data.gt(100 * study.user_attrs['best_filter'])
                short_filter = ahead_d_data.lt(-100 * study.user_attrs['best_filter'])
                if market_to_trade == 'RV':
                    atm_interval_width = study.user_attrs['best_atm_interval_width']
                long = ahead_d_data.mul(long_filter)
                short = ahead_d_data.mul(short_filter)
                if market_to_trade == 'returns':
                    prices = \
                        FrameworkApplication._reader_obj.prices_read().resample(freq).last().loc[ahead_d_data.index, :]
                    long_returns = prices.subtract(prices.shift())
                    long_fees = (prices.add(prices.shift())).mul(FEES_SPOT)
                    long_returns = (long_returns.subtract(long_fees)).div(prices.shift())
                    long_returns = (np.sign(long).mul(long_returns)).mean(axis=1)
                    short_returns = prices.shift().subtract(prices)
                    short_fees = (prices.shift().add(prices)).mul(FEES_SPOT)
                    short_returns = (short_returns.subtract(short_fees)).div(prices)
                    short_returns = (np.sign(short).mul(short_returns)).mean(axis=1)
                    returns = long_returns.add(short_returns).div(2)
                else:
                    try:
                        prices, moneyness = FrameworkApplication._option_prices[date.strftime(DATE_FORMAT)],\
                            FrameworkApplication._option_moneyness[date.strftime(DATE_FORMAT)]
                    except KeyError:
                        continue
                    atm = moneyness.where((moneyness.gt(np.log(1 - atm_interval_width)) &
                                           moneyness.lt(np.log(1 + atm_interval_width))), 0)
                    atm = atm.where(atm == 0, 1)
                    prices = prices.mul(atm)
                    ####################################################################################################
                    ## Long
                    ####################################################################################################
                    long_returns = prices.subtract(prices.shift())
                    long_fees = prices.add(prices.shift()).mul(FEES_OPTIONS)
                    underlying_prices = \
                        FrameworkApplication._reader_obj.prices_read().resample(freq).last().loc[long_fees.index, :]
                    for pair in ['BTCUSDT', 'ETHUSDT']:
                        long_fees.loc[:, long_fees.columns.get_level_values(0) == pair] = \
                            long_fees.loc[:, long_fees.columns.get_level_values(0) == pair].apply(
                                lambda x, y: x.where(x.lt(FEES_STRADDLE_CAP * (y[pair].add(y[pair].shift()))),
                                                     FEES_STRADDLE_CAP * (y[pair].add(y[pair].shift()))),
                                y=underlying_prices)
                        long = long.reindex(long_returns.index)
                        long_returns.loc[:, long_returns.columns.get_level_values(0) == pair] = \
                            long_returns.loc[:, long_returns.columns.get_level_values(0) == pair].apply(
                                lambda x, y: x.mul(np.sign(y)), y=long[pair])
                    long_returns = (long_returns.subtract(long_fees)).div(prices.shift())
                    long_returns = pd.concat(
                        [long_returns.loc[:, long_returns.columns.get_level_values(0) == pair].mean(axis=1) for
                         pair in ['BTCUSDT', 'ETHUSDT']], axis=1).rename(columns={idx: pair for idx, pair in
                                                                                  enumerate(['BTCUSDT', 'ETHUSDT'])})
                    long_returns = \
                        (long_returns.mul(np.sign(long)).replace(np.inf, np.nan).replace(np.inf, np.nan)).mean(axis=1)
                    ####################################################################################################
                    ## Short
                    ####################################################################################################
                    short_returns = prices.shift().subtract(prices)
                    short_fees = (prices.shift().add(prices)).mul(FEES_OPTIONS)
                    for pair in ['BTCUSDT', 'ETHUSDT']:
                        short_fees.loc[:, long_fees.columns.get_level_values(0) == pair] = \
                            short_fees.loc[:, short_fees.columns.get_level_values(0) == pair].apply(
                                lambda x, y: x.where(x.lt(FEES_STRADDLE_CAP * (y[pair].add(y[pair].shift()))),
                                                     FEES_STRADDLE_CAP * (y[pair].add(y[pair].shift()))),
                                y=underlying_prices)
                        short = short.reindex(short_returns.index)
                        short_returns.loc[:, short_returns.columns.get_level_values(0) == pair] = \
                            short_returns.loc[:, short_returns.columns.get_level_values(0) == pair].apply(
                                lambda x, y: x.mul(np.sign(y)), y=long[pair])
                    short_returns = (short_returns.subtract(long_fees)).div(prices.shift())
                    short_returns = pd.concat(
                        [short_returns.loc[:, short_returns.columns.get_level_values(0) == pair].mean(axis=1) for
                         pair in ['BTCUSDT', 'ETHUSDT']], axis=1).rename(columns={idx: pair for idx, pair in
                                                                                  enumerate(['BTCUSDT', 'ETHUSDT'])})
                    short_returns = \
                        (short_returns.mul(np.sign(short)).replace(-np.inf, np.nan).replace(np.inf, np.nan)).mean(
                            axis=1)
                    returns = long_returns.add(short_returns).div(2).iloc[1:]
                if market_to_trade == 'rv':
                    pnl[date] = pd.concat([returns,
                                           pd.Series(data=study.user_attrs['best_filter'], index=returns.index),
                                           pd.Series(data=study.user_attrs['best_atm_interval_width'],
                                                     index=returns.index)], axis=1)
                else:
                    pnl[date] = pd.concat([returns, pd.Series(data=study.user_attrs['best_filter'],
                                                              index=returns.index)], axis=1)
            pnl = \
                pd.concat(pnl.values()).rename(
                    columns={
                        idx: col for idx, col in enumerate({True: ['returns', 'filter'],
                                                            False: ['returns', 'filter', 'atm_interval_width']}
                                                           [market_to_trade == 'returns'])})
        pnl.index = pd.to_datetime(pnl.index, utc=True)
        pnl.sort_index(inplace=True)
        pnl['returns'].fillna(0).cumsum().plot()
        print(self._best_models)
        plt.show()

    @staticmethod
    def iv_regression(data: pd.DataFrame) -> List[float]:
        lr = LinearRegression()
        y = data.pop('mark_iv')
        try:
            lr.fit(X=data, y=y)
        except ValueError:
            return [np.nan] * 6
        else:
            return [lr.intercept_] + lr.coef_.tolist()

    def vrp(self, D: str, freq: str) -> None:
        dates = list(np.unique(self._data.index.date))
        dates = dates[dates.index(dates[0] + relativedelta(days=pd.to_timedelta(D) // pd.to_timedelta('1D'))):-1]
        iv = dict([(pair, list()) for pair in ['BTCUSDT', 'ETHUSDT']])
        ################################################################################################################
        ## IV regression + interpolation
        ################################################################################################################
        for date in dates:
            data = FrameworkApplication._vrp[date.strftime(DATE_FORMAT)].dropna()
            data.columns.name = None
            for pair in ['BTCUSDT', 'ETHUSDT']:
                pair_data = data.query(f'symbol == "{pair}"').drop('symbol', axis=1)
                pair_data = pair_data.assign(maturity2=pair_data.maturity**2, moneyness2=pair_data.moneyness**2,
                                             maturity_moneyness=pair_data.maturity.mul(pair_data.moneyness))
                coef = \
                    pair_data.groupby(by=pd.Grouper(level=0, freq=freq,
                                                    sort=True)).apply(lambda x: self.iv_regression(x))
                coef = \
                    pd.DataFrame(data=coef.values.tolist(), index=coef.index,
                                 columns=['intercept']+pair_data.columns[~pair_data.columns.isin(['mark_iv'])].tolist())
                coef = coef.fillna(coef.expanding().mean())
                fixed_maturity = pd.to_timedelta(D) / pd.to_timedelta('360D')
                coef['maturity'], coef['maturity2'], coef['maturity_moneyness'] = \
                    np.log(1), np.log(1)**2, np.log(1)*fixed_maturity
                coef['maturity'], coef['maturity2'] = fixed_maturity, fixed_maturity**2
                iv[pair].append(coef.sum(axis=1))
        for pair in ['BTCUSDT', 'ETHUSDT']:
            iv[pair] = pd.concat(iv[pair])
        iv = pd.concat(iv)
        iv = iv.where(iv >= .01, .01).reset_index()
        iv = pd.pivot_table(columns='level_0', values=0, index='timestamp', data=iv).dropna()
        iv = iv.apply(lambda x: x.clip(lower=x.quantile(.05), upper=x.quantile(.95)))
        ################################################################################################################
        ## VRP calculation
        ################################################################################################################
        rv = (self._reader_obj.rv_read().resample(freq).sum().loc[iv.index,
              :] ** (.5)).mul((360*pd.to_timedelta('1D')//pd.to_timedelta(freq))**.5)
        return iv.subtract(rv)


if __name__ == '__main__':
    framework_application_obj = FrameworkApplication()
    ####################################################################################################################
    ## Model selection
    ####################################################################################################################
    mase = pd.read_sql(con=framework_application_obj._db_connect, sql='SELECT * FROM metric;')
    mase.loc[:, 'model_type'] = \
        [model_tag.replace('returns, ', '').replace('RV, ', '').replace(', S', '').replace(', O', '')
         for model_tag in mase.loc[:, 'model_tag']]
    best_model_type = mase.groupby(by=pd.Grouper(key='model_type'))[['mase']].mean().idxmin().iloc[0]
    model_tag_list = ','.join(mase.query(f'model_type == "{best_model_type}"').model_tag.unique().tolist())
    mase = \
        mase.loc[mase.model_tag.isin(mase.query(f'model_type == "{best_model_type}"').model_tag.unique().tolist()), :]
    mase.loc[:, 'target'] = \
        [model_tag[1].split(',')[0] for model_tag in mase.loc[:, 'model_tag'].str.split('(')]
    best_models = mase.groupby(by=[pd.Grouper(key='target'), pd.Grouper(key='model_tag')])[['mase']].mean().groupby(
        pd.Grouper(level='target')).apply(lambda x: x.reset_index(0)[['mase']].idxmin()).values.flatten().tolist()
    mase = mase.loc[mase.model_tag.isin(best_models), :]
    pattern = r"\(RV, ([^,]+), O, ([^)]+)\)"
    match = re.search(pattern, best_models[0])
    D, freq = match.group(1), match.group(2)
    best_models = list(map(lambda x: f'"{x}"', best_models))
    data = pd.read_sql(con=framework_application_obj._db_connect,
                       sql=f'SELECT * FROM forecasts WHERE model_tag IN ({",".join(best_models)});')
    data.loc[:, 'target'] = \
        [model_tag[1].split(',')[0] for model_tag in data.loc[:, 'model_tag'].str.split('(')]
    mase.loc[:, 'target'] = \
        [model_tag[1].split(',')[0] for model_tag in mase.loc[:, 'model_tag'].str.split('(')]
    mase = pd.pivot_table(data=mase, index='index', columns='target', values='mase')
    mase.index = pd.to_datetime(mase.index)
    mase = mase.resample('1D').mean()
    data = pd.pivot_table(data=data, columns=['symbol', 'target'], values=['values', 'forecast'],
                          index='timestamp')
    data = data.swaplevel(1, 0, axis=1).swaplevel(-1, 1, axis=1)
    data.index = pd.to_datetime(data.index, utc=True)
    framework_application_obj.data = data
    framework_application_obj.mase = mase
    framework_application_obj.best_models = best_models
    framework_application_obj.create_option_chain_data(freq=freq)
    ####################################################################################################################
    ## Application 1: Volatility Risk Premium (VRP)
    ####################################################################################################################
    framework_application_obj.application == 'vrp'
    vrp = framework_application_obj.vrp(D=D, freq=freq).unstack()
    rv = \
        (framework_application_obj._reader_obj.rv_read().loc[vrp.index.get_level_values(1).unique(), :]**(.5)).unstack()
    rv = rv.mul(((pd.to_timedelta('360D')//pd.to_timedelta(freq))**.5))
    returns = framework_application_obj._reader_obj.returns_read().loc[vrp.index.get_level_values(1).unique(), :]
    returns = returns.cumsum().expanding().mean().unstack()
    pdb.set_trace()
    vrp_crypto_observables = \
        pd.concat([vrp, rv, returns],
                  axis=1).rename(columns={idx: col for idx, col in enumerate(['vrp', 'rv', 'returns'])}).droplevel(1)
    vrp_crypto_observables = \
        pd.melt(frame=vrp_crypto_observables, ignore_index=False, value_vars=['rv', 'returns'], id_vars='vrp',
                value_name='observables').reset_index().rename(columns={'index': 'symbol', 'vrp': 'VRP'})
    fig = px.scatter(data_frame=vrp_crypto_observables, x='observables', y='VRP', facet_col='symbol', trendline='ols',
                     facet_row='variable', title='VRP vs observables', facet_col_spacing=.05, facet_row_spacing=.1)
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].replace('USDT', '')}),
                            selector=lambda x: 'symbol' in x['text'])
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].upper()}),
                            selector=lambda x: ('variable' in x['text']) & ('rv' in x['text']))
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].title()}),
                            selector=lambda x: ('variable' in x['text']) & ('returns' in x['text']))
    fig.update_xaxes(title_text='')
    fig.show()
    fig = px.scatter(data_frame=vrp_crypto_observables, x='VRP', y='observables', facet_col='symbol', trendline='ols',
                     facet_row='variable', title='Observables vs VRP', facet_col_spacing=.05, facet_row_spacing=.1)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].replace('USDT', '')}),
                            selector=lambda x: 'symbol' in x['text'])
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].upper()}),
                            selector=lambda x: ('variable' in x['text']) & ('rv' in x['text']))
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].title()}),
                            selector=lambda x: ('variable' in x['text']) & ('returns' in x['text']))
    fig.update_yaxes(title_text='')
    fig.show()
    for pair_observable in itertools.product(['BTCUSDT', 'ETHUSDT'], ['returns', 'rv']):
        pair_data = vrp_crypto_observables.query(f'symbol == "{pair_observable[0]}" &'
                                                 f'variable == "{pair_observable[1]}"').drop(['symbol', 'variable'],
                                                                                             axis=1)
        pair_data = \
            pair_data.rename(columns={'observables': pair_observable[1].upper() if pair_observable[1] == 'rv' \
    else pair_observable[1].title(), 'vrp': 'VRP'})
        rres = \
            sm.ols(formula=
                   f'VRP ~ {pair_observable[1].upper() if pair_observable[1] == "rv" else pair_observable[1].title()}',
                   data=pair_data).fit()
        print(rres.summary().as_latex())
    vrp_crypto_observables.loc[:, 'quantile'] = \
        vrp_crypto_observables.groupby(
            by=[pd.Grouper(key='symbol'),
                pd.Grouper(key='variable')]).apply(
            lambda x: pd.qcut(x.observables, q=10, labels=list(range(1, 11)))).values
    decile_analysis = dict([(pair, dict([(observable, list()) for observable in ['returns', 'rv']]))
                            for pair in ['BTCUSDT', 'ETHUSDT']])
    for pair_observable in itertools.product(['BTCUSDT', 'ETHUSDT'], ['returns', 'rv']):
        pair_data = vrp_crypto_observables.query(f'symbol == "{pair_observable[0]}" &'
                                                 f'variable == "{pair_observable[1]}"').drop(['symbol', 'variable'],
                                                                                             axis=1)
        pair_data = pair_data.rename(columns={'observables': 'observable'})
        for quantile in range(1, 11):
            quantile_pair_data = pair_data.query(f'quantile == {quantile}').drop('quantile', axis=1)
            quantile_pair_data = quantile_pair_data.rename(columns={'vrp': 'VRP'})
            rres = sm.ols(formula=f'VRP ~ observable', data=quantile_pair_data).fit()
            table = pd.concat([rres.params, rres.tvalues, rres.pvalues], axis=1).rename(columns={idx: col for idx,
            col in enumerate(['estimate', 't-values', 'p-values'])}).unstack()
            table = table.swaplevel(1, 0).sort_index(level=0)
            table.loc['R2'] = rres.rsquared
            table.loc[f'avg. observable'] = quantile_pair_data.observable.mean()
            decile_analysis[pair_observable[0]][pair_observable[1]].append(table)
        decile_analysis[pair_observable[0]][pair_observable[1]] = \
            pd.concat(decile_analysis[pair_observable[0]][pair_observable[1]],
                      axis=1).round(4).rename(columns={idx: idx+1 for idx in range(0, 10)})
        decile_analysis[pair_observable[0]][pair_observable[1]].loc[:, 'symbol'] = pair_observable[0]
    decile_analysis_returns = \
        pd.concat([decile_analysis['BTCUSDT']['returns'], decile_analysis['ETHUSDT']['returns']], axis=0)
    decile_analysis_returns = \
        decile_analysis_returns.set_index('symbol', append=True).swaplevel(-1, 0, axis=0).swaplevel(1, -1, axis=0)
    print(decile_analysis_returns.to_latex())
    decile_analysis_rv = pd.concat([decile_analysis['BTCUSDT']['rv'], decile_analysis['ETHUSDT']['rv']], axis=0)
    decile_analysis_rv = \
        decile_analysis_rv.set_index('symbol', append=True).swaplevel(-1, 0, axis=0).swaplevel(1, -1, axis=0)
    print(decile_analysis_rv.to_latex())
    ####################################################################################################################
    ## Application 2: Economic benefits
    ####################################################################################################################
    framework_application_obj.application == 'trading'
    framework_application_obj.backtest(D=D, freq=freq)


