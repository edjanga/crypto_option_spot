import concurrent.futures
import glob
import pdb
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import RandomSampler
from data_centre.data import Reader
from dateutil.relativedelta import relativedelta
from constants import DATE_FORMAT, N_JOBS, N_TRIALS, FEES_SPOT, FEES_OPTIONS, FEES_STRADDLE_CAP, MARKETS,\
    OPTION_CHAIN_DATA_COLUMNS
from functools import partial
import matplotlib.pyplot as plt
from typing import Dict
from tqdm import tqdm


class Trader:
    _reader_obj = Reader()
    _option_prices = None
    _option_moneyness = None

    def __init__(self, data: pd.DataFrame = None):
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        self._data = data

    @staticmethod
    def process_daily_option_chain_date(freq: str, data: Dict[str, pd.DataFrame],
                                        underlying_prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        processed_data = dict()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x, y: (x, y), x=name, y=crypto) for name, crypto in data.items()]
            for future in concurrent.futures.as_completed(futures):
                name, tmp = future.result()
                if 'option_tag' not in tmp.columns:
                    tmp.reset_index('option_tag', inplace=True)
                tmp = \
                    tmp.groupby(by=[pd.Grouper(key='option_tag'), pd.Grouper(key='type'),
                                    pd.Grouper(level='timestamp', freq=freq, sort=True)]).agg(
                        {'bid_price': 'last', 'ask_price': 'last', 'moneyness': 'last'})
                tmp.loc[:, 'mid_price'] = tmp.filter(regex='_price').mean(axis=1).values
                try:
                    exchange_rate = \
                        underlying_prices.loc[tmp.index.get_level_values(2).unique().sort_values(), name]
                except KeyError:
                    exchange_rate = \
                        underlying_prices.loc[tmp.index.get_level_values(2).unique().sort_values()[1:], name]
                exchange_rate = exchange_rate.reindex(tmp.index.get_level_values(2))
                tmp.loc[:, 'mid_price'] = tmp.loc[:, 'mid_price'].mul(exchange_rate.values)
                tmp.drop(['bid_price', 'ask_price'], axis=1, inplace=True)
                tmp.loc[:, 'moneyness'] = np.log(tmp.loc[:, 'moneyness'])
                tmp = tmp.groupby(by=[pd.Grouper(level='option_tag'),
                                      pd.Grouper(level='timestamp', sort=True)]).agg({'moneyness': 'mean',
                                                                                      'mid_price': 'sum'})
                processed_data[name.replace('USDT', '').lower()] = tmp
        return processed_data

    @staticmethod
    def create_option_chain_data(freq: str) -> None:
        option_chain_data = dict([('prices', dict()), ('moneyness', dict())])
        underlying_prices = Trader._reader_obj.prices_read().resample(freq).last()
        for year in tqdm([2021]):
            btc_files, eth_files = glob.glob(f'../data_centre/BTC/{year}/*'), glob.glob(f'../data_centre/ETH/{year}/*')
            btc_files.sort()
            eth_files.sort()
            for btc, eth in zip(btc_files, eth_files):
                date = btc.split("/")[-1]
                print(f'Option chain datat for {date} is generated................................')
                btc, eth = pd.read_parquet(btc, columns=OPTION_CHAIN_DATA_COLUMNS), \
                    pd.read_parquet(eth, columns=OPTION_CHAIN_DATA_COLUMNS)
                crypto = Trader.process_daily_option_chain_date(freq=freq,
                                                                data=dict([('BTCUSDT', btc), ('ETHUSDT', eth)]),
                                                                underlying_prices=underlying_prices)
                btc, eth = crypto['btc'], crypto['eth']
                prices = pd.concat({'BTCUSDT': pd.pivot(btc[['mid_price']].reset_index(level=0), columns='option_tag',
                                                        values='mid_price'),
                                    'ETHUSDT': pd.pivot(eth[['mid_price']].reset_index(level=0), columns='option_tag',
                                                        values='mid_price')}, axis=1)
                moneyeness = pd.concat({'BTCUSDT': pd.pivot(btc[['moneyness']].reset_index(level=0),
                                                            columns='option_tag', values='moneyness'),
                                        'ETHUSDT': pd.pivot(eth[['moneyness']].reset_index(level=0),
                                                            columns='option_tag', values='moneyness')}, axis=1)
                option_chain_data['prices'][date] = prices
                option_chain_data['moneyness'][date] = moneyeness
                print(f'Option chain data for {date} has been generated.')
        Trader._option_prices = option_chain_data['prices']
        Trader._option_moneyness = option_chain_data['moneyness']
        return

    @staticmethod
    def objective(trial: optuna.trial.Trial, data: pd.DataFrame, market: str, freq: str, D: str) -> float:
        param = {'filter': trial.suggest_float('filter', 1e-1, 9e-1, log=False)}
        if market == 'options':
            param.update({'atm_interval_width': trial.suggest_float('atm_interval_width', 1e-1, 9e-1, log=False)})
        long_filter = data.loc[:, data.columns.get_level_values(2) == 'signal'].gt(100*param['filter'])
        short_filter = data.loc[:, data.columns.get_level_values(2) == 'signal'].lt(-100*param['filter'])
        long = \
            data.loc[:, data.columns.get_level_values(2) == 'signal'].mul(long_filter).droplevel(axis=1, level=[1, 2])
        short = \
            data.loc[:, data.columns.get_level_values(2) == 'signal'].mul(short_filter).droplevel(axis=1, level=[1, 2])
        if market == 'spot':
            prices = Trader._reader_obj.prices_read().resample(freq).last()
        elif market == 'options':
            prices, moneyness = list(), list()
            end_date = pd.to_datetime(trial.study.study_name.split('for ')[-1].split(' has')[0], format=DATE_FORMAT,
                                      utc=True)
            date_range = pd.date_range(start=end_date-relativedelta(days=pd.to_timedelta(D)//pd.to_timedelta('1D')),
                                       end=end_date, inclusive='left', freq='1D')
            date_range = np.intersect1d(np.unique(long_filter.index.date), date_range.date)
            for date in date_range:
                try:
                    prices.append(Trader._option_prices[date.strftime(DATE_FORMAT)])
                    moneyness.append(Trader._option_moneyness[date.strftime(DATE_FORMAT)])
                except KeyError:
                    continue
            try:
                prices, moneyness = pd.concat(prices), pd.concat(moneyness)
            except ValueError:
                pdb.set_trace()
            atm = moneyness.where(moneyness.abs().lt(param['atm_interval_width']), 0)
            atm = atm.where(atm == 0, 1)
        try:
            prices = prices.loc[long_filter.index, :] if market == 'spot' else prices.mul(atm)
        except KeyError:
            pdb.set_trace()
        ################################################################################################################
        ## Long
        ################################################################################################################
        if market == 'spot':
            long_pnl = prices.subtract(prices.shift()).sum(axis=1)
            long_fees = prices.add(prices.shift()).sum(axis=1).mul(FEES_SPOT)
            long_pnl = long_pnl.subtract(long_fees.values).values.reshape(-1, 1)
            long_pnl = np.sign(long).mul(long_pnl).cumsum().sum(axis=1)
        else:
            long_pnl = prices.subtract(prices.shift())
            long_fees = prices.add(prices.shift()).mul(FEES_OPTIONS)
            underlying_prices = Trader._reader_obj.prices_read().resample(freq).last().loc[long_fees.index, :]
            for pair in ['BTCUSDT', 'ETHUSDT']:
                long_fees.loc[:, long_fees.columns.get_level_values(0) == pair] = \
                    long_fees.loc[:, long_fees.columns.get_level_values(0) == pair].apply(
                    lambda x, y: x.where(x.lt(FEES_STRADDLE_CAP*(y[pair].add(y[pair].shift()))),
                                         FEES_STRADDLE_CAP*(y[pair].add(y[pair].shift()))), y=underlying_prices)
                try:
                    long_pnl.loc[:, long_pnl.columns.get_level_values(0) == pair] = \
                        long_pnl.loc[:, long_pnl.columns.get_level_values(0) == pair].apply(lambda x, y: x.mul(np.sign(y)),
                                                                                            y=long[pair])
                except Exception as e:
                    print(e)
                    return 0.0
            long_pnl = long_pnl.subtract(long_fees).sum(axis=1).cumsum()
        ################################################################################################################
        ## Short
        ################################################################################################################
        if market == 'spot':
            short_pnl = prices.shift().subtract(prices).sum(axis=1)
            short_fees = prices.shift().add(prices).sum(axis=1).mul(FEES_SPOT)
            short_pnl = short_pnl.subtract(short_fees.values).values.reshape(-1, 1)
            short_pnl = np.sign(short).mul(short_pnl).cumsum().sum(axis=1)
        else:
            short_pnl = prices.shift().subtract(prices)
            short_fees = prices.shift().add(prices).mul(FEES_OPTIONS)
            for pair in ['BTCUSDT', 'ETHUSDT']:
                short_fees.loc[:, short_fees.columns.get_level_values(0) == pair] = \
                    short_fees.loc[:, short_fees.columns.get_level_values(0) == pair].apply(
                        lambda x, y: x.where(x.lt(FEES_STRADDLE_CAP * (y[pair].add(y[pair].shift()))),
                                             FEES_STRADDLE_CAP * (y[pair].add(y[pair].shift()))), y=underlying_prices)
                short_pnl.loc[:, short_pnl.columns.get_level_values(0) == pair] = \
                    short_pnl.loc[:, short_pnl.columns.get_level_values(0) == pair].apply(
                        lambda x, y: x.mul(np.sign(y)), y=long[pair]
                    )
            short_pnl = short_pnl.subtract(short_fees).sum(axis=1).cumsum()
        pnl = long_pnl.add(short_pnl)
        trial.set_user_attr(key='best_filter', value=param['filter'])
        if market == 'options':
            trial.set_user_attr(key='best_atm_interval_width', value=param['atm_interval_width'])
        return pnl.sum()

    @staticmethod
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key='best_filter', value=trial.user_attrs['best_filter'])
            if 'atm_interval_width' in trial.user_attrs.keys():
                study.set_user_attr(key='best_atm_interval_width', value=trial.user_attrs['atm_interval_width'])

    def backtest(self, D: str, freq: str):
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
                market_to_trade = 'returns'
                data = data.loc[:, data.columns.get_level_values(1) == market_to_trade]
                data.loc[:, data.columns.get_level_values(2) == 'obs'] = \
                    data.loc[:, data.columns.get_level_values(2) == 'obs'].shift()
                ahead_d_data = data.loc[data.index.date == date, :]
                data.drop(ahead_d_data.index, inplace=True)
                insample_data = data.iloc[data.shape[0]//2:, :].copy()
                rolling_mean_ins = data.iloc[:data.shape[0] // 2, :].loc[:,
                                   (data.columns.get_level_values(1) == market_to_trade) &
                                   (data.columns.get_level_values(2) == 'obs')].mean().droplevel(axis=0, level=[1, 2])
                rolling_mean_oos = data.iloc[data.shape[0]//2:, :].loc[:,
                                   (data.columns.get_level_values(1) == market_to_trade) &
                                   (data.columns.get_level_values(2) == 'obs')].mean().droplevel(axis=0, level=[1, 2])
                for pair in ['BTCUSDT', 'ETHUSDT']:
                    insample_data[pair, market_to_trade, 'signal'] = \
                        insample_data[pair, market_to_trade, 'fcst'].subtract(
                            insample_data[pair, market_to_trade, 'obs']).div(rolling_mean_ins[pair])
                    insample_data[pair, market_to_trade, 'signal'] = \
                        insample_data[pair, market_to_trade, 'fcst'].subtract(
                            insample_data[pair, market_to_trade, 'obs']).div(rolling_mean_ins[pair])
                    ahead_d_data[pair, market_to_trade, 'signal'] = \
                        ahead_d_data[pair, market_to_trade, 'fcst'].subtract(
                            ahead_d_data[pair, market_to_trade, 'obs']).div(rolling_mean_oos[pair])
                    ahead_d_data[pair, market_to_trade, 'signal'] = \
                        ahead_d_data[pair, market_to_trade, 'fcst'].subtract(
                            ahead_d_data[pair, market_to_trade, 'obs']).div(rolling_mean_oos[pair])
                ahead_d_data = \
                    ahead_d_data.droplevel(axis=1, level=1).drop(axis=1, level=1,
                                                                 columns=['obs', 'fcst']).droplevel(axis=1, level=1)
                ########################################################################################################
                ## Parameter optimisation
                ########################################################################################################
                study = \
                    optuna.create_study(direction='maximize',
                                        study_name=f'Parameter optimisation: '
                                                   f'process for {date.strftime(DATE_FORMAT)} has started.',
                                        sampler=RandomSampler(123))
                try:
                    objective = partial(Trader.objective, data=insample_data, market=MARKETS[market_to_trade], freq=freq,
                                        D=D)
                    study.optimize(objective, n_trials=N_TRIALS // 20, callbacks=[Trader.callback],
                                   show_progress_bar=True,
                                   n_jobs=N_JOBS)
                except KeyError:
                    continue
                ########################################################################################################
                ## Backtesting out-of-sample
                ########################################################################################################
                long_filter = ahead_d_data.gt(100 * study.user_attrs['best_filter'])
                short_filter = ahead_d_data.lt(-100 * study.user_attrs['best_filter'])
                if market_to_trade == 'rv':
                    try:
                        atm_interval_width = study.user_attrs['best_atm_interval_width']
                    except KeyError:
                        continue
                long = ahead_d_data.mul(long_filter)
                short = ahead_d_data.mul(short_filter)
                if market_to_trade == 'returns':
                    prices = Trader._reader_obj.prices_read().resample(freq).last().loc[ahead_d_data.index, :]
                    long_pnl = prices.subtract(prices.shift()).sum(axis=1)
                    long_fees = prices.add(prices.shift()).sum(axis=1).mul(FEES_SPOT)
                    long_pnl = long_pnl.subtract(long_fees).values.reshape(-1, 1)
                    short_pnl = prices.shift().subtract(prices).sum(axis=1)
                    short_fees = prices.shift().add(prices).sum(axis=1).mul(FEES_SPOT)
                    short_pnl = short_pnl.subtract(short_fees).values.reshape(-1, 1)
                else:
                    prices, moneyness = Trader._option_prices[date.strftime(DATE_FORMAT)],\
                        Trader._option_moneyness[date.strftime(DATE_FORMAT)]
                    pdb.set_trace()
                    long_pnl, short_pnl = prices.subtract(prices.shift()), prices.shift().subtract(prices)
                long_pnl, short_pnl = np.sign(long).mul(long_pnl).cumsum().sum(axis=1),\
                    np.sign(short).mul(short_pnl).cumsum().sum(axis=1)
                pnl[date] = \
                    [(long_pnl.add(short_pnl)).div(2).resample('1D').sum().iloc[0], study.user_attrs['best_filter']]
        pnl = pd.DataFrame(pnl).transpose().sort_index()
        pnl.index = pd.to_datetime(pnl.index, utc=True)
        pnl[0].resample('1W', convention='end').mean().cumsum().plot()
        plt.show()
        pdb.set_trace()


if __name__ == '__main__':
    trader_obj = Trader()
    trader_obj.create_option_chain_data(freq='1h')
    data = dict([(symbol, {}) for symbol in ['BTCUSDT', 'ETHUSDT']])
    data['BTCUSDT'].update(
        {'returns':
             pd.concat([trader_obj._reader_obj.returns_read(symbol='BTCUSDT').rename(columns={'BTCUSDT': 'obs'}),
                        trader_obj._reader_obj.returns_read(symbol='BTCUSDT').rename(columns={'BTCUSDT': 'fcst'})],
                       axis=1)})
    data['BTCUSDT'].update(
        {'rv':
             pd.concat([trader_obj._reader_obj.rv_read(symbol='BTCUSDT').rename(columns={'BTCUSDT': 'obs'}),
                        trader_obj._reader_obj.rv_read(symbol='BTCUSDT').rename(columns={'BTCUSDT': 'fcst'})],
                       axis=1)})
    data['ETHUSDT'].update(
        {'returns':
             pd.concat([trader_obj._reader_obj.returns_read(symbol='ETHUSDT').rename(columns={'ETHUSDT': 'obs'}),
                        trader_obj._reader_obj.returns_read(symbol='ETHUSDT').rename(columns={'ETHUSDT': 'fcst'})],
                       axis=1)})
    data['ETHUSDT'].update(
        {'rv':
             pd.concat([trader_obj._reader_obj.rv_read(symbol='ETHUSDT').rename(columns={'ETHUSDT': 'obs'}),
                        trader_obj._reader_obj.rv_read(symbol='ETHUSDT').rename(columns={'ETHUSDT': 'fcst'})],
                       axis=1)})
    data['BTCUSDT'] = pd.concat(data['BTCUSDT'], axis=1)
    data['ETHUSDT'] = pd.concat(data['ETHUSDT'], axis=1)
    data = pd.concat(data, axis=1)
    data.loc[:, data.columns.get_level_values(2) == 'fcst'] = \
        data.loc[:, data.columns.get_level_values(2) == 'fcst'].add(np.random.normal(size=(data.shape[0],
                                                                                           data.shape[1]//2)))
    ####################################################################################################################
    ## Model selection
    ####################################################################################################################
    trader_obj.data = data.resample('1h').sum()
    trader_obj.backtest(D='7D', freq='1h')


