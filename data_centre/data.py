import pdb
from datetime import datetime
from tardis_dev import datasets
from typing import Union, List
import pandas as pd
from dotenv import load_dotenv
import os
import concurrent
import argparse
import pytz
import numpy as np
from scipy.stats.mstats import winsorize

DATE_FORMAT = '%Y-%m-%d'

load_dotenv()
API_KEY = os.getenv('API_KEY')

data_centre_dir_data = os.path.abspath(__file__).replace('/data.py', '/tmp/underlying')

agg_features = {feature: 'last' for feature in ['type', 'strike_price', 'expiration', 'last_price',
                                                'bid_price', 'bid_amount', 'bid_iv', 'ask_price', 'ask_amount',
                                                'ask_iv', 'mark_price', 'mark_iv', 'underlying_index',
                                                'underlying_price', 'delta', 'gamma', 'vega', 'theta', 'rho']}


class Reader:

    def __init__(self):

        # self._num_sym = num_sym
        self._directory = data_centre_dir_data
        self._df = \
            pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}') for year in [2021, 2022, 2023]])
        self._df = self._df.fillna(self._df.ewm(span=12, min_periods=12).mean())
        self._df.dropna(axis=1, inplace=True)
        self._df.index = pd.to_datetime(self._df.index, utc=pytz.UTC)
        self._df = self._df[['BTCUSDT', 'ETHUSDT']]

    def prices_read(self, symbol: Union[str, List[str]] = ['BTCUSDT', 'ETHUSDT']) -> pd.DataFrame:
        if symbol:
            if isinstance(symbol, str):
                symbol = [symbol]
            return self._df[symbol]
        else:
            return self._df

    def returns_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False,
                     resampled: bool = True, symbol: Union[List[str], str] = ['BTCUSDT', 'ETHUSDT']) -> pd.DataFrame:
        returns_df = np.log(self._df.div(self._df.shift()))
        if symbol:
            if isinstance(symbol, str):
                symbol = [symbol]
            returns_df = returns_df[symbol]
        returns_df.dropna(inplace=True)
        """Winsorise returns"""
        if not raw:
            returns_df = returns_df.apply(lambda x: winsorize(x, (cutoff_low, cutoff_high)))
        if resampled:
            returns_df = returns_df.resample('5min').sum()
        returns_df = returns_df.fillna(returns_df.ewm(span=12, min_periods=1).mean())
        return returns_df

    def rv_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False,
                symbol: Union[List[str], str] = ['BTCUSDT', 'ETHUSDT'], variance: bool = True)-> pd.DataFrame:
        rv_df = \
            self.returns_read(cutoff_low=cutoff_low, cutoff_high=cutoff_high, raw=raw, resampled=False,
                              symbol=symbol)**2
        rv_df.ffill(inplace=True)
        rv_df = rv_df.resample('5min').sum() if variance else rv_df.resample('5min').sum() ** .5
        rv_df.replace(0, np.nan, inplace=True)
        rv_df = rv_df.fillna(rv_df.ewm(span=12, min_periods=1).mean())
        return rv_df


class OptionChainData:

    def __init__(self, target_path: str):
        self._target_path = target_path

    def download_option_chain_data(self, start_date: str, end_date: str, freq: str, chunksize: int,
                                   universe: str) -> None:
        start_date = datetime.strptime(start_date, DATE_FORMAT).date()
        end_date = datetime.strptime(end_date, DATE_FORMAT).date()
        dates = list(pd.date_range(start=start_date, end=end_date, inclusive='left', freq='1D'))
        universe = universe.split('-')
        if isinstance(universe, str):
            universe = [universe]
        for date in dates:
            print(f'[FETCHING]: Option chain data between {date.strftime(DATE_FORMAT)} is being fetched...')
            datasets.download(exchange='deribit', data_types=['options_chain'],
                              from_date=date.strftime(DATE_FORMAT), to_date=date.strftime(DATE_FORMAT),
                              symbols=['OPTIONS'], api_key=API_KEY, format='csv', download_dir='../data_centre')
            print(f'[PROCESSING]: Option chain data between {date.strftime(DATE_FORMAT)} is being processed...')
            tmp = pd.read_csv(f'../data_centre/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz',
                              compression='gzip', chunksize=chunksize)
            data = (OptionChainData.process_option_chain_data_chunk(chunk, universe) for chunk in tmp)
            os.remove(path=os.path.abspath(
                f'../data_centre/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz'))
            for chunk in data:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(lambda x: x, x=symbol) for symbol in universe]
                    for future in concurrent.futures.as_completed(futures):
                        symbol = future.result()
                        symbol_option_chain = chunk.loc[chunk.option_tag.str.contains(symbol), :].copy()
                        if not os.path.exists(f'../data_centre/{symbol}'):
                            os.mkdir(f'../data_centre/{symbol}')
                        symbol_option_chain.to_parquet(
                            f'../data_centre/{symbol}/{date.strftime(DATE_FORMAT)}', engine='fastparquet',
                            append=os.path.exists(f'../data_centre/{symbol}/{date.strftime(DATE_FORMAT)}'))
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(lambda x: x, x=symbol) for symbol in universe]
                for future in concurrent.futures.as_completed(futures):
                    symbol = future.result()
                    symbol_option_chain = pd.read_parquet(f'../data_centre/{symbol}/{date.strftime(DATE_FORMAT)}')
                    os.remove(path=os.path.abspath(f'../data_centre/{symbol}/{date.strftime(DATE_FORMAT)}'))
                    symbol_option_chain = \
                        OptionChainData.resample_option_chain_data(data=symbol_option_chain, freq=freq)
                    symbol_option_chain.to_parquet(
                        f'../data_centre/{symbol}/{date.year}/{date.strftime(DATE_FORMAT)}', compression='gzip')
            print(f'[END PROCESSING]: Option chain data for {date.strftime(DATE_FORMAT)} '
                  f'has been processed.')

    @staticmethod
    def resample_option_chain_data(data: pd.DataFrame, freq: str) -> pd.DataFrame:
        data.set_index(['option_tag', 'timestamp'], append=False, inplace=True)
        resample_option_chain = list()
        for type in ['call', 'put']:
            tmp = data.query(f'type == "{type}"').copy()
            tmp = tmp.groupby(by=[pd.Grouper(level='option_tag'),
                                  pd.Grouper(level='timestamp', freq=freq, sort=True)],
                              group_keys=False).agg(agg_features).dropna()
            tmp.loc[:, 'moneyness'] = tmp['underlying_price'].values / tmp['strike_price'].values
            resample_option_chain.append(tmp)
        return pd.concat(resample_option_chain)

    @staticmethod
    def process_option_chain_data_chunk(data: pd.DataFrame, universe: List[str]) -> pd.DataFrame:
        data = data.loc[data.symbol.str.contains('|'.join(universe)), :]
        data.loc[:, 'option_tag'] = ['-'.join(option_tag[:-1]) for option_tag in data.loc[:, 'symbol'].str.split('-')]
        data.loc[:, 'mark_iv'] = data.loc[:, 'mark_iv'].values / 100
        data.loc[:, 'timestamp'] = [int(idx / 1_000_000) for idx in data.loc[:, 'timestamp']]
        data.loc[:, 'expiration'] = [int(idx / 1_000_000) for idx in data.loc[:, 'expiration']]
        timestamp = [datetime.fromtimestamp(idx) for idx in data.loc[:, 'timestamp']]
        expiration = [datetime.fromtimestamp(idx) for idx in data.loc[:, 'expiration']]
        data.drop(['timestamp', 'expiration'], axis=1, inplace=True)
        data.loc[:, 'timestamp'] = pd.to_datetime(timestamp, format='ISO8601', utc=True, errors='coerce')
        data.loc[:, 'expiration'] = pd.to_datetime(expiration, format='ISO8601', utc=True, errors='coerce')
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script handling data fetching from Tardis.')
    parser.add_argument('--universe', type=str, help='Symbols to fetch data for.', default='ETH-BTC-BNB-SOL-XRP')
    parser.add_argument('--target_path', type=str, help='Target directory to store data.', default='./data_centre/tmp')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date (excluded).', default='2023-07-01')
    parser.add_argument('--freq', type=str, help='Frequency used to resample option chain data', default='5min')
    parser.add_argument('--chunksize', type=int, help='Size of chunk to speed up fetching process', default=100_000)
    args = parser.parse_args()
    option_chain_data_obj = OptionChainData(target_path=args.target_path)
    option_chain_data_obj.download_option_chain_data(start_date=args.start_date, end_date=args.end_date,
                                                     multithreading=bool(args.multithreading), freq=args.freq,
                                                     chunksize=args.chunksize, universe=args.universe)