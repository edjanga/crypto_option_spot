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

load_dotenv()
API_KEY = os.getenv('API_KEY')

data_centre_dir_data = os.path.abspath(__file__).replace('/data.py', '/tmp/underlying')


class Reader:

    def __init__(self, num_sym: int = 20):

        self._num_sym = num_sym
        self._directory = data_centre_dir_data
        self._df = \
            pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}') for year in [2021, 2022, 2023]])
        self._df = self._df.fillna(self._df.ewm(span=12, min_periods=12).mean())
        self._df.dropna(axis=1, inplace=True)
        self._volume_df = \
            pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}_volume') for
                       year in [2021, 2022, 2023]])[self._df.columns]
        self._df.index = pd.to_datetime(self._df.index, utc=pytz.UTC)
        self._volume_df.index = pd.to_datetime(self._volume_df.index, utc=pytz.UTC)
        most_liquid_pairs = self._volume_df.mul(self._df).sum().sort_values(ascending=False)[:self._num_sym].index
        self._df = self._df[most_liquid_pairs]
        self._volume_df = self._volume_df[most_liquid_pairs]

    def prices_read(self, symbol: Union[str, List[str]] = 'ETHUSDT') -> pd.DataFrame:
        if symbol:
            if isinstance(symbol, str):
                symbol = [symbol]

            return self._df[symbol]
        else:
            return self._df

    # def volumes_read(self) -> pd.DataFrame:
    #     volumes = pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}_volume')
    #                          for year in [2021, 2022, 2023]])
    #     volumes = volumes[self._df.columns]
    #     volumes.index = pd.to_datetime(volumes.index, utc=pytz.UTC)
    #     return volumes

    def returns_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False,
                     resampled: bool = True, symbol: Union[List[str], str] = 'ETHUSDT') -> pd.DataFrame:
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
                symbol: Union[List[str], str] = 'ETHUSDT', variance: bool = True)-> pd.DataFrame:
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

    def download_option_chain_data(self, start_date: str, end_date: str, multithreading: bool = True,
                                   universe: Union[str, List[str]] = 'ETH') -> None:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        dates = list(pd.date_range(start=start_date, end=end_date, inclusive='left', freq='1D'))
        if isinstance(universe, str):
            universe = [universe]
        if multithreading:
            for idx in range(0, len(dates), 7):
                print(f'[FETCHING]: Option chain data between {dates[idx].strftime("%Y-%m-%d")} and '
                      f'{dates[idx+6].strftime("%Y-%m-%d")} is being processed...')
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(lambda x: x, x=date) for date in dates[idx:idx + 7]]
                    for future in concurrent.futures.as_completed(futures):
                        data = list()
                        date = future.result()
                        datasets.download(exchange='deribit', data_types=['options_chain'],
                                          from_date=date.strftime('%Y-%m-%d'),
                                          to_date=date.strftime('%Y-%m-%d'), symbols=['OPTIONS'],
                                          api_key=API_KEY, format='csv', download_dir='../data_centre/tmp/datasets')
                        print(f'[PROCESSING]: Option chain data {date.strftime("%Y-%m-%d")} is being processed...')
                        tmp = \
                            pd.read_csv(
                                f'../data_centre/tmp/datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}'
                                f'_OPTIONS.csv.gz',
                                compression='gzip', chunksize=100_000)
                        os.remove(path=os.path.abspath(
                            f'../data_centre/tmp/datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}'
                            f'_OPTIONS.csv.gz'))
                        for _, chunk in enumerate(tmp):
                            chunk = chunk.loc[chunk.symbol.str.contains('|'.join(universe)), :]
                            # chunk = chunk.drop(['local_timestamp', 'underlying_index', 'open_interest', 'exchange',
                            #                     'last_price', 'ask_iv', 'bid_iv', 'bid_price', 'ask_price'], axis=1)
                            # chunk['timestamp'] = [datetime.fromtimestamp(idx / 1_000_000) for idx in chunk['timestamp']]
                            # chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='ISO8601', utc=True)
                            # chunk['mark_iv'] = chunk['mark_iv'].div(100) ** .5
                            # chunk['volume'] = (chunk['bid_amount'] + chunk['ask_amount']).mul(chunk['mark_price']).mul(
                            #     chunk['underlying_price'])
                            # chunk = chunk.drop(['bid_amount', 'ask_amount'], axis=1)
                            # chunk = chunk.set_index('timestamp')
                            data.append(pd.DataFrame(chunk))
                        data = pd.concat(data)
                        # data = OptionChainData.resample_option_chain_data(data=data)
                        print(f'[END PROCESSING]: Option chain data for {date.strftime("%Y-%m-%d")} '
                              f'has been processed.')
                        data.to_parquet(f'../data_centre/tmp/datasets/{date.strftime("%Y-%m-%d")}')
        else:
            for date in dates:
                data = list()
                print(f'[FETCHING]: Option chain data between {date.strftime("%Y-%m-%d")} is being processed...')
                datasets.download(exchange='deribit', data_types=['options_chain'],
                                  from_date=date.strftime('%Y-%m-%d'),
                                  to_date=date.strftime('%Y-%m-%d'), symbols=['OPTIONS'],
                                  api_key=API_KEY, format='csv',
                                  download_dir='../data_centre/tmp/datasets')
                                  #f'{self._target_path}/tmp/')
                print(f'[PROCESSING]: Option chain data between {date.strftime("%Y-%m-%d")} is being processed...')
                tmp = pd.read_csv(
                    f'../data_centre/tmp/datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}_OPTIONS.csv.gz',
                    compression='gzip', chunksize=100_000)
                os.remove(path=os.path.abspath(
                    f'../data_centre/tmp/datasets/deribit_options_chain_{date.strftime("%Y-%m-%d")}_OPTIONS.csv.gz'))
                for _, chunk in enumerate(tmp):
                    chunk['volume'] = \
                        (chunk['bid_amount'].mul(chunk['bid_price'])).add(chunk['ask_amount'].mul(chunk['ask_price']))
                    chunk = chunk.loc[chunk.symbol.str.contains('|'.join(universe)), :]
                    # chunk = chunk.drop(['local_timestamp', 'underlying_index', 'open_interest', 'exchange',
                    #                     'last_price', 'ask_iv', 'bid_iv', 'bid_price', 'ask_price'], axis=1)
                    # chunk['timestamp'] = [datetime.fromtimestamp(idx / 1_000_000) for idx in chunk['timestamp']]
                    # chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='ISO8601', utc=True)
                    chunk['mark_iv'] = chunk['mark_iv'].div(100)
                    # chunk = chunk.drop(['bid_amount', 'ask_amount'], axis=1)
                    # chunk = chunk.set_index('timestamp')
                    data.append(pd.DataFrame(chunk))
                data = pd.concat(data)
                # data = self.resample_option_chain_data(data=data)
                print(f'[END PROCESSING]: Option chain data for {date.strftime("%Y-%m-%d")} has been processed.')
                data.to_parquet(f'../data_centre/tmp/datasets/{date.strftime("%Y-%m-%d")}')

    @staticmethod
    def resample_option_chain_data(data: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
        call = data.query('type == "call"')
        put = data.query('type == "put"')
        data = list()
        for tmp in [call, put]:
            tmp = tmp.assign(option_tag=['-'.join(option_tag[:-1]) for option_tag in tmp['symbol'].str.split('-')])
            tmp = tmp.drop('symbol', axis=1)
            tmp = tmp.set_index(['option_tag'], append=True).swaplevel(i=-1, j=0)
            tmp['mark_price'] = tmp['mark_price'] * tmp['underlying_price']
            tmp['expiration'] = [datetime.fromtimestamp(idx / 1_000_000) for idx in tmp['expiration']]
            tmp['expiration'] = pd.to_datetime(tmp['expiration'], format='ISO8601', utc=True)
            tmp = tmp.assign(moneyness=tmp['underlying_price'].div(tmp['strike_price']))
            tmp = tmp.groupby(by=[pd.Grouper(level='option_tag'),
                                  pd.Grouper(level='timestamp', freq=freq)]).agg({'type': 'last',
                                                                                  'strike_price': 'last',
                                                                                  'mark_price': 'last',
                                                                                  'mark_iv': 'last',
                                                                                  'underlying_price': 'last',
                                                                                  'delta': 'last',
                                                                                  'gamma': 'last', 'vega': 'last',
                                                                                  'theta': 'last', 'rho': 'last',
                                                                                  'volume': 'sum',
                                                                                  'expiration': 'last'})
            data.append(tmp)
        data = pd.concat(data)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script handling data fetching from Tardis.')
    parser.add_argument('--target_path', type=str, help='Target directory to store data.',
                        default='./data_centre/tmp')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date (excluded).', default='2023-07-01')
    parser.add_argument('--multithreading', type=int, help='Use of multithreading or not', default=1)
    args = parser.parse_args()
    option_chain_data_obj = OptionChainData(target_path=args.target_path)
    option_chain_data_obj.download_option_chain_data(start_date=args.start_date, end_date=args.end_date,
                                                     multithreading=bool(args.multithreading))