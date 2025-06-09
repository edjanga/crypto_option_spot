import glob
import pdb
from datetime import datetime
from tardis_dev import datasets
from typing import Union, List, Generator
import pandas as pd
from dotenv import load_dotenv
import os
import concurrent
import argparse
import pytz
import numpy as np
from scipy.stats.mstats import winsorize
from dateutil.relativedelta import relativedelta

DATE_FORMAT = '%Y-%m-%d'

load_dotenv()
API_KEY = os.getenv('API_KEY')

data_centre_dir_data = os.path.abspath(__file__).replace('/data.py', '/tmp/underlying')

agg_features = {feature: 'last' for feature in ['type', 'strike_price', 'expiration', 'last_price',
                                                'bid_price', 'bid_amount', 'bid_iv', 'ask_price', 'ask_amount',
                                                'ask_iv', 'mark_price', 'mark_iv', 'underlying_index',
                                                'underlying_price', 'delta', 'gamma', 'vega', 'theta', 'rho']}
INT_COLUMNS = ['strike_price', 'expiration', 'timestamp']
FLOAT_COLUMNS = ['last_price', 'bid_price', 'bid_amount', 'bid_iv', 'ask_price', 'ask_amount', 'mark_iv', 'delta',
                 'gamma', 'vega', 'theta', 'rho']
STR_COLUMNS = ['type', 'symbol']
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT']


class Reader:

    def __init__(self):
        self._directory = data_centre_dir_data
        try:
            self._df = \
                pd.concat([pd.read_parquet(f'{self._directory}/aggregate{str(year)}') for year in range(2023, 2026)])
        except Exception:
            self._df = \
                pd.concat([pd.read_csv(f'{self._directory}/aggregate{str(year)}',
                                       index_col='timestamp') for year in range(2023, 2026)])
        self._df = self._df.fillna(self._df.ewm(span=12, min_periods=12).mean())
        self._df.index = pd.to_datetime(self._df.index, utc=pytz.UTC)

    def prices_read(self, symbol: Union[str, List[str]] = SYMBOLS) -> pd.DataFrame:
        if symbol:
            if isinstance(symbol, str):
                symbol = [symbol]
            return self._df[symbol]
        else:
            return self._df

    def returns_read(self, cutoff_low: float = .01, cutoff_high: float = .01, raw: bool = False,
                     resampled: bool = True, symbol: Union[List[str], str] = SYMBOLS) -> pd.DataFrame:
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
                symbol: Union[List[str], str] = SYMBOLS, variance: bool = True) -> pd.DataFrame:
        rv_df = \
            self.returns_read(cutoff_low=cutoff_low, cutoff_high=cutoff_high, raw=raw, resampled=False,
                              symbol=symbol)**2
        rv_df.ffill(inplace=True)
        rv_df = rv_df.resample('5min').sum() if variance else rv_df.resample('5min').sum() ** .5
        rv_df.replace(0, np.nan, inplace=True)
        rv_df = rv_df.fillna(rv_df.ewm(span=12, min_periods=1).mean())
        return rv_df


class OptionChainData:

    _reader_obj = Reader()

    def __init__(self, target_path: str):
        self._target_path = target_path

    def download_option_chain_data(self, start_date: str, end_date: str, freq: str, chunksize: int,
                                   universe: str) -> None:
        dates = list(pd.date_range(start=start_date, end=end_date, inclusive='left', freq='1D'))
        symbol_directories = dict()
        for symbol in SYMBOLS:
            symbol_directories[symbol.replace('USDT', '')] = \
                [filename.split('/')[-1] for filename in glob.glob(f'../data_centre/{symbol.replace("USDT", "")}/*/*')]
        universe = universe.split('-')
        underlying_prices = self._reader_obj.prices_read().resample('5min').last()
        if isinstance(universe, str):
            universe = [universe]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x: x, x=date) for date in dates[::2]]
            for future in concurrent.futures.as_completed(futures):
                tmp_start_date = future.result().strftime(DATE_FORMAT)
                tmp_end_date = (future.result() + relativedelta(days=2)).strftime(DATE_FORMAT)
                print(f'[FETCHING]: Option chain data between {tmp_start_date} and {tmp_end_date} is being fetched....')
                datasets.download(exchange='deribit', data_types=['options_chain'], from_date=tmp_start_date,
                                  to_date=tmp_end_date, symbols=['OPTIONS'], api_key=API_KEY, format='csv',
                                  concurrency=7, download_dir=self._target_path)
                print(f'[FETCHING]: Option chain data between {tmp_start_date} and {tmp_end_date} has been fetched....')
                week = list(pd.date_range(start=tmp_start_date, end=tmp_end_date, inclusive='left', freq='1D'))
                for date in week:
                    print(f'[PROCESSING]: Option chain data {date.strftime(DATE_FORMAT)} is being processed...')
                    data = \
                        pd.read_csv(
                            f'{self._target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz',
                            compression='gzip', chunksize=chunksize)
                    os.remove(path=os.path.abspath(
                        f'{self._target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz'))
                    for chunk in data:
                        for symbol in universe:
                            if not date.strftime(DATE_FORMAT) in symbol_directories[symbol]:
                                if not os.path.exists(f'{self._target_path}/{symbol}/{date.strftime(DATE_FORMAT)}'):
                                    symbol_option_chain = chunk.loc[chunk.symbol.str.contains(symbol), :].dropna()
                                    if not symbol_option_chain.empty:
                                        if not os.path.exists(f'{self._target_path}/{symbol}'):
                                            os.mkdir(f'{self._target_path}/{symbol}')
                                        symbol_option_chain.to_csv(
                                            f'{self._target_path}/{symbol}/{date.strftime(DATE_FORMAT)}',
                                            mode='a', compression='gzip',
                                            header=not os.path.exists(
                                                f'{self._target_path}/{symbol}/{date.strftime(DATE_FORMAT)}')
                                        )
                    for symbol in universe:
                        if not date.strftime(DATE_FORMAT) in symbol_directories[symbol]:
                            try:
                                symbol_option_chain = \
                                    pd.read_csv(f'{self._target_path}/{symbol}/{date.strftime(DATE_FORMAT)}',
                                                chunksize=chunksize, compression='gzip',
                                                usecols=INT_COLUMNS + FLOAT_COLUMNS + STR_COLUMNS)
                            except FileNotFoundError:
                                continue
                            else:
                                symbol_option_chain = self.process_option_chain_data_chunk(symbol_option_chain)
                                os.remove(
                                    path=os.path.abspath(f'{self._target_path}/{symbol}/{date.strftime(DATE_FORMAT)}'))
                                symbol_option_chain = \
                                    OptionChainData.resample_option_chain_data(data=symbol_option_chain, freq=freq)
                                symbol_option_chain.loc[:, 'moneyness'] = \
                                    np.log(underlying_prices[f'{symbol}USDT'].reindex(symbol_option_chain.index).div(
                                        symbol_option_chain['strike_price'].values))
                                if not os.path.exists(f'{self._target_path}/{symbol}/{date.year}'):
                                    os.mkdir(f'{self._target_path}/{symbol}/{date.year}')
                                symbol_option_chain.to_parquet(
                                    f'{self._target_path}/{symbol}/{date.year}/{date.strftime(DATE_FORMAT)}',
                                    compression='gzip')
                            print(f'[END PROCESSING]: Option chain data for {date.strftime(DATE_FORMAT)} '
                                  f'has been processed.')

    @staticmethod
    def resample_option_chain_data(data: pd.DataFrame, freq: str) -> pd.DataFrame:
        data.set_index(['symbol', 'timestamp'], append=False, inplace=True)
        data = data.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=freq, sort=True)],
                            group_keys=False).last().dropna().reset_index('symbol')
        data['option_tag'] = ['-'.join(option_tag[:-1]) for option_tag in data['symbol'].str.split('-')]
        return data[INT_COLUMNS[:-1]+FLOAT_COLUMNS+['option_tag', 'type']]

    @staticmethod
    def process_option_chain_data_chunk(data: Union[pd.DataFrame, Generator]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            for col_name, series in data.items():
                if col_name in ['mark_iv', 'timestamp', 'expiration']:
                    data.loc[:, col_name] = series.div(1_000_000).astype('int64').astype('datetime64[s, UTC]') \
                        if col_name in ['timestamp', 'expiration'] else series.div(100)
            return data
        elif isinstance(data, pd.io.parsers.readers.TextFileReader):
            tmp = list()
            for chunk in data:
                for col_name, series in chunk.items():
                    if col_name in ['mark_iv', 'timestamp', 'expiration']:
                        series = pd.to_numeric(series, errors='coerce')
                        chunk[col_name] = series.div(1_000_000).astype('int64').astype('datetime64[s, UTC]') \
                            if col_name in ['timestamp', 'expiration'] else series.div(100)
                tmp.append(chunk)
            return pd.concat(tmp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script handling data fetching from Tardis.')
    parser.add_argument('--universe', type=str, help='Symbols to fetch data for.', default='ETH-BTC-BNB-SOL-XRP')
    parser.add_argument('--target_path', type=str, help='Target directory to store data.', default='../data_centre')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date (excluded).', default='2023-07-01')
    parser.add_argument('--freq', type=str, help='Frequency used to resample option chain data', default='5min')
    parser.add_argument('--chunksize', type=int, help='Size of chunk to speed up fetching process', default=100_000)
    args = parser.parse_args()
    option_chain_data_obj = OptionChainData(target_path=args.target_path)
    option_chain_data_obj.download_option_chain_data(start_date=args.start_date, end_date=args.end_date,
                                                     freq=args.freq, chunksize=args.chunksize, universe=args.universe)