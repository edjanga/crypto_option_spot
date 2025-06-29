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
import polars as pl
import time

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
        # returns_df.dropna(inplace=True)
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

    def save_daily_option_chain(self, symbol: str, date: datetime, freq: str, underlying_price: pl.DataFrame) -> None:
        print(f'[PROCESSING]: Option chain data {symbol}USDT-{date.strftime(DATE_FORMAT)} is being processed...')
        option_chain_data = \
            pl.scan_csv(
                source=f'{self._target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz',
                schema_overrides=dict([(col, pl.Float64) for col in FLOAT_COLUMNS+['open_interest']+INT_COLUMNS] +
                                      [(col, pl.String) for col in ['symbol', 'type']]),
            ).select(['symbol', 'timestamp', 'type', 'strike_price', 'expiration', 'open_interest', 'last_price',
                      'bid_price', 'bid_amount', 'ask_price', 'ask_amount', 'mark_iv', 'delta', 'gamma', 'vega',
                      'theta', 'rho']).sort(['symbol', 'timestamp'])
        option_chain_data = option_chain_data.filter(pl.col('symbol').str.contains(f'{symbol.replace("USDT", "")}'))
        option_chain_data = option_chain_data.with_columns([
            (pl.col('timestamp') // 1_000).cast(pl.Datetime('ms', 'UTC')),
            (pl.col('expiration') // 1_000).cast(pl.Datetime('ms', 'UTC')), (pl.col('mark_iv') / 100)
        ])
        option_chain_data = \
            (option_chain_data.group_by_dynamic(index_column='timestamp', every=freq.replace('in', ''),
                                                group_by='symbol', include_boundaries=False,
                                                period=None).agg([pl.all().last()]).drop_nulls())
        option_chain_data = option_chain_data.collect()
        option_chain_data = \
            option_chain_data.join(underlying_price.with_columns(pl.col('timestamp').cast(pl.Datetime('ms', 'UTC')),
                                                                 (pl.col(f'{symbol}USDT')).alias('underlying_price')),
                                   on='timestamp', how='left')
        option_chain_data = option_chain_data.with_columns([
            (pl.col('underlying_price') / pl.col('strike_price')).log().alias('moneyness')])
        os.makedirs(f'{self._target_path}/{symbol.replace("USDT", "")}', exist_ok=True)
        os.makedirs(f'{self._target_path}/{symbol.replace("USDT", "")}/{date.year}', exist_ok=True)
        option_chain_data.write_parquet(
            f'{self._target_path}/{symbol.replace("USDT", "")}/{date.year}/{date.strftime(DATE_FORMAT)}',
            compression='gzip')
        print(f'[END PROCESSING]: Option chain data {symbol}USDT-{date.strftime(DATE_FORMAT)} has been processed.')
        return

    def download_option_chain_data(self, start_date: str, end_date: str, freq: str, universe: str) -> None:
        dates = list(pd.date_range(start=start_date, end=end_date, inclusive='left', freq='1D'))
        universe = universe.split('-')
        underlying_prices = \
            pl.from_dataframe(self._reader_obj.prices_read().resample('5min').last()).with_columns(
            [(pl.col('timestamp')).cast(pl.Datetime('ms', 'UTC'))])
        if isinstance(universe, str):
            universe = [universe]
        for tmp_start_date in dates[::7]:
            tmp_end_date = (tmp_start_date + relativedelta(days=7)).strftime(DATE_FORMAT)
            tmp_start_date = tmp_start_date.strftime(DATE_FORMAT)
            print(f'[FETCHING]: Option chain data between {tmp_start_date} and {tmp_end_date} is being fetched....')
            datasets.download(exchange='deribit', data_types=['options_chain'], from_date=tmp_start_date,
                              to_date=tmp_end_date, symbols=['OPTIONS'], api_key=API_KEY, format='csv',
                              concurrency=7, download_dir=self._target_path)
            print(f'[FETCHING]: Option chain data between {tmp_start_date} and {tmp_end_date} has been fetched.')
            week = list(pd.date_range(start=tmp_start_date, end=tmp_end_date, inclusive='both', freq='1D'))
            for date in week:
                for symbol in universe:
                    self.save_daily_option_chain(symbol, date, freq,
                                                 underlying_prices.select([f'{symbol}USDT', 'timestamp']))
                print(f'[END PROCESSING]: Option chain data for {date.strftime(DATE_FORMAT)} '
                      f'has been processed.')
                os.remove(f'{self._target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script handling data fetching from Tardis.')
    parser.add_argument('--universe', type=str, help='Symbols to fetch data for.', default='ETH-BTC-BNB-SOL-XRP')
    parser.add_argument('--target_path', type=str, help='Target directory to store data.', default='../data_centre')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date (excluded).', default='2023-07-01')
    parser.add_argument('--freq', type=str, help='Frequency used to resample option chain data', default='5min')
    args = parser.parse_args()
    option_chain_data_obj = OptionChainData(target_path=args.target_path)
    option_chain_data_obj.download_option_chain_data(start_date=args.start_date, end_date=args.end_date,
                                                     freq=args.freq, universe=args.universe)