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
from multiprocessing import Pool

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


def save_daily_option_chain(target_path: str, date: datetime, freq: str) -> datetime:
    underlying_price = OptionChainData._underlying_prices.filter(pl.col('timestamp').dt.date() == date)
    underlying_price = underlying_price.to_pandas().set_index('timestamp')
    underlying_price = \
        pd.melt(underlying_price, value_name='underlying_price', var_name='underlying_name', ignore_index=False)
    underlying_price.reset_index(inplace=True)
    underlying_price = pl.from_dataframe(underlying_price).lazy()
    print(f'[PROCESSING]: Option chain data {date.strftime(DATE_FORMAT)} is being processed...')
    option_chain_data = \
        pl.scan_csv(
            source=f'{target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz',
            schema_overrides=dict([(col, pl.Float32) for col in FLOAT_COLUMNS + ['open_interest', 'strike_price']] +
                                  [(col, pl.Int64) for col in INT_COLUMNS[1:]] +
                                  [(col, pl.String) for col in ['symbol', 'type']]),
            low_memory=True
        ).select(
            ['symbol', 'timestamp', 'type', 'strike_price', 'expiration', 'open_interest', 'last_price',
             'bid_price', 'bid_amount', 'ask_price', 'ask_amount', 'mark_iv', 'delta', 'gamma', 'vega',
             'theta', 'rho']
        ).with_columns(
            (pl.col('symbol').str.replace('_', '-').str.split('-').list.get(0) + 'USDT').alias('underlying_name')
        ).sort(['symbol', 'timestamp'])
    option_chain_data = option_chain_data.with_columns([(pl.col('timestamp') // 1_000).cast(pl.Datetime('ms', 'UTC'))])
    option_chain_data = \
        (option_chain_data.group_by_dynamic(index_column='timestamp', every=freq.replace('in', ''),
                                            group_by='symbol', include_boundaries=False,
                                            period=None).agg([pl.all().last()]))
    option_chain_data = option_chain_data.with_columns(pl.col('underlying_name').str.replace('MATICUSDT', 'XRPUSDT'))
    option_chain_data = option_chain_data.join(underlying_price, on=['underlying_name', 'timestamp'], how='left')
    option_chain_data = option_chain_data.with_columns([
        (pl.col('expiration') // 1_000).cast(pl.Datetime('ms', 'UTC')).alias('expiration'), (pl.col('mark_iv') / 100),
        (pl.col('underlying_price') / pl.col('strike_price')).log().alias('moneyness'),
        (pl.col('ask_price') * pl.col('underlying_price')).alias('ask_price'),
        (pl.col('bid_price') * pl.col('underlying_price')).alias('bid_price'),
        (pl.col('last_price') * pl.col('underlying_price')).alias('last_price')
    ])
    option_chain_data = option_chain_data.with_columns([
        (.5 * (pl.col('bid_price') + pl.col('ask_price'))).alias('mid_price'),
        (pl.col('ask_price') - pl.col('bid_price')).alias('spread')
    ]).drop(['bid_price', 'ask_price'])
    os.makedirs(f'{target_path}/{date.year}', exist_ok=True)
    option_chain_data.sink_parquet(path=f'{target_path}/{date.year}/{date.strftime(DATE_FORMAT)}')
    print(f'[PROCESSING]: Option chain data {date.strftime(DATE_FORMAT)} has been processed.')
    return date


class OptionChainData:

    _reader_obj = Reader()
    _underlying_prices = _reader_obj.prices_read().resample('5min').last()
    _underlying_prices = \
        pl.from_dataframe(_underlying_prices).with_columns([(pl.col('timestamp')).cast(pl.Datetime('ms', 'UTC'))])

    def __init__(self, target_path: str):
        self._target_path = target_path

    def download_option_chain_data(self, start_date: str, end_date: str, freq: str) -> None:
        dates = list(pd.date_range(start=start_date, end=end_date, inclusive='left', freq='1D'))
        for tmp_start_date in dates[::min(7, len(dates))]:
            print(f'[FETCHING]: Option chain data between {tmp_start_date.strftime(DATE_FORMAT)} and '
                  f'{(tmp_start_date + relativedelta(days=min(6, len(dates)))).strftime(DATE_FORMAT)} '
                  f'is being fetched....')
            datasets.download(exchange='deribit', data_types=['options_chain'],
                              from_date=tmp_start_date.strftime(DATE_FORMAT),
                              to_date=(tmp_start_date + relativedelta(days=min(7, len(dates)))).strftime(DATE_FORMAT),
                              symbols=['OPTIONS'], api_key=API_KEY,
                              format='csv', concurrency=os.cpu_count(), download_dir=self._target_path)
            print(f'[FETCHING]: Option chain data between {tmp_start_date.strftime(DATE_FORMAT)} and '
                  f'{(tmp_start_date + relativedelta(days=min(6, len(dates)))).strftime(DATE_FORMAT)} '
                  f'has been fetched.')
            week = list(pd.date_range(start=tmp_start_date,
                                      end=(tmp_start_date + relativedelta(days=min(6, len(dates)))),
                                      inclusive='both', freq='1D'))
            for date in week:
                _ = save_daily_option_chain(self._target_path, date, freq)
                os.remove(f'{self._target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz')
            # with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            #     futures = [executor.submit(save_daily_option_chain, self._target_path, date, freq) for date in week]
            #     for future in concurrent.futures.as_completed(futures):
            #         date = future.result()
            #         os.remove(f'{self._target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script handling data fetching from Tardis.')
    parser.add_argument('--target_path', type=str, help='Target directory to store data.', default='../data_centre')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date (excluded).', default='2023-07-01')
    parser.add_argument('--freq', type=str, help='Frequency used to resample option chain data', default='5min')
    args = parser.parse_args()
    option_chain_data_obj = OptionChainData(target_path=args.target_path)
    option_chain_data_obj.download_option_chain_data(start_date=args.start_date, end_date=args.end_date, freq=args.freq)
    pdb.set_trace()