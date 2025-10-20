import itertools
from datetime import datetime
from tardis_dev import datasets
from typing import List
import pandas as pd
from dotenv import load_dotenv
import os
import argparse
from dateutil.relativedelta import relativedelta
import polars as pl
from constants import ENGINE

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


def column_levels_book_snapshot(level: int = 25) -> List[str]:
    column_levels = [[f'asks[{lvl}].price', f'asks[{lvl}].amount', f'bids[{lvl}].price', f'bids[{lvl}].amount']
                     for lvl in range(0, level)]
    column_levels = list(itertools.chain.from_iterable(column_levels))
    return column_levels


def save_daily_orderbook_snapshot(target_path: str, date: datetime, freq: str) -> None:
    book_snapshot_data = list()
    column_levels = column_levels_book_snapshot()
    for idx, symbol in enumerate(SYMBOLS):
        tmp = \
            pl.scan_csv(
                source=f'{target_path}/binance_book_snapshot_25_{date.strftime(DATE_FORMAT)}_{symbol}.csv.gz',
                schema_overrides=dict([(col, pl.Float32) for col in column_levels] +
                                      [(col, pl.Int64) for col in ['timestamp']]),
                low_memory=True).select(['symbol', 'timestamp'] + column_levels).sort(['timestamp'])
        tmp = tmp.with_columns([(pl.col('timestamp')/1_000).cast(pl.Datetime('ms', 'UTC'))])
        volume_expr = [
            (pl.col(f'asks[{i}].amount') + pl.col(f'bids[{i}].amount')).alias(f'volume[{i}]') for i in range(0, 25)
        ]
        spread_expr = [
            (pl.col(f'asks[{i}].price') - pl.col(f'bids[{i}].price')).alias(f'spreads[{i}]') for i in range(0, 25)
        ]
        mid_expr = \
            [(.5*(pl.col(f'bids[{i}].price') + pl.col(f'asks[{i}].price'))).alias(f'mid_price[{i}]') for i in
             range(0, 25)]
        tmp = tmp.group_by_dynamic(
            index_column='timestamp', every=freq, period=freq, closed='left', label='left').agg(pl.all().last())
        tmp = tmp.with_columns(volume_expr)
        tmp = tmp.with_columns(spread_expr)
        tmp = tmp.with_columns(mid_expr)
        book_snapshot_data.append(tmp.collect(engine=ENGINE))
        os.remove(f'{target_path}/binance_book_snapshot_25_{date.strftime(DATE_FORMAT)}_{symbol}.csv.gz')
    book_snapshot_data = pl.concat(book_snapshot_data)
    os.makedirs(f'{target_path}/{date.year}/spot', exist_ok=True)
    book_snapshot_data.lazy().sink_parquet(path=f'{target_path}/{date.year}/spot/{date.strftime(DATE_FORMAT)}')
    print(f'[GENERATED]: Orderbook snapshots for {date.strftime(DATE_FORMAT)} and has been computed.')


def save_daily_option_chain(target_path: str, date: datetime, freq: str) -> None:
    print(f'[PROCESSING]: Option chain data {date.strftime(DATE_FORMAT)} is being processed...')
    underlying_data = pl.scan_parquet(source=f'{target_path}/{date.year}/spot/{date.strftime(DATE_FORMAT)}').select(
        ['timestamp', 'symbol', 'mid_price[0]']).rename({'symbol': 'underlying_name',
                                                         'mid_price[0]': 'underlying_price'})
    option_chain_data = \
        pl.scan_csv(
            source=f'{target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz',
            schema_overrides=dict([(col, pl.Float64) for col in FLOAT_COLUMNS + ['open_interest', 'strike_price']] +
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
    option_chain_data = option_chain_data.join(underlying_data, on=['underlying_name', 'timestamp'], how='left')
    option_chain_data = option_chain_data.with_columns([
        (pl.col('expiration') // 1_000).cast(pl.Datetime('ms', 'UTC')).alias('expiration'), (pl.col('mark_iv') / 100),
        (pl.col('underlying_price') / pl.col('strike_price')).log().alias('moneyness'),
        (pl.when(pl.col('underlying_name').is_in(SYMBOLS[:2])).then(
            pl.col('ask_price') * pl.col('underlying_price')).otherwise(pl.col('ask_price'))).alias('ask_price'),
        (pl.when(pl.col('underlying_name').is_in(SYMBOLS[:2])).then(
            pl.col('bid_price') * pl.col('underlying_price')).otherwise(pl.col('bid_price'))).alias('bid_price'),
        (pl.when(pl.col('underlying_name').is_in(SYMBOLS[:2])).then(
            pl.col('last_price') * pl.col('underlying_price')).otherwise(pl.col('last_price'))).alias('last_price'),
        (pl.col('bid_amount') + pl.col('ask_amount')).alias('volume'),
    ])
    option_chain_data = \
        option_chain_data.with_columns([(.5 * (pl.col('bid_price') + pl.col('ask_price'))).alias('mid_price'),
                                        (pl.col('ask_price') - pl.col('bid_price')).alias('spread')]).drop(
            ['bid_price', 'ask_price', 'bid_amount', 'ask_amount'])
    os.makedirs(f'{target_path}/{date.year}/options', exist_ok=True)
    option_chain_data.sink_parquet(path=f'{target_path}/{date.year}/options/{date.strftime(DATE_FORMAT)}')
    print(f'[PROCESSING]: Option chain data {date.strftime(DATE_FORMAT)} has been processed.')
    return None


class CryptoData:

    def __init__(self, target_path: str):
        self._target_path = target_path

    def download_spot_data(self, start_date: str, end_date: str, freq: str) -> None:
        dates = list(pd.date_range(start=start_date, end=end_date, inclusive='left', freq='1D'))
        for tmp_start_date in dates[::min(7, len(dates))]:
            print(f'[FETCHING]: Orderbook snapshot data between {tmp_start_date.strftime(DATE_FORMAT)} and '
                  f'{(tmp_start_date + relativedelta(days=min(6, len(dates)))).strftime(DATE_FORMAT)} '
                  f'is being fetched....')
            datasets.download(exchange='binance', data_types=['book_snapshot_25'],
                              from_date=tmp_start_date.strftime(DATE_FORMAT),
                              to_date=(tmp_start_date + relativedelta(days=min(7, len(dates)))).strftime(DATE_FORMAT),
                              symbols=SYMBOLS, api_key=API_KEY,
                              format='csv', concurrency=os.cpu_count(), download_dir=self._target_path)
            print(f'[FETCHING]: Option chain data between {tmp_start_date.strftime(DATE_FORMAT)} and '
                  f'{(tmp_start_date + relativedelta(days=min(6, len(dates)))).strftime(DATE_FORMAT)} '
                  f'has been fetched.')
            week = list(pd.date_range(start=tmp_start_date,
                                      end=(tmp_start_date + relativedelta(days=min(6, len(dates)))),
                                      inclusive='both', freq='1D'))
            for date in week:
                save_daily_orderbook_snapshot(self._target_path, date, freq)

    def download_option_chain_data(self, start_date: str, end_date: str, freq: str) -> None:
        dates = list(pd.date_range(start=start_date, end=end_date, inclusive='left', freq='1D'))
        for tmp_start_date in dates[::min(7, len(dates))]:
            print(f'[FETCHING]: Option chain data between {tmp_start_date.strftime(DATE_FORMAT)} and '
                  f'{(tmp_start_date + relativedelta(days=min(7, len(dates)))).strftime(DATE_FORMAT)} '
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
                save_daily_option_chain(self._target_path, date, freq)
                os.remove(f'{self._target_path}/deribit_options_chain_{date.strftime(DATE_FORMAT)}_OPTIONS.csv.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script handling data fetching from Tardis.')
    parser.add_argument('--target_path', type=str, help='Target directory to store data.', default='../data_centre')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date (excluded).', default='2023-07-01')
    parser.add_argument('--freq', type=str, help='Frequency used to resample option chain data', default='1m')
    parser.add_argument('--type', type=str, help='Nature of data to download: sport or option chain data',
                        default='options')
    args = parser.parse_args()
    crypto_data_obj = CryptoData(target_path=args.target_path)
    if args.type == 'spot':
        crypto_data_obj.download_spot_data(start_date=args.start_date, end_date=args.end_date, freq=args.freq)
    elif args.type == 'options':
        crypto_data_obj.download_option_chain_data(start_date=args.start_date, end_date=args.end_date, freq=args.freq)