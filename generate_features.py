import argparse
import os
import pdb
import sqlite3
import pytz
from constants import *
import polars as pl
import pandas as pd
import glob
import concurrent.futures
from model.lab import FeatureBuilder
from itertools import product
from typing import List
from datetime import datetime
from dateutil.relativedelta import relativedelta


def return_filename(filename: str) -> str:
    return filename


def return_option_chain_filename_interval(date: datetime, symbol: str, target_path: str) -> List[str]:
    dates = pd.date_range(start=date, end=date+relativedelta(days=30), freq='1D', inclusive='left')
    tmp = list(product([target_path], [symbol], dates))
    return [f'{path}/{sym}/{d.year}/options/{d.strftime(DATE_FORMAT)}' for path, sym, d in tmp]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all option market based features presented in '
                                                 'the associated paper.\n')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2025-04-01')
    parser.add_argument('--freq', type=str, help='Frequency used to resample and aggregate data', default='15min')
    parser.add_argument('--options', type=int,
                        help='Tag indicating whether options dynamics are included in the models', default=0)
    parser.add_argument('--D', type=str, help='Number of days ahead to forecast values for', default='1M')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    parser.add_argument('--collect', type=int, help='Whether to have all data stored in memory or not',
                        default=1)
    args = parser.parse_args()
    print(f'[Generation]: Features {args} generation process has started..............................................')
    db_connect = sqlite3.connect(database=f'{args.target_path}/crypto_option_spot.db', check_same_thread=False,
                                 timeout=30)
    db_connect.execute('PRAGMA journal_mode=WAL;')
    feature_builder = FeatureBuilder()
    collect = bool(args.collect)
    ####################################################################################################################
    ## Spot market based features
    ####################################################################################################################
    if args.options == 0:
        spot = \
            pl.scan_parquet(f'{args.target_path}/*/*/spot/*-*-*').select([pl.col('symbol').alias('underlying_name'),
                                                                          pl.col('timestamp'), pl.col('mid_price[0]')])
        spot = spot.sort(by=[pl.col('underlying_name'), pl.col('timestamp')])
        spot = spot.group_by_dynamic(index_column='timestamp', every=args.freq.replace('in', ''),
                                     group_by='underlying_name').agg(
            (pl.col('mid_price[0]')/pl.col('mid_price[0]').shift()).log().sum().alias('returns'),
            ((pl.col('mid_price[0]') / pl.col('mid_price[0]').shift()).log()**2).sum().alias('RV'),
        )
        spot = feature_builder.returns_moments(data=spot, freq=args.freq)
        spot = feature_builder.lags(spot)
        features_spot = \
            spot.drop(['returns', 'RV']).unpivot(index=['underlying_name', 'timestamp'],
                                                 variable_name='features', value_name='features_values').with_columns(
            (pl.when(pl.col('features').str.contains('RV')).then(pl.lit('RV')).otherwise(pl.lit('returns')).alias(
                'target'))
        )
        spot = spot.select(['returns', 'RV', 'timestamp', 'underlying_name']).unpivot(
            index=['underlying_name', 'timestamp'], variable_name='target', value_name='target_values')
        spot = spot.join(features_spot, on=['target', 'underlying_name', 'timestamp']).to_pandas()
        spot['lag'] = [feature[-1] if feature[-1] in LAGS else 'contemporaneous' for feature in
                       spot.features.str.split('_')]
        spot['freq'] = args.freq
        spot.to_sql(name='spot_features', con=db_connect, index=False, if_exists='append')
        print(f'[Insertion]: Spot features have been inserted into the database.')
    ####################################################################################################################
    ## Options market based features
    ####################################################################################################################
    else:
        date_interval = pd.date_range(start=args.start_date, end=args.end_date, freq='30D', inclusive='both')
        for symbol in SYMBOLS:
            print(f'[Insertion]: Process to generate option features for {symbol} has started.........................')
            for idx, date in enumerate(date_interval):
                print(
                    f'[Features]: Generating option features for {symbol} between {date.strftime(DATE_FORMAT)} and '
                    f'{(date+relativedelta(days=30)).strftime(DATE_FORMAT)}.........................')
                option_chain_filenames = \
                    return_option_chain_filename_interval(date=date, symbol=symbol, target_path=args.target_path)
                option_chain_filenames = \
                    [option_chain_filename for option_chain_filename in option_chain_filenames if
                     os.path.exists(option_chain_filename)]
                if len(option_chain_filenames) > 0:
                    options = pl.scan_parquet(option_chain_filenames)
                    tmp_options = \
                        options.with_columns(
                            ((pl.col('expiration') - pl.col('timestamp')) / pl.duration(days=360)).alias('tau'))
                    tmp_options = \
                        tmp_options.with_columns(pl.when(pl.col('last_price') > pl.col('mid_price'))
                                                 .then(1).when(pl.col('last_price') < pl.col('mid_price'))
                                                 .then(-1).otherwise(None).alias('trades'))
                    is_put = pl.col('type') == 'put'
                    is_call = ~is_put
                    # tmp_options = \
                    #     tmp_options.with_columns(
                    #         pl.when(is_put).then(pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS))
                    #         .when(is_call).then(pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS[::-1]))
                    #         .otherwise(None).alias('moneyness_label'))
                    moneyness_dtype = pl.Enum(LABELS)
                    tmp_options = tmp_options.with_columns(
                        pl.when(is_put).then(
                            pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS).cast(moneyness_dtype))
                        .when(is_call).then(
                            pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS[::-1]).cast(moneyness_dtype))
                        .otherwise(None).alias('moneyness_label'))
                    tmp_options = tmp_options.filter(pl.col('trades').is_not_null())
                    tmp_options = \
                        tmp_options.with_columns(-(pl.col('volume') * pl.col('trades')).alias('netDealerExposure'))
                    exposure = feature_builder.compute_exposure(data=tmp_options, freq=args.freq, collect=collect)
                    daily_option_chain_data = tmp_options.drop(['trades', 'netDealerExposure'])
                    pcr = feature_builder.put_call_ratio(data=tmp_options, freq=args.freq, collect=collect)
                    pcr = feature_builder.lags(pcr, ['pcr'])
                    exposure = feature_builder.lags(exposure, ['gex', 'vax', 'vex', 'vox'])
                    ####################################################################################################
                    ## First aggregation in USDT and level terms
                    ####################################################################################################
                    agg_option_chain_data = exposure.join(pcr, on=['underlying_name', 'timestamp'])
                    ####################################################################################################
                    ## Aggregation per contract type: call and put
                    ####################################################################################################
                    spread = feature_builder.spread(data=tmp_options, freq=args.freq, collect=collect)
                    spread = feature_builder.lags(spread, ['spread_call', 'spread_put'])
                    mkt_participants = \
                        daily_option_chain_data.select(
                            [pl.col('underlying_name'), pl.col('timestamp'),
                             pl.when(pl.col('moneyness_label').is_in(['ATM', 'ITM'])).then(pl.lit('informed'))
                             .otherwise(pl.when(pl.col('moneyness_label') == 'OTM')
                                        .then(pl.lit('institutional')).otherwise(pl.lit('retail')))
                             .alias('mkt_participants')]).sort(by='timestamp')
                    mkt_participants = \
                        mkt_participants.group_by_dynamic(index_column='timestamp', every=args.freq.replace('in', ''))\
                            .agg([pl.col('underlying_name').last(), pl.col('mkt_participants').mode().get(-1).get(0)])
                    mkt_participants = \
                        mkt_participants.with_columns(
                            [(pl.col('mkt_participants') == pl.lit(mkt_participant)).cast(pl.UInt8)
                             .alias(f'is_{mkt_participant}') for mkt_participant in
                             ['retail', 'informed', 'institutional']]).drop('mkt_participants')
                    agg_option_chain_data = \
                        agg_option_chain_data.join(mkt_participants, on=['underlying_name', 'timestamp'],
                                                   how='left').join(spread, on=['underlying_name', 'timestamp'])
                    agg_option_chain_data = \
                        agg_option_chain_data.unpivot(index=['underlying_name', 'timestamp'], variable_name='features',
                                                      value_name='features_values')
                    agg_option_chain_data = agg_option_chain_data.with_columns(
                        [pl.when(pl.col('features').str.split('_').list.last().is_in(list(LAGS)))
                        .then(pl.col('features').str.split('_').list.last())
                        .otherwise(pl.lit('contemporaneous'))
                        .alias('lag'), pl.lit(args.freq).alias('freq')]
                    )
                    agg_option_chain_data = agg_option_chain_data.collect().to_pandas()
                    agg_option_chain_data.to_sql(name='option_features', con=db_connect, if_exists='append',
                                                 index=False, chunksize=10_000)
                    print(
                        f'[Features]: Option features for {symbol} between {date.strftime(DATE_FORMAT)} and '
                        f'{(date+relativedelta(days=30)).strftime(DATE_FORMAT)} have been inserted.')
                else:
                    print(f'[Features]: No data for {symbol} between {date.strftime(DATE_FORMAT)} and '
                          f'{(date+relativedelta(days=30)).strftime(DATE_FORMAT)}.')
            print(f'[Insertion]: Option features for {symbol} have been inserted into the database.')

