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

LAGS = INTRADAY_LAGS+INTERDAY_LAGS

def return_filename(filename: str):
    return filename


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
    db_connect = sqlite3.connect(database=f'{args.target_path}/crypto_option_spot.db', check_same_thread=False)
    feature_builder = FeatureBuilder()
    collect = bool(args.collect)
    ####################################################################################################################
    ## Spot market based features
    ####################################################################################################################
    if args.options == 0:
        spot = \
            pl.scan_parquet(f'{args.target_path}/*/spot/*-*-*').select([pl.col('symbol').alias('underlying_name'),
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
        spot.to_sql(name='spot_features', con=db_connect, index=False)
        print(f'[Insertion]: Spot features have been inserted into the database.')
    ####################################################################################################################
    ## Options market based features
    ####################################################################################################################
    else:
        date_interval = pd.date_range(start=args.start_date, end=args.end_date, freq='1MS', inclusive='both')
        for symbol in SYMBOLS:
            options = pl.scan_parquet(f'{args.target_path}/{symbol}/*/options/*-*-*')
            print(f'[Insertion]: Process to generate option features for {symbol} has started.........................')
            for idx, date in enumerate(date_interval[::2]):
                tmp_start, tmp_end = date, date_interval[idx+2]
                tmp_options = \
                    options.filter((pl.col('timestamp') >= pl.lit(tmp_start.date())) &
                                   (pl.col('timestamp') < pl.lit(tmp_end.date()))).drop_nulls()
                tmp_options = \
                    tmp_options.with_columns(
                        ((pl.col('expiration') - pl.col('timestamp')) / pl.duration(days=360)).alias('tau'))
                tmp_options = \
                    tmp_options.with_columns(pl.when(pl.col('last_price') > pl.col('mid_price'))
                                             .then(1).when(pl.col('last_price') < pl.col('mid_price'))
                                             .then(-1).otherwise(None).alias('trades'))
                is_put = pl.col('type') == 'put'
                is_call = ~is_put
                tmp_options = \
                    tmp_options.with_columns(
                        pl.when(is_put).then(pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS))
                        .when(is_call).then(pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS[::-1]))
                        .otherwise(None).alias('moneyness_label'))
                tmp_options = tmp_options.filter(pl.col('trades').is_not_null())
                tmp_options = \
                    tmp_options.with_columns(-(pl.col('volume') * pl.col('trades')).alias('netDealerExposure'))
                exposure = feature_builder.compute_exposure(data=tmp_options, freq=args.freq, collect=collect)
                daily_option_chain_data = tmp_options.drop(['trades', 'netDealerExposure'])
                pcr = feature_builder.put_call_ratio(data=tmp_options, freq=args.freq, collect=collect)
                pcr = feature_builder.lags(pcr, ['pcr'])
                exposure = feature_builder.lags(exposure, ['gex', 'vax', 'vex', 'vox'])
                ########################################################################################################
                ## First aggregation in USDT and level terms
                ########################################################################################################
                agg_option_chain_data = exposure.join(pcr, on=['underlying_name', 'timestamp'])
                ########################################################################################################
                ## Aggregation per contract type: call and put
                ########################################################################################################
                spread = feature_builder.spread(data=tmp_options, freq=args.freq, collect=collect)
                spread = feature_builder.lags(spread, ['spread_call', 'spread_put'])
                skew_call, skew_put = feature_builder.volatility_skew(data=tmp_options, freq=args.freq, collect=collect)
                skew_call, skew_put = feature_builder.lags(skew_call, skew_call.collect_schema().names()[2:]),\
                    feature_builder.lags(skew_put, skew_put.collect_schema().names()[2:])
                mkt_participants = \
                    daily_option_chain_data.select(
                        [pl.col('underlying_name'), pl.col('timestamp'),
                         pl.when(pl.col('moneyness_label').is_in(['ATM', 'ITM'])).then(pl.lit('informed'))
                         .otherwise(pl.when(pl.col('moneyness_label') == 'OTM')
                                    .then(pl.lit('institutional')).otherwise(pl.lit('retail')))
                         .alias('mkt_participants')]).sort(by='timestamp')
                mkt_participants = \
                    mkt_participants.group_by_dynamic(index_column='timestamp', every=args.freq.replace('in', '')).agg(
                    [pl.col('underlying_name').last(), pl.col('mkt_participants').mode().get(-1).get(0)]
                )
                mkt_participants = \
                    mkt_participants.with_columns(
                        [(pl.col('mkt_participants') == pl.lit(mkt_participant)).cast(pl.UInt8)
                         .alias(f'is_{mkt_participant}') for mkt_participant in ['retail', 'informed', 'institutional']
                ]).drop('mkt_participants')
                agg_option_chain_data = \
                    agg_option_chain_data.join(mkt_participants, on=['underlying_name', 'timestamp'],
                                               how='left').join(spread, on=['underlying_name', 'timestamp'])
                agg_option_chain_data = \
                    agg_option_chain_data.unpivot(index=['underlying_name', 'timestamp'], variable_name='features',
                                                  value_name='features_values')
                agg_option_chain_data = agg_option_chain_data.with_columns(
                    pl.when(pl.col('features').str.split('_').list.last().is_in(list(LAGS)))
                    .then(pl.col('features').str.split('_').list.last())
                    .otherwise(pl.lit('contemporaneous'))
                    .alias('lag')
                )
                pdb.set_trace()
                agg_option_chain_data.sink_csv(f'{args.target_path}/tmp_{symbol}')
                agg_option_chain_data = pd.read_csv(f'{args.target_path}/tmp_{symbol}', chunksize=100_000)
                os.remove(f'{args.target_path}/tmp_{symbol}')
                try:
                    while True:
                        chunk = next(agg_option_chain_data)
                        chunk.to_sql(name='option_features', con=db_connect, if_exists='append', index=False)
                except StopIteration:
                    break
                print(f'[Insertion]: Option features for {symbol} have been inserted into the database.')
                pdb.set_trace()
                # agg_option_chain_data = pd.read_parquet(f'{args.target_path}/tmp_{symbol}',)
                # for batch in agg_option_chain_data_stream.iter_slices(100_000):
                #     pdb.set_trace()
                #     batch.to_pandas().to_sql(name='option_features', con=db_connect, if_exists='append', index=False)
                # agg_option_chain_data.lazy().collect(streaming=True).to_pandas().to_sql(name='option_features',
                #                                                                         con=db_connect, if_exists='append',
                #                                                                         index=False)

