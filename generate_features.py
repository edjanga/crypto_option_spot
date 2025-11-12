import argparse
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
        options = pl.scan_parquet(f'{args.target_path}/*/options/*-*-*')
        for symbol in SYMBOLS:
            print(f'[Insertion]: Process to generate option features for {symbol} has started.........................')
            options = options.filter(pl.col('underlying_name') == symbol).drop_nulls()
            options = \
                options.with_columns(((pl.col('expiration') - pl.col('timestamp')) / pl.duration(days=360)).alias('tau'))
            options = options.with_columns(pl.when(pl.col('last_price') > pl.col('mid_price'))
                                           .then(1).when(pl.col('last_price') < pl.col('mid_price'))
                                           .then(-1).otherwise(None).alias('trades'))
            is_put = pl.col('type') == "put"
            is_call = ~is_put
            options = \
                options.with_columns(pl.when(is_put).then(pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS))
                                     .when(is_call).then(pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS[::-1]))
                                     .otherwise(None).alias('moneyness_label'))
            options = options.filter(pl.col('trades').is_not_null())
            options = options.with_columns(-(pl.col('volume') * pl.col('trades')).alias('netDealerExposure'))
            exposure = feature_builder.compute_exposure(data=options, freq=args.freq, collect=collect)
            daily_option_chain_data = options.drop(['trades', 'netDealerExposure'])
            pcr = feature_builder.put_call_ratio(data=options, freq=args.freq, collect=collect)
            pcr = feature_builder.lags(pcr, ['pcr'])
            exposure = feature_builder.lags(exposure, ['gex', 'vax', 'vex', 'vox'])
            ####################################################################################################
            ## First aggregation in USDT and level terms
            ####################################################################################################
            agg_option_chain_data = exposure.join(pcr, on=['underlying_name', 'timestamp'])
            ####################################################################################################
            ## Aggregation per contract type: call or put
            ####################################################################################################
            spread = feature_builder.spread(data=options, freq=args.freq, collect=collect)
            spread = feature_builder.lags(spread, ['spread_call', 'spread_put'])
            skew = feature_builder.volatility_skew(data=options, freq=args.freq, collect=collect)
            skew_columns = list(product(['call', 'put'], ['OTM', 'ATM', 'ITM', 'DITM']))
            skew_columns = \
                [f'{option_type_moneyness[0]}_{option_type_moneyness[1]}' for option_type_moneyness in skew_columns]
            skew = feature_builder.lags(skew, skew_columns)
            mkt_participants = feature_builder.label_market_participants(freq=args.freq, data=daily_option_chain_data)
            # agg_option_chain_data = agg_option_chain_data.join(skew, on=['underlying_name', 'timestamp']).join(
            #     mkt_participants, on=['underlying_name', 'timestamp']).join(spread, on=['underlying_name', 'timestamp'])
            agg_option_chain_data = agg_option_chain_data.melt(id_vars=['underlying_name', 'timestamp'],
                                                               variable_name='features',
                                                               value_name='features_values')

            agg_option_chain_data = agg_option_chain_data.with_columns(
                pl.when(pl.col('features').str.split('_').list.last().is_in(list(LAGS)))
                .then(pl.col('features').str.split('_').list.last())
                .otherwise(pl.lit('contemporaneous'))
                .alias('lag')
            )
            agg_option_chain_data_stream = agg_option_chain_data.lazy().collect(streaming=True)

            for batch in agg_option_chain_data_stream.iter_slices(100_000):
                batch.to_pandas().to_sql(name='option_features', con=db_connect, if_exists='append', index=False)
            # agg_option_chain_data.lazy().collect(streaming=True).to_pandas().to_sql(name='option_features',
            #                                                                         con=db_connect, if_exists='append',
            #                                                                         index=False)
            print(f'[Insertion]: Option features for {symbol} have been inserted into the database.')
