import argparse
import sqlite3
import pytz
from constants import *
import polars as pl
import pandas as pd
import glob
import concurrent.futures
from model.lab import FeatureBuilder


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
    parser.add_argument('--D', type=str, help='Number of days ahead to forecast values for', default='1D')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    args = parser.parse_args()
    print(f'[Generation]: Features {args} generation process has started..............................................')
    db_connect = sqlite3.connect(database=f'{args.target_path}/database.db', check_same_thread=False)
    feature_builder = FeatureBuilder()
    ####################################################################################################################
    ## Options Market based features
    ####################################################################################################################
    for year in range(int(args.start_date[:4]), int(args.end_date[:4])+1):
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            futures = \
                [executor.submit(return_filename, filename) for filename in glob.glob(f'../data_centre/{year}/*')
                 if (pd.to_datetime(filename.split('/')[-1]).date() >= pd.to_datetime(args.start_date).date()) &
                 (pd.to_datetime(filename.split('/')[-1]).date() < pd.to_datetime(args.end_date).date())]
            for future in concurrent.futures.as_completed(futures):
                overall_option_dynamics = list()
                filename = future.result()
                daily_option_chain_data = pl.scan_parquet(filename)
                timestamp_daily = pd.date_range(start=filename.split('/')[-1], freq=args.freq,
                                                periods=pd.to_timedelta('1D')//pd.to_timedelta(args.freq), tz=pytz.UTC)
                daily_option_chain_data = daily_option_chain_data.with_columns(
                    ((pl.col('expiration') - pl.col('timestamp')) / pl.duration(days=360))
                    .alias('tau')
                )
                daily_option_chain_data = daily_option_chain_data.with_columns(
                    pl.when(pl.col('last_price') > pl.col('mid_price')).then(1)
                    .when(pl.col('last_price') < pl.col('mid_price')).then(-1).otherwise(None).alias('trades')
                )
                is_put = pl.col('type') == "put"
                is_call = ~is_put
                daily_option_chain_data = \
                    daily_option_chain_data.with_columns(pl.when(is_put).then(pl.col('moneyness').cut(MONEYNESS_BINS,
                                                                                                      labels=LABELS))
                                            .when(is_call)
                                            .then(pl.col('moneyness').cut(MONEYNESS_BINS, labels=LABELS[::-1]))
                                            .otherwise(None)
                                            .alias('moneyness_label')
                )
                daily_option_chain_data = daily_option_chain_data.filter(pl.col('trades').is_not_null())
                daily_option_chain_data = daily_option_chain_data.with_columns((pl.when(pl.col('trades') == 1)
                                                                                .then(pl.col('ask_amount'))
                                                                                .otherwise(pl.col('bid_amount'))
                                                                                * pl.col('mid_price')).alias('volume'))
                daily_option_chain_data = \
                    daily_option_chain_data.with_columns(
                        -(pl.col('volume') * pl.col('trades')).alias('netDealerExposure'))
                exposure = feature_builder.compute_exposure(data=daily_option_chain_data, freq=args.freq)
                daily_option_chain_data = daily_option_chain_data.drop(['trades', 'netDealerExposure'])
                pcr = feature_builder.put_call_ratio(data=daily_option_chain_data, freq=args.freq)
                ########################################################################################################
                ## First aggregation in USDT and level terms
                ########################################################################################################
                agg_daily_option_chain_data = exposure.join(pcr, on=['underlying_name', 'timestamp'])
                ########################################################################################################
                ## Aggregation per contract type: call or put
                ########################################################################################################
                spread = feature_builder.spread(data=daily_option_chain_data, freq=args.freq)
                skew = feature_builder.volatility_skew(data=daily_option_chain_data, freq=args.freq)
                mkt_participants = \
                    feature_builder.label_market_participants(freq=args.freq, data=daily_option_chain_data)
                agg_daily_option_chain_data = \
                    agg_daily_option_chain_data.join(skew, on=['underlying_name', 'timestamp']).join(
                        mkt_participants, on=['underlying_name', 'timestamp']).join(spread,
                                                                                    on=['underlying_name', 'timestamp'])
                agg_daily_option_chain_data = \
                    agg_daily_option_chain_data.to_pandas().set_index(['underlying_name', 'timestamp'])
                agg_daily_option_chain_data = \
                    pd.melt(frame=agg_daily_option_chain_data, value_name='values', var_name='features',
                            ignore_index=False)
                agg_daily_option_chain_data.to_sql(name='features', if_exists='append', con=db_connect)
                print(f'[Insertion]: '
                      f'Features {filename.split("/")[-1]} have been inserted into the database.')
