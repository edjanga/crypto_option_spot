import argparse
import pdb
import sqlite3
import pytz
from constants import *
from data_centre.data import Reader
import pandas as pd
import glob
import concurrent.futures
from model.lab import FeatureBuilder


def return_filename(filename: str):
    return filename

# RESAMPLING_MAT_MONEYNESS = ['gex', 'vax', 'vex', 'vox', 'volume', 'gex_exp', 'vax_exp', 'vex_exp', 'vox_exp']
# RESAMPLING_MAT_MONEYNESS = {feature: 'sum' for feature in RESAMPLING_MAT_MONEYNESS}
# RESAMPLING_MAT_MONEYNESS.update({feature: 'last' for feature in ['tau', 'moneyness', 'mark_iv']})
# resampling = RESAMPLING.copy()
# for key in ['ask_iv', 'bid_iv', 'rho', 'theta', 'mark_price']:
#     resampling.pop(key)


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
                daily_option_chain_data = pd.read_parquet(filename)
                symbol_daily = daily_option_chain_data.underlying_name.unique()
                timestamp_daily = pd.date_range(start=filename.split('/')[-1], freq=args.freq,
                                                periods=pd.to_timedelta('1D')//pd.to_timedelta(args.freq), tz=pytz.UTC)
                multi_idx = pd.MultiIndex.from_product([symbol_daily, timestamp_daily])
                daily_option_chain_data['tau'] = \
                    (daily_option_chain_data.expiration - daily_option_chain_data.timestamp)
                daily_option_chain_data['tau'] /= pd.to_timedelta('360D')
                daily_option_chain_data['trades'] = np.select(
                    [
                        daily_option_chain_data['last_price'].gt(daily_option_chain_data['mid_price']),
                        daily_option_chain_data['last_price'].lt(daily_option_chain_data['mid_price']),
                    ], [1, -1], default=np.nan
                )
                is_put = daily_option_chain_data['type'].eq('put')
                is_call = ~is_put
                daily_option_chain_data['moneyness_label'] = np.nan
                daily_option_chain_data.loc[is_put, 'moneyness_label'] = pd.cut(
                    daily_option_chain_data.loc[is_put, 'moneyness'], bins=MONEYNESS_BINS, labels=LABELS)
                daily_option_chain_data.loc[is_call, 'moneyness_label'] = pd.cut(
                    daily_option_chain_data.loc[is_call, 'moneyness'], bins=MONEYNESS_BINS, labels=LABELS[::-1])
                daily_option_chain_data = \
                    daily_option_chain_data.loc[~daily_option_chain_data['trades'].isnull(), :]
                daily_option_chain_data['volume'] = np.where(
                    daily_option_chain_data['trades'].eq(1), daily_option_chain_data['ask_amount'],
                    daily_option_chain_data['bid_amount']) * daily_option_chain_data['mid_price']
                daily_option_chain_data = daily_option_chain_data.set_index(['symbol', 'timestamp'])
                daily_option_chain_data = feature_builder.define_market_participants(daily_option_chain_data)
                daily_option_chain_data['netDealerExposure'] = \
                    -1 * (daily_option_chain_data['volume'].mul(daily_option_chain_data['trades']))
                netExposure = feature_builder.compute_exposure(data=daily_option_chain_data)
                daily_option_chain_data = daily_option_chain_data.drop(['trades'], axis=1)
                netExposure = \
                    netExposure.groupby(by=pd.Grouper(level='timestamp'))[EXPOSURE].sum().reindex(timestamp_daily)
                netExposure = netExposure.fillna(netExposure.expanding().mean())
                netExposure = pd.concat([netExposure] * len(symbol_daily))
                netExposure.index = multi_idx
                put_call_ratio = feature_builder.put_call_ratio(data=daily_option_chain_data[['type', 'volume',
                                                                                              'underlying_name']],
                                                                freq=args.freq)
                put_call_ratio = \
                    put_call_ratio.groupby(by=pd.Grouper(level='timestamp')).mean().reindex(timestamp_daily)
                put_call_ratio = put_call_ratio.fillna(put_call_ratio.expanding().mean())
                put_call_ratio = pd.concat([put_call_ratio] * len(symbol_daily))
                put_call_ratio.index = multi_idx
                volatility_skew = \
                    feature_builder.volatility_skew(
                        data=daily_option_chain_data[['type', 'volume', 'mark_iv', 'moneyness_label',
                                                      'underlying_name']], freq=args.freq)
                volatility_skew = volatility_skew.stack().groupby(
                    by=[pd.Grouper(level='timestamp', sort=True), pd.Grouper(level='type')]).mean().unstack(1)
                volatility_skew = volatility_skew.reindex(timestamp_daily)
                volatility_skew = volatility_skew.fillna(volatility_skew.expanding().mean())
                volatility_skew = pd.concat([volatility_skew]*netExposure.index.get_level_values(0).unique().shape[0])
                volatility_skew.index = multi_idx
                mkt_participants = \
                    feature_builder.label_market_participants(freq=args.freq,
                        data=daily_option_chain_data[['type', 'market_participants', 'underlying_name']])
                mkt_participants = \
                    mkt_participants.groupby(by=pd.Grouper(level='timestamp')).apply(lambda x: x.mode().iloc[0])
                mkt_participants = mkt_participants.reindex(timestamp_daily)
                mkt_participants = mkt_participants.ffill()
                mkt_participants = pd.concat([mkt_participants] * len(symbol_daily))
                mkt_participants.index = multi_idx
                overall_option_dynamics.append(
                    pd.concat([netExposure, put_call_ratio, volatility_skew, mkt_participants], axis=1))
                overall_option_dynamics = pd.concat(overall_option_dynamics)
                overall_option_dynamics.index.names = ['symbol', 'timestamp']
                overall_option_dynamics = \
                    pd.melt(frame=overall_option_dynamics, value_name='values', var_name='features', ignore_index=False)
                overall_option_dynamics.to_sql(name='features', if_exists='append', con=db_connect)
                print(f'[Insertion]: '
                      f'Features {filename.split("/")[-1]} have been inserted into the database.')
