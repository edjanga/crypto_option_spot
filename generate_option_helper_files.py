import argparse
import pdb

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import os
from constants import DATE_FORMAT, SYMBOLS
import polars as pl
from datetime import datetime
import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)

    d1 = (np.log(S / K) + ( 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - S * norm.cdf(-d1)

def daily_option_return_file(date: datetime, underlying_name: str, target_path: str) -> None:
    if os.path.exists(f'{target_path}/{date.year}/options/{date.strftime(DATE_FORMAT)}'):
        option_chain = (pl.scan_parquet(f'{target_path}/{date.year}/options/{date.strftime(DATE_FORMAT)}'))
        option_chain = option_chain.filter(pl.col('underlying_name').eq(underlying_name))
        if option_chain.limit(1).collect().height == 0:
            return None
        option_chain = option_chain.with_columns(
                        ((pl.col('expiration')-pl.col('timestamp')).dt.total_days() / 360).alias('tau'))
        for option_type in ['call', 'put']:
            lr = LinearRegression()
            option_chain_type = option_chain.filter(pl.col('type') == option_type)
            option_chain_type = option_chain_type.with_columns(
                [pl.col('moneyness').pow(2).alias('moneyness2'), pl.col('tau').pow(2).alias('tau2'),
                 (pl.col('tau').mul(pl.col('moneyness'))).alias('moneyness_tau')])
            tmp = option_chain_type.select(['mark_iv', 'moneyness', 'tau', 'moneyness2', 'tau2',
                                            'moneyness_tau']).drop_nulls()
            tmp = tmp.collect().to_numpy()
            mark_iv = tmp[:, 0].reshape(-1, 1)
            lr.fit(tmp[:, 1:], mark_iv)
            iv_coefficients = lr.intercept_.tolist()+lr.coef_.tolist()[0]
            option_chain_type = option_chain_type.with_columns(
                ((pl.col('moneyness').mul(iv_coefficients[1])+pl.col('tau').mul(iv_coefficients[2])+
                 pl.col('moneyness2').mul(iv_coefficients[3])+pl.col('tau2').mul(iv_coefficients[4])+
                 pl.col('moneyness_tau').mul(iv_coefficients[5])+(iv_coefficients[0])).clip(lower_bound=.01))
                .alias('iv_hat')).drop_nulls(subset=['moneyness', 'tau', 'moneyness2', 'tau2', 'moneyness_tau'])
            option_chain_type = option_chain_type.with_columns(
                pl.when(pl.col('mid_price').is_null())
                .then(pl.struct(['underlying_price', 'strike_price', 'tau', 'iv_hat', 'type']).map_elements(
                        lambda x: bs_price(x['underlying_price'], x['strike_price'], x['tau'], x['iv_hat'],
                                           x['type']), return_dtype=pl.Float64)).otherwise(pl.col('mid_price')))
            tau_min = pd.to_timedelta(15, unit='min')/pd.to_timedelta(360, unit='d')
            option_chain_type = (
                option_chain_type.select(['symbol', 'timestamp', 'mid_price', 'tau', 'moneyness', 'type']))
            option_chain_type = option_chain_type.filter(pl.col('mid_price').is_not_null() & (pl.col('tau') >= tau_min))
            option_chain_type = option_chain_type.sort(['symbol', 'timestamp'])
            option_chain_type = option_chain_type.with_columns(pl.col('mid_price').log()
                                                               .over('symbol').diff().alias('returns'))
            option_chain_type = (
                option_chain_type.sort('timestamp').group_by_dynamic(index_column='timestamp', every='15m',
                                                                     group_by='symbol'))
            option_chain_type = (
                option_chain_type.agg([pl.col('mid_price').last(), pl.col('returns').sum(), pl.col('tau').last(),
                                       pl.col('moneyness').last(), pl.col('type').last()]))
            os.makedirs(name=f'{args.target_path}/option_helper_files/{date.year}/{underlying_name}',
                        exist_ok=True)
            file_name = \
                f'{args.target_path}/option_helper_files/{date.year}/{underlying_name}/{date.strftime(DATE_FORMAT)}'
            file_exists = os.path.isfile(file_name)
            option_chain_type.collect().to_pandas().to_csv(file_name, header=not file_exists, mode='a')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all performance tables.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2025-07-01')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='./data_centre')
    parser.add_argument('--cpu_count', type=int, default=3,
                        help='Number of processes to use for the pipeline.')
    args = parser.parse_args()
    print(args)
    dates = pd.date_range(args.start_date, args.end_date, freq='1D',  inclusive='left').date.tolist()
    for underlying_name in tqdm(SYMBOLS):
        underlying_name_dates = list(itertools.product([underlying_name], dates))
        with tqdm_joblib(tqdm(desc='Option helper files', total=len(underlying_name_dates))):
            results = Parallel(n_jobs=-args.cpu_count, backend='loky')(
                delayed(daily_option_return_file)(underlying_name_date[1], underlying_name_date[0], args.target_path)
                for underlying_name_date in underlying_name_dates)
