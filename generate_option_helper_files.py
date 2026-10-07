# # import argparse
# # import pdb
# # from joblib import Parallel, delayed
# # from tqdm.auto import tqdm
# # from tqdm_joblib import tqdm_joblib
# # import os
# # from constants import DATE_FORMAT, SYMBOLS
# # import polars as pl
# # from datetime import datetime
# # import pandas as pd
# # import itertools
# # from sklearn.linear_model import LinearRegression
# # import numpy as np
# # from scipy.stats import norm
# #
# #
# # def bs_price(S, K, T, sigma, option_type):
# #     if T <= 0 or sigma <= 0:
# #         return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
# #
# #     d1 = (np.log(S / K) + ( 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
# #     d2 = d1 - sigma * np.sqrt(T)
# #
# #     if option_type == 'call':
# #         return S * norm.cdf(d1) - K * norm.cdf(d2)
# #     else:
# #         return K * norm.cdf(-d2) - S * norm.cdf(-d1)
# #
# #
# # def daily_option_return_file(date: datetime, underlying_name: str, target_path: str) -> None:
# #     if os.path.exists(f'{target_path}/{date.year}/options/{date.strftime(DATE_FORMAT)}'):
# #         option_chain = (pl.scan_parquet(f'{target_path}/{date.year}/options/{date.strftime(DATE_FORMAT)}'))
# #         option_chain = option_chain.filter(pl.col('underlying_name').eq(underlying_name))
# #         if option_chain.limit(1).collect().height == 0:
# #             return None
# #         option_chain = option_chain.with_columns(
# #                         ((pl.col('expiration')-pl.col('timestamp')).dt.total_days() / 360).alias('tau'))
# #         for option_type in ['call', 'put']:
# #             lr = LinearRegression()
# #             option_chain_type = option_chain.filter(pl.col('type') == option_type)
# #             option_chain_type = option_chain_type.with_columns(
# #                 [pl.col('moneyness').pow(2).alias('moneyness2'), pl.col('tau').pow(2).alias('tau2'),
# #                  (pl.col('tau').mul(pl.col('moneyness'))).alias('moneyness_tau')])
# #             tmp = option_chain_type.select(['mark_iv', 'moneyness', 'tau', 'moneyness2', 'tau2',
# #                                             'moneyness_tau']).drop_nulls()
# #             tmp = tmp.collect().to_numpy()
# #             mark_iv = tmp[:, 0].reshape(-1, 1)
# #             lr.fit(tmp[:, 1:], mark_iv)
# #             iv_coefficients = lr.intercept_.tolist()+lr.coef_.tolist()[0]
# #             option_chain_type = option_chain_type.with_columns(
# #                 ((pl.col('moneyness').mul(iv_coefficients[1])+pl.col('tau').mul(iv_coefficients[2])+
# #                  pl.col('moneyness2').mul(iv_coefficients[3])+pl.col('tau2').mul(iv_coefficients[4])+
# #                  pl.col('moneyness_tau').mul(iv_coefficients[5])+(iv_coefficients[0])).clip(lower_bound=.01))
# #                 .alias('iv_hat')).drop_nulls(subset=['moneyness', 'tau', 'moneyness2', 'tau2', 'moneyness_tau'])
# #             option_chain_type = option_chain_type.with_columns(
# #                 pl.when(pl.col('mid_price').is_null())
# #                 .then(pl.struct(['underlying_price', 'strike_price', 'tau', 'iv_hat', 'type']).map_elements(
# #                         lambda x: bs_price(x['underlying_price'], x['strike_price'], x['tau'], x['iv_hat'],
# #                                            x['type']), return_dtype=pl.Float64)).otherwise(pl.col('mid_price')))
# #             tau_min = pd.to_timedelta(15, unit='min')/pd.to_timedelta(360, unit='d')
# #             option_chain_type = (
# #                 option_chain_type.select(['symbol', 'timestamp', 'mid_price', 'tau', 'moneyness', 'type',
# #                                           'underlying_price', 'spread']))
# #             option_chain_type = option_chain_type.filter(pl.col('mid_price').is_not_null() & (pl.col('tau') >= tau_min))
# #             option_chain_type = option_chain_type.sort(['symbol', 'timestamp'])
# #             option_chain_type = option_chain_type.with_columns(pl.col('mid_price').log()
# #                                                                .over('symbol').diff().alias('returns'))
# #             option_chain_type = (
# #                 option_chain_type.sort('timestamp').group_by_dynamic(index_column='timestamp', every='15m',
# #                                                                      group_by='symbol'))
# #             option_chain_type = (
# #                 option_chain_type.agg([pl.col('mid_price').last(), pl.col('returns').sum(), pl.col('tau').last(),
# #                                        pl.col('moneyness').last(), pl.col('type').last(),
# #                                        pl.col('underlying_price').last(), pl.col('spread').last()]))
# #             os.makedirs(name=f'{args.target_path}/crypto_option_spot/option_helper_files/{date.year}/{underlying_name}',
# #                         exist_ok=True)
# #             file_name = \
# #                 f'{args.target_path}/crypto_option_spot/option_helper_files/{date.year}/{underlying_name}' \
# #                 f'/{date.strftime(DATE_FORMAT)}'
# #             file_exists = os.path.isfile(file_name)
# #             option_chain_type.collect().to_pandas().to_csv(file_name, header=not file_exists, mode='a')
# #
# #
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description='Script generating all performance tables.')
# #     parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-01')
# #     parser.add_argument('--end_date', type=str, help='End date.', default='2025-07-01')
# #     parser.add_argument('--target_path', type=str, help='Directory where database is located',
# #                         default='./data_centre')
# #     parser.add_argument('--cpu_count', type=int, default=3, help='Number of processes to use for the pipeline.')
# #     args = parser.parse_args()
# #     print(args)
# #     dates = pd.date_range(args.start_date, args.end_date, freq='1D',  inclusive='left').date.tolist()
# #     for underlying_name in tqdm(SYMBOLS):
# #         underlying_name_dates = list(itertools.product([underlying_name], dates))
# #         with tqdm_joblib(tqdm(desc='Option helper files', total=len(underlying_name_dates))):
# #             results = Parallel(n_jobs=args.cpu_count, backend='loky')(
# #                 delayed(daily_option_return_file)(underlying_name_date[1], underlying_name_date[0], args.target_path)
# #                 for underlying_name_date in underlying_name_dates)
# import argparse
# import itertools
# import os
# from datetime import datetime
# import numpy as np
# import pandas as pd
# import polars as pl
# from joblib import Parallel, delayed
# from scipy.stats import norm
# from tqdm.auto import tqdm
# from tqdm_joblib import tqdm_joblib
# from constants import DATE_FORMAT, SYMBOLS
#
# # True: a leg without a usable quote is priced with Black-Scholes from the implied-volatility surface fitted at the
# # same timestamp (Dumas et al. regression). False: only quoted prices are kept.
# FILL_MISSING_WITH_BLACK_SCHOLES = True
# # A quote is not usable when its bid-ask spread exceeds this multiple of its mid price ...
# MAX_RELATIVE_SPREAD = 1.0
# # ... or when its mid price violates the no-arbitrage bound (call above the underlying, put above the strike).
# ENFORCE_PRICE_BOUNDS = True
# # Minimum number of implied volatilities needed to fit the surface of one timestamp; otherwise the surface fitted
# # on the whole day (same option type) is used.
# MIN_SURFACE_POINTS = 12
# INPUT_COLUMNS = ['symbol', 'timestamp', 'type', 'strike_price', 'expiration', 'mark_iv', 'underlying_name',
#                  'underlying_price', 'moneyness', 'mid_price', 'spread']
# OUTPUT_COLUMNS = ['symbol', 'timestamp', 'mid_price', 'returns', 'tau', 'moneyness', 'type', 'underlying_price',
#                   'spread', 'model_price']
#
#
# def bs_price(S, K, T, sigma, is_call):
#     """Black-Scholes price with a zero rate, for arrays."""
#     S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
#     intrinsic = np.where(is_call, np.maximum(S - K, 0), np.maximum(K - S, 0))
#     ok = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         vol = sigma * np.sqrt(T)
#         d1 = (np.log(S / K) + .5 * sigma ** 2 * T) / vol
#         d2 = d1 - vol
#         call = S * norm.cdf(d1) - K * norm.cdf(d2)
#         put = K * norm.cdf(-d2) - S * norm.cdf(-d1)
#     return np.where(ok, np.where(is_call, call, put), intrinsic)
#
#
# def surface_design(moneyness, tau) -> np.ndarray:
#     return np.column_stack([np.ones(len(moneyness)), moneyness, tau, moneyness ** 2, tau ** 2, moneyness * tau])
#
#
# def fitted_iv(chain: pd.DataFrame) -> pd.Series:
#     """Implied volatility of every row from the surface fitted on the quotes of its own timestamp."""
#     iv_hat = pd.Series(np.nan, index=chain.index)
#     usable = chain['mark_iv'].notna() & chain['moneyness'].notna() & chain['tau'].notna() & (chain['tau'] > 0)
#     for option_type, type_rows in chain.groupby('type'):
#         fit_rows = type_rows[usable.loc[type_rows.index]]
#         if len(fit_rows) < 6:
#             continue                                      # no surface for this option type today: nothing is filled
#         day_coef = np.linalg.lstsq(surface_design(fit_rows['moneyness'].to_numpy(), fit_rows['tau'].to_numpy()),
#                                    fit_rows['mark_iv'].to_numpy(), rcond=None)[0]
#         for _, rows in type_rows.groupby('timestamp'):
#             fit = rows[usable.loc[rows.index]]
#             coef = day_coef
#             if len(fit) >= MIN_SURFACE_POINTS:
#                 coef = np.linalg.lstsq(surface_design(fit['moneyness'].to_numpy(), fit['tau'].to_numpy()),
#                                        fit['mark_iv'].to_numpy(), rcond=None)[0]
#             values = surface_design(rows['moneyness'].to_numpy(dtype=float), rows['tau'].to_numpy(dtype=float)) @ coef
#             iv_hat.loc[rows.index] = np.clip(values, .01, None)
#     return iv_hat
#
#
# def daily_option_return_file(date: datetime, underlying_name: str, target_path: str) -> None:
#     # the raw file of the day: one folder per pair, or one folder for all pairs
#     candidates = [f'{target_path}/{underlying_name}/{date.year}/options/{date.strftime(DATE_FORMAT)}',
#                   f'{target_path}/{date.year}/options/{date.strftime(DATE_FORMAT)}']
#     source = next((path for path in candidates if os.path.exists(path)), None)
#     if source is None:
#         return None
#     # only the columns that are needed are loaded
#     chain = pl.scan_parquet(source).filter(pl.col('underlying_name').eq(underlying_name)).select(INPUT_COLUMNS) \
#         .collect().to_pandas()
#     if chain.empty:
#         return None
#     # time to expiry in years (360-day year), with its intraday part
#     chain['tau'] = (chain['expiration'] - chain['timestamp']).dt.total_seconds() / (360 * 86400)
#     is_call = (chain['type'] == 'call').to_numpy()
#     # quotes that cannot be used are treated as missing
#     unusable = chain['mid_price'].isna() | (chain['mid_price'] <= 0)
#     unusable |= chain['spread'] > MAX_RELATIVE_SPREAD * chain['mid_price']
#     if ENFORCE_PRICE_BOUNDS:
#         bound = np.where(is_call, chain['underlying_price'], chain['strike_price'])
#         unusable |= chain['mid_price'] > bound
#     chain.loc[unusable, ['mid_price', 'spread']] = np.nan
#     chain['model_price'] = False
#     if FILL_MISSING_WITH_BLACK_SCHOLES:
#         iv_hat = fitted_iv(chain)
#         to_fill = unusable & iv_hat.notna() & (chain['tau'] > 0)
#         chain.loc[to_fill, 'mid_price'] = bs_price(chain.loc[to_fill, 'underlying_price'],
#                                                    chain.loc[to_fill, 'strike_price'], chain.loc[to_fill, 'tau'],
#                                                    iv_hat[to_fill], is_call[to_fill.to_numpy()])
#         chain.loc[to_fill, 'model_price'] = True
#     tau_min = pd.to_timedelta(15, unit='min') / pd.to_timedelta(360, unit='d')
#     chain = chain[chain['mid_price'].notna() & (chain['mid_price'] > 0) & (chain['tau'] >= tau_min)]
#     if chain.empty:
#         return None
#     chain = chain.sort_values(['symbol', 'timestamp'])
#     # log return of each contract (never across two contracts)
#     chain['returns'] = np.log(chain['mid_price']).groupby(chain['symbol']).diff()
#     chain['bar'] = chain['timestamp'].dt.floor('15min')
#     bars = chain.groupby(['symbol', 'bar'], sort=True).agg(
#         mid_price=('mid_price', 'last'), returns=('returns', lambda x: x.sum(min_count=1)), tau=('tau', 'last'),
#         moneyness=('moneyness', 'last'), type=('type', 'last'), underlying_price=('underlying_price', 'last'),
#         spread=('spread', 'last'), model_price=('model_price', 'last')).reset_index()
#     bars = bars.rename(columns={'bar': 'timestamp'}).sort_values(['type', 'symbol', 'timestamp'])
#     bars = bars[OUTPUT_COLUMNS].reset_index(drop=True)
#     folder = f'{target_path}/crypto_option_spot/option_helper_files/{date.year}/{underlying_name}'
#     os.makedirs(name=folder, exist_ok=True)
#     # the file is written once and replaces any earlier version
#     bars.to_csv(f'{folder}/{date.strftime(DATE_FORMAT)}', header=True, mode='w')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Script generating the daily option files used by the backtest.')
#     parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-01')
#     parser.add_argument('--end_date', type=str, help='End date.', default='2025-07-01')
#     parser.add_argument('--target_path', type=str, help='Directory where database is located',
#                         default='./data_centre')
#     parser.add_argument('--cpu_count', type=int, default=3, help='Number of processes to use for the pipeline.')
#     args = parser.parse_args()
#     print(args)
#     dates = pd.date_range(args.start_date, args.end_date, freq='1D', inclusive='left').date.tolist()
#     for underlying_name in tqdm(SYMBOLS):
#         underlying_name_dates = list(itertools.product([underlying_name], dates))
#         with tqdm_joblib(tqdm(desc='Option helper files', total=len(underlying_name_dates))):
#             results = Parallel(n_jobs=args.cpu_count, backend='loky')(
#                 delayed(daily_option_return_file)(underlying_name_date[1], underlying_name_date[0], args.target_path)
#                 for underlying_name_date in underlying_name_dates)
#         folder = f'{args.target_path}/crypto_option_spot/option_helper_files'
#         n_files = sum(len(files) for path, _, files in os.walk(folder) if os.path.basename(path) == underlying_name)
#         print(f'{underlying_name}: {n_files} daily file(s) present under {folder}.', flush=True)
import argparse
import itertools
import os
from datetime import datetime
import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm.auto import tqdm
from constants import DATE_FORMAT, SYMBOLS

# True: a leg without a usable quote is priced with Black-Scholes from the implied-volatility surface fitted at the
# same timestamp (Dumas et al. regression). False: only quoted prices are kept.
FILL_MISSING_WITH_BLACK_SCHOLES = True
# A quote is not usable when its bid-ask spread exceeds this multiple of its mid price ...
MAX_RELATIVE_SPREAD = 1.0
# ... or when its mid price violates the no-arbitrage bound (call above the underlying, put above the strike).
ENFORCE_PRICE_BOUNDS = True
# Minimum number of implied volatilities needed to fit the surface of one timestamp; otherwise the surface fitted
# on the whole day (same option type) is used.
MIN_SURFACE_POINTS = 12
INPUT_COLUMNS = ['symbol', 'timestamp', 'type', 'strike_price', 'expiration', 'mark_iv', 'underlying_name',
                 'underlying_price', 'moneyness', 'mid_price', 'spread']
OUTPUT_COLUMNS = ['symbol', 'timestamp', 'mid_price', 'returns', 'tau', 'moneyness', 'type', 'underlying_price',
                  'spread', 'model_price']


def bs_price(S, K, T, sigma, is_call):
    """Black-Scholes price with a zero rate, for arrays."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    intrinsic = np.where(is_call, np.maximum(S - K, 0), np.maximum(K - S, 0))
    ok = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        vol = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + .5 * sigma ** 2 * T) / vol
        d2 = d1 - vol
        call = S * norm.cdf(d1) - K * norm.cdf(d2)
        put = K * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(ok, np.where(is_call, call, put), intrinsic)


def surface_design(moneyness, tau) -> np.ndarray:
    return np.column_stack([np.ones(len(moneyness)), moneyness, tau, moneyness ** 2, tau ** 2, moneyness * tau])


def fitted_iv(chain: pd.DataFrame) -> pd.Series:
    """Implied volatility of every row from the surface fitted on the quotes of its own timestamp."""
    iv_hat = pd.Series(np.nan, index=chain.index)
    usable = chain['mark_iv'].notna() & chain['moneyness'].notna() & chain['tau'].notna() & (chain['tau'] > 0)
    for option_type, type_rows in chain.groupby('type'):
        fit_rows = type_rows[usable.loc[type_rows.index]]
        if len(fit_rows) < 6:
            continue                                      # no surface for this option type today: nothing is filled
        day_coef = np.linalg.lstsq(surface_design(fit_rows['moneyness'].to_numpy(), fit_rows['tau'].to_numpy()),
                                   fit_rows['mark_iv'].to_numpy(), rcond=None)[0]
        for _, rows in type_rows.groupby('timestamp'):
            fit = rows[usable.loc[rows.index]]
            coef = day_coef
            if len(fit) >= MIN_SURFACE_POINTS:
                coef = np.linalg.lstsq(surface_design(fit['moneyness'].to_numpy(), fit['tau'].to_numpy()),
                                       fit['mark_iv'].to_numpy(), rcond=None)[0]
            values = surface_design(rows['moneyness'].to_numpy(dtype=float), rows['tau'].to_numpy(dtype=float)) @ coef
            iv_hat.loc[rows.index] = np.clip(values, .01, None)
    return iv_hat


def daily_option_return_file(date: datetime, underlying_name: str, target_path: str) -> None:
    # the raw file of the day: one folder per pair, or one folder for all pairs
    candidates = [f'{target_path}/{underlying_name}/{date.year}/options/{date.strftime(DATE_FORMAT)}',
                  f'{target_path}/{date.year}/options/{date.strftime(DATE_FORMAT)}']
    source = next((path for path in candidates if os.path.exists(path)), None)
    if source is None:
        return None
    # only the columns that are needed are loaded
    chain = pl.scan_parquet(source).filter(pl.col('underlying_name').eq(underlying_name)).select(INPUT_COLUMNS) \
        .collect().to_pandas()
    if chain.empty:
        return None
    # time to expiry in years (360-day year), with its intraday part
    chain['tau'] = (chain['expiration'] - chain['timestamp']).dt.total_seconds() / (360 * 86400)
    is_call = (chain['type'] == 'call').to_numpy()
    # quotes that cannot be used are treated as missing
    unusable = chain['mid_price'].isna() | (chain['mid_price'] <= 0)
    unusable |= chain['spread'] > MAX_RELATIVE_SPREAD * chain['mid_price']
    if ENFORCE_PRICE_BOUNDS:
        bound = np.where(is_call, chain['underlying_price'], chain['strike_price'])
        unusable |= chain['mid_price'] > bound
    chain.loc[unusable, ['mid_price', 'spread']] = np.nan
    chain['model_price'] = False
    if FILL_MISSING_WITH_BLACK_SCHOLES:
        iv_hat = fitted_iv(chain)
        to_fill = unusable & iv_hat.notna() & (chain['tau'] > 0)
        chain.loc[to_fill, 'mid_price'] = bs_price(chain.loc[to_fill, 'underlying_price'],
                                                   chain.loc[to_fill, 'strike_price'], chain.loc[to_fill, 'tau'],
                                                   iv_hat[to_fill], is_call[to_fill.to_numpy()])
        chain.loc[to_fill, 'model_price'] = True
    tau_min = pd.to_timedelta(15, unit='min') / pd.to_timedelta(360, unit='d')
    chain = chain[chain['mid_price'].notna() & (chain['mid_price'] > 0) & (chain['tau'] >= tau_min)]
    if chain.empty:
        return None
    chain = chain.sort_values(['symbol', 'timestamp'])
    # log return of each contract (never across two contracts)
    chain['returns'] = np.log(chain['mid_price']).groupby(chain['symbol']).diff()
    chain['bar'] = chain['timestamp'].dt.floor('15min')
    bars = chain.groupby(['symbol', 'bar'], sort=True).agg(
        mid_price=('mid_price', 'last'), returns=('returns', lambda x: x.sum(min_count=1)), tau=('tau', 'last'),
        moneyness=('moneyness', 'last'), type=('type', 'last'), underlying_price=('underlying_price', 'last'),
        spread=('spread', 'last'), model_price=('model_price', 'last')).reset_index()
    bars = bars.rename(columns={'bar': 'timestamp'}).sort_values(['type', 'symbol', 'timestamp'])
    bars = bars[OUTPUT_COLUMNS].reset_index(drop=True)
    folder = f'{target_path}/crypto_option_spot/option_helper_files/{date.year}/{underlying_name}'
    os.makedirs(name=folder, exist_ok=True)
    # the file is written once and replaces any earlier version
    bars.to_csv(f'{folder}/{date.strftime(DATE_FORMAT)}', header=True, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating the daily option files used by the backtest.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2025-07-01')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='./data_centre')
    parser.add_argument('--cpu_count', type=int, default=3, help='Number of processes to use for the pipeline.')
    args = parser.parse_args()
    print(args)
    dates = pd.date_range(args.start_date, args.end_date, freq='1D', inclusive='left').date.tolist()
    for underlying_name in tqdm(SYMBOLS):
        underlying_name_dates = list(itertools.product([underlying_name], dates))
        jobs = (delayed(daily_option_return_file)(underlying_name_date[1], underlying_name_date[0], args.target_path)
                for underlying_name_date in underlying_name_dates)
        try:
            # joblib >= 1.3: results are yielded as they finish, which drives the progress bar
            finished = Parallel(n_jobs=args.cpu_count, backend='loky', return_as='generator_unordered')(jobs)
            results = list(tqdm(finished, desc=f'Option helper files ({underlying_name})',
                                total=len(underlying_name_dates)))
        except (TypeError, ValueError):
            # older joblib: same work, without a per-day progress bar
            results = Parallel(n_jobs=args.cpu_count, backend='loky')(
                delayed(daily_option_return_file)(underlying_name_date[1], underlying_name_date[0],
                                                  args.target_path)
                for underlying_name_date in underlying_name_dates)
        folder = f'{args.target_path}/crypto_option_spot/option_helper_files'
        n_files = sum(len(files) for path, _, files in os.walk(folder) if os.path.basename(path) == underlying_name)
        print(f'{underlying_name}: {n_files} daily file(s) present under {folder}.', flush=True)