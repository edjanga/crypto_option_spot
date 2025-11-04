import datetime
import itertools
import pdb
import sqlite3
import os
import numpy as np
import pandas as pd
import optuna
import pytz
from optuna.samplers import RandomSampler
from dateutil.relativedelta import relativedelta
from constants import DATE_FORMAT, N_JOBS, N_TRIALS, FEES_SPOT, FEES_OPTIONS, DAYS_CONVERTER, TITLE_FONT_SIZE,\
    LABEL_AXIS_FONT_SIZE, HEIGHT, WIDTH, SYMBOLS, LOW_ATM_MONEYNESS, HIGH_ATM_MONEYNESS, VOL_REGIMES, ATM_MONEYNESSES, \
    SUPERSCRIPT, ENGINE
from functools import partial
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from scipy.stats.distributions import norm
import statsmodels.api as sm
from pandas.api.types import CategoricalDtype
import polars as pl
from itertools import product
import argparse
from applications.apps import FrameworkApplication
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all P&L series based.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2023-07-01')
    parser.add_argument('--target', type=str, help='RV or returns.', default='returns')
    # parser.add_argument('--freq', type=str, help='Frequency used to resample and aggregate data', default='15min')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    args = parser.parse_args()
    print(args)
    framework_application_obj = FrameworkApplication()
    # with open('../applications/mase_per_regression_freq_pair.txt', 'r') as mase_per_regression_freq_pair_file:
    #     mase_per_regression_freq_pair_lines = \
    #         [l.strip() for l in mase_per_regression_freq_pair_file.readlines() if not \
    #             re.match(r'\\(begin|end|toprule|midrule|bottomrule)', l.strip()) and '&' in l]
    #     mase_per_regression_freq_pair_rows = [re.split(r'\s*&\s*', l.replace('\\\\', '').strip()) for l in \
    #                                           mase_per_regression_freq_pair_lines]
    #     mase_per_regression_freq_pair_columns = [c.strip() for c in mase_per_regression_freq_pair_rows[0]]
    #     mase_per_regression_freq_pair_data = [[c.strip() for c in r] for r in mase_per_regression_freq_pair_rows[1:]]
    # mase_per_regression_freq_pair = pd.DataFrame(mase_per_regression_freq_pair_data,
    #                                              columns=mase_per_regression_freq_pair_columns)
    # mase_per_regression_freq_pair.columns = mase_per_regression_freq_pair.columns.str.replace('regression', '')
    # mase_per_regression_freq_pair = mase_per_regression_freq_pair.iloc[1:, :]
    # mase_per_regression_freq_pair = mase_per_regression_freq_pair.set_index('').stack()
    # mase_per_regression_freq_pair.index = mase_per_regression_freq_pair.index.set_names(['freq', 'regression'])
    # freq, regression = mase_per_regression_freq_pair.idxmin()
    # mase_comparison = \
    #     pd.read_sql(con=framework_application_obj.db_connect,
    #                 sql=f'SELECT "timestamp","MASE","regression","target","D","market","freq" '
    #                     f'FROM metric WHERE regression="{regression}" AND freq="{freq}";')
    # mase_comparison['timestamp'] = pd.to_datetime(mase_comparison['timestamp'], utc=True)
    # mase_comparison = mase_comparison.set_index(['D', 'timestamp'])
    # mase_comparison = mase_comparison.groupby(by=pd.Grouper(key='market'))['MASE'].mean()
    # best_market = mase_comparison.idxmin()
    # data = pd.read_sql(con=framework_application_obj.db_connect,
    #                    sql=f'SELECT * FROM forecasts '
    #                        f'WHERE regression = "{regression}" '
    #                        f'AND freq = "{freq}" AND market = {best_market};').set_index('timestamp')
    ####################################################################################################################
    ## Crypto SVIX and VRP
    ####################################################################################################################
    data_pnl = pd.read_sql(con=framework_application_obj.db_connect, sql='SELECT * FROM pnl;')
    data_pnl['timestamp'] = pd.to_datetime(data_pnl['timestamp'], utc=pytz.UTC)
    data = data_pnl.set_index(['aggregated', 'target', 'timestamp'])
    pdb.set_trace()


    data_pnl = framework_application_obj.backtest(vrp=vrp, transaction_cost=True, spread=True,
                                               freq=args.freq, target=args.target)
    if args.target == 'RV':
        pnl, trading_volume = \
            framework_application_obj.backtest(vrp=vrp, transaction_cost=True, spread=True,
                                               freq=args.freq, target=args.target)
        pnl, trading_volume = pnl.swaplevel(0, 1, axis=1), trading_volume.swaplevel(0, 1, axis=1)
        # pnl.columns = pnl.columns.set_levels(pnl.columns.levels[-1].str.upper(), level=1)
        pnl.columns = pnl.columns.set_names(['underlying_name', 'D'])
        pnl.index = pnl.index.set_names(['timestamp'])
        # trading_volume.columns = \
        #     trading_volume.columns.set_levels(trading_volume.columns.levels[-1].str.lower(), level=1)
    else:
        pnl = \
            framework_application_obj.backtest(transaction_cost=True, spread=False, freq=args.freq, target=args.target)