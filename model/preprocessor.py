import pdb
from data_centre.data import Reader
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Tuple, List, Union, Dict, Callable, Any
import glob
import numpy as np
import itertools
from scipy.spatial.distance import euclidean
from scipy.interpolate import splrep, splev, bisplrep, bisplev, griddata, RectBivariateSpline
from scipy.optimize import minimize
import time
import concurrent.futures
from scipy.stats import norm
import torch
import optuna
from optuna.samplers import RandomSampler
from functools import partial
import swifter


class DataPreProcessor(object):

    global _deltas
    global _tau
    _deltas = [.05, .1, .25, .5, .75, .9]
    _tau = [1/720, 1/360, 7/360, 30/360, 180/360, .9]
    global _lattice
    _lattice = list(map(lambda x: tuple(x), list(itertools.product(_deltas, _tau))))
    _lattice = [_lattice[i * len(_tau):(i + 1) * len(_tau)] for i in range(len(_deltas))]
    _reader = Reader()
    _underlying = _reader.rv_read()

    def __init__(self, L: str = None, h: str = None) -> None:
        self._L = L
        self._h = h
        self._input_call = None
        self._input_put = None

    @property
    def input_call(self) -> pd.DataFrame:
        return self._input_call

    @property
    def input_put(self) -> pd.DataFrame:
        return self._input_put

    @property
    def L(self) -> str:
        return self._L

    @property
    def h(self) -> str:
        return self._h

    @input_call.setter
    def input_call(self, data: pd.DataFrame) -> None:
        self._input_call = data
        # if 'type' in self._input_call.columns:
        #     self._input_call = \
        #         self._input_call.query('type == "call"').drop(['gamma', 'vega', 'theta', 'rho', 'volume', 'type',
        #                                                        'expiration'], axis=1)
        #     self._input_call['tau'] = self._input_call['tau'].replace(0, np.nan)
        #     self._input_call = \
        #         self._input_call.assign(total_variance=(self._input_call['mark_iv'] ** 2).mul(self._input_call['tau']))
        #     self._input_call.dropna(inplace=True)
        #     self._input_call.index = list(range(0, self._input_call.shape[0]))

    @input_put.setter
    def input_put(self, data: pd.DataFrame) -> None:
        self._input_put = data
        # self._input_put['tau'] = self._input_put['tau'].replace(0, np.nan)
        # self._input_put = \
        #     self._input_put.assign(total_variance=(self._input_put['mark_iv'] ** 2).mul(self._input_put['tau']))
        # self._input_put.dropna(inplace=True)
        # self._input_put.index = list(range(0, self._input_put.shape[0]))

    @L.setter
    def L(self, L: str):
        self._L = L

    @h.setter
    def h(self, h: str):
        self._h = h

    @staticmethod
    def compute_closest_point(delta: float, tau: float, grid: np.ndarray) -> None:
        # Initialisation
        row, col = grid.shape
        row -= 1
        col -= 1
        old_min = np.inf
        while True:
            if grid[-1, -1] == np.inf:
                grid[-1, -1] = euclidean((delta, tau), _lattice[-1][-1])
            if grid[row-1, col] == np.inf:
                north = euclidean((delta, tau), _lattice[row-1][col])
                grid[row-1, col] = north
            if grid[row, col-1] == np.inf:
                west = euclidean((delta, tau), _lattice[row][col-1])
                grid[row, col-1] = west
            if grid[row-1, col-1] == np.inf:
                north_west = euclidean((delta, tau), _lattice[row-1][col-1])
                grid[row-1, col-1] = north_west
            coordinates = np.where(grid == grid.min())
            row, col = int(coordinates[0][0]), int(coordinates[1][0])
            if old_min == grid.min():
                break
            old_min = grid.min()

    @staticmethod
    def interpolation_per_bucket(df: pd.DataFrame, grid_size: Tuple[int] = (100, 100)) -> np.ndarray:
        total_variance_plan = df[['k', 'tau', 'total_variance']].values
        k_min, k_max = float(total_variance_plan[:, 0].min()), float(total_variance_plan[:, 0].max())
        tau_max = float(total_variance_plan[:, 1].max())
        tau = np.linspace(float(np.log(1/360)), max(np.log(tau_max+1), 0), grid_size[1])
        k = np.linspace(-(-2 * k_min) ** (1/3), (2 * k_max) ** (1/3), grid_size[0])
        k, tau = list(map(lambda x: float(x ** 3), k)), list(map(lambda x: float(np.exp(x)), tau))
        k, tau = np.meshgrid(k, tau)
        return griddata(points=(total_variance_plan[:, 0], total_variance_plan[:, 1]),
                        values=total_variance_plan[:, 2], xi=(k, tau), method='nearest')

    @staticmethod
    def split_data(df: pd.DataFrame, date: datetime, extraction_method: str) -> Tuple[pd.DataFrame]:
        test = df.loc[df.index.date == date]
        df.drop(axis=0, index=test.index, inplace=True)
        if extraction_method != 'pca':
            valid = df.iloc[-int(df.shape[0] / 5):, ].copy()
            df.drop(axis=0, index=valid.index, inplace=True)
            return df, valid, test
        else:
            return df, None, test

    # def rv_matrix2_buckets(self, dim: int, sym: Union[str, List[str]], start: datetime,
    #                        end: datetime, h: str) -> np.ndarray:
    #     if isinstance(sym, str):
    #         sym = [sym]
    #     rv_matrix_ls = list()
    #     underlying = \
    #         self._underlying.loc[(self._underlying.index.date >= start) & (self._underlying.index.date < end), :]
    #     underlying = underlying.resample(h).sum().mul((pd.to_timedelta('360D')//pd.to_timedelta(h)))
    #     underlying = \
    #         pd.DataFrame(underlying.stack()).reset_index(level=-1).rename(columns={'level_1': 'symbol', 0: 'rv'})
    #     underlying = underlying.assign(hour=underlying.index.hour, minute=underlying.index.minute)
    #     rv_symbol_group = \
    #         underlying.groupby(by=[pd.Grouper(key='symbol'), pd.Grouper(key='hour'), pd.Grouper(key='minute')]).mean()
    #     rv_symbol_group.reset_index(0, inplace=True)
    #     for i in range(0, len(sym)):
    #         diag = rv_symbol_group.reset_index(0).query(f'symbol == "{sym[i]}"').rv
    #         rv_matrix = np.eye(N=dim, dtype=float)
    #         np.fill_diagonal(rv_matrix, diag)
    #         rv_matrix_ls.append(rv_matrix)
    #     return np.array(rv_matrix_ls)



