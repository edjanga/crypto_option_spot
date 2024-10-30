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
from preprocessor import DataPreProcessor

DATE_FORMAT = '%Y-%m-%d'
BUCKET = 288
L_train_converter_days = {'1W': 7, '1M': 30, '6M': 180}


class SDEModel:
    ####################################################################################################################
    ## CARR-MADAN Formula constants
    ####################################################################################################################
    N, ALPHA, ETA = 4096, 1.5, .25
    LAMBDA = 2 * np.pi / N / ETA
    B = LAMBDA * N / 2
    K = np.arange(start=-B, stop=B, step=LAMBDA)
    KK = np.exp(K)
    V = np.arange(0, N * ETA, ETA)
    SW = (3 + (-1) ** np.arange(1, N+1, 1))
    SW[0] = 1
    SW = SW/3
    ####################################################################################################################
    ## CONSTANTS
    ####################################################################################################################
    NUM_ITER = 1000
    TOL = 1e-4
    ####################################################################################################################
    ## USEFULL OBJECTS
    ####################################################################################################################
    _reader_obj = Reader()
    _data_pre_processor = DataPreProcessor()

    def __init__(self, L: str, symbol: Union[str, List[str]] = 'ETHUSDT', h: str = '30min'):
        self._parameters = pd.DataFrame()
        if isinstance(symbol, str):
            self._symbol = [symbol]
        self._S0 = SDEModel._reader_obj.prices_read(symbol=self._symbol)
        self._rv = SDEModel._reader_obj.rv_read(symbol=self._symbol)
        self._h = h
        self._L = L
        self._scaling_vol_pa = np.sqrt(pd.to_timedelta('360D') // self._h)

    def characteristic_function(self):
        pass

    def theoretical_price(self, mkt_data: pd.DataFrame) -> None:
        pass

    def params_init(self) -> Union[float, Tuple[float]]:
        pass

    def model_bounds(self) -> List[Tuple]:
        pass

    def model_constraints(self) -> Tuple[dict]:
        pass

    def model_calibration(self) -> None:
        pass

    @staticmethod
    def select_loss_function(prices: Union[pd.DataFrame, Tuple[float]], loss_function: str = 'mse') -> float:
        loss_function_catalogue = dict([('mse', lambda x: float((x.diff(axis=1)[0]**2).values.mean())),
                                        ('ape', lambda x: float(np.abs(x.diff(axis=1)[0]).mean()/x[0].mean())),
                                        ('arpe', lambda x: float(np.abs(x.diff(axis=1)[0]).div(x[0]).mean()))])
        return loss_function_catalogue[loss_function](prices)

    def loss(self, params: Union[None, float], mkt_data: pd.DataFrame, option_type: str = 'call',
             loss_function: str = 'mse') -> float:
        prices = self.theoretical_price(params=params, mkt_data=mkt_data,  option_type=option_type)
        prices[0] = prices[0].where(prices[0].gt(0), 0)
        return self.select_loss_function(prices, loss_function)

    def params_estimation(self, mkt_data: pd.DataFrame, option_type: str = 'call', loss_function: str = 'mse') -> None:
        self._data_pre_processor.L = self._L
        self._data_pre_processor.h = self._h
        self._parameters = \
            pd.DataFrame(columns=self.PARAMS_NAME,
                         index=np.unique(np.array([date.date() for date in mkt_data.timestamp]))).sort_index()
        self._parameters.index = pd.to_datetime(self._parameters.index, utc=True)
        self._parameters.index.name = 'timestamp'
        dates = list(self._parameters.index)
        start_date = dates[0]+relativedelta(days=L_train_converter_days[self._L])
        for date in dates[dates.index(start_date):]:
            print(f'[Model Calibration]: Parameter calibration for {date.strftime(DATE_FORMAT)} has started...')
            start = date-relativedelta(days=L_train_converter_days[self._L])
            self._data_pre_processor.input_call = \
                mkt_data.loc[(mkt_data.timestamp >= start) & (mkt_data.timestamp < date), :]
            print(f'[Grid]: Grid construction has started...')
            train = Heston._data_pre_processor.build_grid(option_type=option_type)
            print(f'[Grid]: Grid construction is now completed.')
            self._parameters.loc[date, :] = \
                self.model_calibration(mkt_data=train, option_type=option_type, loss_function=loss_function,
                                       params=self._parameters.loc[date-relativedelta(days=1), :])
            print(f'[Model Calibration]: Parameter calibration for {date.strftime(DATE_FORMAT)} is now completed.')
        self._parameters.dropna(inplace=True)

    def option_chain_pricer(self, mkt_data: pd.DataFrame, option_type: str = 'call',
                            loss_function: str = 'mse') -> pd.DataFrame:
        self.params_estimation(mkt_data=mkt_data, option_type=option_type, loss_function=loss_function)
        test = mkt_data.loc[(mkt_data.timestamp >= self._parameters.index[0]) &
                            (mkt_data.timestamp < self._parameters.index[-1]), :].query(f'type == "{option_type}"')
        test['tau'] = test['tau'].replace(0, np.nan)
        test.dropna(inplace=True)
        test.set_index('timestamp', inplace=True)
        test = test.groupby(by=[pd.Grouper(level='timestamp', freq=self._h), pd.Grouper(key='option_tag')]).last()
        test.reset_index(1, inplace=True)

        test = \
            pd.concat([test, self._parameters.reindex(test.index).ffill()], axis=1).set_index('option_tag', append=True)
        test = test.assign(model_price=np.nan)
        for row in test.itertuples():
            test.loc[row.Index, 'model_price'] = self.theoretical_price(params=None, mkt_data=row, chain=False)
        return test

    @staticmethod
    def bs_formula(data: pd.Series, vol: float) -> float:
        d1 = np.log(data.underlying_price / data.strike_price) + data.tau * (data.r - data.q + .5 * vol ** 2)
        d1 /= (vol ** 2 * data.tau) ** .5
        d2 = d1 - (vol ** 2 * data.tau) ** .5
        return data.underlying_price * norm.cdf(d1) - \
            data.strike_price * norm.cdf(d2) * np.exp(-(data.r - data.q) * data.tau)

    def find_iv(self, data: pd.Series) -> float:
        vol0 = self.params_init(date=data.Index[0].date()).iloc[0]
        fun = lambda x: float((data.mark_price - self.bs_formula(data=data, vol=x))[0] ** 2)
        result = minimize(fun, x0=[vol0], options={'maxiter': self.NUM_ITER}, method='Nelder-Mead', tol=self.TOL,
                          bounds=self.model_bounds())
        return result.x

    # @njit
    # def find_iv_optimised(self, data: np.ndarray) -> float:
    #     vol0 = self.params_init(date=data.name[0].date()).iloc[0]
    #     pdb.set_trace()
    #     fun = lambda x: float((data.mark_price - self.bs_formula(data=data, vol=x)) ** 2)
    #     result = minimize(fun, x0=[vol0], options={'maxiter': self.NUM_ITER}, method='Nelder-Mead', tol=self.TOL,
    #                       bounds=self.model_bounds())
    #     return result.x

    def compute_iv(self, mkt_data: pd.DataFrame, option_type: str = 'call', loss_function: str = 'mse') -> None:
        test = self.option_chain_pricer(mkt_data=mkt_data, option_type=option_type, loss_function=loss_function)
        test.drop(['gamma', 'vega', 'theta', 'rho', 'volume', 'expiration', 'delta',
                   'type'] + self._parameters.columns.tolist(), axis=1, inplace=True)
        test['model_price'] = test['model_price'].where(test['model_price']>0, 0)
        test['iv_hat'] = np.nan
        for row in test.itertuples():
            test.loc[row.Index, 'iv_hat'] = self.find_iv(data=row)
        return test


class BlackScholes(SDEModel):
    OPTIMISATION_ALGOS = ['SLSQP']  # ['COBYLA', 'COBYQA', 'SLSQP', 'trust-constr']
    PARAMS_NAME = ['vol0']

    def __init__(self, L: str, symbol: Union[str, List[str]] = 'ETHUSDT', h: str='30min'):
        super().__init__(symbol=symbol, h=h, L=L)

    def characteristic_function(self, vol: Union[float, pd.Series], T: Union[float, pd.Series], u: np.ndarray,
                                S0: Union[float, pd.Series], q: Union[float, pd.Series], r: Union[float, pd.Series])\
            -> Union[float, pd.DataFrame]:
        if isinstance(vol, pd.Series):
            u = np.broadcast_to(u, (S0.shape[0], u.shape[0]))
            vol = pd.concat([vol] * BlackScholes.N, ignore_index=True, axis=1)
            T = pd.concat([T] * BlackScholes.N, ignore_index=True, axis=1)
            S0 = pd.concat([S0] * BlackScholes.N, ignore_index=True, axis=1)
            q = pd.concat([q] * BlackScholes.N, ignore_index=True, axis=1)
            r = pd.concat([r] * BlackScholes.N, ignore_index=True, axis=1)
            tmp = pd.concat([(np.log(S0) + T * (r - q - .5 * vol ** 2))] * BlackScholes.N, ignore_index=True, axis=1)
        else:
            tmp = np.log(S0) + T * (r - q - .5 * vol ** 2)
        return np.exp(1j * u * tmp) * np.exp(-.5 * vol ** 2 * T * u ** 2)

    def params_init(self, date: Union[None, datetime] = None) -> float:
        if date is None:
            start_date = self._rv.index[0].date()
            date = start_date + relativedelta(days=L_train_converter_days[self._L])
        else:
            start_date = date - relativedelta(days=L_train_converter_days[self._L])
        params = self._rv.loc[(self._rv.index.date >= start_date) & (self._rv.index.date < date)]
        return params.resample(self._h).sum().mean() ** .5 * self._scaling_vol_pa

    def model_bounds(self) -> List[Tuple]:
        return [(0, None)]

    def theoretical_price(self, params: Union[None, List[float]], mkt_data: pd.DataFrame, q: float = 0.0,
                          r: Union[float, pd.Series, pd.DataFrame] = 0.0, option_type: str = 'call',
                          chain: bool = True) -> Union[float, np.array]:
        """
            params: vol: float
        """
        if chain:
            vol = float(params[0])
            T = mkt_data[['tau']].copy()
            S0 = self._S0.copy().reindex(index=T.index.get_level_values(0))
            S0.index = T.index
            modelPrice = list()
            for row in T.itertuples():
                tau, s0, strike = row[1], float(S0.loc[row[0], :].ETHUSDT.values[0]),\
                    mkt_data.loc[row[0], 'strike_price'].values[0]
                rho = \
                    self.characteristic_function(vol=vol, T=tau, S0=s0,
                                                 u=BlackScholes.V - (BlackScholes.ALPHA + 1) * 1j, q=q, r=r)/\
                    (BlackScholes.ALPHA ** 2 + BlackScholes.ALPHA - BlackScholes.V ** 2 + 1j *
                     (2 * BlackScholes.ALPHA + 1) * BlackScholes.V)
                A = rho * np.exp(1j * BlackScholes.V * BlackScholes.B) * BlackScholes.ETA * BlackScholes.SW
                Z = np.real(np.fft.fft(A))
                call_prices = np.exp(-BlackScholes.ALPHA * BlackScholes.K) / np.pi * Z
                spline_rep = splrep(BlackScholes.KK, call_prices, s=0)
                option_price_fft = splev(mkt_data.strike_price.loc[row[0]], spline_rep, der=0)
                if option_type == 'put':
                    option_price_fft = \
                        option_price_fft - s0 * np.exp(-q * tau) + mkt_data.strike_price.loc[row[0]] * np.exp(-r * tau)
                modelPrice.append(option_price_fft)
            model_price = pd.concat([mkt_data[['mark_price']], pd.DataFrame(index=mkt_data.index, data=modelPrice)],
                                    axis=1)
        else:
            rho = \
                self.characteristic_function(vol=mkt_data.vol0, T=mkt_data.tau, S0=mkt_data.underlying_price,
                                             u=BlackScholes.V - (BlackScholes.ALPHA + 1) * 1j, q=mkt_data.q,
                                             r=mkt_data.r) / \
                (BlackScholes.ALPHA ** 2 + BlackScholes.ALPHA - BlackScholes.V ** 2 + 1j *
                 (2 * BlackScholes.ALPHA + 1) * BlackScholes.V)
            A = rho * np.exp(1j * BlackScholes.V * BlackScholes.B) * BlackScholes.ETA * BlackScholes.SW
            Z = np.real(np.fft.fft(A))
            global call_price
            call_price = np.exp(-BlackScholes.ALPHA * BlackScholes.K) / np.pi * Z
            try:
                spline_rep = [splrep(BlackScholes.KK, call_price[idx, :], s=0) for idx in range(0, call_price.shape[0])]
                model_price = splev(mkt_data.strike_price, spline_rep[0], der=0)
            except IndexError:
                spline_rep = splrep(BlackScholes.KK, call_price, s=0)
                model_price = splev(mkt_data.strike_price, spline_rep, der=0)
            # model_price = splev(mkt_data.strike_price, spline_rep[0], der=0)
            if option_type == 'put':
                model_price = model_price - mkt_data.underlying_price * np.exp(-mkt_data.q * mkt_data.tau) + \
                              mkt_data.strike_price * np.exp(-(mkt_data.r - mkt_data.q) * mkt_data.tau)
        return model_price

    def model_calibration(self, params: pd.Series, mkt_data: pd.DataFrame, option_type: str = 'call',
                          loss_function: str = 'mse') -> float:
        if params.isnull().sum() != 0:
            vol0 = self.params_init()
            x0 = [float(vol0.values[0])]
        else:
            x0 = list(params)
        ################################################################################################################
        ## Optimisation
        ################################################################################################################
        if len(self._symbol) == 1:
            result = minimize(self.loss, x0=x0, args=(mkt_data, option_type, loss_function), tol=BlackScholes.TOL,
                              method='Nelder-Mead', options={'maxiter': BlackScholes.NUM_ITER},
                              bounds=self.model_bounds())
        return float(result.x[0])


class Heston(SDEModel):
    OPTIMISATION_ALGOS = ['SLSQP'] #['COBYLA', 'COBYQA', 'SLSQP', 'trust-constr']
    PARAMS_NAME = ['vol0', 'theta_param', 'eta_param', 'kappa_param', 'p_param']

    def __init__(self, L: str, symbol: Union[str, List[str]] = 'ETHUSDT', h: str = '30min'):
        super().__init__(symbol=symbol, h=h, L=L)

    def characteristic_function(self, vol: float, theta: float, eta: float, kappa: float, p: float,
                                T: float, u: np.ndarray, S0: float, q: float, r: float) -> float:
        d = ((p * theta * u * 1j) ** 2 - theta ** 2 * (-1j * u - u ** 2)) ** .5
        g = (kappa - p * theta * u * 1j - d) / (kappa - p * theta * u * 1j + d)
        first = np.exp(1j * u * (np.log(S0) + (r - q) * T))
        second = \
            np.exp(eta * kappa / theta ** 2 * (T * (kappa - p * theta * u * 1j - d) -
                                             2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
        third = \
            np.exp(vol ** 2 / theta ** 2 * (kappa - p * theta * 1j * u - d) *
                   (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        return first * second * third

    def theoretical_price(self, params: Union[None, List[float]], mkt_data: pd.DataFrame, option_type: str = 'call',
                          chain: bool = True) -> Union[pd.DataFrame, float]:
        """
            params: vol: float, theta: float, eta: float, kappa: float, p: float
        """
        if chain:
            vol, theta, eta, kappa, p = params[0], params[1], params[2], params[3], params[4]
            T = mkt_data[['tau']].copy()
            S0 = self._S0.copy().reindex(index=T.index.get_level_values(0))
            S0.index = T.index
            modelPrice = list()
            for row in T.itertuples():
                tau, s0 = row[1], float(S0.loc[row[0], :].ETHUSDT.values[0])
                r, q = mkt_data.loc[row[0], :].r.values[0], mkt_data.loc[row[0], :].q.values[0]
                rho = \
                    self.characteristic_function(vol=vol, T=tau, S0=s0, theta=theta, eta=eta, kappa=kappa, p=p,
                                                 u=Heston.V - (Heston.ALPHA + 1) * 1j, q=q, r=r)/\
                    (Heston.ALPHA ** 2 + Heston.ALPHA - Heston.V ** 2 + 1j *
                     (2 * Heston.ALPHA + 1) * Heston.V)
                A = rho * np.exp(1j * Heston.V * Heston.B) * Heston.ETA * Heston.SW
                Z = np.real(np.fft.fft(A))
                CallPrices = np.exp(-Heston.ALPHA * Heston.K) / np.pi * Z
                spline_rep = splrep(Heston.KK, CallPrices, s=0)
                OptionPriceFFT = splev(mkt_data.strike_price.loc[row[0]], spline_rep, der=0)
                if option_type == 'put':
                    OptionPriceFFT = \
                        OptionPriceFFT - s0 * np.exp(-q * tau) + mkt_data.strike_price.loc[row[0]] * np.exp(-r * tau)
                modelPrice.append(OptionPriceFFT)
            model_price = pd.concat([mkt_data[['mark_price']], pd.DataFrame(index=mkt_data.index, data=modelPrice)],
                                    axis=1)
        else:
            rho = \
                self.characteristic_function(vol=mkt_data.vol0, T=mkt_data.tau, S0=mkt_data.underlying_price,
                                             theta=mkt_data.theta_param, eta=mkt_data.eta_param,
                                             kappa=mkt_data.kappa_param, p=mkt_data.p_param,
                                             u=Heston.V - (Heston.ALPHA + 1) * 1j, q=mkt_data.q, r=mkt_data.r) / \
                (Heston.ALPHA ** 2 + Heston.ALPHA - Heston.V ** 2 + 1j *
                 (2 * Heston.ALPHA + 1) * Heston.V)
            A = rho * np.exp(1j * Heston.V * Heston.B) * Heston.ETA * Heston.SW
            Z = np.real(np.fft.fft(A))
            call_price = np.exp(-Heston.ALPHA * Heston.K) / np.pi * Z
            spline_rep = splrep(Heston.KK, call_price, s=0)
            model_price = splev(mkt_data.strike_price, spline_rep, der=0)
            if option_type == 'put':
                model_price = model_price - mkt_data.underlying_price * np.exp(-mkt_data.q * mkt_data.tau) + \
                              mkt_data.strike_price * np.exp(-(mkt_data.r - mkt_data.q) * mkt_data.tau)
        return model_price

    def params_init(self) -> Tuple[float]:
        return (np.log(self._S0.div(self._S0.shift())) ** 2).resample(self._h).sum().mean() ** .5, 1e-2, 1e-2, 1e-1, .1

    def model_bounds(self) -> List[Tuple]:
        return [(0, None), (0, None), (0, None), (0, None), (-1, 1)]

    def model_constraints(self) -> dict[str, str | Callable[[Any], int | Any]]:
        return {'type': 'ineq', 'fun': lambda x: 2 * x[-2] * x[-3] - x[1] ** 2}

    # def loss(self, params: List[float], mkt_data: pd.DataFrame, q: float = 0.0,
    #          r: Union[float, pd.Series, pd.DataFrame] = 0.0, option_type: str = 'call',
    #          loss_function: str = 'mse') -> float:
    #     prices = self.theoretical_price(params=params, mkt_data=mkt_data, q=q, r=r,  option_type=option_type)
    #     return Heston.select_loss_function(prices, loss_function)

    def model_calibration(self, params: pd.Series, mkt_data: pd.DataFrame, option_type: str = 'call',
                          loss_function: str = 'mse') -> Union[List[float], dict[str, List[float]]]:
        ################################################################################################################
        ## Initialisation
        ################################################################################################################
        if params.isnull().sum() != 0:
            vol0, theta0, eta0, kappa0, p0 = self.params_init()
            x0 = [float(vol0.loc[self._symbol[0]]), theta0, eta0, kappa0, p0]
        else:
            x0 = list(params)
        best_fun = np.inf
        if len(self._symbol) == 1:
            for method in Heston.OPTIMISATION_ALGOS:
                result = \
                    minimize(self.loss, x0=x0, args=(mkt_data, option_type, loss_function),
                             tol=1e-4, method=method, constraints=self.model_constraints(), bounds=self.model_bounds(),
                             options={'maxiter': Heston.NUM_ITER//len(Heston.OPTIMISATION_ALGOS), 'disp': False})
                if result.fun < best_fun:
                    best_result = result
                    best_fun = best_result.fun
        else:
            result_symbols = dict()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(lambda x: x, x=symbol) for symbol in self._symbol]
                for future in concurrent.futures.as_completed(futures):
                    symbol = future.result()
                    pdb.set_trace()
        return best_result.x

    # def params_estimation(self, mkt_data: pd.DataFrame, q: float = 0.0, r: Union[float, pd.Series, pd.DataFrame] = 0.0,
    #                       option_type: str = 'call', loss_function: str = 'mse') -> None:
    #     Heston._data_pre_processor.L = self._L
    #     Heston._data_pre_processor.h = self._h
    #     self._parameters = \
    #         pd.DataFrame(columns=Heston.PARAMS_NAME,
    #                      index=np.unique(np.array([date.date() for date in mkt_data.timestamp]))).sort_index()
    #     self._parameters.index = pd.to_datetime(self._parameters.index, utc=True)
    #     self._parameters.index.name = 'timestamp'
    #     dates = list(self._parameters.index)
    #     start_date = dates[0]+relativedelta(days=L_train_converter_days[self._L])
    #     for date in dates[dates.index(start_date):]:
    #         print(f'[Model Calibration]: Parameter calibration for {date.strftime(DATE_FORMAT)} has started...')
    #         start = date-relativedelta(days=L_train_converter_days[self._L])
    #         Heston._data_pre_processor.input_call = \
    #             mkt_data.loc[(mkt_data.timestamp >= start) & (mkt_data.timestamp < date), :]
    #         print(f'[Grid]: Grid construction has started...')
    #         train = Heston._data_pre_processor.build_grid(option_type=option_type)
    #         print(f'[Grid]: Grid construction is now completed.')
    #         self._parameters.loc[date, :] = \
    #             self.model_calibration(mkt_data=train, q=q, r=r, option_type=option_type, loss_function=loss_function,
    #                                    params=self._parameters.loc[date-relativedelta(days=1), :])
    #         print(f'[Model Calibration]: Parameter calibration for {date.strftime(DATE_FORMAT)} is now completed.')
    #     self._parameters.dropna(inplace=True)

    # def option_chain_pricer(self, mkt_data: pd.DataFrame, q: float = 0.0,
    #                         r: Union[float, pd.Series, pd.DataFrame] = 0.0, option_type: str = 'call',
    #                         loss_function: str = 'mse') -> pd.DataFrame:
    #     self.params_estimation(mkt_data=mkt_data, q=q, r=r, option_type=option_type, loss_function=loss_function)
    #     test = mkt_data.loc[(mkt_data.timestamp >= self._parameters.index[0]) &
    #                         (mkt_data.timestamp < self._parameters.index[-1]), :].query(f'type == "{option_type}"')
    #     test['tau'] = test['tau'].replace(0, np.nan)
    #     test.dropna(inplace=True)
    #     test.set_index('timestamp', inplace=True)
    #     test = test.groupby(by=[pd.Grouper(level='timestamp', freq=self._h), pd.Grouper(key='option_tag')]).last()
    #     test.reset_index(1, inplace=True)
    #     test = pd.concat([test, self._parameters.reindex(test.index).ffill()], axis=1).set_index('option_tag',
    #                                                                                              append=True)
    #     test = \
    #         test.assign(model_price=test.transpose().apply(
    #             lambda x: self.theoretical_price(params=None, mkt_data=x, chain=False)).values)
    #     test['model_price'] = test['model_price'].where(test['model_price'].gt(0), 0)
    #     return test