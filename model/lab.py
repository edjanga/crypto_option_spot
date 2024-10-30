import pdb
import matplotlib.pyplot as plt
from data_centre.data import Reader
# from catalogue import RealNVP
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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
from scipy.stats import truncnorm
import time
import concurrent.futures
from scipy.stats import norm, truncnorm
import torch
import optuna
from optuna.samplers import RandomSampler
from functools import partial
from preprocessor import DataPreProcessor
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

NUM_BOOST_ROUND = 50
STOPPING_ROUNDS = 3
MAX_ITER = 10
N_TRIALS = 3
N_JOBS = 1
EPOCH = 1_000
EARLY_STOPPING_PATIENCE = 5
DEVICE = 'cpu'
# if torch.backends.mps.is_available():
#     DEVICE = 'mps'
# elif torch.cuda.is_available():
#     DEVICE = 'cuda'
DATE_FORMAT = '%Y-%m-%d'
BUCKET = 288
L_train_converter_days = {'1W': 7, '1M': 30, '6M': 180}


def custom_loss_metric(tv_pred, data):
    tv_true = data.get_label()  # Ground truth labels
    pdb.set_trace()
    # mae = np.mean(np.abs(y_true - y_pred))  # Calculate MAE
    # return 'custom_mae', mae, False  # False means lower is better


def objective(trial: optuna.trial.Trial, X_train: pd.DataFrame, X_valid: pd.DataFrame):
    pdb.set_trace()
    train_loader = lgb.Dataset(data=X_train, label=X_train['total_variance'], free_raw_data=False,
                               categorical_feature=X_train.filter(regex='is').columns.tolist())
    valid_loader = lgb.Dataset(data=X_valid, label=X_valid['total_variance'], free_raw_data=False,
                               categorical_feature=X_train.filter(regex='is').columns.tolist())
    param = {
        'objective': 'regression', 'verbosity': -1,
        'max_depth': trial.suggest_int('max_depth', 1, 3),
        'lr': trial.suggest_categorical(name='lr', choices=[.01, .05, .1]),
        'tree_learner': 'data',
        'extra_tree': True, 'boosting_type': 'goss',
        #'metric': 'mse' # to be removed when adding custom metric, i.e. arbitrage free total variance feval=custom_mae_metric in lgb.train
    }
    tmp_rres = lgb.train(params=param, train_set=train_loader, valid_sets=[valid_loader], feval=custom_loss_metric,
                         num_boost_round=NUM_BOOST_ROUND,
                         callbacks=[lgb.early_stopping(STOPPING_ROUNDS,
                                                       first_metric_only=True, verbose=True, min_delta=0.0)])
    loss = mean_squared_error(X_valid['total_variance'], tmp_rres.predict(X_valid))
    trial.set_user_attr(key='best_estimator', value=tmp_rres)
    return loss


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_estimator', value=trial.user_attrs['best_estimator'])


class HelperIVS(object):

    _deltas = np.linspace(1, 9, 9) / 10
    _taus = np.array([0, 7, 30, 180, 360])#np.array([0, 1, 7, 30, 45, 60, 90, 180, 360])
    @staticmethod
    def regress_implied_q_and_r(df: pd.DataFrame) -> np.ndarray:
        """
        :return: Two column vectors where the first = dividend yield and the second one = the rate
        """
        y = df.mark_price.values.reshape(-1, 1)
        X = df[['underlying_price', 'strike_price']].values
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X=X, y=y)
        return np.hstack((np.array(-np.log(lr.coef_[0][0])), np.array(-np.log(lr.coef_[0][1]))), dtype=float)

    @staticmethod
    def implied_q_and_r(df: pd.DataFrame, start: datetime, date: datetime, freq: str) -> pd.DataFrame:
        call = df.loc[(df.index.date >= start) & (df.index.date <= date), :].query(
            f'type == "call"').set_index('option_tag', append=True).sort_index()
        put = df.loc[(df.index.date >= start) & (df.index.date <= date), :].query(
            f'type == "put"').set_index('option_tag', append=True).sort_index()
        call = call.loc[~call.index.duplicated(keep='first'), :]
        put = put.loc[~put.index.duplicated(keep='first'), :]
        common_idx = call.index.intersection(put.index)
        call, put = call.loc[common_idx, :], put.loc[common_idx, :]
        assert call.shape == put.shape
        target = call.mark_price.subtract(put.mark_price)
        implied_q_and_r = pd.concat([target, call[['underlying_price', 'strike_price', 'tau']]], axis=1).groupby(
            by=pd.Grouper(level='timestamp', freq=freq)).apply(lambda x: HelperIVS.regress_implied_q_and_r(x))
        implied_q_and_r = pd.DataFrame(implied_q_and_r.tolist(), index=implied_q_and_r.index, columns=['q', 'r'])
        return implied_q_and_r

    @staticmethod
    def preprocess_train_data(df: pd.DataFrame, start: datetime, date: datetime, freq: str) -> pd.DataFrame:
        implied_q_and_r = HelperIVS.implied_q_and_r(df, start, date, freq)
        implied_q_and_r = implied_q_and_r.reindex(df.index)
        implied_q_and_r.ffill(inplace=True)
        df = pd.concat([df, implied_q_and_r], axis=1)
        df[['q', 'r']] = df[['q', 'r']].apply(lambda x, y: x.div(y), y=df.tau)
        df.fillna(0, inplace=True)
        df.loc[:, 'm'] = HelperIVS.delta2m(df[['mark_iv', 'tau', 'q', 'delta']])
        return df

    @staticmethod
    def delta2m(df: pd.DataFrame) -> pd.Series:
        return .5 * df.mark_iv ** 2 * df.tau - \
            df.mark_iv * df.mark_iv ** .5 * norm.pdf(np.exp(df.q * df.tau) * df.delta)

    @staticmethod
    def dfw_coefficients(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        X = df.copy()
        X = X.assign(m2=X.m ** 2, tau2=X.tau ** 2, m_tau=X.m * X.tau).dropna()
        lrs = X.groupby(by=pd.Grouper(level='timestamp', freq=freq)).apply(
            lambda x: LinearRegression(fit_intercept=True).fit(y=x['mark_iv'].values.reshape(-1, 1),
                                                               X=x[['m', 'tau', 'm2', 'tau2', 'm_tau']].values))
        lrs = lrs.apply(lambda x: np.vstack((x.intercept_, x.coef_.reshape(-1, 1))).tolist())
        lrs = lrs.apply(lambda x: list(map(lambda y: y[0], x)))
        idx = X.index.unique()
        lrs = pd.DataFrame(lrs.values.tolist(), index=idx.to_series().resample(freq).asfreq())
        return lrs.reset_index().dropna().set_index('timestamp')

    @staticmethod
    def generate_grid(method: str = 'uniform') -> Tuple[np.ndarray]:
        if method == 'uniform':
            deltas = HelperIVS._deltas
        elif method == 'dense_atm':
            np.random.seed(333)
            delta_min, delta_max = 0, 1
            delta_atm, sigma = 0.5, 0.5
            deltas = \
                truncnorm.rvs((delta_min - delta_atm) / sigma, (delta_max - delta_atm) / sigma, loc=delta_atm,
                              scale=sigma, size=50)
            deltas = np.sort(deltas).tolist()
        grid = \
            np.meshgrid(deltas, HelperIVS._taus) if method == 'uniform' else \
                np.meshgrid(deltas, np.linspace(0, 360, len(deltas))/360)
        return grid

    @staticmethod
    def ivs_interpolation(grid: Tuple[np.ndarray], coef: pd.DataFrame) -> np.ndarray:
        term1 = coef.iloc[:, 0]
        term2 = coef.iloc[:, 1].apply(lambda x, y: x * y, y=grid[0])
        term3 = coef.iloc[:, 2].apply(lambda x, y: x * y, y=grid[1])
        term4 = coef.iloc[:, 3].apply(lambda x, y: x * y ** 2, y=grid[0])
        term5 = coef.iloc[:, 4].apply(lambda x, y: x * y ** 2, y=grid[1])
        term6 = coef.iloc[:, 5].apply(lambda x, y: x * grid[0] * grid[1], y=grid)
        tmp_grid = term1 + term2 + term3 + term4 + term5 + term6
        tmp_grid = tmp_grid.apply(lambda x: np.where(x > .01, x, .01))
        tmp_grid = np.array(list(map(lambda x: x, tmp_grid.values)))
        return tmp_grid


class ForecastIVS(DataPreProcessor, HelperIVS):

    def __init__(self, L: str, symbol: Union[str, List[str]], h: str, option_type: str):
        super().__init__(L=L, h=h)
        self._symbol = symbol
        self._option_type = option_type

    @property
    def option_type(self):
        return self._option_type

    @option_type.setter
    def option_type(self, option_type: str):
        self._option_type = option_type

    @staticmethod
    def reshape_ivs(ivs_time_of_day: np.ndarray) -> torch.Tensor:
        len_tau = len(ivs_time_of_day.iloc[0][0])
        ivs_time_of_day = \
            np.array([ivs[0].reshape(np.cumprod(ivs[0].shape)[1]) for ivs in ivs_time_of_day.values.tolist()])
        ivs_time_of_day = \
            ivs_time_of_day.reshape((ivs_time_of_day.shape[0], len_tau, ivs_time_of_day.shape[-1] // len_tau))
        return torch.from_numpy(ivs_time_of_day)

    @staticmethod
    def objective(trial: optuna.trial.Trial, train: torch.Tensor, valid: torch.Tensor, I0: np.ndarray, **kwargs):
        generative_model = trial.study.study_name.split('_')[:2]
        tag = '_'.join(trial.study.study_name.split('_')[2:])
        generative_model = \
            '_'.join(generative_model) if generative_model == ['normalising', 'flows'] else generative_model[0]
        tag = '_'.join([generative_model, tag])
        print(f'[Start of Training]: Training of {tag} has started.')
        if generative_model == 'normalising_flows':
            start = kwargs['date'] - relativedelta(days=L_train_converter_days[kwargs['L_train']])
            underlying = kwargs['underlying_price']
            underlying = underlying.filter(regex=kwargs['symbol']).loc[(underlying.index.date >= start) &
                                                                       (underlying.index.date < kwargs['date']), :]
            underlying = \
                underlying.resample(kwargs['h']).sum().mul((pd.to_timedelta('360D') // pd.to_timedelta(kwargs['h'])))
            rv_matrix = underlying.mean().loc[kwargs['symbol']] * np.eye(N=I0[0].shape[0], M=I0[0].shape[-1])
            param = {
                'lr': trial.suggest_categorical(name='lr', choices=[1e-04, 5e-04, 1e-03]),
                'num_layers': trial.suggest_categorical(name='num_layers', choices=list(range(1, 5))),
                'size_layers': I0[0].shape[0]
            }
            param.update({'sigma2': rv_matrix})
            prior = distributions.MultivariateNormal(torch.zeros(I0[0].shape[0]), torch.from_numpy(rv_matrix))
            # prior = np.random.multivariate_normal(np.zeros(I0[0].shape[0]), rv_matrix, I0[0].shape[0])
            tmp_mask1 = [1 if i % 2 != 0 else 0 for i in range(1, I0[0].shape[0] + 1)]
            tmp_mask2 = [1 if i % 2 == 0 else 0 for i in range(1, I0[0].shape[0] + 1)]
            mask = [tmp_mask1, tmp_mask2]
            mask = torch.from_numpy(np.array(mask * (I0[0].shape[0] // 2)))
            nets = \
                lambda: nn.Sequential(nn.Linear(mask.shape[0], param['size_layers'], device=DEVICE,
                                                dtype=torch.float64),
                                      nn.LeakyReLU(),
                                      nn.Linear(param['size_layers'], param['size_layers'], device=DEVICE,
                                                dtype=torch.float64),
                                      nn.LeakyReLU(),
                                      nn.Linear(param['size_layers'], param['size_layers'], device=DEVICE,
                                                dtype=torch.float64))
            nett = \
                lambda: nn.Sequential(nn.Linear(mask.shape[0], param['size_layers'], device=DEVICE,
                                                dtype=torch.float64),
                                      nn.LeakyReLU(),
                                      nn.Linear(param['size_layers'], param['size_layers'], device=DEVICE,
                                                dtype=torch.float64),
                                      nn.LeakyReLU(),
                                      nn.Linear(param['size_layers'], param['size_layers'], device=DEVICE,
                                                dtype=torch.float64))
            flow = RealNVP(prior=prior, num_layers=param['num_layers'], mask=mask, nets=nets, nett=nett)
            optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad is True], lr=1e-4)
            best_val_loss = np.inf
            epochs_without_improvement = 0
            for epoch in range(0, EPOCH):
                loss = -flow.log_prob(train).mean()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                pdb.set_trace()
                if epoch % 100 == 0:
                    print('iter %s:' % epoch, 'loss = %.3f' % loss)

        elif generative_model == 'vae':
            pass

    @staticmethod
    def callback(study: optuna.study.study.Study, trial: optuna.trial.Trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key='best_estimator', value=trial.user_attrs['best_estimator'])

    def latent_features(self, train: torch.Tensor, valid: torch.Tensor,
                        extraction_method: str, symbol: str, date: datetime, time_bucket: int, I0: Tuple[np.ndarray]):
        if extraction_method == 'pca':
            pdb.set_trace()
        else:
            study_name = f'{extraction_method}_{self._L}_{symbol}_{date.strftime("%Y-%m-%d")}_{time_bucket}'
            study = optuna.create_study(direction='minimize', study_name=study_name, sampler=RandomSampler(123))
            objective = partial(self.objective, train=train, valid=valid, I0=I0) \
                if extraction_method != 'normalising_flows' else \
                partial(self.objective, train=train, valid=valid, h=self._h, date=date, L_train=self._L, symbol=symbol,
                        underlying_price=self._underlying, I0=I0)
            study.optimize(objective, n_trials=N_TRIALS, callbacks=[self.callback], n_jobs=N_JOBS)
            rres = study.user_attrs['best_estimator']
            return date, rres
        pdb.set_trace()

    def extract_latent_features_symbol(self, mkt_data: pd.DataFrame, symbol: str, extraction_method: str):
        dates = list(np.unique([timestamp.date() for timestamp in mkt_data['timestamp']]))
        mkt_data.set_index('timestamp', inplace=True)
        start_date = dates[0] + relativedelta(days=L_train_converter_days[self._L])
        start_idx = dates.index(start_date)
        # I0 = self.generate_grid(method='dense_atm')
        eigenvalues_ls = list()
        for date in dates[start_idx:]:
            print(f'[Model Calibration]: Parameter calibration for {date.strftime(DATE_FORMAT)} has started...')
            start = date - relativedelta(days=L_train_converter_days[self._L])
            train = mkt_data.loc[(mkt_data.index.date >= start) & (mkt_data.index.date <= date), :].copy()
            train = train.query(f'type == "{self._option_type}"')
            train = pd.pivot_table(train[['option_tag', 'mark_iv']].reset_index(), columns='option_tag',
                                   values='mark_iv', index='timestamp')
            # train = train.subtract(train.min()).div(train.max().subtract(train.min()))
            train.fillna(0, inplace=True)
            train_cov = np.cov(train.values.transpose())
            S = np.linalg.svd(train_cov)[1]
            S = S / S.sum()
            eigenvalues_ls.append(S)
        pd.DataFrame(eigenvalues_ls)[[0]].plot()
        plt.show()
        pdb.set_trace()

    def spectral_analysis(self, mkt_data: pd.DataFrame, normalise: bool = False):
        mkt_data.set_index('timestamp', inplace=True)
        eigenvalues = pd.DataFrame()
        for L in ['1W', '1M', '6M']:
            if L != '1W':
                self.L = L
            if L == '6M':
                break
            dates = list(np.unique([timestamp.date() for timestamp in mkt_data.index]))
            start_date = dates[0] + relativedelta(days=L_train_converter_days[self._L])
            start_idx = dates.index(start_date)
            eigenvalues_ls = list()
            for date in dates[start_idx:]:
                start = date - relativedelta(days=L_train_converter_days[self._L])
                train = mkt_data.loc[(mkt_data.index.date >= start) & (mkt_data.index.date <= date), :].copy()
                train = train.query(f'type == "{self._option_type}"')
                train = pd.pivot_table(train[['option_tag', 'mark_iv']].reset_index(), columns='option_tag',
                                       values='mark_iv', index='timestamp')
                if normalise:
                    train = train.subtract(train.min()).div(train.max().subtract(train.min()))
                train.fillna(0, inplace=True)
                train_cov = np.cov(train.values.transpose())
                S = np.linalg.svd(train_cov)[1]
                S = S / S.sum()
                eigenvalues_ls.append(S)
            tmp = \
                pd.DataFrame(eigenvalues_ls, index=dates[start_idx:]).cumsum(axis=1).iloc[:, :5].unstack().reset_index()
            tmp['L'] = self._L
            eigenvalues = pd.concat([eigenvalues, tmp])
        fig = make_subplots(rows=5, cols=2, shared_xaxes=False, shared_yaxes=False, vertical_spacing=0.05,
                            horizontal_spacing=0.1, row_titles=list(range(1, 6, 1)), column_titles=['1W', '1M'])
        tmp_fig = px.line(data_frame=eigenvalues, facet_row='level_0', facet_col='L', y=0, x='level_1')
        for idx in range(0, len(tmp_fig.data), 2):
            fig.add_trace(go.Scatter(x=tmp_fig.data[idx].x, y=tmp_fig.data[idx].y), row=(idx // 2 + 1),
                          col=idx % 2 + 1)
            fig.add_trace(go.Scatter(x=tmp_fig.data[idx + 1].x, y=tmp_fig.data[idx + 1].y), row=(idx // 2 + 1),
                          col=idx % 2 + 2)
        fig.update_layout(showlegend=False)
        fig.show()


if __name__ == '__main__':
    reader_obj = Reader()
    underlying_price = reader_obj.prices_read().resample('5min').last()
    files = [file for file in glob.glob('../data_centre/tmp/options/*-*-*')]
    files.sort()
    data = pd.concat([pd.read_parquet(option_chain) for option_chain in files])
    data.reset_index(inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
    data = data.set_index('timestamp').join(underlying_price, how='left')
    data.reset_index(inplace=True)
    data.drop('underlying_price', axis=1, inplace=True)
    data = data.rename(columns={'ETHUSDT': 'underlying_price'})
    data.loc[:, 'tau'] = data['expiration'].subtract(data['timestamp']).div(pd.to_timedelta('1D'))
    latent_features_obj = ForecastIVS(L='1W', h='1H', symbol='ETHUSDT', option_type='call')
    #latent_features_obj.extract_latent_features_symbol(mkt_data=data, symbol='ETHUSDT', extraction_method='pca')
    latent_features_obj.spectral_analysis(mkt_data=data)

