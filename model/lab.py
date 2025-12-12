import os
import pdb
from constants import *
from itertools import product
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV, ElasticNet
from sklearn.metrics import mean_squared_error
from typing import List, Union, Dict, Tuple
import optuna
from optuna.samplers import RandomSampler
from functools import partial
import lightgbm as lgb
from torch.utils.data import DataLoader
from .lstm_room import TimeSeriesDataset, LSTMModel
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.model_selection import ParameterSampler
import polars as pl
import pytorch_lightning as p_light
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'

tscv = TimeSeriesSplit(n_splits=2)


def shuffle_single_feature(test, feature: str) -> Tuple[pd.DataFrame, str]:
    np.random.seed(123)
    if isinstance(test, pd.DataFrame):
        shuffled_test = test.copy()
    for symbol in shuffled_test.index.get_level_values(0).unique():
        feature_value = np.random.permutation(shuffled_test.xs(symbol, axis=0, level=0)[feature].values)
        shuffled_test.loc[shuffled_test.index.get_level_values(0) == symbol, feature] = feature_value
    return shuffled_test, feature


def run_pnl_ols(x, data, targets, pnl) -> pd.Series|pd.DataFrame:
    list_factors = x.features.values.tolist()
    target = x.target.unique()[0]
    targets = targets.xs(target, axis=1, level=-1).filter(regex='|'.join(list_factors)).unstack()
    targets = targets.loc[targets.index.get_level_values(0).isin(list_factors)]
    X = data.filter(regex='|'.join(list_factors))
    pnl = pnl.xs(x.name[1], axis=1, level=1).xs(x.name[-1], axis=0, level=1).stack()
    pnl.name = 'P&L'
    if not targets.empty:
        pdb.set_trace()
        # x = x.join(targets.filter(regex='|'.join(list_factors)), axis=1, how='left')
    X = pnl.reset_index(0).swaplevel(0, 1).join(X, on=['underlying_name', 'timestamp']).set_index('aggregated',
                                                                                                  append=True)
    X['P&L'] = X['P&L'] + np.random.uniform(0, 1e-8, X.shape[0])
    X = X.apply(lambda x: pd.to_numeric(x, errors='coerce').astype('float32'))
    X = (X - X.min()).div(X.max() - X.min())
    params = sm.OLS(X['P&L'], sm.add_constant(X.drop(columns=['P&L']))).fit().params
    params.index = ['α'] + [f'β<sub>{i}</sub>' for i in range(1, params.shape[0])]
    return params


def run_pnl_decile_ols(x, ranking, data, targets) -> pd.Series:
    ranking = ranking.query(f'(volatility_regime == "{x.name[1]}") & (D == "{x.name[2]}") & (target == "{x.name[3]}")')
    list_factors = ranking.features.unique().tolist()
    data = data.filter(regex='|'.join(list_factors))
    target = targets.xs(x.name[3], axis=1, level=1).xs(x.name[2].upper(), axis=1,
                                                       level=1).filter(regex='|'.join(list_factors))
    if not target.empty:
        pdb.set_trace()
    x = x.drop(['aggregated', 'target', 'D', 'volatility_regime', 'P&L decile'], axis=1)
    X = x.set_index('underlying_name', append=True).swaplevel(0, 1).join(data, on=['underlying_name', 'timestamp'])
    X = X.apply(lambda x: pd.to_numeric(x, errors='coerce').astype('float32'))
    X = X.groupby(by=[pd.Grouper(level='underlying_name')], group_keys=False).apply(
        lambda x: (x - x.min()).div(x.max() - x.min())).dropna()
    params = sm.OLS(X['P&L'], sm.add_constant(X.drop(columns=['P&L']))).fit().params
    params.index = ['α'] + [f'β<sub>{i}</sub>' for i in range(1, params.shape[0])]
    return params




def run_pnl_ols_copy(x):
    pdb.set_trace()
    model = sm.OLS(x['P&L'], sm.add_constant(x.drop(columns=['P&L']))).fit()
    return model.params


class FeatureBuilder:
    def __init__(self):
        pass

    @staticmethod
    def returns_moments(data: pl.DataFrame | pl.LazyFrame, freq: str, upperbound: int = 6) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        pl_freq = freq[:-2] if 'min' in freq else freq
        moments = \
            data.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
                index_column='timestamp', every=pl_freq, by='underlying_name').sum()
        K = (pd.to_timedelta('1D') // pd.to_timedelta(freq))
        for m in range(2, upperbound + 1):
            if m > 2:
                tmp = data.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
                    index_column='timestamp', every=pl_freq,
                    by='underlying_name').agg([
                    (pl.col('returns') ** 2).sum().alias(f'm2'), (pl.col('returns') ** m).sum().alias(f'm{m}')])
                tmp = tmp.with_columns((((K / pl.col(f'm2')) ** m) ** .5).alias(f'm{m}')).drop('m2')
            else:
                tmp = data.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
                    index_column='timestamp', every=pl_freq,
                    by='underlying_name').agg([(pl.col('returns') ** m).sum().alias(f'm{m}')])
            moments = moments.join(tmp, on=['underlying_name', 'timestamp'])
        moments = \
            moments.with_columns([
                pl.when(pl.col(m) == float("inf")).then(np.finfo(np.float32).max).when(
                    pl.col(m) == -float("inf")).then(np.finfo(np.float32).min).otherwise(pl.col(m)).alias(m) for m in
                ['returns']+[f'm{m}' for m in range(2, upperbound + 1)]])
        return moments.collect(engine=ENGINE)

    @staticmethod
    def lags(data: pl.DataFrame, features: List[str] = ['RV']+[f'm{i}' for i in range(2, 7)]) -> pl.LazyFrame:
        intraday_lags_exprs = [pl.col(feature) for feature in features]
        for lag in INTRADAY_LAGS:
            window_size = int(pd.to_timedelta(lag) // pd.to_timedelta('5m'))
            for feature in features:
                intraday_lags_exprs.append((pl.col(f'{feature}').rolling_mean(window_size)).shift().over
                                           ('underlying_name').alias(f'{feature}_{lag}'))
        for lag in INTERDAY_LAGS:
            feature_lag = list(product(features, [lag]))
            pl_lag = f'{lag[:-1]}mo' if 'M' in lag else lag.lower()
            if 'feature_lags' not in vars():
                feature_lags = data.with_columns(intraday_lags_exprs)
            feature_interday_lag = \
                (data.group_by_dynamic(index_column='timestamp', every=pl_lag.lower(),  by='underlying_name')
                 .agg([(pl.col(f'{feature}').mean()).alias(f'{feature}_{lag}') for feature, lag in feature_lag]))
            feature_interday_lag = feature_interday_lag.select(
                [pl.col('underlying_name'), pl.col('timestamp')]+
                [(pl.col(f'{feature}_{lag}').shift(1).over('underlying_name')) for feature, lag in feature_lag])
        feature_lags = feature_lags.join(feature_interday_lag, on=['underlying_name', 'timestamp'], how='left')
        feature_lags = \
            feature_lags.with_columns([pl.col(f'{feature}_{lag}').fill_null(strategy='forward').over('underlying_name')
                 for feature, lag in feature_lag])
        return feature_lags

    @staticmethod
    def define_market_participants(data: pl.DataFrame) -> pd.DataFrame:
        data = data.with_columns(
            pl.when(pl.col('moneyness_label').is_in(['ATM', 'ITM', 'DITM']))
            .then(pl.lit('informed_traders')).when(pl.col('moneyness_label') == 'OTM')
            .then(pl.lit('institutions')).otherwise(pl.lit('retail')).alias('market_participants'))
        return data

    @staticmethod
    def compute_exposure(data: pl.DataFrame, freq: str, collect: bool = True) -> pd.DataFrame:
        exposure = data
        for exp, greek in GREEKS_EXPOSURE.items():
            if exp == 'dex':
                expr = (pl.col('netDealerExposure') * pl.col(greek) * pl.col('underlying_price'))
            elif exp == 'gex':
                expr = (pl.col('netDealerExposure') * pl.col(greek) * (pl.col('underlying_price') ** 2))
            elif exp == "vex":
                expr = (pl.col('gex') * pl.col('tau') * (pl.col('mark_iv') ** 2))
            elif exp == "vax":
                expr = (pl.col('gex') * (pl.col('moneyness') + 0.5 * pl.col('tau') * (pl.col('mark_iv') ** 2)))
            elif exp == "vox":
                expr = (pl.col('vax') * (-pl.col('moneyness') - 0.5 * pl.col('tau') * (pl.col('mark_iv') ** 2)))
            exposure = exposure.with_columns(expr.alias(exp))
        if 'min' in freq:
            pl_freq = freq.replace('min', 'm')
        exposure = exposure.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by='underlying_name').agg([pl.col(exp).sum() for exp in
                                                                                ['gex', 'vax', 'vex', 'vox']])
        if collect:
            exposure = exposure.collect(engine=ENGINE)
        return exposure

    @staticmethod
    def put_call_ratio(data: pl.DataFrame, freq: str, collect: bool = True) -> pl.DataFrame|pl.LazyFrame:
        if 'min' in freq:
            pl_freq = freq.replace('min', 'm')
        data.filter(pl.col('type') == 'put')
        pcr = data.filter(pl.col('type') == 'put').sort(by=[pl.col('underlying_name'), pl.col('type'),
                                                            pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('volume').sum())
        call = data.filter(pl.col('type') == 'call').sort(by=[pl.col('underlying_name'), pl.col('type'),
                                                              pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('volume').sum())
        pcr = pcr.join(call, on=['underlying_name', 'timestamp'], how='left', suffix='_call')
        pcr = pcr.with_columns((pl.col('volume')/pl.col('volume_call')).alias('pcr')).drop(
            ['type', 'volume', 'volume_call', 'type_call'])
        if collect:
            pcr = pcr.collect(engine=ENGINE)
        return pcr

    @staticmethod
    def spread(data: pl.DataFrame, freq: str, collect: bool = True) -> pl.DataFrame:
        if 'min' in freq:
            pl_freq = freq.replace('min', 'm')
        spread = data
        if collect:
            spread = spread.collect(engine=ENGINE)
        spread = spread.select(['underlying_name', 'timestamp', 'type', 'spread'])
        spread = spread.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('spread').mean())
        spread_put = spread.filter(pl.col('type') == 'put').rename({'spread': 'spread_put'}).drop('type')
        spread = spread.filter(pl.col('type') == 'call').rename({'spread': 'spread_call'}).drop('type')
        spread = spread.join(spread_put, on=['underlying_name', 'timestamp'])
        return spread

    @staticmethod
    def label_market_participants(data: pl.DataFrame, freq: str) -> pl.DataFrame:
        if 'min' in freq:
            pl_freq = freq.replace('min', 'm')
        mkt_participants = data.sort(by=[pl.col('underlying_name'), pl.col('type'),
                                         pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(
            pl.col('moneyness_label').mode())
        mkt_participants = mkt_participants.collect(engine=ENGINE).to_pandas()
        mkt_participants.moneyness_label = [participants[0] for participants in mkt_participants.moneyness_label]
        mkt_participants = pl.from_dataframe(mkt_participants)
        mkt_participants_call = mkt_participants.filter(pl.col('type') == 'call').drop('type')
        mkt_participants_put = mkt_participants.filter(pl.col('type') == 'put').drop('type')
        mkt_participants = \
            mkt_participants_call.with_columns(pl.when(pl.col('moneyness_label').is_in(['ATM', 'ITM']))
                                               .then(pl.lit('informed_traders_call'))
                                               .when(pl.col('moneyness_label') == 'OTM')
                                               .then(pl.lit('institutional_call'))
                                               .otherwise(pl.lit('retail_call')).alias('market_participants_call'))
        mkt_participants = mkt_participants.drop('moneyness_label')
        mkt_participants_put = \
            mkt_participants_put.with_columns(pl.when(pl.col("moneyness_label").is_in(['ATM', 'ITM']))
                                              .then(pl.lit('informed_traders_put'))
                                              .when(pl.col('moneyness_label') == 'OTM')
                                              .then(pl.lit('institutional_put'))
                                              .otherwise(pl.lit('retail_put')).alias('market_participants_put'))
        mkt_participants_put = mkt_participants_put.drop('moneyness_label')
        mkt_participants = mkt_participants.join(mkt_participants_put, on=['underlying_name', 'timestamp'])
        return mkt_participants


class Regressor:

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def model_train(self, features: Union[pd.DataFrame, np.ndarray],
                    target: Union[pd.DataFrame, np.ndarray], study_name: str, **kwargs):
        study = optuna.create_study(direction='minimize', study_name=study_name, sampler=RandomSampler(123))
        objective = partial(self.objective, train_X=features, train_y=target)
        study.optimize(objective, n_trials=N_TRIALS // 20, callbacks=[self.callback], show_progress_bar=True,
                       n_jobs=N_JOBS)
        return study

    @staticmethod
    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key='best_estimator', value=trial.user_attrs['best_estimator'])

    @staticmethod
    def objective(trial: optuna.trial.Trial, train_X: Union[pd.DataFrame, np.ndarray],
                  train_y: Union[pd.Series, np.ndarray]) -> float:
        train_dates = list(np.unique(train_X.index.get_level_values(1)))
        train_dates, valid_dates = train_dates[:int(.8*len(train_dates))], train_dates[int(.8*len(train_dates)):]
        train_X, valid_X = train_X.loc[train_X.index.get_level_values(1).isin(train_dates), :], \
            train_X.loc[train_X.index.get_level_values(1).isin(valid_dates), :]
        train_y, valid_y = train_y.loc[train_y.index.get_level_values(1).isin(train_dates)], \
            train_y.loc[train_y.index.get_level_values(1).isin(valid_dates)]
        if 'elastic' in trial.study.study_name:
            alpha = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', low=.1, high=.9, step=.1)
            rres = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=3_000, fit_intercept=True, selection='random',
                              tol=1e-2, random_state=123)
            rres.fit(train_X, train_y)
        elif ('lightgbm' in trial.study.study_name) | ('booster' in trial.study.study_name):
            train_loader, valid_loader = lgb.Dataset(train_X, label=train_y), lgb.Dataset(valid_X, label=valid_y)
            param = {'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                     'max_depth': trial.suggest_int('max_depth', 1, 3), 'tree_learner': 'data',
                     'lr': trial.suggest_float(name='lr', low=0.01, high=0.1, log=True),
                     'feature_fraction': trial.suggest_float('feature_fraction', low=.1, high=.5, log=True),
                     'boosting_type': 'goss', 'early_stopping_round': STOPPING_ROUNDS,
                     'num_threads': os.cpu_count()//2, 'force_row_wise': False,
                     'max_bin': 32, 'min_data_in_leaf': 20, 'feature_pre_filter': False, 'pre_partition': True,
                     'lambda_l2': trial.suggest_categorical(name='lambda_l2', choices=np.linspace(.1, .9, 9)),
            }
            rres = lgb.train(param, train_set=train_loader, valid_sets=[valid_loader],
                             num_boost_round=NUM_BOOST_ROUND//10,
                             callbacks=[lgb.early_stopping(STOPPING_ROUNDS, first_metric_only=True, verbose=True)])
        loss = mean_squared_error(train_y.values.reshape(-1, 1), rres.predict(train_X).reshape(-1, 1))
        trial.set_user_attr(key='best_estimator', value=rres)
        trial.set_user_attr(key='best_score', value=loss)
        return loss


class LassoRegressor(Regressor):

    def __init__(self, name: str = 'lasso'):
        super(LassoRegressor, self).__init__(name=name)

    def model_train(self, features: Union[pd.DataFrame, np.ndarray], target: Union[pd.DataFrame, np.ndarray],
                    **kwargs) -> LassoCV:
        rres = LassoCV(alphas=np.linspace(0.1, 0.9, 9, dtype=np.float64), cv=tscv, random_state=123, max_iter=MAX_ITER,
                       n_jobs=os.cpu_count()//2)
        rres.fit(features, target)
        return rres


class ElasticRegressor(Regressor):

    def __init__(self, name: str = 'elastic'):
        super(ElasticRegressor, self).__init__(name=name)
    # def model_train(self, features: Union[pd.DataFrame, np.ndarray], target: Union[pd.DataFrame, np.ndarray],
    #                 **kwargs) -> ElasticNet:
    #     rres = ElasticNet(l1_ratio=np.linspace(1e-1, 5e-1, 5),
    #                         cv=tscv, random_state=123, max_iter=5_000, n_jobs=-1, n_alphas=5)
    #     rres.fit(features, target)
    #     return rres


class BoostingRegressor(Regressor):

    def __init__(self, name: str = 'lightgbm'):
        super(BoostingRegressor, self).__init__(name=name)


class LSTMRegressor(Regressor):
    def __init__(self, name: str = 'lstm', **kwargs):
        super(LSTMRegressor, self).__init__(name=name)
        if len(kwargs) > 0:
            self._model = LSTMModel(**kwargs)
        self._minmaxscaler = dict()

    @property
    def minmaxscaler(self) -> Dict[(str, MinMaxScaler)]:
        return self._minmaxscaler

    @property
    def model(self) -> LSTMModel:
        return self._model

    @model.setter
    def model(self, model: LSTMModel) -> None:
        self._model = model

    def model_train(self, train_X: pd.DataFrame, train_y: pd.Series, valid_X: pd.DataFrame, valid_y: pd.Series,
                    positiveness: bool,  cols_ignore_in_scaling: List[str], n_trials: int = N_TRIALS,
                    max_epochs: int = EPOCH) -> None:
        params_list = list(ParameterSampler(PARAMS_LSTM, n_iter=n_trials, random_state=123))
        for symbol in train_X.index.get_level_values(0).unique():
            self._minmaxscaler[symbol] = MinMaxScaler().set_output(transform='pandas')
            train_X.loc[
                train_X.index.get_level_values(0) == symbol, ~train_X.columns.isin(cols_ignore_in_scaling)] = \
                self._minmaxscaler[symbol].fit_transform(X=train_X.loc[train_X.index.get_level_values(0) == symbol,
                ~train_X.columns.isin(cols_ignore_in_scaling)])
            valid_X.loc[
                valid_X.index.get_level_values(0) == symbol, ~valid_X.columns.isin(cols_ignore_in_scaling)] = \
                self._minmaxscaler[symbol].transform(
                    X=valid_X.loc[
                        valid_X.index.get_level_values(0) == symbol, ~valid_X.columns.isin(cols_ignore_in_scaling)])
        train_dataset = TimeSeriesDataset(train_X.values, train_y.values)
        valid_dataset = TimeSeriesDataset(valid_X.values, valid_y.values)
        train_loader = DataLoader(train_dataset, batch_size=1_024, num_workers=7, persistent_workers=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1_024, num_workers=7, persistent_workers=True)
        best_valid_loss = float('inf')
        best_params = None
        best_model = None
        for trial, params in enumerate(params_list):
            print(f'\nTrial {trial + 1}/{n_trials}')
            print(f'Parameters: {params}')
            model = LSTMModel(input_size=train_X.shape[1], hidden_size=params['hidden_size'], output_size=1,
                              num_layers=params['num_layers'], learning_rate=params['learning_rate'],
                              beta=params['beta'], feature_names=train_X.columns.tolist(), positiveness=positiveness,
                              dropout=params['dropout'])
            trainer = \
                p_light.Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping(monitor='valid_loss', patience=5)],
                                logger=TensorBoardLogger('lightning_logs', name=f'trial_{trial}'),
                                enable_progress_bar=True, num_sanity_val_steps=0)
            trainer.fit(model, train_loader, valid_loader)
            valid_loss = trainer.callback_metrics.get('valid_loss', float('inf'))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_params = params
                best_model = model
            print(f'Trial {trial + 1} completed. Validation loss: {valid_loss:.4f}')
        print('\nBest parameters:', best_params)
        print(f'Best validation loss: {best_valid_loss:.2f}')
        self._model = best_model

    @staticmethod
    def model_fine_tune(train_fine_tune_X: pd.DataFrame, train_fine_tune_y: pd.Series, model: LSTMModel,
                        date: datetime, L: str, target: str, n_trials: int = N_TRIALS,
                        max_epochs: int = EPOCH) -> LSTMModel:
        print(f'Fine tuning: '
              f'{model.__class__.__name__.lower()}_{L}_{target}_{date.strftime("%Y-%m-%d")} has started...............')
        model.eval()
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
                print(f'Weights in {name} layer has been frozen')
        train_dataset = TimeSeriesDataset(train_fine_tune_X.values, train_fine_tune_y.values)
        best_train_loss = float('inf')
        best_params = None
        best_model = None
        params_list = list(ParameterSampler(PARAMS_LSTM, n_iter=n_trials, random_state=123))
        for trial, params in enumerate(params_list):
            print(f"\nTrial {trial + 1}/{n_trials}")
            print(f"Parameters: {params}")
            train_loader = \
                DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=7, persistent_workers=True)
            trainer = \
                p_light.Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping(monitor='train_loss', patience=5)],
                                logger=TensorBoardLogger('lightning_logs', name=f'trial_{trial}'),
                                enable_progress_bar=True)
            trainer.fit(model, train_loader)
            train_loss = trainer.callback_metrics.get('train_loss', float('inf'))
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_params = params
                best_model = model
            print(f'Trial {trial + 1} completed. Validation loss: {train_loss:.4f}')
        print(f'\nBest parameters:', best_params)
        print(f'Best training loss: {best_train_loss:.2f}')
        print(f'Fine tuning: {model.__class__.__name__.lower()}_{L}_{target}_{date.strftime("%Y-%m-%d")} completed.')
        return best_model


class PermutationFeatureImportance:
    def __init__(self, test: Union[pd.DataFrame, np.ndarray, DataLoader] = None,
                 model: Union[LassoRegressor, ElasticRegressor, BoostingRegressor, LSTMRegressor] = None,
                 test_y: pd.Series = None, train_y: pd.Series = None,  freq: str = None):
        self._model = model
        self._test = test
        self._test_y = test_y
        self._train_y = train_y
        self.score = dict()
        self._freq = freq

    @property
    def model(self) -> Union[LassoRegressor, BoostingRegressor, LSTMRegressor]:
        return self._model

    @model.setter
    def model(self, model: Union[LassoRegressor, BoostingRegressor, LSTMRegressor]) -> None:
        self._model = model


    @property
    def test(self) -> Union[pd.DataFrame, np.ndarray, DataLoader]:
        return self._test

    @test.setter
    def test(self, test: Union[pd.DataFrame, np.ndarray, DataLoader]) -> None:
        self._test = test

    @property
    def test_y(self) -> pd.Series:
        return self._test_y

    @test_y.setter
    def test_y(self, test_y: pd.Series) -> None:
        self._test_y = test_y

    @property
    def train_y(self) -> pd.Series:
        return self._train_y

    @train_y.setter
    def train_y(self, train_y: pd.Series) -> None:
        self._train_y = train_y

    def compute_score(self) -> pd.Series:
        avg1s = self._train_y.groupby(by=pd.Grouper(level=0))
        avg1s = avg1s.apply(lambda x: (x - x.shift()).abs().sum() / (x.shape[0] - 1))
        feature_importance = dict([(feature, 0) for feature in self._test.columns])
        count = 0
        while count < 50:
            for feature in self._test.columns:
                shuffled_test, feature = shuffle_single_feature(self._test, feature)
                self._test_y.iloc[:, 1] = \
                    self.model.predict(shuffled_test) if not isinstance(self.model, LSTMModel) else \
                        self.model(torch.from_numpy(
                            shuffled_test.values.reshape(1, shuffled_test.values.shape[0],
                                                         shuffled_test.values.shape[1])).to(torch.float32)
                                   ).flatten().unsqueeze(0).detach().numpy()[0]
                feature_importance[feature] += \
                (self._test_y.diff(axis=1).iloc[:, -1].div(avg1s)).abs().groupby(by=pd.Grouper(level=0)).mean().mean()
            count += 1
        feature_importance = pd.Series(feature_importance)/50
        return feature_importance


class PnLExplained:

    def __init__(self, underlying_returns: pd.DataFrame, underlying_rv: pd.DataFrame):
        self._returns = underlying_returns.set_index('timestamp')
        self._rv = underlying_rv.set_index('timestamp')

    def resample_moments(self, freq: str) -> None:
        self._returns = self._returns.resample(freq).sum()
        self._rv = self._rv.resample(freq).sum()

    def pnl_analysis(self, pnl: pd.DataFrame, ranking: pd.DataFrame,
                     features: pd.DataFrame, targets: pd.DataFrame, freq: str) -> Tuple[pd.DataFrame]:
        self.resample_moments(freq)
        rv_mkt = pd.qcut(self._rv.mean(axis=1), q=[0, .45, .9, 1], labels=VOL_REGIMES)
        rv_mkt.name = 'volatility_regime'
        ranking['features'] = ranking.features.str.replace(r'^is_(BTCUSDT|ETHUSDT|XRPUSDT|BNBUSDT|SOLUSDT)$', r'\1',
                                                           regex=True)
        ranking['features'] = \
            ['_'.join(name.split('_')[:-1]+[name.split('_')[-1].upper()])
             if ('_1m' in name) | ('_1w' in name) | ('_1d' in name) else name for name in ranking['features']]
        factor_analysis = ranking.groupby(by=[pd.Grouper(key='volatility_regime'), pd.Grouper(key='D'),
                                              pd.Grouper(key='target')],
                                          group_keys=True).apply(
            lambda x, data, targets, pnl: run_pnl_ols(x, data, targets, pnl), data=features,
            targets=targets.xs('1D', axis=1, level=-1), pnl=pnl).unstack()
        tail_analysis = pnl.stack([0, 1])
        tail_analysis.name = 'P&L'
        tail_analysis = pd.DataFrame(tail_analysis)
        tail_analysis['P&L'] = tail_analysis['P&L']+np.random.uniform(0, 1e-8, tail_analysis.shape[0])
        tail_analysis['P&L decile'] = \
            tail_analysis.groupby(by=[pd.Grouper(level='target')], group_keys=False)['P&L'].apply(
                lambda x: pd.qcut(x, q=10, labels=list(range(1, 11))))
        tail_analysis = tail_analysis.reset_index([0, 1, 3, 4]).join(rv_mkt)
        tail_analysis = tail_analysis.groupby(
            by=[pd.Grouper(key='P&L decile'), pd.Grouper(key='volatility_regime'), pd.Grouper(key='D'),
                pd.Grouper(key='target'), pd.Grouper(key='aggregated')]).apply(
            lambda x, ranking, data, targets: run_pnl_decile_ols(x, ranking, data, targets),
            ranking=ranking, data=features, targets=targets).unstack().dropna()
        return factor_analysis, tail_analysis





