import pdb

import numpy as np
import pytz
from constants import *
from itertools import product
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV, ElasticNetCV
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

tscv = TimeSeriesSplit(n_splits=5)


def shuffle_single_feature(test, feature: str) -> Tuple[pd.DataFrame, str]:
    np.random.seed(123)
    if isinstance(test, pd.DataFrame):
        shuffled_test = test.copy()
    for symbol in shuffled_test.index.get_level_values(0).unique():
        feature_value = np.random.permutation(shuffled_test.xs(symbol, axis=0, level=0)[feature].values)
        shuffled_test.loc[shuffled_test.index.get_level_values(0) == symbol, feature] = feature_value
    return shuffled_test, feature


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
    def avg_volatility(data: pd.DataFrame, moneyness_label: str, freq: str) -> pd.Series:
        vol = data.query(f'moneyness_label == "{moneyness_label}"').drop('moneyness_label', axis=1)
        vol['weighted_mark_iv'] = vol['mark_iv'].mul(vol['volume'])
        vol_weighted_mark_iv = \
            vol.groupby([pd.Grouper(key='underlying_name'),
                         pd.Grouper(key='type'),
                         pd.Grouper(level='timestamp', sort=True, freq=freq)])['weighted_mark_iv'].sum()
        vol_volume = \
            vol.groupby(([pd.Grouper(key='underlying_name'),
                         pd.Grouper(key='type'),
                         pd.Grouper(level='timestamp', sort=True, freq=freq)]))['volume'].sum()
        return vol_weighted_mark_iv.div(vol_volume).unstack(level=1)

    @staticmethod
    def volatility_skew(data: pl.LazyFrame, freq: str,
                        collect: bool = True) -> Union[Tuple[pl.DataFrame], Tuple[pl.LazyFrame]]:
        if 'min' in freq:
            pl_freq = freq.replace('min', 'm')
        dotm = data.filter(pl.col('moneyness_label') == 'DOTM')
        dotm = dotm.sort(by=[pl.col('underlying_name'), pl.col('type'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq,
            by=['underlying_name', 'type']).agg(pl.col('mark_iv').mean().alias('DOTM'))
        monyness_label_frames = []
        for monyness_label in LABELS[1:]:
            tmp = (data.filter(pl.col('moneyness_label') == monyness_label))
            tmp = tmp.sort(by=[pl.col('underlying_name'), pl.col('type'), pl.col('timestamp')])
            tmp = tmp.group_by_dynamic(index_column='timestamp', every=pl_freq, by=['underlying_name', 'type'])\
                .agg(pl.col('mark_iv').mean().alias(f'{monyness_label}'))\
                .join(dotm, on=['underlying_name', 'type', 'timestamp'], how='left')\
                .with_columns((pl.col(f'{monyness_label}') - pl.col('DOTM')).alias(monyness_label))\
                .select(["underlying_name", "timestamp", "type", monyness_label])
            monyness_label_frames.append(tmp)
        skew = monyness_label_frames[0]
        for tmp in monyness_label_frames[1:]:
            skew = skew.join(tmp, on=['underlying_name', 'type', 'timestamp'], how='left')
        skew_call = (skew.filter(pl.col('type') == 'call').drop('type')
                     .rename({monyness_label: f'call_{monyness_label}' for monyness_label in LABELS[1:]}))
        skew_put = (skew.filter(pl.col('type') == 'put').drop('type')
                    .rename({monyness_label: f'put{monyness_label}' for monyness_label in LABELS[1:]}))
        if collect:
            return skew_call.collect(engine=ENGINE), skew_put.collect(engine=ENGINE)
        return skew_call, skew_put

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
        if ('lightgbm' in trial.study.study_name) | ('booster' in trial.study.study_name):
            train_loader, valid_loader = lgb.Dataset(train_X, label=train_y), lgb.Dataset(valid_X, label=valid_y)
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'lr': trial.suggest_categorical(name='lr', choices=[.01, .05, .1]),
                'tree_learner': 'data',
                'feature_fraction': trial.suggest_float('feature_fraction', .1, .5, step=.1, log=False),
                'extra_tree': True, 'boosting_type': 'goss', 'early_stopping_round': STOPPING_ROUNDS,
                'num_threads': 0, 'force_row_wise': torch.mps.is_available(), 'max_bin': 255,
                'min_data_in_leaf': 20, 'lambda_l2': trial.suggest_categorical(name='lambda_l2',
                                                                               choices=np.linspace(.1, .9, 9)),
                'feature_pre_filter': False
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
        rres = \
            LassoCV(alphas=np.linspace(0.1, 0.9, 9, dtype=np.float64), cv=tscv, random_state=123, max_iter=MAX_ITER,
                    n_jobs=-1)
        rres.fit(features, target)
        return rres


class ElasticRegressor(Regressor):

    def __init__(self, name: str = 'elastic'):
        super(ElasticRegressor, self).__init__(name=name)

    def model_train(self, features: Union[pd.DataFrame, np.ndarray], target: Union[pd.DataFrame, np.ndarray],
                    **kwargs) -> ElasticNetCV:
        rres = ElasticNetCV(alphas=np.linspace(0.1, 0.9, 9, dtype=np.float64), l1_ratio=np.linspace(1e-1, 5e-1, 5),
                            cv=tscv, random_state=123, max_iter=MAX_ITER, n_jobs=-1)
        rres.fit(features, target)
        return rres


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
                              beta=params['beta'], feature_names=train_X.columns.tolist(), positiveness=positiveness)
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
        print(f'Fine tuning: {model.__class__.__name__.lower()}_{L}_{target}_{date.strftime("%Y-%m-%d")} has started..')
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
                    self.model.predict(shuffled_test) if not isinstance(self.model, LSTMModel) else pdb.set_trace()
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
                     features: pd.DataFrame, targets: pd.DataFrame, freq: str, moment: str) -> None:
        self.resample_moments(freq)
        rv_mkt = pd.qcut(self._rv.mean(axis=1), q=[0, .45, .9, 1], labels=VOL_REGIMES)
        rv_mkt.name = 'volatility_regime'
        coefficients = dict()
        tmp_target = targets.xs(dict([('first', 'returns'), ('second', 'RV')])[moment], axis=0, level=0)
        fig_tails = make_subplots(rows=3, cols=3, shared_xaxes=True, shared_yaxes=True,
                                  column_titles=['low', 'normal', 'high'],
                                  row_titles=[f'<i>{d}</i>' for d in ['1d', '1w', '1m']])
        for d, vol in list(product(['1d', '1w', '1m'], ['Low', 'Normal', 'High'])):
            tmp_ranking = ranking.query(f'(D == "{d}") & '
                                        f'(target == "{dict([("first", "returns"), ("second", "RV")])[moment]}") & '
                                        f'(volatility_regime == "{vol}")')
            if tmp_ranking.empty:
                continue
            tmp_ranking = tmp_ranking.drop(['D', 'target'], axis=1)
            tmp_ranking['features'] = \
                tmp_ranking.features.str.replace(r"^is_(BTCUSDT|ETHUSDT|XRPUSDT|BNBUSDT|SOLUSDT)$", r"\1",
                                                 regex=True)
            list_factors = tmp_ranking.features.values.tolist()  # Order is the ranking
            try:
                tmp_features = features.filter(regex='|'.join(list_factors), axis=1)
                tmp_target = tmp_target.filter(regex='|'.join(list_factors), axis=1)
            except TypeError:
                continue
            tmp_pnl = \
                pd.melt(frame=pnl, ignore_index=False).set_index('underlying_name', append=True).swaplevel(0, 1)
            tmp_pnl = tmp_pnl.rename(columns={'value': 'P&L'}).dropna().query(
                f'D == "{dict([("first", d.upper()),("second", d)])[moment]}"').drop('D', axis=1)
            if tmp_pnl.empty:
                continue
            tmp_pnl.index = tmp_pnl.index.set_names(['underlying_name', 'timestamp'])
            tmp = tmp_pnl.join(tmp_target, on='timestamp').join(tmp_features).dropna()
            for analysis in ['factors', 'tails']:
                if analysis == 'factors':
                    tmp_factors = tmp.groupby(by=pd.Grouper(level=0), group_keys=False).apply(
                        lambda x: (x - x.max()).div(x.max() - x.min()))
                    tmp_factors = tmp_factors.join(rv_mkt, on='timestamp')
                    tmp_factors = sm.add_constant(tmp_factors).query(f'volatility_regime == "{vol}"')
                    coefficients[(d, vol)] = \
                        sm.OLS(tmp_factors['P&L'],
                               tmp_factors.drop(['P&L', 'volatility_regime'], axis=1)).fit().params
                    coefficients[(d, vol)].index = \
                        ['α'] + [f'β<sub>{i}</sub>' for i in range(1, coefficients[(d, vol)].shape[0])]
                elif analysis == 'tails':
                    zmin, zmax = np.inf, -np.inf
                    tmp = tmp.join(rv_mkt, on='timestamp')
                    if moment == 'first':
                        tmp['P&L decile'] = tmp.groupby(by=pd.Grouper(level=0), group_keys=False)['P&L'].apply(
                            lambda x: pd.qcut(x, q=10, labels=list(range(1, 11))))
                    else:
                        tmp['P&L decile'] = tmp.groupby(by=pd.Grouper(level=0), group_keys=False)['P&L'].apply(
                            lambda x: pd.cut(x, bins=10, labels=list(range(1, 11))))
                    tmp = tmp.query(f'volatility_regime == "{vol}"').drop('volatility_regime', axis=1)
                    heatmap = list()
                    for decile in range(1, 11):
                        tmp_decile = tmp.query(f'`P&L decile` == {decile}').drop('P&L decile', axis=1)
                        tmp_decile = \
                            tmp_decile.groupby(by=pd.Grouper(level=0), group_keys=False).apply(
                                lambda x: (x - x.max()).div(x.max() - x.min()))
                        tmp_decile = tmp_decile.replace([np.inf, -np.inf], np.nan).dropna()
                        tmp_decile = sm.add_constant(tmp_decile)
                        try:
                            coefficient_values = \
                                sm.OLS(tmp_decile['P&L'], tmp_decile.drop('P&L', axis=1)).fit().params.tolist()
                        except ValueError:
                            continue
                        heatmap.append(coefficient_values)
                        if zmin > min(coefficient_values):
                            zmin = min(coefficient_values)
                        if zmax < max(coefficient_values):
                            zmax = max(coefficient_values)
                    fig_tails.add_trace(
                        trace=go.Heatmap(z=heatmap, x=['α'] + [f'β<sub>{i}</sub>' for i in range(1, 6)],
                                         y=['1<sup>st</sup>', '2<sup>nd</sup>', '3<sup>rd</sup>'] +
                                           [f'{q}<sup>th</sup>' for q in range(4, 11)],
                                         colorscale='Viridis', zmin=zmin, zmax=zmax, showlegend=False),
                        col=['Low', 'Normal', 'High'].index(vol) + 1, row=['1d', '1w', '1m'].index(d) + 1)
        ############################################################################################################
        ## Factor analysis
        ############################################################################################################
        coefficients = pd.DataFrame(coefficients).unstack()
        try:
            coefficients.index = coefficients.index.set_names(['D', 'volatility_regime', 'Factors'])
        except ValueError:
            pdb.set_trace()
        coefficients = coefficients.dropna().reset_index().rename(columns={0: 'values'})
        coefficients['colour'] = np.where(coefficients['values'] > 0, 'green', 'red')
        fig_factors = \
            px.bar(data_frame=coefficients, y='values', x='Factors', facet_row='D',
                   facet_col='volatility_regime', color='colour',
                   color_discrete_map={'green': px.colors.qualitative.Plotly[2],
                                       'red': px.colors.qualitative.Plotly[1]},
                   category_orders={'volatility_regime': ['Low', 'Normal', 'High'],
                                    'Factors': ['α'] + [f'β<sub>{i}</sub>' for i in range(1, 6)]},
                   title=f'Aggregated P&L Attribution via Factor Analysis: '
                 f'{dict([(True, "Second-Moment Strategy"), (False, "First-Moment Strategy")])[moment == "second"]}')
        fig_factors.update_yaxes(matches=None, title='')
        fig_factors.for_each_annotation(lambda x: x.update({'text': f'{x["text"].split("=")[-1].lower()}'}),
                                        selector=lambda x: 'volatility_regime' in x['text'])
        fig_factors.for_each_annotation(lambda x: x.update({'text': f'<i>{x["text"].split("=")[-1]}</i>'}),
                                        selector=lambda x: 'D' in x['text'])
        fig_factors.update_layout(
            showlegend=False, font=dict(size=LABEL_AXIS_FONT_SIZE), title=dict(font=dict(size=30)))
        fig_factors.show()
        ############################################################################################################
        ## Tail analysis
        ############################################################################################################
        for idx, _ in enumerate(fig_tails.data):
            fig_tails.data[idx].zmin, fig_tails.data[idx].zmax = zmin, zmax
        fig_tails.update_layout(
            font=dict(size=LABEL_AXIS_FONT_SIZE),
            title=dict(
                text=f'Aggregated P&L Tail Analysis via Factor Analysis: '
                     f'{dict([(True, "Second-Moment Strategy"), (False, "First-Moment Strategy")])[moment=="second"]}',
                font=dict(size=30)))
        fig_tails.show()















        # if moment == 'first':
        #     for d, vol in list(product(['1d', '1w', '1m'], ['Low', 'Normal', 'High'])):
        #         tmp_ranking = ranking.query(f'(D == "{d}") & (target == "returns") & (volatility_regime == "{vol}")')
        #         tmp_ranking = tmp_ranking.drop(['D', 'target'], axis=1)
        #         tmp_ranking['features'] = \
        #             tmp_ranking.features.str.replace(r"^is_(BTCUSDT|ETHUSDT|XRPUSDT|BNBUSDT|SOLUSDT)$", r"\1",
        #                                              regex=True)
        #         list_factors = tmp_ranking.features.values.tolist() # Order is the ranking
        #         try:
        #             tmp_features = features.filter(regex='|'.join(list_factors), axis=1)
        #             tmp_target = tmp_target.filter(regex='|'.join(list_factors), axis=1)
        #         except TypeError:
        #             continue
        #         tmp_pnl = \
        #             pd.melt(frame=pnl, ignore_index=False).set_index('underlying_name', append=True).swaplevel(0, 1)
        #         tmp_pnl = tmp_pnl.rename(columns={'value': 'P&L'}).dropna().query(f'D == "{d}"').drop('D', axis=1)
        #         if tmp_pnl.empty:
        #             continue
        #         tmp_pnl.index = tmp_pnl.index.set_names(['underlying_name', 'timestamp'])
        #         tmp = tmp_pnl.join(tmp_target, on='timestamp').join(tmp_features).dropna()
        #         for analysis in ['factors', 'tails']:
        #             if analysis == 'factors':
        #                 tmp_factors = tmp.groupby(by=pd.Grouper(level=0), group_keys=False).apply(
        #                     lambda x: (x - x.max()).div(x.max() - x.min()))
        #                 tmp_factors = tmp_factors.join(rv_mkt, on='timestamp')
        #                 tmp_factors = sm.add_constant(tmp_factors).query(f'volatility_regime == "{vol}"')
        #                 coefficients[(d, vol)] = \
        #                     sm.OLS(tmp_factors['P&L'],
        #                            tmp_factors.drop(['P&L', 'volatility_regime'], axis=1)).fit().params
        #                 coefficients[(d, vol)].index = \
        #                     ['α']+[f'β<sub>{i}</sub>' for i in range(1, coefficients[(d, vol)].shape[0])]
        #             elif analysis == 'tails':
        #                 zmin, zmax = np.inf, -np.inf
        #                 tmp = tmp.join(rv_mkt, on='timestamp')
        #                 tmp['P&L decile'] = tmp.groupby(by=pd.Grouper(level=0), group_keys=False)['P&L'].apply(
        #                         lambda x: pd.qcut(x, q=10, labels=list(range(1, 11))))
        #                 tmp = tmp.query(f'volatility_regime == "{vol}"').drop('volatility_regime', axis=1)
        #                 heatmap = list()
        #                 for decile in range(1, 11):
        #                     pdb.set_trace()
        #                     tmp_decile = tmp.query(f'`P&L decile` == {decile}').drop('P&L decile', axis=1)
        #                     tmp_decile = \
        #                         tmp_decile.groupby(by=pd.Grouper(level=0), group_keys=False).apply(
        #                             lambda x: (x - x.max()).div(x.max() - x.min()))
        #                     tmp_decile = sm.add_constant(tmp_decile)
        #                     coefficient_values = \
        #                         sm.OLS(tmp_decile['P&L'], tmp_decile.drop('P&L', axis=1)).fit().params.tolist()
        #                     heatmap.append(coefficient_values)
        #                     if zmin > min(coefficient_values):
        #                         zmin = min(coefficient_values)
        #                     if zmax < max(coefficient_values):
        #                         zmax = max(coefficient_values)
        #                 fig_tails.add_trace(
        #                     trace=go.Heatmap(z=heatmap, x=['α'] + [f'β<sub>{i}</sub>' for i in range(1, 6)],
        #                                      y=['1<sup>st</sup>', '2<sup>nd</sup>', '3<sup>rd</sup>']+
        #                                        [f'{q}<sup>th</sup>' for q in range(4, 11)],
        #                                      colorscale='Viridis', zmin=zmin, zmax=zmax, showlegend=False),
        #                     col=['Low', 'Normal', 'High'].index(vol)+1, row=['1d', '1w', '1m'].index(d)+1)
        #     ############################################################################################################
        #     ## Factor analysis
        #     ############################################################################################################
        #     coefficients = pd.DataFrame(coefficients).unstack()
        #     try:
        #         coefficients.index = coefficients.index.set_names(['D', 'volatility_regime', 'Factors'])
        #     except ValueError:
        #         pdb.set_trace()
        #     coefficients = coefficients.dropna().reset_index().rename(columns={0: 'values'})
        #     coefficients['colour'] = np.where(coefficients['values'] > 0, 'green', 'red')
        #     fig_factors = \
        #         px.bar(data_frame=coefficients, y='values', x='Factors', facet_row='D', facet_col='volatility_regime',
        #                color='colour', color_discrete_map={'green': px.colors.qualitative.Plotly[2],
        #                                                    'red': px.colors.qualitative.Plotly[1]},
        #                category_orders={'volatility_regime': ['Low', 'Normal', 'High'],
        #                                 'Factors': ['α']+[f'β<sub>{i}</sub>' for i in range(1, 6)]},
        #                title=
        #                f'Aggregated P&L Attribution via Factor Analysis: '
        #                f'{dict([(True, "Second-Moment Strategy"),(False, "First-Moment Strategy")])[target=="RV"]}')
        #     fig_factors.update_yaxes(matches=None, title='')
        #     fig_factors.for_each_annotation(lambda x: x.update({'text': f'{x["text"].split("=")[-1].lower()}'}),
        #                                     selector=lambda x: 'volatility_regime' in x['text'])
        #     fig_factors.for_each_annotation(lambda x: x.update({'text': f'<i>{x["text"].split("=")[-1]}</i>'}),
        #                                     selector=lambda x: 'D' in x['text'])
        #     fig_factors.update_layout(
        #         showlegend=False, font=dict(size=LABEL_AXIS_FONT_SIZE), title=dict(font=dict(size=30)))
        #     fig_factors.show()
        #     ############################################################################################################
        #     ## Tail analysis
        #     ############################################################################################################
        #     if target == 'returns':
        #         for idx, _ in enumerate(fig_tails.data):
        #             fig_tails.data[idx].zmin, fig_tails.data[idx].zmax = zmin, zmax
        #         fig_tails.update_layout(
        #             font=dict(size=LABEL_AXIS_FONT_SIZE),
        #             title=dict(
        #                 text=f'Aggregated P&L Tail Analysis via Factor Analysis: '
        #                      f'{dict([(True, "Second-Moment Strategy"), (False, "First-Moment Strategy")])[target=="RV"]}',
        #                 font=dict(size=30)))
        #         # fig_tails.update_xaxes(title='Factors')
        #         # fig_tails.update_yaxes(title='Deciles')
        #         fig_tails.show()
        # elif moment == 'second':
        #
        # pdb.set_trace()
        # return None












