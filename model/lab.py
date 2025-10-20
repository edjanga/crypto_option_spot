import concurrent.futures
import pdb
from constants import *
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
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
        moments = \
            data.sort(by=[pl.col('symbol'), pl.col('timestamp')]).group_by_dynamic(index_column='timestamp',
                                                                                   every=freq, by='symbol').sum()
        K = (pd.to_timedelta('1D') // pd.to_timedelta(freq))
        for m in range(2, upperbound + 1):
            if m > 2:
                tmp = data.sort(by=[pl.col('symbol'), pl.col('timestamp')]).group_by_dynamic(
                    index_column='timestamp', every=freq, by='symbol').agg([(pl.col('returns') ** 2).sum().alias(f'm2'),
                                                                            (pl.col('returns') ** m).sum().alias(
                                                                                f'm{m}')])
                tmp = tmp.with_columns((((K / pl.col(f'm2')) ** m) ** .5).alias(f'm{m}')).drop('m2')
            else:
                tmp = data.sort(by=[pl.col('symbol'), pl.col('timestamp')]).group_by_dynamic(
                    index_column='timestamp', every=freq,
                    by='symbol').agg([(pl.col('returns') ** m).sum().alias(f'm{m}')])
            moments = moments.join(tmp, on=['symbol', 'timestamp'])
        moments = \
            moments.with_columns([
                pl.when(pl.col(m) == float("inf")).then(np.finfo(np.float32).max).when(
                    pl.col(m) == -float("inf")).then(np.finfo(np.float32).min).otherwise(pl.col(m)).alias(m) for m in
                ['returns']+[f'm{m}' for m in range(2, upperbound + 1)]])
        return moments.collect(engine=ENGINE)

    @staticmethod
    def rv_lags(data: pl.DataFrame, freq: str, D: str):
        all_lags = INTRADAY_LAGS + INTERDAY_LAGS
        start_idx = all_lags.index(freq)
        end_idx = all_lags.index(D) + 1
        lags = all_lags[start_idx:end_idx]
        intraday_lags_exprs = [pl.col('RV')]
        for lag in lags:
            if lag in INTRADAY_LAGS:
                window_size = int(pd.to_timedelta(lag) // pd.to_timedelta('5m'))
                intraday_lags_exprs.append(pl.col('RV').rolling_mean(window_size).over('symbol').alias(f'RV_{lag}'))
            elif lag in INTERDAY_LAGS:
                pl_lag = f'{lag[:-1]}mo' if 'M' in lag else lag.lower()
                if 'rv_lags' not in vars():
                    rv_lags = data.with_columns(intraday_lags_exprs)
                rv_interday_lag = (data.group_by_dynamic(index_column='timestamp', every=pl_lag.lower(), by='symbol')
                                   .agg(pl.col('RV').mean().alias(f'RV_mean'))
                                   .with_columns(pl.col('RV_mean').shift(1).over('symbol').alias(f'RV_{lag}'))
                                   .drop('RV_mean'))
                rv_lags = rv_lags.join(rv_interday_lag, on=['symbol', 'timestamp'], how='left')
        rv_lags = \
            rv_lags.with_columns([pl.col(f'RV_{lag}').fill_null(strategy='forward').over('symbol') for lag in lags])
        return rv_lags

    @staticmethod
    def define_market_participants(data: pl.DataFrame) -> pd.DataFrame:
        data = data.with_columns(
            pl.when(pl.col('moneyness_label').is_in(['ATM', 'ITM', 'DITM']))
            .then(pl.lit('informed_traders')).when(pl.col('moneyness_label') == 'OTM')
            .then(pl.lit('institutions')).otherwise(pl.lit('retail')).alias('market_participants'))
        return data

    @staticmethod
    def compute_exposure(data: pl.DataFrame, freq: str) -> pd.DataFrame:
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
        return exposure.collect(engine=ENGINE)

    @staticmethod
    def put_call_ratio(data: pl.DataFrame, freq: str) -> pl.DataFrame:
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
            ['type', 'volume', 'volume_call', 'type_call']).collect(engine=ENGINE)
        return pcr

    @staticmethod
    def spread(data: pl.DataFrame, freq: str) -> pl.DataFrame:
        if 'min' in freq:
            pl_freq = freq.replace('min', 'm')
        spread = data.collect(engine=ENGINE).select(['underlying_name', 'timestamp', 'type', 'spread'])
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
    def volatility_skew(data: pl.DataFrame, freq: str) -> pl.DataFrame:
        skew = data
        if 'min' in freq:
            pl_freq = freq.replace('min', 'm')
        dotm = data.filter(pl.col('moneyness_label') == 'DOTM')
        dotm = dotm.sort(by=[pl.col('underlying_name'), pl.col('type'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('mark_iv').mean())
        for idx, label in enumerate(LABELS[1:]):
            tmp = data.filter(pl.col('moneyness_label') == label)
            tmp = tmp.sort(by=[pl.col('underlying_name'), pl.col('type'), pl.col('timestamp')]).group_by_dynamic(
                index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('mark_iv').mean())
            tmp = tmp.join(dotm, on=['underlying_name', 'type', 'timestamp'], how='left', suffix=f'_DOTM')
            tmp = tmp.with_columns((pl.col('mark_iv') - pl.col('mark_iv_DOTM')).alias(f'{label}')).select(
                ['underlying_name', 'timestamp', f'{label}', 'type'])
            skew = skew.join(tmp, on=['underlying_name', 'type', 'timestamp'], how='left')
        skew = skew.select(['timestamp', 'underlying_name', 'type', 'OTM', 'ATM', 'ITM', 'DITM'])
        skew = skew.sort(by=[pl.col('underlying_name'), pl.col('type'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']
        ).agg([pl.col(f'{label}').mean() for label in ['OTM', 'ATM', 'ITM', 'DITM']])
        skew_call = skew.filter(pl.col('type') == 'call').drop('type').rename({col: f'call_{col}' for col in
                                                                               ['OTM', 'ATM', 'ITM', 'DITM']})
        skew_put = skew.filter(pl.col('type') == 'put').drop('type').rename({col: f'put_{col}' for col in
                                                                             ['OTM', 'ATM', 'ITM', 'DITM']})
        skew = skew_call.join(skew_put, on=['underlying_name', 'timestamp'])
        return skew.collect(engine=ENGINE)

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
            mkt_participants_call.with_columns(pl.when(pl.col("moneyness_label").is_in(['ATM', 'ITM']))
                                               .then(pl.lit('informed_traders_call'))
                                               .when(pl.col('moneyness_label') == 'OTM')
                                               .then(pl.lit('institutional_call'))
                                               .otherwise(pl.lit('retail_call')).alias('market_participants_call'))
        mkt_participants = mkt_participants.drop('moneyness_label')
        mkt_participants_put = \
            mkt_participants_put.with_columns(pl.when(pl.col("moneyness_label").is_in(['ATM', 'ITM']))
                                              .then(pl.lit('informed_traders_put'))
                                              .when(pl.col('moneyness_label') == 'OTM')
                                              .then(pl.lit('institutional_call'))
                                              .otherwise(pl.lit('retail_call')).alias('market_participants_put'))
        mkt_participants_put = mkt_participants_put.drop('moneyness_label')
        mkt_participants = mkt_participants.join(mkt_participants_put, on=['underlying_name', 'timestamp'])
        return mkt_participants

    @staticmethod
    def lags(data: pd.DataFrame, min_lag: str, max_lag: str):
        if isinstance(data.index, pd.core.indexes.multi.MultiIndex):
            data = data.droplevel(0)
        lags_features = INTRADAY_LAGS+INTERDAY_LAGS
        lags_features = lags_features[lags_features.index(min_lag):lags_features.index(max_lag)+1]
        tmp = data.copy()
        for lag in lags_features:
            if lag in INTRADAY_LAGS:
                data = data.join(tmp.rolling(window=lag).mean().shift(), how='left', rsuffix=f'_{lag}')
            else:
                data = \
                    data.join(
                        tmp.resample({False: lag,
                                      True: lag.replace('M', 'ME')}['M' in lag]).mean().shift().reindex(data.index),
                        how='left', rsuffix=f'_{lag}')
        data = data.ffill()
        return data.filter(regex='RV|returns|symbol|_([0-9]|[1-9][0-9])([a-zA-Z])')


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
                 model: Union[LassoRegressor, BoostingRegressor, LSTMRegressor] = None,
                 test_y: pd.Series = None, train_y: pd.Series = None,  freq: str = None):
        self._model = model
        self._test = test
        self._test_y = test_y
        self._train_y = train_y
        self.score = dict()
        # self.mase_score = MeanAbsoluteScaledError()
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

    def compute_score(self):
        mase = dict([(symbol, dict([(feature, list()) for feature in self._test.columns])) for symbol in SYMBOLS])
        with concurrent.futures.ProcessPoolExecutor(4) as executor:
            futures = [executor.submit(shuffle_single_feature, self._test, feature) for feature in
                       self._test.columns]
            for future in concurrent.futures.as_completed(futures):
                shuffled_test, feature = future.result()
                if isinstance(self.model, LSTMModel):
                    shuffled_test = torch.from_numpy(
                        shuffled_test.values.reshape((shuffled_test.shape[0], 1,
                                                      shuffled_test.shape[1])).astype(np.float32))
                if isinstance(self.model, (LassoCV, ElasticNetCV, lgb.basic.Booster)):
                    y_hat = pd.Series(self.model.predict(shuffled_test))
                else:
                    y_hat = pd.Series(self.model(shuffled_test).detach().numpy().reshape(shuffled_test.shape[0]))
                y_hat.index = self._test.index
                y_hat = y_hat.groupby(by=[pd.Grouper(level='symbol'),
                                          pd.Grouper(level='timestamp', sort=True, freq=self._freq)]).sum()
                y_train = self._train_y.groupby(by=[pd.Grouper(level='symbol'),
                                                    pd.Grouper(level='timestamp', sort=True, freq=self._freq)]).sum()
                self._test_y = \
                    self._test_y.groupby(by=[pd.Grouper(level='symbol'),
                                             pd.Grouper(level='timestamp', sort=True, freq=self._freq)]).sum()
                self._test_y.iloc[:, 1] = y_hat.values
                for symbol in self._test_y.index.get_level_values(0).unique():
                    y_true = self._test_y.xs(symbol, 0, 0).iloc[:, 0].values
                    y_pred = self._test_y.xs(symbol, 0, 0).iloc[:, 1].values
                    avg1s = \
                        sum(np.abs(y_train.xs(symbol, 0, 0).values[1:]-y_train.xs(symbol, 0, 0).values[:-1]))/\
                        (y_train.xs(symbol, 0, 0).values.shape[0]-1)
                    mase[symbol][feature].append(sum(np.abs((y_true - y_pred)/avg1s)))
        for symbol in mase.keys():
            mase[symbol] = pd.DataFrame(mase[symbol])
        return pd.concat(mase).droplevel(axis=0, level=1)


class PnLExplained:

    def __init__(self, data: pd.DataFrame, ):
        pass










