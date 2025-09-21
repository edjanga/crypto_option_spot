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
import pytorch_lightning as pl
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
    def returns_moments(data: pd.DataFrame, freq: str, upperbound: int = 6) -> pd.DataFrame:
        """
        :param lowerbound:  lowerbound (inclusive)
        :param upperbound:  upperbound (inclusive)
        """
        rv = (data.copy() ** 2).groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=freq)]).sum()
        returns = data.groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=freq)]).sum()
        ret_orders = pd.concat([returns, rv.rename(columns={'returns': 'm2'})], axis=1)
        K = pd.to_timedelta('1D')//pd.to_timedelta(freq)
        for moment_mth in range(3, upperbound+1):
            tmp_ret_order = \
                (returns ** moment_mth).groupby(by=[pd.Grouper(level='symbol'), pd.Grouper(level='timestamp',
                                                                                           freq=freq)]).sum()
            tmp_ret_order = tmp_ret_order * (K / (rv ** moment_mth)) ** (1 / 2)
            ret_orders = pd.concat([ret_orders, tmp_ret_order.rename(columns={'returns': f'm{moment_mth}'})], axis=1)
        return ret_orders

    @staticmethod
    def rv_lags(data: pd.DataFrame, freq: str, D: str) -> pd.DataFrame:
        all_lags = INTRADAY_LAGS + INTERDAY_LAGS
        lags = all_lags[all_lags.index(freq):all_lags.index(D) + 1]
        symbol_group = data.stack().rename('RV').droplevel(axis=0, level=2).to_frame()
        tmp = symbol_group.copy()
        lag_dfs = [tmp.copy()]
        symbol_grouped = tmp.groupby(level='symbol', group_keys=False)
        for lag in lags:
            if lag in INTRADAY_LAGS:
                window_size = int(pd.to_timedelta(lag) // pd.to_timedelta('5min')) + 1
                rv_lag = \
                    symbol_grouped['RV'].transform(lambda x: x.rolling(window=window_size).mean()).rename(f'RV_{lag}')
            else:
                grouped = \
                    tmp.groupby([pd.Grouper(level='symbol'), pd.Grouper(level='timestamp', freq=lag)],
                                group_keys=False)['RV'].mean()
                rv_lag = (grouped.groupby(level='symbol').shift().reindex(tmp.index).ffill().rename(f'RV_{lag}'))
            lag_dfs.append(rv_lag.to_frame())
        result = pd.concat(lag_dfs, axis=1)
        return result

    @staticmethod
    def define_market_participants(data: pd.DataFrame) -> pd.DataFrame:
        data.loc[:, 'market_participants'] = 'retail'
        data.loc[data.moneyness_label == 'OTM', 'market_participants'] = 'institutions'
        data.loc[data.moneyness_label.isin(['ATM', 'ITM', 'DITM']), 'market_participants'] = 'informed_traders'
        return data

    @staticmethod
    def compute_exposure(data: pd.DataFrame) -> pd.DataFrame:
        for exp, greek in GREEKS_EXPOSURE.items():
            if exp == 'dex':
                data[exp] = data['netDealerExposure'].mul(data[GREEKS_EXPOSURE[exp]]).mul(data['underlying_price'])
            elif exp == 'gex':
                data[exp] = data['netDealerExposure'].mul(data[GREEKS_EXPOSURE[exp]]).mul(data['underlying_price'] ** 2)
            elif exp == 'vex':
                data[exp] = data['gex'].mul(data['tau']).mul(data['mark_iv'] ** 2)
            elif exp == 'vax':
                data[exp] = data['gex'].mul(np.log(data['strike_price'].div(data['underlying_price'])) +
                                            .5 * (data['tau'].mul(data['mark_iv'] ** 2)))
            elif exp == 'vox':
                data[exp] = data['vax'].mul(-np.log(data['strike_price'].div(data['underlying_price'])) -
                                            .5 * (data['tau'].mul(data['mark_iv'] ** 2)))
        return data

    @staticmethod
    def put_call_ratio(data: pd.DataFrame, freq: str) -> pd.Series:
        put_call_ratio = \
            data.groupby(by=[pd.Grouper(key='underlying_name'),
                             pd.Grouper(key='type'),
                             pd.Grouper(level='timestamp', freq=freq, sort=True)])[['volume']].sum()
        calls, puts = put_call_ratio.xs('call', axis=0, level=1), put_call_ratio.xs('put', axis=0, level=1)
        return puts.div(calls).rename(columns={'volume': 'pcr'})

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
    def volatility_skew(data: pd.DataFrame, freq: str) -> pd.DataFrame:
        moneyness_data = data.copy()
        volatility_skew = list()
        dotm = FeatureBuilder.avg_volatility(data=moneyness_data, moneyness_label='DOTM', freq=freq)
        for idx, label in enumerate(LABELS[1:]):
            tmp = FeatureBuilder.avg_volatility(data=moneyness_data, moneyness_label=label, freq=freq)
            tmp = dotm.subtract(tmp).rename(columns={option: f'{option}_{label}' for option in tmp.columns})
            volatility_skew.append(tmp)
        return pd.concat(volatility_skew, axis=1)

    @staticmethod
    def label_market_participants(data: pd.DataFrame, freq: str) -> pd.DataFrame:
        mkt_participants = data.rename(columns={'underlying_name': 'symbol'}).groupby(
            [pd.Grouper(key='symbol'), pd.Grouper(level='timestamp', sort=True, freq=freq)]
        )['market_participants'].apply(lambda x: x.mode().iloc[0])
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
            trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping(monitor='valid_loss', patience=5)],
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
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=7, persistent_workers=True)
            trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping(monitor='train_loss', patience=5)],
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








