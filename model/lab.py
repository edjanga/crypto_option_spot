import concurrent.futures
from constants import *
import pandas as pd
import torch
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from typing import List, Union
import itertools
import optuna
from optuna.samplers import RandomSampler
from functools import partial
import lightgbm as lgb
from torch.utils.data import DataLoader
from .lstm_room import TimeSeriesDataset, LSTMModel
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import ParameterSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
import pdb

DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
RESAMPLING_MAT_MONEYNESS = ['gex', 'vax', 'vex', 'vox', 'volume', 'gex_exp', 'vax_exp', 'vex_exp', 'vox_exp']
RESAMPLING_MAT_MONEYNESS = {feature: 'sum' for feature in RESAMPLING_MAT_MONEYNESS}
RESAMPLING_MAT_MONEYNESS.update({feature: 'last' for feature in ['tau', 'moneyness', 'mark_iv']})
resampling = RESAMPLING.copy()
for key in ['ask_iv', 'bid_iv', 'rho', 'theta']:
    resampling.pop(key)
mkt_participants_resampling = {'volume': 'sum', 'mark_iv': 'mean'}


class FeatureBuilder:
    def __init__(self):
        pass

    @staticmethod
    def iv_mat_moneyness(tags: List[List[str]], data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()
        for idx, tag in enumerate(tags):
            tag_mark_iv = data.query(f'tau == "{tag[0]}" & moneyness == "{tag[1]}" & level_3 == "gex"')
            tag_mark_iv = tag_mark_iv.drop(['tau', 'moneyness', 'level_3'], axis=1).set_index('timestamp')
            tag_mark_iv = tag_mark_iv.rename(columns={'mark_iv': f'mark_iv_{tag[0]}_{tag[1]}'})
            features = \
                features.join(tag_mark_iv, how={True: 'left', False: 'right'}[features.shape[0] > tag_mark_iv.shape[0]])
        features.ffill(inplace=True)
        return features

    @staticmethod
    def dealer_exposure(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features = pd.DataFrame()
        if 'market_participants' in data.columns:
            tags = data[['market_participants']].values.tolist()
            tags.sort()
            tags = list(tag for tag, _ in itertools.groupby(tags))
            for idx, tag in enumerate(tags):
                tag_exp = data.query(f'market_participants == "{tag[0]}"')[['exp', 'timestamp']]
                tag_exp = tag_exp.groupby(by=pd.Grouper(key='timestamp')).mean()
                tag_exp = tag_exp.rename(columns={'exp': f'exp_{tag[0]}'})
                features = pd.concat([features, tag_exp], axis=1)
        else:
            tags = kwargs['tags']
            for idx, tag in enumerate(tags):
                tag_exp = data.query(f'tau == "{tag[0]}" & moneyness == "{tag[1]}"')
                tag_exp['tag'] = \
                    [f'{row["level_3"]}_{row["tau"]}_{row["moneyness"]}' for _, row in
                     tag_exp[['level_3', 'tau', 'moneyness']].iterrows()]
                tag_exp.drop(['tau', 'moneyness', 'level_3'], axis=1, inplace=True)
                tag_exp = tag_exp.groupby(by=[pd.Grouper(key='tag'), pd.Grouper(key='timestamp')]).mean().reset_index()
                tag_exp = pd.pivot_table(data=tag_exp, columns='tag', values='exp', index='timestamp')
                features = \
                    features.join(tag_exp, how={True: 'left', False: 'right'}[features.shape[0] > tag_exp.shape[0]])
            features.ffill(inplace=True)
        return features

    @staticmethod
    def option_imbalance(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features = pd.DataFrame()
        if 'market_participants' in data.columns:
            tags = data[['market_participants']].values.tolist()
            tags.sort()
            tags = list(tag for tag, _ in itertools.groupby(tags))
            for idx, tag in enumerate(tags):
                tag_volume = data.query(f'market_participants == "{tag[0]}"')[['volume', 'timestamp']]
                tag_volume = tag_volume.groupby(by=pd.Grouper(key='timestamp')).sum()
                tag_volume = tag_volume.rename(columns={'volume': f'volume_{tag[0]}'})
                features = pd.concat([features, tag_volume], axis=1)
        else:
            tags = kwargs['tags']
            for idx, tag in enumerate(tags):
                tag_volume = data.query(f'tau == "{tag[0]}" & moneyness == "{tag[1]}" & level_3 == "gex"')
                tag_volume = tag_volume.drop(['tau', 'moneyness', 'level_3'], axis=1).set_index('timestamp')
                tag_volume = tag_volume.rename(columns={'volume': f'volume_{tag[0]}_{tag[1]}'})
                features = \
                    features.join(tag_volume,
                                  how={True: 'left', False: 'right'}[features.shape[0] > tag_volume.shape[0]])
        features.ffill(inplace=True)
        return features

    @staticmethod
    def returns_moments(data: pd.DataFrame, freq: str, lowerbound: int = 2, upperbound: int = 6) -> pd.DataFrame:
        """
        :param lowerbound:  lowerbound (inclusive)
        :param upperbound:  upperbound (inclusive)
        """
        symbol = data.columns[0]
        returns = (np.log(data.div(data.shift())))
        rv = (returns**2).resample(freq).sum().rename(columns={symbol: 'RV'})
        returns = returns.resample(freq).sum().rename(columns={symbol: 'returns'})
        ret_orders = pd.DataFrame()
        K = pd.to_timedelta('1D')//pd.to_timedelta(freq)
        for moment_mth in range({True: lowerbound + 1, False: lowerbound}[lowerbound == 2], upperbound+1):
            tmp_ret_order = (np.log(data.div(data.shift())) ** moment_mth).resample(freq).sum()
            tmp_ret_order = tmp_ret_order.div(np.sqrt(rv**moment_mth).values).mul(K ** ((moment_mth-2)/2))
            ret_orders = pd.concat([ret_orders, tmp_ret_order], axis=1)
        ret_orders.columns = [f'm{moment_mth}' for moment_mth in range(lowerbound + 1, upperbound+1)]
        ret_orders = pd.concat([ret_orders, rv, returns], axis=1) if lowerbound == 2 else ret_orders
        return ret_orders

    @staticmethod
    def define_market_participants(data: pd.DataFrame) -> pd.DataFrame:
        tmp = data.copy()
        tmp.loc[:, 'market_participants'] = 'retail'
        tmp.loc[data.moneyness == 'OTM', 'market_participants'] = 'institutions'
        tmp.loc[data.moneyness.isin(['ATM', 'ITM', 'DITM']), 'market_participants'] = 'informed_traders'
        return tmp

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
    def put_call_ratio(data: pd.DataFrame) -> pd.Series:
        put_call_ratio = \
            data.groupby(by=[pd.Grouper(key='type'), pd.Grouper(level='timestamp')])[['volume']].sum()
        put_call_ratio.reset_index(level=0, inplace=True)
        put_call_ratio = \
            put_call_ratio.query('type == "put"').volume.div(put_call_ratio.query('type == "call"').volume)
        put_call_ratio.name = 'put_call_ratio'
        return put_call_ratio

    @staticmethod
    def avg_volatility(data: pd.DataFrame, moneyness_label: str, freq: str) -> pd.Series:
        vol = data.query(f'moneyness == "{moneyness_label}"').drop('moneyness', axis=1)
        vol['weighted_mark_iv'] = vol['mark_iv'].mul(vol['volume'])
        vol_weighted_mark_iv = \
            vol.groupby([pd.Grouper(key='type'),
                         pd.Grouper(level='timestamp', sort=True, freq=freq)])['weighted_mark_iv'].sum()
        vol_volume = \
            vol.groupby([pd.Grouper(key='type'), pd.Grouper(level='timestamp', sort=True, freq=freq)])['volume'].sum()
        vol = vol_weighted_mark_iv.div(vol_volume).reset_index()
        vol = \
            pd.concat([vol.query(f'type == "call"').pivot(index='timestamp', columns='type', values=0),
                       vol.query(f'type == "put"').pivot(index='timestamp', columns='type', values=0)], axis=1)
        return vol

    @staticmethod
    def volatility_skew(data: pd.DataFrame, freq: str) -> pd.DataFrame:
        moneyness_data = data.copy()
        volatility_skew = list()
        dotm = FeatureBuilder.avg_volatility(data=moneyness_data, moneyness_label='DOTM', freq=freq)
        for idx, label in enumerate(LABELS[1:]):
            tmp = FeatureBuilder.avg_volatility(data=moneyness_data, moneyness_label=label, freq=freq)
            tmp = dotm.subtract(tmp).rename(columns={option: f'{option}_DOTM_{label}' for option in tmp.columns})
            volatility_skew.append(tmp)
        return pd.concat(volatility_skew, axis=1)

    @staticmethod
    def mkt_participants_pivot_table(group: pd.Grouper) -> pd.DataFrame:
        data = group.agg(mkt_participants_resampling).reset_index('type')
        data = \
            pd.concat([data.query(f'type == "{option_type}"').drop('type', axis=1).rename(
                columns={col: f'{col}_{option_type}' for col in ['volume', 'mark_iv']}) for
                option_type in ['call', 'put']], axis=1).reset_index('market_participants')
        data = \
            pd.concat([data.query(f'market_participants == "{mkt_participants}"').drop(
                'market_participants', axis=1).rename(columns={col: f'{mkt_participants}_{col}' for
                                                               col in data.columns}) for
                       mkt_participants in MKT_PARTICIPANTS], axis=1)
        return data

    @staticmethod
    def lags(data: pd.DataFrame, min_lag: str, max_lag: str):
        lags_features = INTRADAY_LAGS+INTERDAY_LAGS
        lags_features = lags_features[lags_features.index(min_lag):lags_features.index(max_lag)]
        tmp = data.drop(['symbol', 'RV', 'returns'], axis=1)
        for lag in lags_features:
            if lag in INTRADAY_LAGS:
                data = data.join(tmp.shift(periods=1, freq=lag), how='left', rsuffix=f'_{lag}')
            else:
                data = data.join(tmp.rolling(pd.to_timedelta(DAYS_CONVERTER[lag], unit='D')).mean(), how='left',
                                 rsuffix=f'_{lag}')
        return data.filter(regex='RV|returns|symbol|_([0-9]|[1-9][0-9])([a-zA-Z])')


class Regressor:

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def model_train(self, features: Union[pd.DataFrame, np.ndarray],
                    target: Union[pd.DataFrame, np.ndarray], study_name: str):
        study = optuna.create_study(direction='minimize', study_name=study_name, sampler=RandomSampler(123))
        objective = partial(self.objective, train_X=features, train_y=target)
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[self.callback], show_progress_bar=True,
                       n_jobs=N_JOBS)
        return study

    @staticmethod
    def model_fine_tune(train_fine_tune_X: pd.DataFrame, train_fine_tune_y: pd.Series,
                        model: Union[Lasso, lgb.LGBMRegressor], n_trials: int,
                        L: str, target: str, date: datetime, **kwargs):
        study = optuna.create_study(direction='minimize',
                                    study_name=f'Fine tuning: {model.__class__.__name__.lower()}_{L}_{target}_'
                                               f'{date.strftime("%Y-%m-%d")}',
                                    sampler=RandomSampler(123))
        study.set_user_attr(key='best_estimator', value=model)
        objective = partial(Regressor.objective, train_X=train_fine_tune_X, train_y=train_fine_tune_y)
        study.optimize(objective, n_trials=n_trials, callbacks=[Regressor.callback], show_progress_bar=True,
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
        if 'lasso' in trial.study.study_name:
            param = {'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                     'alpha': trial.suggest_float('alpha', 1e-1, 3e-1, log=False)
                     }
            rres = Lasso(alpha=param['alpha'], fit_intercept=True, random_state=123, max_iter=MAX_ITER,
                         warm_start=True, selection='random') if len(trial.study.user_attrs) == 0 else \
                Lasso(**trial.study.user_attrs['best_estimator'].get_params())
            rres.fit(train_X, train_y.values.reshape(-1, 1))
        elif ('lightgbm' in trial.study.study_name) | ('booster' in trial.study.study_name):
            train_loader, valid_loader = lgb.Dataset(train_X, label=train_y), lgb.Dataset(valid_X, label=valid_y)
            param = {
                'objective': 'regression', 'metric': 'mse', 'verbosity': -1,
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'lr': trial.suggest_categorical(name='lr', choices=[.01, .05, .1]),
                'tree_learner': 'data',
                'feature_fraction': trial.suggest_float('feature_fraction', .1, .8, step=.1, log=False),
                'extra_tree': True, 'boosting_type': 'goss', 'early_stopping_round': STOPPING_ROUNDS
            }
            if 'Fine' not in trial.study.study_name:
                rres = lgb.train(param, train_set=train_loader, valid_sets=[valid_loader],
                                 num_boost_round=NUM_BOOST_ROUND,
                                 callbacks=[lgb.early_stopping(STOPPING_ROUNDS, first_metric_only=True, verbose=True,
                                                               min_delta=0.0)])
            else:
                rres = lgb.train(params=param, init_model=trial.study.user_attrs['best_estimator'],
                                 train_set=train_loader, valid_sets=[valid_loader], num_boost_round=NUM_BOOST_ROUND,
                                 callbacks=[lgb.early_stopping(STOPPING_ROUNDS, first_metric_only=True, verbose=True,
                                                               min_delta=0.0)])
        loss = mean_squared_error(train_y.values.reshape(-1, 1), rres.predict(train_X).reshape(-1, 1))
        trial.set_user_attr(key='best_estimator', value=rres)
        trial.set_user_attr(key='best_score', value=loss)
        return loss


class LassoRegressor(Regressor):

    def __init__(self, name: str = 'lasso'):
        super(LassoRegressor, self).__init__(name=name)


class BoostingRegressor(Regressor):

    def __init__(self, name: str = 'lightgbm'):
        super(BoostingRegressor, self).__init__(name=name)


class LSTMRegressor(Regressor):
    def __init__(self, name: str = 'lstm', **kwargs):
        super(LSTMRegressor, self).__init__(name=name)
        if len(kwargs) > 0:
            self._model = LSTMModel(**kwargs)
        self._maxabsscaler = MaxAbsScaler()

    @property
    def maxabsscaler(self) -> MaxAbsScaler:
        return self._maxabsscaler

    @property
    def model(self) -> LSTMModel:
        return self._model

    @model.setter
    def model(self, model: LSTMModel) -> None:
        self._model = model

    def model_train(self, train_X: pd.DataFrame, train_y: pd.Series, valid_X: pd.DataFrame, valid_y: pd.Series,
                    n_trials: int = N_TRIALS, max_epochs: int = EPOCH) -> None:
        train_X.iloc[:, :-2] = self._maxabsscaler.fit_transform(X=train_X.iloc[:, :-2])
        valid_X.iloc[:, :-2] = self._maxabsscaler.transform(X=valid_X.iloc[:, :-2])
        train_dataset = TimeSeriesDataset(train_X.values, train_y.values)
        valid_dataset = TimeSeriesDataset(valid_X.values, valid_y.values)
        best_valid_loss = float('inf')
        best_params = None
        best_model = None
        params_list = list(ParameterSampler(PARAMS_LSTM, n_iter=n_trials, random_state=123))
        for trial, params in enumerate(params_list):
            print(f'\nTrial {trial + 1}/{n_trials}')
            print(f'Parameters: {params}')
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=7, persistent_workers=True)
            valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=7, persistent_workers=True)
            model = LSTMModel(input_size=train_X.shape[1], hidden_size=params['hidden_size'], output_size=1,
                              num_layers=params['num_layers'], learning_rate=params['learning_rate'],
                              feature_names=train_X.columns.tolist())
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

    def model_fine_tune(self, train_fine_tune_X: pd.DataFrame, train_fine_tune_y: pd.Series, model: LSTMModel,
                        date: datetime, L: str, target: str, n_trials: int = N_TRIALS,
                        max_epochs: int = EPOCH) -> LSTMModel:
        print(f'Fine tuning: {model.__class__.__name__.lower()}_{L}_{target}_{date.strftime("%Y-%m-%d")} has started..')
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
                print(f'Network in {name} layer has been frozen')
        train_fine_tune_X.iloc[:, :-2] = self._maxabsscaler.fit_transform(X=train_fine_tune_X.iloc[:, :-2])
        train_dataset = TimeSeriesDataset(train_fine_tune_X.values, train_fine_tune_y.values)
        best_train_loss = float('inf')
        best_params = None
        best_model = None
        params_list = list(ParameterSampler(PARAMS_LSTM, n_iter=n_trials, random_state=123))
        for trial, params in enumerate(params_list):
            print(f"\nTrial {trial + 1}/{n_trials}")
            print(f"Parameters: {params}")
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=7, persistent_workers=True)
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
                 test_y: pd.Series = None, train_y: pd.Series = None):
        self._model = model
        self._test = test
        self._test_y = test_y
        self._train_y = train_y
        self.score = dict()
        self.mase_score = MeanAbsoluteScaledError()

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

    def shuffle_single_feature(self, feature: str) -> pd.DataFrame:
        if isinstance(self._test, pd.DataFrame):
            shuffled_test = self._test.copy()
        shuffled_test[feature] = np.random.permutation(shuffled_test[feature].values)
        return shuffled_test

    def compute_score(self) -> pd.Series:
        feature_importance = dict([(feature, list()) for feature in self._test.columns])
        for feature in tqdm(self._test.columns):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(lambda x: x) for _ in range(0, N_SHUFFLES)]
                for _ in concurrent.futures.as_completed(futures):
                    shuffled_test = self.shuffle_single_feature(feature)
                    if isinstance(self._model, LSTMModel):
                        shuffled_test = \
                            torch.from_numpy(shuffled_test.values.reshape((shuffled_test.shape[0], 1,
                                                                           shuffled_test.shape[1])).astype(np.float32))
                    self._test_y.iloc[:, 1] = self._model.predict(shuffled_test) \
                        if isinstance(self._model, (Lasso, lgb.basic.Booster)) else \
                        self._model(shuffled_test).detach().numpy()
                    feature_importance[feature].append(
                        self.mase_score(y_true=self._test_y.iloc[:, 0].droplevel(axis=0, level=0).values,
                                        y_pred=self._test_y.iloc[:, 1].droplevel(axis=0, level=0).values,
                                        y_train=self._train_y.droplevel(axis=0, level=0).values)
                    )
        return pd.DataFrame(feature_importance).mean()






