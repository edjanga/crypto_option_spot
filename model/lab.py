import os
import pdb
from constants import *
from itertools import product
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from typing import List, Union, Dict, Tuple, Optional, Type
import optuna
from optuna.samplers import RandomSampler
from functools import partial
import lightgbm as lgb
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import polars as pl
import pytorch_lightning as p_light
from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
import statsmodels.api as sm
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')
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
    X = pnl.reset_index(0).swaplevel(0, 1).join(X, on=['underlying_name', 'timestamp']).set_index('aggregated',
                                                                                                  append=True)
    X['P&L'] = X['P&L'] + np.random.uniform(0, 1e-8, X.shape[0])
    X = X.apply(lambda x: pd.to_numeric(x, errors='coerce').astype('float32'))
    X = (X - X.min()).div(X.max()-X.min()).replace([-np.inf, np.inf], np.nan).dropna()
    params = sm.OLS(X['P&L'], sm.add_constant(X.drop(columns=['P&L']))).fit().params
    params.index = ['α'] + [f'β<sub>{i}</sub>' for i in range(1, params.shape[0])]
    return params


class FeatureBuilder:
    def __init__(self):
        pass

    @staticmethod
    def returns_moments(data: pl.DataFrame | pl.LazyFrame, freq: str, upperbound: int = 4) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        pl_freq = freq.replace('in', '') if 'min' in freq else freq
        data = data.sort(['underlying_name', 'timestamp'])
        aggs = [pl.sum('returns').alias('returns')]
        K = (pd.to_timedelta('1D') // pd.to_timedelta(freq))
        for m in range(2, upperbound + 1):
            aggs.append((pl.col('returns') ** m).sum().alias(f'm{m}'))
            if m == 2:
                aggs.append((pl.col('returns') ** 2).sum().alias('RV'))
        moments = data.group_by_dynamic(index_column='timestamp', every=pl_freq, by='underlying_name').agg(aggs)
        for m in range(1, upperbound - 1):
            moments = moments.with_columns((((pl.lit(K) ** m / pl.col('m2')) ** (m + 2)) ** 0.5).alias(f'm{m+2}'))
        return moments.collect()

    @staticmethod
    def lags(data: pl.DataFrame, features: List[str] = ['RV']+[f'm{i}' for i in range(2, 5)]) -> pl.LazyFrame:
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
        pl_freq = freq.replace('min', 'm')
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
        exposure = exposure.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by='underlying_name').agg([pl.col(exp).sum() for exp in
                                                                                ['gex', 'vax', 'vex', 'vox']])
        if collect:
            exposure = exposure.collect()
        return exposure

    @staticmethod
    def put_call_ratio(data: pl.DataFrame, freq: str, collect: bool = True) -> pl.DataFrame|pl.LazyFrame:
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
            pcr = pcr.collect()
        return pcr

    @staticmethod
    def spread(data: pl.DataFrame, freq: str, collect: bool = True) -> pl.DataFrame:
        pl_freq = freq.replace('min', 'm')
        spread = data
        if collect:
            spread = spread.collect()
        spread = spread.select(['underlying_name', 'timestamp', 'type', 'spread'])
        spread = spread.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('spread').mean())
        spread_put = spread.filter(pl.col('type') == 'put').rename({'spread': 'spread_put'}).drop('type')
        spread = spread.filter(pl.col('type') == 'call').rename({'spread': 'spread_call'}).drop('type')
        spread = spread.join(spread_put, on=['underlying_name', 'timestamp'])
        return spread

    @staticmethod
    def label_market_participants(data: pl.DataFrame, freq: str) -> pl.DataFrame:
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
        self._standardscaler: Dict[str, StandardScaler] = {}
        self._model: Union[LassoRegressor, ElasticRegressor, BoostingRegressor] = None

    @property
    def name(self):
        return self._name

    @property
    def standardscaler(self):
        return self._standardscaler

    @standardscaler.setter
    def standardscaler(self, standardscaler: Dict[str, StandardScaler]) -> None:
        self._standardscaler = standardscaler

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: object):
        self._model = model

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
        train_X = train_X.replace([-np.inf, np.inf], [np.nan, np.nan]).dropna()
        train_y = train_y.loc[train_X.index]
        train_X, valid_X = train_X.loc[train_X.index.get_level_values(1).isin(train_dates), :], \
            train_X.loc[train_X.index.get_level_values(1).isin(valid_dates), :]
        train_y, valid_y = train_y.loc[train_y.index.get_level_values(1).isin(train_dates)], \
            train_y.loc[train_y.index.get_level_values(1).isin(valid_dates)]
        if 'lasso' in trial.study.study_name:
            alpha = trial.suggest_float('alpha', 1e-4, 1.0, log=True)
            rres = Lasso(alpha=alpha)
            rres.fit(train_X, train_y)
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


class ElasticRegressor(Regressor):

    def __init__(self, name: str = 'elastic'):
        super(ElasticRegressor, self).__init__(name=name)


class BoostingRegressor(Regressor):

    def __init__(self, name: str = 'lightgbm'):
        super(BoostingRegressor, self).__init__(name=name)


class TimeSeriesDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return self.X[index:index + self.seq_len], self.y[index + self.seq_len - 1]


class DLRegressor:
    def __init__(self, name: str):
        self.name = name
        self._model: Optional[p_light.LightningModule] = None
        self._standardscaler: Dict[str, StandardScaler] = {}

    @property
    def model(self):
        return self._model

    @property
    def standardscaler(self):
        return self._standardscaler

    def _prepare_data(self, df: pd.DataFrame, cols_ignore: List[str], is_train: bool = True) -> pd.DataFrame:
        df_scaled = df.copy()
        idx = pd.IndexSlice
        scaling_cols = [c for c in df.columns if c not in cols_ignore]
        for symbol in df.index.get_level_values(0).unique():
            if is_train:
                self._standardscaler[symbol] = StandardScaler().set_output(transform='pandas')
                df_scaled.loc[idx[symbol, :], scaling_cols] = \
                    self._standardscaler[symbol].fit_transform(df.loc[idx[symbol, :], scaling_cols])
            elif symbol in self._standardscaler:
                df_scaled.loc[idx[symbol, :], scaling_cols] = \
                    self._standardscaler[symbol].transform(df.loc[idx[symbol, :], scaling_cols])
        return df_scaled

    def model_train(self, train_X: pd.DataFrame, train_y: pd.Series, valid_X: pd.DataFrame, valid_y: pd.Series,
                    positiveness: bool, cols_ignore: List[str], n_trials: int, target: str,
                    study_name: str) -> None:
        X_train_s = self._prepare_data(train_X, cols_ignore, is_train=True)
        X_valid_s = self._prepare_data(valid_X, cols_ignore, is_train=False)

        def objective(trial: optuna.Trial):
            params = {
                'hidden_size': trial.suggest_categorical('hidden_size', PARAMS_DL['hidden_size']),
                'num_layers': trial.suggest_categorical('num_layers', PARAMS_DL['num_layers']),
                'learning_rate': trial.suggest_categorical('learning_rate', PARAMS_DL['learning_rate']),
                'dropout': trial.suggest_categorical('dropout', PARAMS_DL['dropout']),
                'batch_size': trial.suggest_categorical('batch_size', PARAMS_DL['batch_size']),
            }
            if positiveness is True:
                params.update({'beta': trial.suggest_categorical('beta', PARAMS_DL['beta'])})
            if self.name == 'gru':
                model = GRUModel(input_size=train_X.shape[1], hidden_size=params['hidden_size'],
                                 num_layers=params['num_layers'], lr=params['learning_rate'],
                                 positiveness=target == 'RV',
                                 dropout=params['dropout'] if params['num_layers'] > 1 else 0.0)
            pruning_cb = PyTorchLightningPruningCallback(trial, monitor='valid_loss')
            early_stop_cb = EarlyStopping(monitor='valid_loss', patience=3)
            accelerator = 'gpu' if 'cuda' in DEVICE else 'cpu'
            precision = 16 if accelerator == 'gpu' else 32
            trainer = p_light.Trainer(max_epochs=EPOCH // 10, accelerator=accelerator,
                                      devices=1, precision=precision, callbacks=[pruning_cb, early_stop_cb],
                                      logger=False, enable_progress_bar=False, check_val_every_n_epoch=1,
                                      enable_checkpointing=False)
            train_loader = DataLoader(TimeSeriesDataset(X_train_s.values, train_y.values),
                                      batch_size=params['batch_size'], num_workers=8, pin_memory=True,
                                      persistent_workers=True)
            valid_loader = DataLoader(TimeSeriesDataset(X_valid_s.values, valid_y.values),
                                      batch_size=params['batch_size'], num_workers=8, pin_memory=True,
                                      persistent_workers=True)
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
            metrics = trainer.callback_metrics
            val_loss = metrics.get('valid_loss')
            if val_loss is None:
                return float('inf')
            return val_loss.item()
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(), study_name=study_name)
        study.optimize(objective, n_trials=n_trials)
        print(f'Best trial parameters: {study.best_params}')
        if self.name == 'gru':
            self._model = \
                GRUModel(input_size=train_X.shape[1], hidden_size=study.best_params['hidden_size'],
                         num_layers=study.best_params['num_layers'], lr=study.best_params['learning_rate'],
                         dropout=study.best_params['dropout'] if study.best_params['num_layers'] > 1 else 0.0,
                         positiveness=target == 'RV')

    def model_fine_tune(self, train_X: pd.DataFrame, train_y: pd.Series, epochs: int = 5) -> None:
        if self._model is None:
            return
        for name, param in self._model.named_parameters():
            param.requires_grad = name.startswith('fc')
        if hasattr(self._model, 'lr'):
            self._model.lr *= 0.1
        loader = DataLoader(TimeSeriesDataset(train_X.values, train_y.values), batch_size=min(1024, len(train_X)),
                            num_workers=8, pin_memory=True)
        accelerator = 'gpu' if 'cuda' in DEVICE else 'cpu'
        precision = 16 if accelerator == 'gpu' else 32
        trainer = p_light.Trainer(max_epochs=epochs, accelerator=accelerator, devices=1, precision=precision,
                                  logger=False, enable_progress_bar=False)
        trainer.fit(self._model, loader)


class GRUModel(p_light.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, lr: float,
                 output_size: int = 1, positiveness: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.positiveness = positiveness

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        if self.positiveness:
            out = F.softplus(out)
        return out.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5, fused=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'valid_loss', 'interval': 'epoch', 'frequency': 1}
                }


class GRURegressor(DLRegressor):
    def __init__(self, name: str = 'gru'):
        super(GRURegressor, self).__init__(name=name)


class LSTMRegressor(DLRegressor):
    def __init__(self, name: str = 'lstm'):
        super(LSTMRegressor, self).__init__(name=name)


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
        self._test = self._test.replace([-np.inf, np.inf], [np.finfo(np.float32).min, np.finfo(np.float32).max])
        while count < 50:
            for feature in self._test.columns:
                shuffled_test, feature = shuffle_single_feature(self._test, feature)
                if not isinstance(self.model, Union[GRURegressor, LSTMRegressor]):
                    self._test_y.iloc[:, 1] = self.model.model.predict(shuffled_test)
                else:
                    tmp_shuffle = torch.from_numpy(
                        shuffled_test.values.reshape(shuffled_test.values.shape[0], 1, shuffled_test.values.shape[1]))
                    self._test_y.iloc[:, 1] = self.model.model(tmp_shuffle).detach().cpu().numpy()
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
        return factor_analysis





