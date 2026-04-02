import argparse
import pdb

import polars as pl
import pytz
import sqlite3
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from constants import *
from model.lab import LassoRegressor, ElasticRegressor, BoostingRegressor,\
    GRURegressor, LSTMRegressor, PermutationFeatureImportance
import torch
import os

# CASTS = [pl.col('underlying_name').cast(pl.Utf8),
#          pl.col('timestamp').cast(pl.Utf8),#.str.to_datetime(format='%Y-%m-%d %H:%M:%S%z', strict=False).dt.replace_time_zone('UTC'),
#          pl.col('features').cast(pl.Utf8), pl.col('features_values').cast(pl.Float32)]

models = [LassoRegressor(), ElasticRegressor(), BoostingRegressor(), GRURegressor(), LSTMRegressor()]
models_factory_dd = {model.name.lower(): model for model in models}
LAGS = INTRADAY_LAGS + INTERDAY_LAGS
contemporanous_features = {'returns': ['m2', 'm3', 'm4', 'm5', 'm6'], 'RV': []}


def train_one_window(date, features, date_mask, model, args):
    train_X = features.filter(date_mask[date]).to_pandas().set_index(['underlying_name', 'timestamp']).dropna()
    numeric_cols = train_X.select_dtypes(include='float32').columns.tolist()
    numeric_cols.remove(args.target)
    study_name = f'{model.name}_{args.L}_{args.target}_{date}'
    if args.regression != 'lightgbm':
        tmp_train_X = list()
        for sym, tmp in train_X.groupby(level='underlying_name'):
            tmp_scaled = model.standardscaler[sym].fit_transform(tmp[numeric_cols])
            tmp = tmp.copy()
            tmp[numeric_cols] = tmp_scaled
            tmp_train_X.append(tmp)
        train_X = pd.concat(tmp_train_X)
    train_y = train_X.pop(args.target)
    if args.regression in ['lasso', 'elastic', 'lightgbm']:
        res = model.model_train(features=train_X, target=train_y, study_name=study_name)
        model.model = res.user_attrs['best_estimator']
    else:
        valid_X = train_X.groupby(level='underlying_name', group_keys=False).apply(
            lambda x: x.iloc[int(.8 * x.shape[0]):])
        valid_y = train_y.groupby(level='underlying_name', group_keys=False).apply(
            lambda x: x.iloc[int(.8 * x.shape[0]):])
        train_X = train_X.loc[~train_X.index.isin(valid_X.index), :]
        train_X = train_X.groupby(level='underlying_name', group_keys=False).apply(lambda x: x.bfill())
        train_y = train_y.loc[~train_y.index.isin(valid_y.index), :]
        model.model_train(train_X=train_X, train_y=train_y, valid_X=valid_X, valid_y=valid_y,
                          n_trials=N_TRIALS // 20, positiveness=(args.target == 'RV'), cols_include=numeric_cols,
                          target=args.target, study_name=study_name)
    return date, model


def evaluate_one_window(date, model, features, train_features, date_mask, args):
    test_X = features.filter(date_mask[date]).to_pandas().set_index(['underlying_name', 'timestamp'])
    numeric_cols = test_X.select_dtypes(include='float32').columns.tolist()
    numeric_cols.remove(args.target)
    if args.regression != 'lightgbm':
        tmp_test_X = list()
        for sym, tmp in test_X.groupby(level='underlying_name'):
            tmp_scaled = model.standardscaler[sym].transform(tmp[numeric_cols])
            tmp = tmp.copy()
            tmp[numeric_cols] = tmp_scaled
            tmp_test_X.append(tmp)
        test_X = pd.concat(tmp_test_X)
    test_y = test_X.pop(args.target)
    if model.name in ['gru', 'lstm']:
        X_tensor = torch.from_numpy(test_X.values.reshape(test_X.shape[0], 1, test_X.shape[1]).astype(np.float32))
    y_hat = model.model.predict(test_X) if hasattr(model.model, 'predict') else \
        model.model(X_tensor).detach().numpy().ravel()
    y_hat = pd.Series(y_hat, index=test_y.index)
    if args.regression != 'lightgbm':
        means = pd.Series({k: v.mean_[0] for k, v in model.standardscaler.items()})
        stds = pd.Series({k: np.sqrt(v.var_[0]) for k, v in model.standardscaler.items()})
        sym = y_hat.index.get_level_values(0)
        y_hat = y_hat * sym.map(stds) + sym.map(means)
    forecast_df = pd.concat([test_y.rename('y'), y_hat.rename('yhat')], axis=1)
    in_sample_avg = \
        train_features.to_pandas().groupby('underlying_name').apply(
            lambda x: ((x-x.shift()).abs().sum())/(x.shape[0]-1))
    mase_mean = \
        forecast_df.diff(axis=1).dropna(how='all', axis=1).div(
            in_sample_avg.rename(columns={args.target: 'yhat'})).abs()
    mase_mean = mase_mean.rename(columns={'yhat': 'MASE'})
    perm_obj = PermutationFeatureImportance()
    perm_obj.in_avg = in_sample_avg.iloc[:, 0].to_dict()
    perm_obj.model = model
    perm_df = list()
    for sym in SYMBOLS:
        perm_obj.underlying_name = sym
        perm_obj.test = test_X.xs(sym, axis=0, level=0, drop_level=True)
        perm_obj.test_y = forecast_df.xs(sym, axis=0, level=0, drop_level=True)
        perm_score = perm_obj.compute_score()
        perm_score.index = \
            pd.MultiIndex.from_product([[sym], [pd.to_datetime(date, utc=True)], perm_score.index.tolist()],
                                       names=['underlying_name', 'timestamp', 'features'])
        perm_df.append(perm_score.reset_index(-1).assign(
            regression=args.regression, D=args.D, freq=args.freq, market=args.options, target=args.target))
    return forecast_df, mase_mean, pd.concat(perm_df).rename(columns={0: 'MASE'})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script generating all results presented in the associated paper.\n'
                                                 'Results are computed model by model.')
    parser.add_argument('--regression', type=str, help='Regression technique to be used for modelling.')
    parser.add_argument('--L', type=str, help='Parameter controlling the size of the training set size.')
    parser.add_argument('--freq', type=str, help='Parameter controlling forcasting horizon.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2023-07-01')
    parser.add_argument('--target', type=str, help='RV or returns.', default='returns')
    parser.add_argument('--target_path', type=str, help='Directory where database is located', default='./data_centre')
    parser.add_argument('--to_csv', type=int, help='Saving results in .csv files or not', default=1)
    parser.add_argument('--options', type=int, default=0, help='Add option features or not.')
    parser.add_argument('--D', type=str, help='Parameter controlling forcasting horizon.')
    parser.add_argument('--cpu_count', type=int, default=10, help='Number of processes to run the whole pipeline on.')
    args = parser.parse_args()
    print(args)
    dates = pd.date_range(args.start_date, args.end_date, freq='1D', tz=pytz.UTC).date.tolist()
    db_connect = sqlite3.connect(f"{args.target_path}/crypto_option_spot.db", check_same_thread=False)
    SYMBOLS_QUERY = "(" + ",".join(f'"{s}"' for s in SYMBOLS) + ")"
    features_query = \
        f'WITH base AS(SELECT DISTINCT * FROM spot_features WHERE "underlying_name" IN {SYMBOLS_QUERY} ' \
        f'AND "target" = "{args.target}" AND "freq" = "{args.freq}") SELECT underlying_name, timestamp, features, ' \
        f'features_values FROM base ' \
        f'UNION ALL SELECT underlying_name, timestamp, target AS features, target_values AS features_values ' \
        f'FROM base;'
    features = pl.read_database(query=features_query, connection=db_connect, iter_batches=True, batch_size=100_000,
                                infer_schema_length=None)
    features = (df.with_columns([pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%z'),
                                 pl.col('features_values').cast(pl.Float32, strict=False)]) for df in features)
    features = (df.filter((pl.col('timestamp') >= pl.lit(pd.to_datetime(args.start_date, utc=True))) &
                          (pl.col('timestamp') < pl.lit(pd.to_datetime(args.end_date, utc=True)))) for df in features)
    features = pl.concat(features)
    if args.options == 1:
        options_features_query = \
            f'SELECT "features","features_values","underlying_name","timestamp" FROM option_features ' \
            f'WHERE "underlying_name" IN {SYMBOLS_QUERY};'
        options_features = \
            pl.read_database(query=features_query, connection=db_connect, iter_batches=True, batch_size=100_000,
                             infer_schema_length=None)
        options_features = \
            (df.with_columns([pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%z'),
                              pl.col('features_values').cast(pl.Float32, strict=False)]) for df in options_features)
        options_features = (df.filter((pl.col('timestamp') >= pl.lit(pd.to_datetime(args.start_date, utc=True))) &
                              (pl.col('timestamp') < pl.lit(pd.to_datetime(args.end_date, utc=True)))) for df in
                            options_features)
        options_features = pl.concat(options_features)
        main_features = features.join(options_features, on=['underlying_name', 'timestamp'], how='right')
    else:
        main_features = features
    one_hot = main_features.select('underlying_name').to_dummies().rename(
        lambda c: c.replace('underlying_name_', 'is_')).cast(pl.Int8)
    main_features = pl.concat([main_features, one_hot], how='horizontal')
    main_features = main_features.with_columns(pl.col('timestamp').dt.replace_time_zone(None))
    main_features = \
        main_features.pivot(values='features_values', index=['underlying_name', 'timestamp'], on='features',
                            aggregate_function='first')
    model = models_factory_dd[args.regression]
    model.standardscaler = {s: StandardScaler().set_output(transform='pandas') for s in SYMBOLS}
    start_idx = np.searchsorted(dates, dates[0] + relativedelta(days=DAYS_CONVERTER[args.L]))
    date_grid = dates[start_idx::DAYS_CONVERTER[args.L]]
    date_masks = {d: ((main_features['timestamp'] >= pl.lit(d - relativedelta(days=DAYS_CONVERTER[args.L]))) &
                      (main_features['timestamp'] < pl.lit(d))) for d in date_grid}
    main_features = main_features.drop(contemporanous_features[args.target])
    trained_models = dict(Parallel(n_jobs=args.cpu_count)(
        delayed(train_one_window)(d, main_features, date_masks, model, args) for d in date_grid
    ))
    forecasts = list()
    metrics = list()
    features_importance = list()
    date_grid_test = dates
    date_grid_test = date_grid_test[date_grid_test.index(date_grid[0])::DAYS_CONVERTER[args.D]]
    pd.Series(trained_models).reindex(date_grid_test)
    date_masks_tests = {d: ((main_features['timestamp'] < pl.lit(d + relativedelta(days=DAYS_CONVERTER[args.D]))) &
                            (main_features['timestamp'] >= pl.lit(d))) for d in date_grid_test}
    for date in date_grid_test:
        in_sample_date = date_grid[np.where(np.array(date_grid) >= date)[0][0]]
        tmp_model = trained_models[in_sample_date]
        forecast, metric, feature_importance = \
            evaluate_one_window(
                date, tmp_model, main_features,
                main_features.filter(date_masks[in_sample_date])[['underlying_name', args.target]],
                date_masks_tests, args)
        forecasts.append(forecast)
        metrics.append(metric)
        features_importance.append(feature_importance)
    forecasts = pd.concat(forecasts)
    forecasts = \
        forecasts.set_index(forecasts.index.set_levels(pd.to_datetime(forecasts.index.levels[1], utc=True), level=1))
    metric = pd.concat(metrics)
    metric = metric.set_index(metric.index.set_levels(pd.to_datetime(metric.index.levels[1], utc=True), level=1))
    features_importance = pd.concat(features_importance)
    features_importance = features_importance.set_index(features_importance.index.set_levels(
        pd.to_datetime(features_importance.index.levels[1], utc=True), level=1))
    underlying_rv = pd.read_sql(
        f'SELECT underlying_name, timestamp, target_values FROM spot_features WHERE target == "{args.target}";',
        con=db_connect
    )
    underlying_rv['timestamp'] = pd.to_datetime(underlying_rv['timestamp'], utc=True)
    underlying_rv['target_values'] = underlying_rv['target_values'].astype(np.float32)
    volatility_regime = underlying_rv.groupby('timestamp')['target_values'].mean().resample(args.freq).sum()
    volatility_regime = volatility_regime.loc[(volatility_regime.index.date >= dates[start_idx]) &
                                             (volatility_regime.index.date <= dates[-1])]
    volatility_regime = pd.qcut(volatility_regime, q=[0, .45, .9, 1], labels=VOL_REGIMES)
    volatility_regime.name = 'volatility_regime'
    metric = metric.join(volatility_regime, how='left', on='timestamp').assign(regression=args.regression,
                                                                               target=args.target, D=args.D,
                                                                               freq=args.freq, market=args.options)
    features_importance = features_importance.join(volatility_regime, how='left').assign(market=args.options)
    forecasts = forecasts.join(volatility_regime, how='left').assign(regression=args.regression,
                                                                     target=args.target, D=args.D, freq=args.freq,
                                                                     market=args.options)
    if args.to_csv:
        forecasts.to_csv(f'{args.target_path}/forecasts.csv', mode='a',
                         header=not os.path.exists(f'{args.target_path}/forecasts.csv'))
        metric.to_csv(f'{args.target_path}/metric.csv', mode='a',
                      header=not os.path.exists(f'{args.target_path}/metric.csv'))
        features_importance.to_csv(
            f'{args.target_path}/features_importance.csv', mode='a',
            header=not os.path.exists(f'{args.target_path}/features_importance.csv'))
    else:
        forecasts.to_sql('forecasts', con=db_connect, if_exists='append')
        metric.to_sql('metric', con=db_connect, if_exists='append')
        features_importance.to_sql('feature_importance', con=db_connect, if_exists='append')
    print('[Done] All forecasts, MASE, and feature importance computed and stored.')