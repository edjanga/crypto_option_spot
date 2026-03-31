import argparse
import polars as pl
import pytz
import sqlite3
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from constants import *
from model.lab import FeatureBuilder, LassoRegressor, ElasticRegressor, BoostingRegressor,\
    GRURegressor, LSTMRegressor, PermutationFeatureImportance
import torch
import os

models = [LassoRegressor(), ElasticRegressor(), BoostingRegressor()]
models_factory_dd = {model.name.lower(): model for model in models}
LAGS = INTRADAY_LAGS + INTERDAY_LAGS
mase_score = MeanAbsoluteScaledError()


def fit_transform_grouped(df, scalers, cols_scale):
    cols_scale = [c for c in cols_scale if c in df.columns]
    if not cols_scale:
        return df
    def _fit(g):
        sym = g.name
        X = g[cols_scale].to_numpy(copy=False)
        X[~np.isfinite(X)] = np.nan
        X = pd.DataFrame(X).bfill().ffill().values
        scalers[sym].fit(X)
        g.loc[:, cols_scale] = scalers[sym].transform(X)
        return g
    return df.groupby(level=0, group_keys=False, group_names=False).apply(_fit)


def transform_grouped(df, scalers, cols_scale):
    cols_scale = [c for c in cols_scale if c in df.columns]
    if not cols_scale:
        return df

    def _transform(g):
        sym = g.name
        X = g[cols_scale].to_numpy(copy=False)
        g.loc[:, cols_scale] = scalers[sym].transform(X)
        return g

    return df.groupby(level=0, group_keys=False, group_names=False).apply(_transform)


def train_one_window(date, features, date_mask, model, args):
    train_X = features.loc[date_mask].copy()
    train_y = train_X.pop(args.target)
    if args.regression in ['lasso', 'elastic']:
        train_X = fit_transform_grouped(train_X, model.standardscaler, train_X.columns.difference(BINARY_PATTERN))
    study_name = f'{model.name}_{args.L}_{args.target}_{date}'
    if model.name in ['lasso', 'elastic', 'lightgbm']:
        res = model.model_train(features=train_X, target=train_y, study_name=study_name)
        model.model = res.user_attrs['best_estimator']
    else:
        dates_unique = train_X.index.get_level_values(1).unique()
        split = int(0.8 * len(dates_unique))
        model.model_train(
            train_X=train_X[train_X.index.get_level_values(1).isin(dates_unique[:split])],
            train_y=train_y[train_y.index.get_level_values(1).isin(dates_unique[:split])],
            valid_X=train_X[train_X.index.get_level_values(1).isin(dates_unique[split:])],
            valid_y=train_y[train_y.index.get_level_values(1).isin(dates_unique[split:])],
            n_trials=N_TRIALS // 20,
            positiveness=(args.target == 'RV'),
            cols_ignore=[col for col in train_X.columns if any(p in col for p in BINARY_PATTERN)],
            target=args.target,
            study_name=study_name
        )
    return date, model


def predict_one(date, model_obj, features, train_y, args):
    cols_ignore = [col for col in features.columns if any(pat in col for pat in BINARY_PATTERN)]
    cols_scale = features.columns.difference(cols_ignore)
    start = date
    end = date + relativedelta(days=DAYS_CONVERTER[args.freq])
    mask = (features.index.get_level_values(1).date >= start) & \
           (features.index.get_level_values(1).date < end)
    test_X = features.loc[mask].copy()
    test_y = test_X.pop(args.target)
    if args.regression != 'lightgbm':
        test_X = transform_grouped(test_X, model_obj.standardscaler, cols_scale)
    if model_obj.name in ['gru', 'lstm']:
        X_tensor = torch.from_numpy(test_X.values.astype(np.float32).reshape(test_X.shape[0], 1, test_X.shape[1]))
        y_hat = model_obj.model(X_tensor).detach().cpu().numpy().ravel()
    else:
        y_hat = model_obj.model.predict(test_X)
    y_hat = pd.Series(y_hat, index=test_y.index)
    if args.regression != 'lightgbm':
        means = pd.Series({k: v.mean_[0] for k, v in model_obj.standardscaler.items()})
        stds = pd.Series({k: np.sqrt(v.var_[0]) for k, v in model_obj.standardscaler.items()})
        sym = y_hat.index.get_level_values(0)
        y_hat = y_hat * sym.map(stds) + sym.map(means)
    df = pd.concat([test_y.rename('y'), y_hat.rename('yhat')], axis=1)
    mase_vals = []
    for sym, g in df.groupby(level=0):
        try:
            mase_vals.append(mase_score(g['y'].values, g['yhat'].values, train_y.xs(sym).values))
        except KeyError:
            continue
    return df, np.mean(mase_vals) if mase_vals else np.nan


def evaluate_one_window(date, model_obj, features, train_features, args):
    cols_ignore = [col for col in features.columns if any(pat in col for pat in BINARY_PATTERN)]
    cols_scale = features.columns.difference(cols_ignore)
    start = date
    end = date + relativedelta(days=DAYS_CONVERTER[args.freq])
    mask = (features.index.get_level_values(1).date >= start) & (features.index.get_level_values(1).date < end)
    test_X = features.loc[mask].copy()
    test_y = test_X.pop(args.target)
    if args.regression != 'lightgbm':
        test_X = transform_grouped(test_X, model_obj.standardscaler, cols_scale)
    if model_obj.name in ['gru', 'lstm']:
        X_tensor = torch.from_numpy(test_X.values.reshape(test_X.shape[0], 1, test_X.shape[1]).astype(np.float32))
        y_hat = model_obj.model(X_tensor).detach().numpy().ravel()
    else:
        y_hat = model_obj.model.predict(test_X)
    y_hat = pd.Series(y_hat, index=test_y.index)
    if args.regression != 'lightgbm':
        means = pd.Series({k: v.mean_[0] for k, v in model_obj.standardscaler.items()})
        stds = pd.Series({k: np.sqrt(v.var_[0]) for k, v in model_obj.standardscaler.items()})
        sym = y_hat.index.get_level_values(0)
        y_hat = y_hat * sym.map(stds) + sym.map(means)
    forecast_df = pd.concat([test_y.rename('y'), y_hat.rename('yhat')], axis=1)
    forecast_df = forecast_df.replace(0, np.nan).groupby(level=0, group_keys=False).apply(lambda x: x.ffill())
    mase_values = []
    for sym, group in forecast_df.groupby(level=0):
        try:
            train_y_sym = \
                train_features.loc[train_features.index.get_level_values(1).isin(
                    train_features.index.get_level_values(1))].xs(sym, level=0)
            mase_values.append(mase_score(group['y'].values, group['yhat'].values, train_y_sym[args.target].values))
        except:
            continue
    mase_mean = np.mean(mase_values) if mase_values else np.nan
    perm_obj = PermutationFeatureImportance()
    perm_obj.model = model_obj
    perm_obj.test = test_X
    perm_obj.test_y = forecast_df
    perm_obj.train_y = train_features[args.target] if args.target in train_features else train_features.pop(args.target)
    perm_score = perm_obj.compute_score()
    perm_df = pd.DataFrame.from_dict(perm_score, orient='index', columns=['MASE'])
    perm_df.index.name = 'timestamp'
    perm_df = perm_df.reset_index().assign(
        regression=args.regression, D=args.freq, freq=args.freq, market=0, target=args.target
    )
    return forecast_df, mase_mean, perm_df


def evaluate_all_windows(trained_models, features, train_features, args):
    date_grid = list(trained_models.keys())
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_one_window)(d, trained_models[d], features, train_features, args)
        for d in date_grid
    )
    forecasts = pd.concat([r[0] for r in results])
    mase_metrics = pd.DataFrame([
        {'timestamp': d, 'MASE': r[1], 'regression': args.regression, 'target': args.target, 'D': args.freq,
         'market': 0, 'freq': args.freq}
        for d, r in zip(date_grid, results)
    ]).set_index('timestamp')
    features_importance = pd.concat([r[2] for r in results])
    return forecasts, mase_metrics, features_importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--regression', type=str, required=True)
    parser.add_argument('--L', type=str, required=True)
    parser.add_argument('--freq', type=str, required=True)
    parser.add_argument('--start_date', type=str, default='2021-01-01')
    parser.add_argument('--end_date', type=str, default='2023-07-01')
    parser.add_argument('--target', type=str, default='returns')
    parser.add_argument('--target_path', type=str, default='./data_centre')
    parser.add_argument('--to_csv', type=int, default=1)
    parser.add_argument('--options', type=int, default=0)
    parser.add_argument('--D', type=str, help='Parameter controlling forcasting horizon.')
    args = parser.parse_args()
    print(args)
    dates = pd.date_range(args.start_date, args.end_date, freq='1D', tz=pytz.UTC).date
    db_connect = sqlite3.connect(f"{args.target_path}/crypto_option_spot.db", check_same_thread=False)
    SYMBOLS_QUERY = "(" + ",".join(f'"{s}"' for s in SYMBOLS) + ")"
    features_query = \
        f'WITH base AS(SELECT DISTINCT * FROM spot_features WHERE "underlying_name" IN {SYMBOLS_QUERY} ' \
        f'AND "target" = "{args.target}" AND "freq" = "{args.freq}") SELECT underlying_name, timestamp, features, ' \
        f'features_values FROM base ' \
        f'UNION ALL SELECT underlying_name, timestamp, target AS features, target_values AS features_values ' \
        f'FROM base;'
    features = pl.read_database(query=features_query, connection=db_connect)
    features = features.with_columns([pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%z'),
                                      pl.col('features_values').cast(pl.Float32, strict=False)])
    features = features.filter(
        (pl.col('timestamp') >= pl.lit(pd.to_datetime(args.start_date, utc=True))) &
        (pl.col('timestamp') < pl.lit(pd.to_datetime(args.end_date, utc=True)))
    ).to_pandas()
    features = \
        features.pivot_table(values='features_values', index=['underlying_name', 'timestamp'], columns='features')
    if args.options == 1:
        options_features_query = \
            f'SELECT "features","features_values","underlying_name","timestamp" FROM option_features ' \
            f'WHERE "underlying_name" IN {SYMBOLS_QUERY};'
        options_features = pl.read_database(query=options_features_query, connection=db_connect)
        options_features = \
            options_features.with_columns([pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%z'),
                                           pl.col('features_values').cast(pl.Float32, strict=False)])
        options_features = \
            options_features.filter((pl.col('timestamp') >= pl.lit(pd.to_datetime(args.start_date, utc=True))) &
                                    (pl.col('timestamp') < pl.lit(pd.to_datetime(args.end_date, utc=True))))
        options_features = options_features.with_columns(pl.all().exclude(['underlying_name', 'timestamp'])
                                                         .forward_fill().over('underlying_name')).to_pandas()
        main_features = features.join(options_features, on=['underlying_name', 'timestamp'], how='right')
    else:
        main_features = features
    one_hot = pd.get_dummies(main_features.index.get_level_values(0), prefix='is').astype(np.int8)
    one_hot.index = main_features.index
    main_features = pd.concat([main_features, one_hot], axis=1).groupby(level=0, group_keys=False)
    main_features = main_features.apply(lambda x: x.ffill()).dropna()
    model = models_factory_dd[args.regression]
    model.standardscaler = {s: StandardScaler().set_output(transform='pandas') for s in SYMBOLS}
    start_idx = np.searchsorted(dates, dates[0] + relativedelta(days=DAYS_CONVERTER[args.L]))
    date_grid = dates[start_idx::DAYS_CONVERTER[args.D]]
    date_masks = {d: (main_features.index.get_level_values(1).date >= d - relativedelta(days=DAYS_CONVERTER[args.L])) &
                      (main_features.index.get_level_values(1).date < d) for d in date_grid}
    trained_models = dict(Parallel(n_jobs=-1)(
        delayed(train_one_window)(d, main_features, date_masks[d], model, args) for d in date_grid
    ))
    forecasts, metric, features_importance = evaluate_all_windows(
        trained_models, main_features, main_features[[args.target]], args
    )
    underlying_rv = pd.read_sql(
        f'SELECT underlying_name, timestamp, target_values FROM spot_features WHERE target == "{args.target}";',
        con=db_connect
    )
    underlying_rv['timestamp'] = pd.to_datetime(underlying_rv['timestamp'], utc=True)
    underlying_rv['target_values'] = underlying_rv['target_values'].astype(np.float32)
    volatility_regime = underlying_rv.groupby("timestamp")["target_values"].mean().resample(args.freq).sum()
    volatility_regime = volatility_regime.loc[(volatility_regime.index.date >= dates[start_idx]) &
                                             (volatility_regime.index.date <= dates[-1])]
    volatility_regime = pd.qcut(volatility_regime, q=[0, .45, .9, 1], labels=VOL_REGIMES)
    volatility_regime.name = 'volatility_regime'
    metric = metric.join(volatility_regime, how='left')
    features_importance = features_importance.join(volatility_regime, how='left')
    if args.to_csv:
        forecasts.to_csv(f'{args.target_path}/forecasts.csv_{args.regression}', mode='a',
                         header=not os.path.exists(f'{args.target_path}/forecasts_{args.regression}.csv'))
        metric.to_csv(f'{args.target_path}/metric.csv_{args.regression}', mode='a',
                      header=not os.path.exists(f'{args.target_path}/metric_{args.regression}.csv'))
        features_importance.to_csv(
            f'{args.target_path}/features_importance_{args.regression}.csv', mode='a',
            header=not os.path.exists(f'{args.target_path}/features_importance_{args.regression}.csv'))
    else:
        forecasts.to_sql('forecasts', con=db_connect, if_exists='append')
        metric.to_sql('metric', con=db_connect, if_exists='append')
        features_importance.to_sql('feature_importance', con=db_connect, if_exists='append')
    print("[Done] All forecasts, MASE, and feature importance computed and stored.")