import argparse
import pdb
import polars as pl
import pytz
import sqlite3
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
from torch._dynamo.polyfills import itertools
import itertools
from constants import *
import os
from mlforecast import MLForecast
import lightgbm as lgb
from sklearn.linear_model import Lasso
import neuralforecast
from neuralforecast import NeuralForecast
from neuralforecast.models import SOFTS
from neuralforecast.losses.pytorch import MSE
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import random


contemporanous_features = {'returns': ['m2', 'm3', 'm4', 'm5', 'm6'], 'RV': []}
regression_dd = {'ml': ['LGBMRegressor', 'Lasso'], 'dl': ['SOFTS']}
rng = random.Random(123)

def winsorise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).ffill()
    q05 = df.quantile(0.05)
    q95 = df.quantile(0.95)
    return df.clip(lower=q05, upper=q95, axis=1)



def train_one_window(date, features, regression: str):
    train_X = features.to_pandas().set_index(['underlying_name', 'timestamp']).dropna()
    train_X = \
        train_X.reset_index().rename(columns={'underlying_name': 'unique_id', 'timestamp': 'ds', args.target: 'y'})
    ds_unique = train_X.ds.unique()
    start, end = ds_unique[0], ds_unique[-1]
    full_dates = pd.date_range(start, end, freq=args.freq, inclusive='both')
    uids = train_X['unique_id'].unique()
    full_index = pd.MultiIndex.from_product([uids, full_dates], names=['unique_id', 'ds'])
    try:
        train_X = \
            train_X.drop_duplicates(subset=['unique_id', 'ds']).set_index(['unique_id', 'ds']).reindex(full_index)
        train_X = train_X.groupby(level=0, group_keys=False).apply(winsorise)
        train_X = train_X.reset_index()
        if train_X.empty:
            raise ValueError
    except ValueError:
        return date, None
    train_X = train_X.groupby('unique_id', group_keys=False).apply(lambda x: x.ffill().bfill())
    if regression == 'dl':
        n_heads = pd.to_timedelta(DAYS_CONVERTER[args.D], unit='d') // pd.to_timedelta(args.freq)
        n_series = train_X['unique_id'].nunique()
        model = SOFTS(h=n_heads, input_size=train_X.shape[1], n_series=n_series, hidden_size=256,
                      loss=MSE(), scaler_type='standard', learning_rate=1e-3, e_layers=1, d_ff=64, batch_size=128,
                      max_steps=100, val_check_steps=10, early_stop_patience_steps=3, start_padding_enabled=True)
        model = NeuralForecast(models=[model], freq=args.freq)
        model.fit(df=train_X, val_size=n_heads)
    else:
        model = \
            MLForecast(models=lgb.LGBMRegressor(force_col_wise=True, learning_rate=1e-1, n_estimators=50),
                       freq=args.freq, target_transforms=[])
        model.fit(train_X, fitted=False, static_features=[])
        model2 = MLForecast(models=Lasso(), freq=args.freq, target_transforms=[])
        num_cols = train_X.select_dtypes(include='number').columns
        train_group = train_X.groupby('unique_id')[num_cols].agg(['mean', 'std'])
        mean = train_group.xs('mean', axis=1, level=1)
        std = train_group.xs('std', axis=1, level=1).clip(lower=1e-8)
        train_X = train_X.set_index(['unique_id', 'ds']).subtract(mean).div(std).reset_index()
        model2.fit(train_X, fitted=False, static_features=[])
        model = [model, model2, train_group]
    return (date, regression), model


def compute_feat_importance(col, test_X, test_y, in_sample_avg, regression,
                            model_names, n_heads, n_steps_ahead_scaler, **kwargs):
    shuffled_X = test_X.copy()
    # shuffled_X[col] = \
    #     (shuffled_X.groupby('unique_id')[col].transform(lambda x: np.random.permutation(x.to_numpy())))
    full_mase_mean_shuffled = pd.DataFrame()
    jitter = np.random.normal(size=(shuffled_X[col].shape[0], 50))
    for i in range(10):
        shuffled_X[col] = shuffled_X[col] + jitter[:, i]
        if regression == 'ml':
            shuffled_lasso_X = kwargs['test_lasso_X'].copy()
            # shuffled_lasso_X[col] = \
            #     (shuffled_lasso_X.groupby('unique_id')[col].transform(
            #         lambda x: np.random.permutation(x.to_numpy())))
            shuffled_lasso_X[col] = shuffled_X[col] + jitter[:, i]
            lightgbm_shuffled_y = kwargs['lightgbm_model'].predict(h=n_heads * n_steps_ahead_scaler, X_df=shuffled_X)
            lightgbm_shuffled_y = lightgbm_shuffled_y.set_index(['unique_id', 'ds'])
            lasso_shuffled_y = kwargs['lasso_model'].predict(h=n_heads * n_steps_ahead_scaler, X_df=shuffled_lasso_X)
            lasso_shuffled_y = lasso_shuffled_y.set_index(['unique_id', 'ds'])
            lasso_shuffled_y['Lasso'] = (
                lasso_shuffled_y['Lasso'].mul(kwargs['std']['y'], axis=0, level=0).add(kwargs['mean']['y'], axis=0,
                                                                                       level=0))
            test_shuffled_y = pd.concat([test_y['y'], lightgbm_shuffled_y, lasso_shuffled_y], axis=1)
        else:
            softs_shuffled_y = kwargs['softs_model'].predict(h=n_heads * n_steps_ahead_scaler, futr_df=shuffled_X)
            softs_shuffled_y = softs_shuffled_y.set_index(['unique_id', 'ds'])
            test_shuffled_y = pd.concat([test_y['y'], softs_shuffled_y], axis=1)
        delta_shuffled_ys = -test_shuffled_y.iloc[:, 1:].subtract(test_shuffled_y['y'], axis=0)
        mase_mean_shuffled = delta_shuffled_ys.div(in_sample_avg.values, axis=0).abs().rename(columns=model_names)
        mase_mean_shuffled = mase_mean_shuffled.stack().reset_index()
        mase_mean_shuffled = mase_mean_shuffled.rename(columns={'unique_id': 'underlying_name',
                                                                'ds': 'timestamp', 'level_2': 'regression', 0: 'MASE'})
        if full_mase_mean_shuffled.empty:
            full_mase_mean_shuffled = mase_mean_shuffled.set_index(['underlying_name', 'timestamp', 'regression'])
        else:
            full_mase_mean_shuffled += mase_mean_shuffled.set_index(['underlying_name', 'timestamp', 'regression'])
    return full_mase_mean_shuffled.div(50).assign(features=col).reset_index()


def evaluate_one_window(model, train_features, regression: str,
                        args , n_steps_ahead_scaler, main_features):
    n_heads = pd.to_timedelta(DAYS_CONVERTER[args.D], unit='d') // pd.to_timedelta(args.freq)
    if regression != 'dl':
        lightgbm_model, lasso_model, stats = model[0], model[1], model[2]
        mean, std = stats.xs('mean', axis=1, level=1), stats.xs('std', axis=1, level=1).clip(lower=1e-8)
        idx_test = lightgbm_model.make_future_dataframe(h=n_heads * n_steps_ahead_scaler)
        test_X = pl.from_pandas(idx_test).with_columns(pl.col('ds').cast(pl.Datetime('us'))).rename(
            {'unique_id': 'underlying_name', 'ds': 'timestamp'}).join(main_features,
                                                                      on=['underlying_name', 'timestamp'], how='left')
        test_X = test_X.rename({'underlying_name': 'unique_id', 'timestamp': 'ds'}).to_pandas()
        test_X = test_X.drop_duplicates(subset=['unique_id', 'ds'])
        test_X = test_X.set_index(['unique_id', 'ds']).groupby(level=0, group_keys=False).apply(
            lambda x: x.fillna(x.ffill().bfill())).reset_index()
        lightgbm_y = \
            lightgbm_model.predict(h=n_heads * n_steps_ahead_scaler,
                                   X_df=test_X.drop(columns=[args.target])).set_index(['unique_id', 'ds'])
        test_lasso_X = test_X.copy().rename(columns={args.target: 'y'}).set_index(['unique_id', 'ds'])
        test_lasso_X = test_lasso_X.subtract(mean, axis=0).div(std, axis=0).reset_index()
        lasso_y = lasso_model.predict(h=n_heads * n_steps_ahead_scaler, X_df=test_lasso_X.drop(columns='y'))
        lasso_y = lasso_y.set_index(['unique_id', 'ds'])
        lasso_y['Lasso'] = lasso_y['Lasso'].mul(std['y'], axis=0, level=0).add(mean['y'], axis=0, level=0)
        tmp_model = [lasso_y, lightgbm_y]
    else:
        softs_model = model
        idx_test = softs_model.make_future_dataframe(h=n_heads * n_steps_ahead_scaler)
        test_X = pl.from_pandas(idx_test).with_columns(pl.col('ds').cast(pl.Datetime('us'))).rename(
            {'unique_id': 'underlying_name', 'ds': 'timestamp'}).join(main_features,
                                                                      on=['underlying_name', 'timestamp'], how='left')
        test_X = test_X.rename({'underlying_name': 'unique_id', 'timestamp': 'ds'}).to_pandas()
        test_X = test_X.drop_duplicates(subset=['unique_id', 'ds'])
        softs_y = softs_model.predict(h=n_heads * n_steps_ahead_scaler,
                                      futr_df=test_X.drop(columns=[args.target])).set_index(['unique_id', 'ds'])
        tmp_model = [softs_y]
    test_y = test_X[[args.target, 'ds', 'unique_id']].set_index(['unique_id', 'ds']).rename(columns={args.target: 'y'})
    test_y = pd.concat([test_y]+tmp_model, axis=1)
    test_y = test_y.groupby([pd.Grouper(level=0), pd.Grouper(level=1, freq=args.freq)]).sum()
    test_X = test_X.drop(args.target, axis=1)
    #################################################### MASE ##########################################################
    train_features = train_features.to_pandas().rename(columns={'underlying_name': 'unique_id', args.target: 'y'})
    in_sample_avg = train_features.groupby('unique_id')['y'].diff().abs().groupby(train_features['unique_id']).mean()
    delta_ys = -1*test_y[regression_dd[regression]].subtract(test_y['y'], axis=0)
    in_sample_avg = in_sample_avg.reindex(delta_ys.index, level=0)
    mase_mean = delta_ys.div(in_sample_avg.values, axis=0).abs().rename(columns=model_names)
    mase_mean = mase_mean.stack().reset_index().rename(
        columns={'unique_id': 'underlying_name', 'ds': 'timestamp', 'level_2': 'regression', 0: 'MASE'}).dropna()
    ############################################# Feature Importance ###################################################
    sample_columns = random.sample(test_X.columns[~test_X.columns.isin(['unique_id', 'ds'])].tolist(), k=5)
    perm_df = list()
    for col in sample_columns:
        perm_df.append(compute_feat_importance(col=col, test_X=test_X,
                                test_lasso_X=test_lasso_X if regression == 'ml' else None, test_y=test_y,
                                in_sample_avg=in_sample_avg, args=args, regression=regression,
                                model_names=model_names, n_heads=n_heads, n_steps_ahead_scaler=n_steps_ahead_scaler,
                                lightgbm_model=lightgbm_model if regression == 'ml' else None,
                                lasso_model=lasso_model if regression == 'ml' else None,
                                softs_model=softs_model if regression == 'dl' else None,
                                mean=mean if regression == 'ml' else None,
                                std=std if regression == 'ml' else None))
    perm_df = pd.concat(perm_df, ignore_index=True)
    ################################################ Forecasts #########################################################
    forecast_df = test_y.set_index('y', append=True).rename(columns=model_names)
    forecast_df = forecast_df.clip(lower=0.000001)
    forecast_df = forecast_df.stack().reset_index().rename(
        columns={'unique_id': 'underlying_name', 'ds': 'timestamp', 'level_3': 'regression', 0: 'y_hat'})
    return forecast_df, mase_mean, perm_df

def process_one_test_window(trained_model, features_df, main_features, args,
                            tmp_sum, date_grid_test, horizon_date, regression: str):
    forecast, metric, feature_importance = evaluate_one_window(model=trained_model,
                                                               train_features=features_df,
                                                               regression=regression,
                                                               args=args,
                                                               n_steps_ahead_scaler=tmp_sum,
                                                               main_features=main_features)
    start = pd.Timestamp(date_grid_test)
    end = horizon_date
    f_mask = (forecast.timestamp >= start) & (forecast.timestamp < end)
    m_mask = (metric.timestamp >= start) & (metric.timestamp < end)
    fi_mask = (feature_importance.timestamp >= start) & (feature_importance.timestamp < end)
    return forecast.loc[f_mask], metric.loc[m_mask], feature_importance.loc[fi_mask]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script generating all results presented in the associated paper.\n'
                                                 'Results are computed model by model.')
    # parser.add_argument('--regression', type=str, help='Regression technique to be used for modelling.')
    parser.add_argument('--L', type=str, help='Parameter controlling the size of the training set size.')
    parser.add_argument('--freq', type=str, help='Parameter controlling forcasting horizon.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2023-07-01')
    parser.add_argument('--target', type=str, help='RV or returns.', default='returns')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    parser.add_argument('--to_csv', type=int, help='Saving results in .csv files or not', default=1)
    parser.add_argument('--options', type=int, default=0, help='Add option features or not.')
    parser.add_argument('--D', type=str, help='Parameter controlling forcasting horizon.')
    parser.add_argument('--cpu_count', type=int, default=10,
                        help='Number of processes to use for the pipeline.')
    args = parser.parse_args()
    print(args)
    dates = pd.date_range(args.start_date, args.end_date, freq='1D', tz=pytz.UTC, inclusive='left').date.tolist()
    db_connect = sqlite3.connect(f"{args.target_path}/crypto_option_spot.db", check_same_thread=False)
    if not os.path.exists(f'{args.target_path}/spot_features.csv'):
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
    else:
        features = pl.read_csv(f'{args.target_path}/spot_features.csv', batch_size=100_000)
        features = features.filter(pl.col('target') == args.target)
        features = features.with_columns([pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%z'),
                                          pl.col('features_values').cast(pl.Float32, strict=False)])
    features = features.pivot(values='features_values',
                              index=['underlying_name', 'timestamp', 'target', 'target_values'], on='features',
                              aggregate_function='first')
    features = features.drop('target').with_columns(pl.col('target_values').alias(args.target)).drop('target_values')
    features = features.unique(maintain_order=True)
    global main_features
    if args.options == 1:
        if not os.path.exists(f'{args.target_path}/option_features.csv'):
            options_features_query = \
                f'SELECT "features","features_values","underlying_name","timestamp" FROM option_features ' \
                f'WHERE "underlying_name" IN {SYMBOLS_QUERY};'
            options_features = \
                pl.read_database(query=options_features_query, connection=db_connect, iter_batches=True,
                                 batch_size=100_000, infer_schema_length=None)
            options_features = \
                (df.with_columns([pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%z'),
                                  pl.col('features_values').cast(pl.Float32, strict=False)]) for df in options_features)
            options_features = (df.filter((pl.col('timestamp') >= pl.lit(pd.to_datetime(args.start_date, utc=True))) &
                                          (pl.col('timestamp') < pl.lit(pd.to_datetime(args.end_date, utc=True)))) for
                                df in
                                options_features)
            options_features = pl.concat(options_features).drop(['is_retail', 'is_informed', 'is_institutional'])
        else:
            options_features = pl.read_csv(f'{args.target_path}/option_features.csv', batch_size=100_000)
            options_features = options_features.with_columns([pl.col('timestamp').str.strptime(pl.Datetime,
                                                                                               '%Y-%m-%d %H:%M:%S%z'),
                                                              pl.col('features_values').cast(pl.Float32, strict=False)])
        options_features = options_features.pivot(values='features_values', index=['underlying_name', 'timestamp'],
                                                  on='features', aggregate_function='first')
        main_features = features.join(options_features, on=['underlying_name', 'timestamp'], how='left')
        main_features = main_features.sort(['underlying_name', 'timestamp'])
        main_features = main_features.sort(['underlying_name', 'timestamp']).with_columns(
            pl.all().forward_fill().over('underlying_name')
        ).drop_nulls()
    else:
        main_features = features
    main_features = main_features.with_columns(pl.col('timestamp').dt.replace_time_zone(None))
    start_idx = np.searchsorted(dates, dates[0] + relativedelta(days=DAYS_CONVERTER[args.L]))
    date_grid = dates[start_idx::DAYS_CONVERTER[args.L]]
    date_masks = {d: ((main_features['timestamp'] >= pl.lit(d - relativedelta(days=DAYS_CONVERTER[args.L]))) &
                      (main_features['timestamp'] < pl.lit(d))) for d in date_grid}
    main_features = main_features.drop(contemporanous_features[args.target])
    forecasts = list()
    metrics = list()
    features_importance = list()
    trained_models = dict()
    date_grid_test = dates[dates.index(date_grid[0])::DAYS_CONVERTER[args.D]]
    date_masks_tests = {d: ((main_features['timestamp'] < pl.lit(d + relativedelta(days=DAYS_CONVERTER[args.D]))) &
                            (main_features['timestamp'] >=
                             pl.lit(d - relativedelta(days=DAYS_CONVERTER[args.D]+DAYS_CONVERTER[args.D]))))
                        for d in date_grid_test}
    date_grid_regressions = list(itertools.product(date_grid, list(regression_dd.keys())))
    with tqdm_joblib(tqdm(desc='Model training windows', total=len(date_grid))):
        trained_models = dict(Parallel(n_jobs=args.cpu_count)(
            delayed(train_one_window)(date_grid_regression[0],
                                      main_features.filter(date_masks[date_grid_regression[0]]),
                                      date_grid_regression[1]) for date_grid_regression in date_grid_regressions
        ))
    forecasts, metrics, features_importance = [], [], []
    date_grid_np = np.array(date_grid)
    date_grid_test = np.array(date_grid_test)
    end_date = pd.Timestamp(args.end_date)
    D_days = DAYS_CONVERTER[args.D]
    L_days = DAYS_CONVERTER[args.L]
    date_grid_np = np.array(date_grid)
    idxs = np.searchsorted(date_grid_np, date_grid_test, side='right')
    in_sample_dates = (
        np.where(idxs < len(date_grid_np), date_grid_np[np.clip(idxs - 1, 0, len(date_grid_np) - 1)],
                 date_grid_np[-1]))
    dates = pd.to_datetime(date_grid_test).to_numpy(dtype='datetime64[ns]')
    end = np.datetime64(args.end_date)
    horizon_dates = np.minimum(dates + np.timedelta64(D_days, 'D'), end)
    feature_cache = {d: main_features.filter(date_masks[d])[['underlying_name', args.target]]
                     for d in np.unique(in_sample_dates)}
    date_regression_grid_test = list(itertools.product(date_grid, regression_dd.keys()))
    with tqdm_joblib(tqdm(desc='Test windows', total=len(date_regression_grid_test))):
        results = Parallel(n_jobs=args.cpu_count, backend='loky')(
            delayed(process_one_test_window)(
                trained_models[(in_sample_dates[idx], date_regression[1])]
                if isinstance(trained_models[(in_sample_dates[idx], date_regression[1])],
                              (list, neuralforecast.core.NeuralForecast)) else
                trained_models[(in_sample_dates[idx] - pd.Timedelta(days=L_days), date_regression[1])],
                feature_cache[in_sample_dates[idx]], main_features, args, idxs[idx], date_grid_test[idx],
                horizon_dates[idx], date_regression[1])
            for idx, date_regression in enumerate(date_regression_grid_test))
    forecasts, metrics, features_importance = map(list, zip(*results))
    forecasts = pd.concat(forecasts).set_index(['underlying_name', 'timestamp'])
    forecasts.index = forecasts.index.set_levels(pd.to_datetime(forecasts.index.levels[1], utc=True), level=1)
    metric = pd.concat(metrics).set_index(['underlying_name', 'timestamp'])
    metric.index = metric.index.set_levels(pd.to_datetime(metric.index.levels[1], utc=True), level=1)
    features_importance = pd.concat(features_importance).set_index(['underlying_name', 'timestamp'])
    features_importance.index = features_importance.index.set_levels(
        pd.to_datetime(features_importance.index.levels[1], utc=True), level=1)
    underlying_rv = pl.scan_csv(f'{args.target_path}/spot_features.csv')
    underlying_rv = (
        underlying_rv.filter((pl.col('freq') == '15min') & (pl.col('target') == 'RV') &
                             (pl.col('features') == 'RV_15min')))
    underlying_rv = underlying_rv.select(['timestamp', 'target_values'])
    underlying_rv = underlying_rv.with_columns(pl.col('timestamp')
                                               .str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S%z', strict=False)
                                               .dt.convert_time_zone('UTC')).collect().to_pandas()
    volatility_regime = underlying_rv.groupby('timestamp')['target_values'].mean().resample(args.freq).sum()
    volatility_regime = \
        volatility_regime.loc[(volatility_regime.index.date < pd.to_datetime(args.end_date, utc=True).date())]
    volatility_regime = pd.qcut(volatility_regime, q=[0, .45, .9, 1], labels=VOL_REGIMES)
    volatility_regime.name = 'volatility_regime'
    metric = metric.join(volatility_regime, how='left', on='timestamp').assign(target=args.target, D=args.D,
                                                                               freq=args.freq, market=args.options)
    features_importance = \
        features_importance.join(volatility_regime, how='left').assign(target=args.target, D=args.D,
                                                                       freq=args.freq, market=args.options)
    forecasts = forecasts.join(volatility_regime, how='left').assign(target=args.target, D=args.D, freq=args.freq,
                                                                     market=args.options)
    if args.to_csv:
        forecasts.to_csv(f'{args.target_path}/crypto_option_spot/forecasts_{args.freq}.csv', mode='a',
                         header=not os.path.exists(f'{args.target_path}/crypto_option_spot/forecasts_{args.freq}.csv'))
        metric.to_csv(f'{args.target_path}/crypto_option_spot/metric_{args.freq}.csv', mode='a',
                      header=not os.path.exists(f'{args.target_path}/crypto_option_spot/metric_{args.freq}.csv'))
        features_importance.to_csv(
            f'{args.target_path}/crypto_option_spot/features_importance_{args.freq}.csv', mode='a',
            header=not os.path.exists(f'{args.target_path}/crypto_option_spot/features_importance_{args.freq}.csv'))
    else:
        forecasts.to_sql('forecasts', con=db_connect, if_exists='append')
        metric.to_sql('metric', con=db_connect, if_exists='append')
        features_importance.to_sql('feature_importance', con=db_connect, if_exists='append')
    print('[Done] All forecasts, MASE, and feature importance computed and stored.')