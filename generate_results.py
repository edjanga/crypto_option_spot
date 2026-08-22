import argparse
import os
import pdb
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import polars as pl
import pytz
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
import itertools
from constants import *
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
from constants import EARLY_STOPPING_PATIENCE


contemporanous_features = {'returns': ['m2', 'm3', 'm4', 'm5', 'm6'], 'RV': []}
regression_dd = {'ml': ['LGBMRegressor', 'Lasso'], 'dl': ['SOFTS']}
rng = random.Random(123)


def winsorise_vectorized(df: pd.DataFrame, group_col: str = 'unique_id', exclude_cols=('ds',)) -> pd.DataFrame:
    df = df.copy()
    excluded = {group_col, *exclude_cols}
    value_cols = [c for c in df.columns if c not in excluded]
    df[value_cols] = df[value_cols].replace([np.inf, -np.inf], np.nan)
    df[value_cols] = df.groupby(df[group_col])[value_cols].ffill()
    q05 = df.groupby(group_col)[value_cols].quantile(0.05)
    q95 = df.groupby(group_col)[value_cols].quantile(0.95)
    lower = df[[group_col]].join(q05, on=group_col)[value_cols]
    upper = df[[group_col]].join(q95, on=group_col)[value_cols]
    df[value_cols] = df[value_cols].clip(lower=lower, upper=upper, axis=0)
    return df


def train_one_window(date, features, regression: str):
    try:
        train_X = features.to_pandas().set_index(['underlying_name', 'timestamp']).dropna()
        train_X = \
            train_X.reset_index().rename(columns={'underlying_name': 'unique_id', 'timestamp': 'ds', args.target: 'y'})
        start, end = train_X.ds.min(), train_X.ds.max()
        full_dates = pd.date_range(start, end, freq=args.freq, inclusive='both')
        uids = train_X['unique_id'].unique()
        full_index = pd.MultiIndex.from_product([uids, full_dates], names=['unique_id', 'ds'])
        train_X = \
            train_X.drop_duplicates(subset=['unique_id', 'ds']).set_index(['unique_id', 'ds']).reindex(full_index)
        train_X = train_X.reset_index()
        train_X = winsorise_vectorized(train_X, group_col='unique_id')
        if train_X.empty:
            raise ValueError('empty training window after reindex/winsorisation')
        fill_cols = [c for c in train_X.columns if c not in ('unique_id', 'ds')]
        train_X[fill_cols] = train_X.groupby('unique_id')[fill_cols].ffill()
        train_X[fill_cols] = train_X.groupby('unique_id')[fill_cols].bfill()
        if regression == 'dl':
            n_heads = pd.to_timedelta(DAYS_CONVERTER[args.D], unit='d') // pd.to_timedelta(args.freq)
            n_series = train_X['unique_id'].nunique()
            model = SOFTS(h=n_heads, input_size=train_X.shape[1], n_series=n_series, hidden_size=256,
                          loss=MSE(), scaler_type='standard', learning_rate=1e-3, e_layers=2, d_ff=512,
                          batch_size=64, max_steps=1_000, val_check_steps=10,
                          early_stop_patience_steps=EARLY_STOPPING_PATIENCE,
                          start_padding_enabled=True, accelerator=ENGINE, devices=1)
            model = NeuralForecast(models=[model], freq=args.freq)
            model.fit(df=train_X, val_size=n_heads)
        else:
            model = \
                MLForecast(models=lgb.LGBMRegressor(force_col_wise=True, learning_rate=1e-1, n_estimators=500,
                                                    n_jobs=1, subsample=0.8, colsample_bytree=0.8),
                           freq=args.freq, target_transforms=[])
            model.fit(train_X, fitted=False, static_features=[])
            model2 = MLForecast(models=Lasso(), freq=args.freq, target_transforms=[])
            num_cols = train_X.select_dtypes(include='number').columns
            train_group = train_X.groupby('unique_id')[num_cols].agg(['mean', 'std'])
            mean = train_group.xs('mean', axis=1, level=1)
            std = train_group.xs('std', axis=1, level=1).clip(lower=1e-8)
            train_X = train_X.set_index(['unique_id', 'ds']).subtract(mean, axis=0).div(std, axis=0).reset_index()
            model2.fit(train_X, fitted=False, static_features=[])
            model = [model, model2, train_group]
        return (date, regression), model
    except Exception as e:
        print(f"WARNING: training window ({date}, {regression}) failed ({type(e).__name__}: {e}) "
              f"-- no model produced for this window.")
        return (date, regression), None


def compute_feat_importance(col, test_X, test_y, in_sample_avg, regression,
                            model_names, n_heads, n_steps_ahead_scaler, **kwargs):
    jittered_X = test_X.copy()
    full_mase_mean_shuffled = pd.DataFrame()
    n_iter = 50
    jitter = np.random.normal(size=(jittered_X[col].shape[0], n_iter))
    original_col = test_X[col].copy()
    for i in range(n_iter):
        jittered_X[col] = original_col + jitter[:, i]
        if regression == 'ml':
            shuffled_lasso_X = kwargs['test_lasso_X'].copy()
            shuffled_lasso_X[col] = jittered_X[col]
            shuffled_lasso_feature_cols = [c for c in shuffled_lasso_X.columns if c not in ('unique_id', 'ds', 'y')]
            if shuffled_lasso_X[shuffled_lasso_feature_cols].isna().any().any():
                shuffled_lasso_X[shuffled_lasso_feature_cols] = \
                    shuffled_lasso_X[shuffled_lasso_feature_cols].fillna(0)
            lightgbm_shuffled_y = kwargs['lightgbm_model'].predict(h=n_heads * n_steps_ahead_scaler, X_df=jittered_X)
            lightgbm_shuffled_y = lightgbm_shuffled_y.set_index(['unique_id', 'ds'])
            lasso_shuffled_y = kwargs['lasso_model'].predict(h=n_heads * n_steps_ahead_scaler, X_df=shuffled_lasso_X)
            lasso_shuffled_y = lasso_shuffled_y.set_index(['unique_id', 'ds'])
            lasso_shuffled_y['Lasso'] = \
                lasso_shuffled_y['Lasso'].mul(kwargs['std']['y'], axis=0, level=0).add(kwargs['mean']['y'], axis=0,
                                                                                       level=0)
            test_shuffled_y = pd.concat([lightgbm_shuffled_y, lasso_shuffled_y], axis=1)
        else:
            softs_shuffled_y = kwargs['softs_model'].predict(h=n_heads * n_steps_ahead_scaler, futr_df=jittered_X)
            softs_shuffled_y = softs_shuffled_y.set_index(['unique_id', 'ds'])
            test_shuffled_y = softs_shuffled_y
        freq = kwargs['args'].freq
        test_shuffled_y = test_shuffled_y.groupby([pd.Grouper(level=0), pd.Grouper(level=1, freq=freq)]).sum()
        test_shuffled_y = pd.concat([test_y['y'], test_shuffled_y], axis=1)
        delta_shuffled_ys = -test_shuffled_y.iloc[:, 1:].subtract(test_shuffled_y['y'], axis=0)
        mase_mean_shuffled = delta_shuffled_ys.div(in_sample_avg.values, axis=0).abs().rename(columns=model_names)
        mase_mean_shuffled = mase_mean_shuffled.stack().reset_index()
        mase_mean_shuffled = mase_mean_shuffled.rename(columns={'unique_id': 'underlying_name',
                                                                'ds': 'timestamp', 'level_2': 'regression', 0: 'MASE'})
        if full_mase_mean_shuffled.empty:
            full_mase_mean_shuffled = mase_mean_shuffled.set_index(['underlying_name', 'timestamp', 'regression'])
        else:
            full_mase_mean_shuffled += mase_mean_shuffled.set_index(['underlying_name', 'timestamp', 'regression'])
    return full_mase_mean_shuffled.div(n_iter).assign(features=col).reset_index()


def evaluate_one_window(model, train_features, regression: str, args , n_steps_ahead_scaler, main_features):
    n_heads = pd.to_timedelta(DAYS_CONVERTER[args.D], unit='d') // pd.to_timedelta(args.freq)
    if regression != 'dl':
        lightgbm_model = model[0]
        idx_test = lightgbm_model.make_future_dataframe(h=n_heads * n_steps_ahead_scaler)
        test_X = pl.from_pandas(idx_test).with_columns(pl.col('ds').cast(pl.Datetime('us'))).rename(
            {'unique_id': 'underlying_name', 'ds': 'timestamp'}).join(main_features,
                                                                      on=['underlying_name', 'timestamp'], how='left')
        test_X = test_X.rename({'underlying_name': 'unique_id', 'timestamp': 'ds'}).to_pandas()
        test_X = test_X.drop_duplicates(subset=['unique_id', 'ds'])
        test_X = test_X.set_index(['unique_id', 'ds'])
        fill_cols = test_X.columns
        test_X[fill_cols] = test_X.groupby(level=0)[fill_cols].ffill()
        test_X[fill_cols] = test_X.groupby(level=0)[fill_cols].bfill()
        test_X = test_X.reset_index()
        lightgbm_y = \
            lightgbm_model.predict(h=n_heads * n_steps_ahead_scaler,
                                   X_df=test_X.drop(columns=[args.target])).set_index(['unique_id', 'ds'])
        lasso_model, stats = model[1], model[2]
        mean, std = stats.xs('mean', axis=1, level=1), stats.xs('std', axis=1, level=1).clip(lower=1e-8)
        test_lasso_X = test_X.copy().rename(columns={args.target: 'y'}).set_index(['unique_id', 'ds'])
        test_lasso_X = test_lasso_X.subtract(mean, axis=0).div(std, axis=0).reset_index()
        lasso_feature_cols = [c for c in test_lasso_X.columns if c not in ('unique_id', 'ds', 'y')]
        n_nan_before = test_lasso_X[lasso_feature_cols].isna().sum().sum()
        if n_nan_before > 0:
            print(f"WARNING: {n_nan_before} NaN value(s) remained in test_lasso_X after ffill/bfill and "
                  f"z-scoring -- filling with 0 (= training mean) so Lasso.predict() doesn't crash. "
                  f"If this happens often, investigate which columns/windows are all-null.")
            test_lasso_X[lasso_feature_cols] = test_lasso_X[lasso_feature_cols].fillna(0)
        lasso_y = lasso_model.predict(h=n_heads * n_steps_ahead_scaler, X_df=test_lasso_X.drop(columns='y'))
        lasso_y = lasso_y.set_index(['unique_id', 'ds'])
        lasso_y['Lasso'] = lasso_y['Lasso'].mul(std['y'], axis=0, level=0).add(mean['y'], axis=0, level=0)
        tmp_model = [lightgbm_y, lasso_y]
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
    train_features = train_features.to_pandas().rename(columns={'underlying_name': 'unique_id', args.target: 'y'})
    in_sample_avg = train_features.groupby('unique_id')['y'].diff().abs().groupby(train_features['unique_id']).mean()
    delta_ys = -1*test_y[regression_dd[regression]].subtract(test_y['y'], axis=0)
    in_sample_avg = in_sample_avg.reindex(delta_ys.index, level=0)
    mase_mean = delta_ys.div(in_sample_avg.values, axis=0).abs().rename(columns=model_names)
    mase_mean = mase_mean.stack().reset_index().rename(
        columns={'unique_id': 'underlying_name', 'ds': 'timestamp', 'level_2': 'regression', 0: 'MASE'}).dropna()
    sample_columns_pool = test_X.columns[~test_X.columns.isin(['unique_id', 'ds'])].tolist()
    k = min(3, len(sample_columns_pool))
    if k == 0:
        print("WARNING: no feature columns available for permutation importance in this "
              "window -- feature_importance will be empty for this window.")
        perm_df = pd.DataFrame(columns=['underlying_name', 'timestamp', 'regression', 'MASE', 'features'])
    else:
        if k < 3:
            print(f"WARNING: only {k} feature column(s) available for permutation importance in "
                  f"this window (expected 3) -- using all {k} instead of failing.")
        sample_columns = random.sample(sample_columns_pool, k=k)
        perm_df = list()
        for col in sample_columns:
            perm_df.append(compute_feat_importance(col=col, test_X=test_X,
                                                   test_lasso_X=test_lasso_X if regression == 'ml' else None, test_y=test_y,
                                                   in_sample_avg=in_sample_avg, args=args, regression=regression,
                                                   model_names=model_names, n_heads=n_heads,
                                                   n_steps_ahead_scaler=n_steps_ahead_scaler,
                                                   lightgbm_model=lightgbm_model if regression == 'ml' else None,
                                                   lasso_model=lasso_model if regression == 'ml' else None,
                                                   softs_model=softs_model if regression == 'dl' else None,
                                                   mean=mean if regression == 'ml' else None,
                                                   std=std if regression == 'ml' else None))
        perm_df = pd.concat(perm_df, ignore_index=True)
    forecast_df = test_y.set_index('y', append=True).rename(columns=model_names)
    forecast_df = forecast_df.stack().reset_index().rename(
        columns={'unique_id': 'underlying_name', 'ds': 'timestamp', 'level_3': 'regression', 0: 'y_hat'})
    return forecast_df, mase_mean, perm_df


def process_one_test_window(trained_model, features_df, main_features, args, tmp_sum, date_grid_test, horizon_date,
                            regression: str):
    if trained_model is None:
        print(f"WARNING: no trained model available (current window or any earlier fallback) "
              f"for test window {date_grid_test} / {regression} -- skipping (no signal produced).")
        return None
    try:
        forecast, metric, feature_importance = evaluate_one_window(model=trained_model,
                                                                   train_features=features_df,
                                                                   regression=regression,
                                                                   args=args,
                                                                   n_steps_ahead_scaler=tmp_sum,
                                                                   main_features=main_features)
    except Exception as e:
        print(f"WARNING: evaluate_one_window failed for test window {date_grid_test} / {regression} "
              f"({type(e).__name__}: {e}) -- skipping this window (no signal produced).")
        return None
    start = pd.Timestamp(date_grid_test)
    end = horizon_date
    f_mask = (forecast.timestamp >= start) & (forecast.timestamp < end)
    m_mask = (metric.timestamp >= start) & (metric.timestamp < end)
    fi_mask = (feature_importance.timestamp >= start) & (feature_importance.timestamp < end)
    return forecast.loc[f_mask], metric.loc[m_mask], feature_importance.loc[fi_mask]


def resolve_trained_model(trained_models, date_grid, date, regression, L_days):
    for candidate_date in sorted((d for d in date_grid if d <= date), reverse=True):
        model = trained_models.get((candidate_date, regression))
        if isinstance(model, (list, neuralforecast.core.NeuralForecast)):
            return model
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script generating all results presented in the associated paper.\n'
                                                 'Results are computed model by model.')
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
    features = pl.scan_csv(f'{args.target_path}/spot_features.csv').filter(pl.col('target') == args.target).collect()
    features = features.with_columns([pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%z'),
                                      pl.col('features_values').cast(pl.Float32, strict=False)])
    features = features.pivot(values='features_values',
                              index=['underlying_name', 'timestamp', 'target', 'target_values'], on='features',
                              aggregate_function='first')
    features = features.drop('target').with_columns(pl.col('target_values').alias(args.target)).drop('target_values')
    features = features.unique(maintain_order=True)
    global main_features
    if args.options == 1:
        options_features = pl.scan_csv(f'{args.target_path}/option_features.csv').collect()
        options_features = options_features.with_columns([pl.col('timestamp').str.strptime(pl.Datetime,
                                                                                           '%Y-%m-%d %H:%M:%S%z'),
                                                          pl.col('features_values').cast(pl.Float32, strict=False)])
        options_features = options_features.pivot(values='features_values', index=['underlying_name', 'timestamp'],
                                                  on='features', aggregate_function='first')
        main_features = features.join(options_features, on=['underlying_name', 'timestamp'], how='left')
        main_features = main_features.sort(['underlying_name', 'timestamp'])
        main_features = main_features.sort(['underlying_name', 'timestamp']).with_columns(
            pl.all().forward_fill().over('underlying_name')
        )
    else:
        main_features = features
    main_features = main_features.with_columns(pl.col('timestamp').dt.replace_time_zone(None))
    start_idx = np.searchsorted(dates, dates[0] + relativedelta(days=DAYS_CONVERTER[args.L]))
    date_grid = dates[start_idx::DAYS_CONVERTER[args.L]]
    date_masks = {d: ((main_features['timestamp'] >= pl.lit(d - relativedelta(days=DAYS_CONVERTER[args.L]))) &
                      (main_features['timestamp'] < pl.lit(d))) for d in date_grid}
    contemporaneous_cols_to_drop = [
        col for col in
        contemporanous_features[args.target] + [f'{c}_{args.D}' for c in contemporanous_features[args.target]]
        if col in main_features.columns
    ]
    main_features = main_features.drop(contemporaneous_cols_to_drop)
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
    retrain_idx = np.clip(np.searchsorted(date_grid_np, date_grid_test, side='right') - 1,
                          0, len(date_grid_np) - 1)
    in_sample_dates = date_grid_np[retrain_idx]
    _test_ts = pd.to_datetime(date_grid_test)
    _retrain_ts = pd.to_datetime(in_sample_dates)
    n_steps_ahead_scaler = (((_test_ts - _retrain_ts).days // D_days) + 1).to_numpy()
    dates = pd.to_datetime(date_grid_test).to_numpy(dtype='datetime64[ns]')
    end = np.datetime64(args.end_date)
    horizon_dates = np.minimum(dates + np.timedelta64(D_days, 'D'), end)
    feature_cache = {d: main_features.filter(date_masks[d])[['underlying_name', args.target]]
                     for d in np.unique(in_sample_dates)}
    main_features_test_cache = {d: main_features.filter(date_masks_tests[d]) for d in date_masks_tests}
    date_regression_grid_test = list(itertools.product(range(len(date_grid_test)), regression_dd.keys()))
    with tqdm_joblib(tqdm(desc='Test windows', total=len(date_regression_grid_test))):
        results = Parallel(n_jobs=args.cpu_count, backend='loky')(
            delayed(process_one_test_window)(
                resolve_trained_model(trained_models, date_grid, in_sample_dates[i], regression, L_days),
                feature_cache[in_sample_dates[i]],
                main_features_test_cache[date_grid_test[i]],
                args, n_steps_ahead_scaler[i], date_grid_test[i],
                horizon_dates[i], regression)
            for i, regression in date_regression_grid_test)
    n_total_windows = len(results)
    results = [r for r in results if r is not None]
    n_skipped = n_total_windows - len(results)
    if n_skipped > 0:
        print(f"WARNING: {n_skipped} of {n_total_windows} test windows were skipped entirely "
              f"(no trained model available, even via fallback, or evaluation failed) -- "
              f"see the WARNING messages above for which ones. Forecasts/MASE/feature "
              f"importance will have gaps for those windows.")
    if not results:
        raise RuntimeError("No test windows produced any results (all were skipped or failed) "
                           "-- nothing to write out. Check the WARNING messages above.")
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
        underlying_rv.filter((pl.col('freq') == '1h') & (pl.col('target') == 'RV') &
                             (pl.col('features') == 'RV_1h')))
    underlying_rv = underlying_rv.select(['timestamp', 'target_values'])
    underlying_rv = underlying_rv.with_columns(pl.col('timestamp')
                                               .str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S%z', strict=False)
                                               .dt.convert_time_zone('UTC')).collect().to_pandas()
    volatility_regime = underlying_rv.groupby('timestamp')['target_values'].mean().resample(args.freq).sum()
    volatility_regime = \
        volatility_regime.loc[(volatility_regime.index.date < pd.to_datetime(args.end_date, utc=True).date())]
    volatility_regime = pd.qcut(volatility_regime, q=[0, .33, .66, 1], labels=VOL_REGIMES)
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
    print('[Done] All forecasts, MASE, and feature importance computed and stored.')