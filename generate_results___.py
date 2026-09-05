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
from lightgbm import LGBMRegressor
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
regression_champion_dd = {'lasso': Lasso, 'lightgbm': LGBMRegressor, 'softs': SOFTS}
rng = random.Random(123)
D_LIST = ['1D', '1W', '1M']
TARGET_LIST = ['RV', 'returns']
REPORT_FREQS = ['15min', '30min', '1h']
FILTERED_COLUMNS = {'RV': r"^RV_(?:\d+min|1h|1D|1W|1M)$", 'returns': r"^m\d+_(?:\d+min|1h|1D|1W|1M)$",}
SOFTS_MAX_INPUT_SIZE = 64
SOFTS_VAL_FRACTION = 5
SOFTS_MAX_VAL_SIZE = 30
SOFTS_H_FRACTION = 0.5


def filter_market_features(features: pl.DataFrame, target: str, market: int) -> pl.DataFrame:
    if market != 0:
        return features
    return features.select(pl.col('underlying_name'), pl.col('timestamp'), pl.col(target),
                           pl.col(FILTERED_COLUMNS[target]))


def compute_softs_sizing(train_X: pd.DataFrame, desired_h: int) -> tuple[int, int, int]:
    series_lengths = train_X.groupby('unique_id').size()
    min_series_len = int(series_lengths.min())
    max_feasible_h = max(1, int(min_series_len * SOFTS_H_FRACTION) - SOFTS_MAX_VAL_SIZE - 2)
    h = max(1, min(desired_h, max_feasible_h))
    min_required = h + 2
    if min_series_len < min_required:
        raise ValueError(
            f'shortest series has {min_series_len} timestamp(s) after preprocessing, '
            f'need at least {min_required} (h={h}) to train SOFTS'
        )
    if h <= SOFTS_MAX_VAL_SIZE and min_series_len - h - SOFTS_MAX_VAL_SIZE >= 2:
        val_size = max(h, min(SOFTS_MAX_VAL_SIZE, min_series_len // SOFTS_VAL_FRACTION))
        val_size = min(val_size, min_series_len - h - 1)
    else:
        val_size = 0
    remaining = min_series_len - val_size - h
    input_size = max(1, min(remaining, SOFTS_MAX_INPUT_SIZE))
    return input_size, val_size, h


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


def train_one_window(date, features, regression: str, market: int, D: str, target: str, freq: str, L_days: int):
    torch.set_num_threads(1)
    try:
        features = filter_market_features(features, target, market)
        train_X = features.to_pandas().set_index(['underlying_name', 'timestamp']).dropna()
        train_X = train_X.reset_index().rename(
            columns={'underlying_name': 'unique_id', 'timestamp': 'ds', target: 'y'})
        avg_1s = train_X[['unique_id', 'ds', 'y']].set_index(['unique_id', 'ds'])
        avg_1s = \
            pd.concat([avg_1s, avg_1s.groupby(['unique_id'])[['y']].shift().rename(columns={'y': 'y_lag'})], axis=1)
        avg_1s = avg_1s.diff(axis=1).dropna(how='all', axis=1).abs()
        avg_1s = avg_1s.groupby('unique_id').agg(lambda x: x.sum()/(x.shape[0]-1)).rename(columns={'y_lag': 'avg_1s'})
        per_uid_range = train_X.groupby('unique_id')['ds'].agg(['min', 'max'])
        full_index = pd.MultiIndex.from_tuples(
            [(uid, d) for uid, (lo, hi) in per_uid_range.iterrows()
             for d in pd.date_range(lo, hi, freq=freq, inclusive='both')],
            names=['unique_id', 'ds']
        )
        train_X = train_X.drop_duplicates(subset=['unique_id', 'ds']).set_index(['unique_id', 'ds']).reindex(full_index)
        train_X = train_X.reset_index()
        train_X = winsorise_vectorized(train_X, group_col='unique_id')
        if train_X.empty:
            raise ValueError('empty training window after reindex/winsorisation')
        fill_cols = [c for c in train_X.columns if c not in ('unique_id', 'ds')]
        train_X[fill_cols] = train_X.groupby('unique_id')[fill_cols].ffill()
        train_X[fill_cols] = train_X.groupby('unique_id')[fill_cols].bfill()
        if regression == 'dl':
            n_series = train_X['unique_id'].nunique()
            freq_delta_days = pd.to_timedelta(freq) / pd.Timedelta(days=1)
            desired_h = int(np.ceil((L_days + DAYS_CONVERTER[D]) / freq_delta_days))
            input_size, val_size, h = compute_softs_sizing(train_X, desired_h)
            model = SOFTS(h=h, input_size=input_size, n_series=n_series, hidden_size=256,
                          loss=MSE(), scaler_type='standard', learning_rate=1e-3, e_layers=2, d_ff=512,
                          batch_size=64, max_steps=100, val_check_steps=10,
                          early_stop_patience_steps=(EARLY_STOPPING_PATIENCE if val_size > 0 else -1),
                          start_padding_enabled=True, accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                          devices=1)
            model = NeuralForecast(models=[model], freq=freq)
            model.fit(df=train_X, val_size=val_size)
        else:
            model = \
                MLForecast(models=lgb.LGBMRegressor(force_col_wise=True, learning_rate=1e-1, n_estimators=500,
                                                    n_jobs=1, subsample=0.8, colsample_bytree=0.8),
                           freq=freq, target_transforms=[])
            model.fit(train_X, fitted=False, static_features=[])
            model2 = MLForecast(models=Lasso(), freq=freq, target_transforms=[])
            num_cols = train_X.select_dtypes(include='number').columns
            train_group = train_X.groupby('unique_id')[num_cols].agg(['mean', 'std'])
            mean = train_group.xs('mean', axis=1, level=1)
            std = train_group.xs('std', axis=1, level=1).clip(lower=1e-8)
            train_X = train_X.set_index(['unique_id', 'ds']).subtract(mean, axis=0).div(std, axis=0).reset_index()
            model2.fit(train_X, fitted=False, static_features=[])
            model = [model, model2, train_group]
        return (date, regression, market, D), (model, avg_1s)
    except Exception as e:
        print(f"WARNING: training window ({date}, {regression}, market={market}, D={D}) failed "
              f"({type(e).__name__}: {e}) -- no model produced for this window.")
        return (date, regression, market, D), None


def get_n_heads(regression: str) -> int:
    return 1


def direct_predict(fitted_mlforecast, model_name: str, X_df: pd.DataFrame) -> pd.DataFrame:
    features_order = fitted_mlforecast.ts.features_order_
    missing = [c for c in features_order if c not in X_df.columns]
    if missing:
        raise ValueError(f'direct_predict: X_df is missing required feature columns {missing}')
    fitted_est = fitted_mlforecast.models_[model_name]
    preds = fitted_est.predict(X_df[features_order])
    out = X_df[['unique_id', 'ds']].copy()
    out[model_name] = preds
    return out


def evaluate_one_window(model, regression: str, target: str, freq: str, n_steps_ahead_scaler, main_features):
    n_heads = 1
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
        lightgbm_y = direct_predict(lightgbm_model, 'LGBMRegressor',
                                    test_X.drop(columns=[target])).set_index(['unique_id', 'ds'])
        lasso_model, stats = model[1], model[2]
        mean, std = stats.xs('mean', axis=1, level=1), stats.xs('std', axis=1, level=1).clip(lower=1e-8)
        test_lasso_X = test_X.copy().rename(columns={target: 'y'}).set_index(['unique_id', 'ds'])
        test_lasso_X = test_lasso_X.subtract(mean, axis=0).div(std, axis=0).reset_index()
        lasso_feature_cols = [c for c in test_lasso_X.columns if c not in ('unique_id', 'ds', 'y')]
        n_nan_before = test_lasso_X[lasso_feature_cols].isna().sum().sum()
        if n_nan_before > 0:
            print(f"WARNING: {n_nan_before} NaN value(s) remained in test_lasso_X after ffill/bfill and "
                  f"z-scoring -- filling with 0 (= training mean) so Lasso.predict() doesn't crash.")
            test_lasso_X[lasso_feature_cols] = test_lasso_X[lasso_feature_cols].fillna(0)
        lasso_y = direct_predict(lasso_model, 'Lasso', test_lasso_X.drop(columns='y'))
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
                                      futr_df=test_X.drop(columns=[target])).set_index(['unique_id', 'ds'])
        tmp_model = [softs_y]
    test_y = test_X[[target, 'ds', 'unique_id']].set_index(['unique_id', 'ds']).rename(columns={target: 'y'})
    test_y = pd.concat([test_y]+tmp_model, axis=1)
    n_before = len(test_y)
    test_y = test_y[test_y.index.get_level_values('ds').notna()]
    n_dropped = n_before - len(test_y)
    if n_dropped > 0:
        print(f"WARNING: dropped {n_dropped} row(s) with NaT ds for regression={regression} "
              f"(h={n_heads * n_steps_ahead_scaler}) before grouping.")
    test_y = test_y.groupby([pd.Grouper(level=0), pd.Grouper(level=1, freq=freq)]).sum()
    forecast_df = test_y.set_index('y', append=True).rename(columns=model_names)
    forecast_df = forecast_df.stack().reset_index().rename(
        columns={'unique_id': 'underlying_name', 'ds': 'timestamp', 'level_3': 'regression', 0: 'y_hat'})
    return forecast_df


def process_one_test_window(trained_entry, main_features, target, freq, tmp_sum, date_grid_test, horizon_date,
                            regression: str, market: int, D: str):
    if trained_entry is None:
        print(f"WARNING: no trained model available (current window or any earlier fallback) "
              f"for test window {date_grid_test} / {regression} / market={market} / D={D} -- "
              f"skipping (no signal produced).")
        return None, None
    trained_model, avg_1s = trained_entry
    try:
        main_features = filter_market_features(main_features, target, market)
        forecast = evaluate_one_window(model=trained_model, regression=regression, target=target, freq=freq,
                                       n_steps_ahead_scaler=tmp_sum, main_features=main_features)
    except Exception as e:
        print(f"WARNING: evaluate_one_window failed for test window {date_grid_test} / {regression} "
              f"/ market={market} / D={D} ({type(e).__name__}: {e}) -- skipping this window.")
        return None, None
    start = pd.Timestamp(date_grid_test)
    end = horizon_date
    f_mask = (forecast.timestamp >= start) & (forecast.timestamp < end)
    forecast = forecast.loc[f_mask]
    if avg_1s is not None and not forecast.empty:
        avg_1s_tagged = avg_1s.reset_index().assign(retrain_date=date_grid_test, D=D)
    else:
        avg_1s_tagged = None
    return forecast.assign(freq=freq), avg_1s_tagged


def resolve_trained_entry(trained_models, date_grid, date, regression, market, D):
    for candidate_date in sorted((d for d in date_grid if d <= date), reverse=True):
        entry = trained_models.get((candidate_date, regression, market, D))
        if entry is None:
            continue
        model, avg_1s = entry
        if isinstance(model, (list, neuralforecast.core.NeuralForecast)):
            return (model, avg_1s)
    return None


def run_for_target(target: str, args):
    dates = pd.date_range(args.start_date, args.end_date, freq='1D', tz=pytz.UTC, inclusive='left').date.tolist()
    volatility_regime = pl.scan_parquet(f'{args.target_path}/spot_features.parquet').filter(
        (pl.col('target') == 'RV') & (pl.col('freq') == '15min')
    ).select(['underlying_name', 'timestamp', 'target_values']).unique(maintain_order=True).collect().to_pandas()
    volatility_regime = volatility_regime.groupby('timestamp')['target_values'].mean()
    volatility_regime = \
        {freq: pd.DataFrame(pd.qcut(volatility_regime.resample(freq).sum(), q=[0, .33, .66, 1],
                                    labels=VOL_REGIMES)).assign(freq=freq) for freq in ['15min', '30min', '1h']}
    volatility_regime = pd.concat(volatility_regime.values()).rename(columns={'target_values': 'volatility_regime'})
    volatility_regime = \
        volatility_regime.loc[(volatility_regime.index.date >= pd.to_datetime(args.start_date, utc=True).date())&
                              (volatility_regime.index.date < pd.to_datetime(args.end_date, utc=True).date())]
    features = \
        pl.scan_parquet(f'{args.target_path}/spot_features.parquet').filter(pl.col('target') == target).collect()
    features = features.with_columns([pl.col('features_values').cast(pl.Float32, strict=False)])
    features = features.pivot(values='features_values',
                              index=['underlying_name', 'timestamp', 'target', 'target_values'], on='features',
                              aggregate_function='first')
    features = features.drop('target').with_columns(pl.col('target_values').alias(target)).drop('target_values')
    features = features.unique(maintain_order=True)
    options_features = pl.scan_parquet(f'{args.target_path}/option_features.parquet').collect()
    options_features = options_features.with_columns([pl.col('features_values').cast(pl.Float32, strict=False)])
    options_features = options_features.pivot(values='features_values', index=['underlying_name', 'timestamp'],
                                              on='features', aggregate_function='first')
    main_features_base = features.join(options_features, on=['underlying_name', 'timestamp'], how='left')
    main_features_base = main_features_base.sort(['underlying_name', 'timestamp']).with_columns(
        pl.all().forward_fill().over('underlying_name')
    )
    main_features_base = main_features_base.with_columns(pl.col('timestamp').dt.replace_time_zone(None))
    main_features_by_D = dict()
    for D in D_LIST:
        n_heads_agg = int(pd.to_timedelta(DAYS_CONVERTER[D], unit='d') // pd.to_timedelta(args.freq))
        rolled = main_features_base.sort(['underlying_name', 'timestamp']).with_columns(
            pl.col(target).rolling_sum(window_size=n_heads_agg, min_samples=n_heads_agg)
              .shift(-(n_heads_agg - 1)).over('underlying_name').alias(target)
        )
        cols_to_drop = [
            col for col in contemporanous_features[target] + [f'{c}_{D}' for c in contemporanous_features[target]]
            if col in rolled.columns
        ]
        main_features_by_D[D] = rolled.drop(cols_to_drop)
    L_days = DAYS_CONVERTER[args.L]
    start_idx = np.searchsorted(dates, dates[0] + relativedelta(days=L_days))
    date_grid = dates[start_idx::L_days]
    date_masks = {d: ((main_features_base['timestamp'] >= pl.lit(d - relativedelta(days=L_days))) &
                      (main_features_base['timestamp'] < pl.lit(d))) for d in date_grid}
    date_grid_regressions_markets_Ds = \
        list(itertools.product(date_grid, list(regression_dd.keys()), [0, 1], D_LIST))
    print(f'[Training] {len(date_grid_regressions_markets_Ds)} (date, regression, market, D) windows '
          f'to train across {args.cpu_count} worker(s) for target={target}.')
    with tqdm_joblib(tqdm(desc=f'Model training windows (target={target})',
                          total=len(date_grid_regressions_markets_Ds))):
        trained_models = dict(Parallel(n_jobs=args.cpu_count, backend='loky')(
            delayed(train_one_window)(
                date, main_features_by_D[D].filter(date_masks[date]), regression, market, D,
                target, args.freq, L_days)
            for date, regression, market, D in date_grid_regressions_markets_Ds
        ))
    date_grid_np = np.array(date_grid)
    end = np.datetime64(args.end_date)
    freq_delta = pd.to_timedelta(args.freq)
    freq_ns = np.timedelta64(freq_delta.value, 'ns')
    date_grid_test_by_D = dict()
    date_masks_tests_by_D = dict()
    in_sample_dates_by_D = dict()
    horizon_dates_by_D = dict()
    n_steps_ahead_scaler_by_D = dict()
    for D in D_LIST:
        D_days = DAYS_CONVERTER[D]
        grid_test = dates[dates.index(date_grid[0])::D_days]
        date_masks_tests_by_D[D] = {
            d: ((main_features_base['timestamp'] < pl.lit(d + relativedelta(days=D_days))) &
                (main_features_base['timestamp'] >= pl.lit(d - relativedelta(days=D_days + D_days))))
            for d in grid_test
        }
        grid_test = np.array(grid_test)
        date_grid_test_by_D[D] = grid_test
        retrain_idx = np.clip(np.searchsorted(date_grid_np, grid_test, side='right') - 1, 0, len(date_grid_np) - 1)
        in_sample_dates_by_D[D] = date_grid_np[retrain_idx]
        _dates = pd.to_datetime(grid_test).to_numpy(dtype='datetime64[ns]')
        horizon_dates_by_D[D] = np.minimum(_dates + np.timedelta64(D_days, 'D'), end)
        _retrain_ts = pd.to_datetime(in_sample_dates_by_D[D]).to_numpy(dtype='datetime64[ns]')
        _horizon_ts = pd.to_datetime(horizon_dates_by_D[D]).to_numpy(dtype='datetime64[ns]')
        raw_steps = (_horizon_ts - _retrain_ts).astype('timedelta64[ns]') / freq_ns
        n_steps_ahead_scaler_by_D[D] = np.clip(np.ceil(raw_steps), 1, None).astype(int)
    for market in [1, 0]:
        all_forecasts = []
        all_avg_1s = []
        for D in D_LIST:
            main_features_test_cache = {
                d: filter_market_features(main_features_by_D[D].filter(mask), target, market)
                for d, mask in date_masks_tests_by_D[D].items()
            }
            date_grid_test = date_grid_test_by_D[D]
            date_regression_grid_test = list(itertools.product(range(len(date_grid_test)), regression_dd.keys()))
            with tqdm_joblib(tqdm(desc=f'Test windows (target={target}, market={market}, D={D})',
                                  total=len(date_regression_grid_test))):
                results = Parallel(n_jobs=args.cpu_count, backend='loky')(
                    delayed(process_one_test_window)(
                        resolve_trained_entry(trained_models, date_grid, in_sample_dates_by_D[D][i], regression,
                                              market, D),
                        main_features_test_cache[date_grid_test[i]],
                        target, args.freq, n_steps_ahead_scaler_by_D[D][i], date_grid_test[i],
                        horizon_dates_by_D[D][i], regression, market, D)
                    for i, regression in date_regression_grid_test)
            n_total_windows = len(results)
            pairs = [(f, a) for f, a in results if f is not None]
            n_skipped = n_total_windows - len(pairs)
            if n_skipped > 0:
                print(f"WARNING: {n_skipped} of {n_total_windows} test windows were skipped entirely "
                      f"for target={target}, market={market}, D={D} -- see WARNING messages above.")
            if not pairs:
                print(f"WARNING: no test windows produced any results for target={target}, market={market}, "
                      f"D={D} -- skipping.")
                continue
            D_forecasts = pd.concat([f for f, _ in pairs])
            D_forecasts['timestamp'] = pd.to_datetime(D_forecasts['timestamp'], utc=True)
            D_forecasts = D_forecasts.assign(target=target, market=market, D=D)
            all_forecasts.append(D_forecasts)
            D_avg_1s = [a for _, a in pairs if a is not None]
            if D_avg_1s:
                all_avg_1s.append(pd.concat(D_avg_1s, ignore_index=True))
        if not all_forecasts:
            print(f"WARNING: no forecasts produced for target={target}, market={market} across any D -- "
                  f"skipping export.")
            continue
        market_forecasts = pd.concat(all_forecasts, ignore_index=True)
        market_forecasts = market_forecasts.set_index(['underlying_name', 'timestamp'])
        market_forecasts_by_freq = []
        for D in D_LIST:
            D_slice = market_forecasts[market_forecasts['D'] == D]
            if D_slice.empty:
                continue
            for freq in REPORT_FREQS:
                grouped = \
                    D_slice.groupby(['underlying_name', 'regression',
                                     pd.Grouper(level='timestamp', freq=freq)])[['y', 'y_hat']].sum().reset_index()
                market_forecasts_by_freq.append(grouped.assign(D=D, freq=freq))
        market_forecasts = pl.concat([pl.from_pandas(df) for df in market_forecasts_by_freq]).to_pandas()
        market_forecasts = market_forecasts.assign(target=target, market=market)
        avg_flat = pd.concat(all_avg_1s, ignore_index=True)
        avg_flat = avg_flat.rename(columns={'unique_id': 'underlying_name', 'retrain_date': 'timestamp'})
        avg_flat = avg_flat.drop_duplicates(subset=['underlying_name', 'timestamp', 'D'])
        market_forecasts['timestamp'] = pd.to_datetime(market_forecasts['timestamp'], utc=True).dt.as_unit('ns')
        avg_flat['timestamp'] = pd.to_datetime(avg_flat['timestamp'], utc=True).dt.as_unit('ns')
        market_forecasts = market_forecasts.sort_values('timestamp').reset_index(drop=True)
        avg_flat = avg_flat.sort_values('timestamp').reset_index(drop=True)
        merged = pd.merge_asof(market_forecasts, avg_flat, on='timestamp', by=['underlying_name', 'D'],
                               direction='backward')
        merged['MASE'] = (merged['y_hat'] / merged['avg_1s']).abs()
        merged['D_days'] = merged['D'].map(DAYS_CONVERTER)
        merged['MASE'] = \
            merged['MASE'] / (pd.to_timedelta(merged['D_days'], unit='d') / pd.to_timedelta(merged['freq']))
        metric = merged.drop(columns=['y_hat', 'avg_1s', 'D_days'], errors='ignore')
        metric = pd.merge(metric, volatility_regime.reset_index(), on=['freq', 'timestamp'])
        metric_summary = \
            metric.groupby(['regression', 'freq', 'D', 'volatility_regime', 'market', 'target'])[['MASE']].mean()
        metric_summary = metric_summary.reset_index()
        forecasts_path = f'{args.target_path}/crypto_option_spot/forecasts_{target}_market{market}.csv'
        market_forecasts.to_csv(forecasts_path, mode='a', header=not os.path.exists(forecasts_path), index=False)
        metric_path = f'{args.target_path}/crypto_option_spot/metric_{target}_market{market}.csv'
        metric_summary.to_csv(metric_path, mode='a', header=not os.path.exists(metric_path), index=False)
        print(f'[Done] target={target}, market={market}: forecasts and metrics computed across D={D_LIST} '
              f'and freq={REPORT_FREQS}.')


def run_feature_importance(target_path: str, target: str, args):
    def parse_latex_table(path: str) -> pd.DataFrame:
        with open(path) as f:
            text = f.read()
        lines = text.splitlines()
        in_table = False
        rows = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(r'\begin{tabular}'):
                in_table = True
                continue
            if stripped.startswith(r'\end{tabular}'):
                break
            if not in_table:
                continue
            if not stripped or stripped.startswith((r'\toprule', r'\midrule', r'\bottomrule', r'\hline')):
                continue
            rows.append(stripped)

        def split_row(row: str) -> list[str]:
            row = row.rstrip()
            if row.endswith(r'\\'):
                row = row[:-2]
            return [cell.strip() for cell in row.split('&')]

        parsed_rows = [split_row(r) for r in rows]
        header = parsed_rows[0]
        col_labels = header[1:]

        data_rows = parsed_rows[1:]
        if data_rows and all(c == '' for c in data_rows[0][1:]):
            data_rows = data_rows[1:]  # skip pandas' index-name spacer row

        index_labels = [r[0] for r in data_rows]
        values = [[float(c) for c in r[1:]] for r in data_rows]
        return pd.DataFrame(values, index=index_labels, columns=col_labels)

    def best_row_col(df: pd.DataFrame) -> tuple[str, str, float]:
        row, col = df.stack().idxmin()
        return row, col
    mase_target_table = parse_latex_table('/'.join([target_path,
                                                    'crypto_option_spot',
                                                    f'mase_per_regression_freq_pair_{target}.txt']))
    freq, regression = best_row_col(mase_target_table)
    dates = pd.date_range(args.start_date, args.end_date, freq='1D', tz=pytz.UTC, inclusive='left').date.tolist()
    volatility_regime = pl.scan_parquet(f'{args.target_path}/spot_features.parquet').filter(
        (pl.col('target') == 'RV') & (pl.col('freq') == '15min')
    ).select(['underlying_name', 'timestamp', 'target_values']).unique(maintain_order=True).collect().to_pandas()
    volatility_regime = volatility_regime.groupby('timestamp')['target_values'].mean()
    volatility_regime = \
        {freq: pd.DataFrame(pd.qcut(volatility_regime.resample(freq).sum(), q=[0, .33, .66, 1],
                                    labels=VOL_REGIMES)).assign(freq=freq) for freq in ['15min', '30min', '1h']}
    volatility_regime = pd.concat(volatility_regime.values()).rename(columns={'target_values': 'volatility_regime'})
    volatility_regime = \
        volatility_regime.loc[(volatility_regime.index.date >= pd.to_datetime(args.start_date, utc=True).date()) &
                              (volatility_regime.index.date < pd.to_datetime(args.end_date, utc=True).date())]
    features = \
        pl.scan_parquet(f'{args.target_path}/spot_features.parquet').filter(pl.col('target') == target).collect()
    features = features.with_columns([pl.col('features_values').cast(pl.Float32, strict=False)])
    features = features.pivot(values='features_values',
                              index=['underlying_name', 'timestamp', 'target', 'target_values'], on='features',
                              aggregate_function='first')
    features = features.drop('target').with_columns(pl.col('target_values').alias(target)).drop('target_values')
    features = features.unique(maintain_order=True)
    options_features = pl.scan_parquet(f'{args.target_path}/option_features.parquet').collect()
    options_features = options_features.with_columns([pl.col('features_values').cast(pl.Float32, strict=False)])
    options_features = options_features.pivot(values='features_values', index=['underlying_name', 'timestamp'],
                                              on='features', aggregate_function='first')
    main_features_base = features.join(options_features, on=['underlying_name', 'timestamp'], how='left')
    main_features_base = main_features_base.sort(['underlying_name', 'timestamp']).with_columns(
        pl.all().forward_fill().over('underlying_name')
    )
    main_features_base = main_features_base.with_columns(pl.col('timestamp').dt.replace_time_zone(None))
    pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script generating all results presented in the associated paper.\n'
                    'Self-contained: runs both targets, all markets, all reporting '
                    'frequencies, and all forecasting horizons in one execution.',
        allow_abbrev=False)
    parser.add_argument('--L', type=str, help='Parameter controlling the size of the training set size.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2023-07-01')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    parser.add_argument('--cpu_count', type=int, default=2, help='Number of processes to use for the pipeline.')
    parser.add_argument('--feature_importance', type=int, default=0, help='Feature importance part of the pipeline.')
    args = parser.parse_args()
    args.freq = '15min'
    print(args)
    print(f'targets={TARGET_LIST}, markets=[1, 0], D={D_LIST}, report_freqs={REPORT_FREQS}')
    for target in TARGET_LIST:
        if args.feature_importance == 0:
            run_for_target(target, args)
        else:
            run_feature_importance(args.target_path, target, args)
    print('[Done] All targets, markets, D, and reporting frequencies processed.')