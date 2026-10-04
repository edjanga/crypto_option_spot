import argparse
import os
import re
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import polars as pl
import pytz
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
import itertools
from constants import *
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import random
from constants import EARLY_STOPPING_PATIENCE
from typing import Dict
contemporanous_features = {'returns': ['m2', 'm3', 'm4', 'm5', 'm6'], 'RV': []}
regression_dd = {'ml': ['LGBMRegressor'], 'dl': ['MLPRegressor']}
model_names = {**model_names, 'MLPRegressor': 'mlp'}
rng = random.Random(123)
D_LIST = ['1D', '1W', '1M']
TARGET_LIST = ['RV', 'returns']
REPORT_FREQS = ['15min', '30min', '1h']
FILTERED_COLUMNS = {'RV': r"^RV_(?:\d+min|1h|1D|1W|1M)$", 'returns': r"^m\d+_(?:\d+min|1h|1D|1W|1M)$",}
MLP_HIDDEN_LAYERS = (64, 32)
MLP_LEARNING_RATE = 1e-3
MLP_VALIDATION_FRACTION = 0.1
MLP_SEED = 123
FEAT_IMP_N_ITER = 10
# Sampling frequency of the rows of spot_features.parquet / option_features.parquet. The target of a horizon D is the
# sum of the per-bar target over the next D, i.e. over D / BASE_FREQ rows of that grid, whatever the reporting
# frequency of the model. (Counting the window in bars of the reporting frequency made the 30min and 1h models
# forecast the next D/2 and D/4.)
BASE_FREQ = '15min'
# True: MASE = |y - y_hat| / in-sample naive error. False: the previous formula |y_hat| / in-sample naive error.
MASE_FROM_FORECAST_ERROR = True
# The target of a row at time t is measured over [t, t + D]: it is only known at t + D. True: a model retrained at
# date d is trained on rows with t < d - D only, so no training target contains data from d onwards (the period it
# forecasts). The forecast span is extended by D accordingly. False: the previous behaviour (rows up to d).
EMBARGO_TRAINING = True


def embargo_days(D: str) -> int:
    return DAYS_CONVERTER[D] if EMBARGO_TRAINING else 0


def check_training_length(L_days: int) -> None:
    longest = max(DAYS_CONVERTER[D] for D in D_LIST)
    if EMBARGO_TRAINING and L_days <= longest:
        raise ValueError(f'--L covers {L_days} day(s) but the longest horizon is {longest} day(s): with '
                         f'EMBARGO_TRAINING the training window [d - L, d - D) would be empty. Use a longer --L.')


def check_base_grid(main_features_base: pl.DataFrame) -> None:
    """Stops the run if the feature rows are not on the BASE_FREQ grid: the target windows are counted in rows."""
    base = pd.to_timedelta(BASE_FREQ)
    for underlying_name, timestamps in main_features_base.select(['underlying_name', 'timestamp']).to_pandas() \
            .groupby('underlying_name')['timestamp']:
        timestamps = pd.to_datetime(timestamps).sort_values()
        n_duplicates = int(timestamps.duplicated().sum())
        steps = timestamps.drop_duplicates().diff().dropna()
        if steps.empty:
            continue
        modal_step = steps.mode().iloc[0]
        if modal_step != base:
            raise ValueError(f'{underlying_name}: the feature rows are {modal_step} apart, not BASE_FREQ={BASE_FREQ}; '
                             f'set BASE_FREQ to the sampling frequency of the feature files.')
        n_gaps = int((steps != base).sum())
        if n_duplicates or n_gaps:
            print(f'WARNING: {underlying_name}: {n_duplicates} duplicated timestamp(s) and {n_gaps} gap(s) on the '
                  f'{BASE_FREQ} grid ({len(timestamps)} rows) -- target windows spanning them do not cover exactly D.',
                  flush=True)
        else:
            print(f'[Grid check] {underlying_name}: {len(timestamps)} rows, all {BASE_FREQ} apart.', flush=True)


def filter_market_features(features: pl.DataFrame, target: str, market: int) -> pl.DataFrame:
    if market != 0:
        return features
    return features.select(pl.col('underlying_name'), pl.col('timestamp'), pl.col(target),
                           pl.col(FILTERED_COLUMNS[target]))


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
        if EMBARGO_TRAINING and features.height > 0:
            last_training_row = features['timestamp'].max()
            first_forbidden_row = pd.Timestamp(date) - pd.Timedelta(days=DAYS_CONVERTER[D])
            if pd.Timestamp(last_training_row) >= first_forbidden_row:
                raise ValueError(f'training rows reach {last_training_row}, inside the embargo starting '
                                 f'{first_forbidden_row}')
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
        feature_cols = [c for c in train_X.columns if c not in ('unique_id', 'ds', 'y')]
        last_date = train_X.groupby('unique_id')['ds'].max()
        if regression == 'dl':
            fill_values = train_X[feature_cols].median().fillna(0.)
            clip_low, clip_high = train_X[feature_cols].min().fillna(0.), train_X[feature_cols].max().fillna(0.)
            mlp = make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=MLP_HIDDEN_LAYERS, activation='relu', solver='adam',
                             learning_rate_init=MLP_LEARNING_RATE, batch_size=64, max_iter=MAX_ITER,
                             early_stopping=True, validation_fraction=MLP_VALIDATION_FRACTION,
                             n_iter_no_change=EARLY_STOPPING_PATIENCE, random_state=MLP_SEED))
            mlp = TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())
            mlp.fit(train_X[feature_cols].fillna(fill_values).to_numpy(dtype=float), train_X['y'].to_numpy(dtype=float))
            model = {'estimator': mlp, 'name': 'MLPRegressor', 'feature_cols': feature_cols,
                     'last_date': last_date, 'fill_values': fill_values, 'clip': (clip_low, clip_high)}
        else:
            lgb_model = lgb.LGBMRegressor(force_col_wise=True, learning_rate=1e-1, n_estimators=500,
                                          n_jobs=1, subsample=0.8, colsample_bytree=0.8)
            lgb_model.fit(train_X[feature_cols], train_X['y'])
            model = {'estimator': lgb_model, 'name': 'LGBMRegressor', 'feature_cols': feature_cols,
                     'last_date': last_date, 'fill_values': None}
        return (date, regression, market, D, freq), (model, avg_1s)
    except Exception as e:
        print(f"WARNING: training window ({date}, {regression}, market={market}, D={D}, freq={freq}) failed "
              f"({type(e).__name__}: {e}) -- no model produced for this window.")
        return (date, regression, market, D, freq), None


def evaluate_one_window(model, regression: str, target: str, freq: str, n_steps_ahead_scaler, main_features):
    n_heads = 1
    h = n_heads * n_steps_ahead_scaler
    estimator = model['estimator']
    feature_cols = model['feature_cols']
    last_date = model['last_date']
    idx_test = pd.concat([
        pd.DataFrame({'unique_id': uid,
                     'ds': pd.date_range(start=ld + pd.Timedelta(freq), periods=h, freq=freq)})
        for uid, ld in last_date.items()
    ], ignore_index=True)
    test_X = pl.from_pandas(idx_test).with_columns(pl.col('ds').cast(pl.Datetime('us'))).rename(
        {'unique_id': 'underlying_name', 'ds': 'timestamp'}).join(main_features,
                                                                  on=['underlying_name', 'timestamp'], how='left')
    test_X = test_X.rename({'underlying_name': 'unique_id', 'timestamp': 'ds'}).to_pandas()
    test_X = test_X.drop_duplicates(subset=['unique_id', 'ds'])
    test_X = test_X.set_index(['unique_id', 'ds'])
    fill_cols = [c for c in test_X.columns if c != target]
    test_X[fill_cols] = test_X.groupby(level=0)[fill_cols].ffill()
    test_X[fill_cols] = test_X.groupby(level=0)[fill_cols].bfill()
    test_X = test_X.reset_index()
    missing_cols = [c for c in feature_cols if c not in test_X.columns]
    if missing_cols:
        raise ValueError(f'evaluate_one_window: test_X is missing required feature columns {missing_cols}')
    predictions = test_X[['unique_id', 'ds']].copy()
    if model['fill_values'] is None:
        predictions[model['name']] = estimator.predict(test_X[feature_cols])
    else:
        inputs = test_X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(model['fill_values'])
        inputs = inputs.clip(lower=model['clip'][0], upper=model['clip'][1], axis=1)
        predictions[model['name']] = estimator.predict(inputs.to_numpy(dtype=float))
    predictions = predictions.set_index(['unique_id', 'ds'])
    tmp_model = [predictions]
    test_y = test_X[[target, 'ds', 'unique_id']].set_index(['unique_id', 'ds']).rename(columns={target: 'y'})
    test_y = pd.concat([test_y]+tmp_model, axis=1)
    n_before = len(test_y)
    test_y = test_y[test_y.index.get_level_values('ds').notna()]
    n_dropped = n_before - len(test_y)
    if n_dropped > 0:
        print(f"WARNING: dropped {n_dropped} row(s) with NaT ds for regression={regression} "
              f"(h={n_heads * n_steps_ahead_scaler}) before grouping.")
    test_y = test_y.groupby([pd.Grouper(level=0), pd.Grouper(level=1, freq=freq)]).sum(min_count=1)
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


def resolve_trained_entry(trained_models, date_grid, date, regression, market, D, freq):
    for candidate_date in sorted((d for d in date_grid if d <= date), reverse=True):
        entry = trained_models.get((candidate_date, regression, market, D, freq))
        if entry is None:
            continue
        model, avg_1s = entry
        if isinstance(model, dict):
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
    check_base_grid(main_features_base)
    main_features_by_D = dict()
    for D, freq in itertools.product(D_LIST, REPORT_FREQS):
        n_heads_agg = int(pd.to_timedelta(DAYS_CONVERTER[D], unit='d') // pd.to_timedelta(BASE_FREQ))
        rolled = main_features_base.sort(['underlying_name', 'timestamp']).with_columns(
            pl.col(target).rolling_sum(window_size=n_heads_agg, min_samples=n_heads_agg)
              .shift(-(n_heads_agg - 1)).over('underlying_name').alias(target)
        )
        cols_to_drop = [
            col for col in contemporanous_features[target] + [f'{c}_{D}' for c in contemporanous_features[target]]
            if col in rolled.columns
        ]
        main_features_by_D[(D, freq)] = rolled.drop(cols_to_drop)
    L_days = DAYS_CONVERTER[args.L]
    check_training_length(L_days)
    start_idx = np.searchsorted(dates, dates[0] + relativedelta(days=L_days))
    date_grid = dates[start_idx::L_days]
    date_masks = \
        {(d, D, freq): ((main_features_by_D[(D, freq)]['timestamp'] >= pl.lit(d - relativedelta(days=L_days))) &
                        (main_features_by_D[(D, freq)]['timestamp'] < pl.lit(d - relativedelta(days=embargo_days(D)))))
         for d, D, freq in itertools.product(date_grid, D_LIST, REPORT_FREQS)}
    date_grid_regressions_markets_Ds_freqs = \
        list(itertools.product(date_grid, list(regression_dd.keys()), [0, 1], D_LIST, REPORT_FREQS))
    print(f'[Training] {len(date_grid_regressions_markets_Ds_freqs)} (date, regression, market, D, freq) windows '
          f'to train across {args.cpu_count} worker(s) for target={target}.')
    with tqdm_joblib(tqdm(desc=f'Model training windows (target={target})',
                          total=len(date_grid_regressions_markets_Ds_freqs))):
        trained_models = dict(Parallel(n_jobs=args.cpu_count, backend='loky')(
            delayed(train_one_window)(
                date, main_features_by_D[(D, freq)].filter(date_masks[(date, D, freq)]), regression, market, D,
                target, freq, L_days)
            for date, regression, market, D, freq in date_grid_regressions_markets_Ds_freqs
        ))
    date_grid_np = np.array(date_grid)
    end = np.datetime64(args.end_date)
    date_grid_test_by_D = dict()
    date_masks_tests_by_D = dict()
    in_sample_dates_by_D = dict()
    horizon_dates_by_D = dict()
    n_steps_ahead_scaler_by_D_freq = dict()
    for D, freq in itertools.product(D_LIST, REPORT_FREQS):
        D_days = DAYS_CONVERTER[D]
        grid_test = dates[dates.index(date_grid[0])::D_days]
        date_masks_tests_by_D[(D, freq)] = {
            d: ((main_features_by_D[(D, freq)]['timestamp'] < pl.lit(d + relativedelta(days=D_days))) &
                (main_features_by_D[(D, freq)]['timestamp'] >= pl.lit(d - relativedelta(days=D_days + D_days))))
            for d in grid_test
        }
        grid_test = np.array(grid_test)
        date_grid_test_by_D[(D, freq)] = grid_test
        retrain_idx = np.clip(np.searchsorted(date_grid_np, grid_test, side='right') - 1, 0, len(date_grid_np) - 1)
        in_sample_dates_by_D[(D, freq)] = date_grid_np[retrain_idx]
        _dates = pd.to_datetime(grid_test).to_numpy(dtype='datetime64[ns]')
        horizon_dates_by_D[(D, freq)] = np.minimum(_dates + np.timedelta64(D_days, 'D'), end)
        _retrain_ts = pd.to_datetime(in_sample_dates_by_D[(D, freq)]).to_numpy(dtype='datetime64[ns]')
        _horizon_ts = pd.to_datetime(horizon_dates_by_D[(D, freq)]).to_numpy(dtype='datetime64[ns]')
        freq_delta = pd.to_timedelta(freq)
        freq_ns = np.timedelta64(freq_delta.value, 'ns')
        raw_steps = (_horizon_ts - _retrain_ts).astype('timedelta64[ns]') / freq_ns
        raw_steps = raw_steps + pd.Timedelta(days=embargo_days(D)) / freq_delta
        n_steps_ahead_scaler_by_D_freq[(D, freq)] = np.clip(np.ceil(raw_steps), 1, None).astype(int)
    for market in [1, 0]:
        all_forecasts = []
        all_avg_1s = []
        for D, freq in itertools.product(D_LIST, REPORT_FREQS):
            main_features_test_cache = {
                d: filter_market_features(main_features_by_D[(D, freq)].filter(mask), target, market)
                for d, mask in date_masks_tests_by_D[(D, freq)].items()
            }
            date_grid_test = date_grid_test_by_D[(D, freq)]
            regression_grid_test = list(itertools.product(range(len(date_grid_test)), regression_dd.keys()))
            with tqdm_joblib(tqdm(desc=f'Test windows (target={target}, market={market}, D={D}, freq={freq})',
                                  total=len(regression_grid_test))):
                results = Parallel(n_jobs=args.cpu_count, backend='loky')(
                    delayed(process_one_test_window)(
                        resolve_trained_entry(trained_models, date_grid, in_sample_dates_by_D[(D, freq)][i],
                                              regression, market, D, freq),
                        main_features_test_cache[date_grid_test[i]],
                        target, freq, n_steps_ahead_scaler_by_D_freq[(D, freq)][i], date_grid_test[i],
                        horizon_dates_by_D[(D, freq)][i], regression, market, D)
                    for i, regression in regression_grid_test)
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
        if MASE_FROM_FORECAST_ERROR:
            merged['MASE'] = ((merged['y'] - merged['y_hat']) / merged['avg_1s']).abs()
        else:
            merged['MASE'] = (merged['y_hat'] / merged['avg_1s']).abs()
        merged['D_days'] = merged['D'].map(DAYS_CONVERTER)
        merged['MASE'] = \
            merged['MASE'] / (pd.to_timedelta(merged['D_days'], unit='d') / pd.to_timedelta(BASE_FREQ))
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
            if not stripped or stripped.startswith(
                (r'\toprule', r'\midrule', r'\bottomrule', r'\hline')
            ):
                continue
            rows.append(stripped)

        def split_row(row: str) -> list[str]:
            row = row.rstrip()
            if row.endswith(r'\\'):
                row = row[:-2]
            return [cell.strip() for cell in row.split('&')]

        def clean_numeric(cell: str) -> float:
            cell = cell.strip()
            cell = cell.replace('$', '')
            cell = re.sub(r'\\(?:mathbf|textbf|mathrm|textit)\{([^{}]*)\}', r'\1', cell)
            cell = re.sub(r'\\[a-zA-Z]+', '', cell)
            return float(cell)
        parsed_rows = [split_row(r) for r in rows]
        header = parsed_rows[0]
        col_labels = header[1:]
        data_rows = parsed_rows[1:]
        if data_rows and all(c == '' for c in data_rows[0][1:]):
            data_rows = data_rows[1:]
        index_labels = [r[0] for r in data_rows]
        values = [[clean_numeric(c) for c in r[1:]] for r in data_rows]
        return pd.DataFrame(values, index=index_labels, columns=col_labels)

    def best_row_col(df: pd.DataFrame) -> tuple[str, str, float]:
        stacked = df.stack()
        idx = stacked.idxmin()
        value = float(stacked.loc[idx])
        row, col = idx
        return row, col, value
    mase_target_table = parse_latex_table('/'.join([target_path, 'crypto_option_spot',
                                                    f'mase_per_regression_freq_pair_{target}.txt']))
    champion_freq, champion_regression, champion_mase = best_row_col(mase_target_table)
    model_names_reverse = {v: k for k, v in model_names.items()}
    champion_model_key = model_names_reverse[champion_regression.lower()]
    champion_regression_type = 'dl' if champion_model_key == 'MLPRegressor' else 'ml'
    champion_freq = re.sub(r'^\$?(\d+)\\mathrm\{([^}]+)\}\$?$', r'\1\2', champion_freq)
    print(
        f'[FeatureImportance] target={target}: '
        f'champion regression={champion_regression!r} '
        f'({champion_model_key}, type={champion_regression_type}), '
        f'champion_mase={champion_mase:.6f}. '
        f'Forecasting every (D, freq) pair with D in {D_LIST} and freq in {REPORT_FREQS}.'
    )
    dates = pd.date_range(args.start_date, args.end_date, freq='1D', tz=pytz.UTC, inclusive='left').date.tolist()
    volatility_regime = pl.scan_parquet(f'{args.target_path}/spot_features.parquet')
    volatility_regime = volatility_regime.filter((pl.col('target') == 'RV') & (pl.col('freq') == '15min'))
    volatility_regime = volatility_regime.select(['underlying_name', 'timestamp', 'target_values'])
    volatility_regime = volatility_regime.unique(maintain_order=True).collect().to_pandas()
    volatility_regime = volatility_regime.groupby('timestamp')['target_values'].mean()
    volatility_regime = {freq: pd.DataFrame(pd.qcut(volatility_regime.resample(freq).sum(), q=[0, .33, .66, 1],
                                                    labels=VOL_REGIMES)).assign(freq=freq)
                         for freq in ['15min', '30min', '1h']}
    volatility_regime = pd.concat(volatility_regime.values()).rename(columns={'target_values': 'volatility_regime'})
    volatility_regime = volatility_regime.loc[(volatility_regime.index.date >=
                                               pd.to_datetime(args.start_date, utc=True).date()) &
                                              (volatility_regime.index.date <
                                               pd.to_datetime(args.end_date, utc=True).date())]
    volatility_regime.index.name = 'timestamp'
    volatility_regime = volatility_regime.reset_index()
    features = pl.scan_parquet(f'{args.target_path}/spot_features.parquet').filter(pl.col('target') == target).collect()
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
    main_features_base = main_features_base.sort(['underlying_name', 'timestamp'])
    main_features_base = main_features_base.with_columns(pl.all().forward_fill().over('underlying_name'))
    main_features_base = main_features_base.with_columns(pl.col('timestamp').dt.replace_time_zone(None))
    check_base_grid(main_features_base)
    main_features_by_D = dict()
    for D, freq in itertools.product(D_LIST, REPORT_FREQS):
        n_heads_agg = int(pd.to_timedelta(DAYS_CONVERTER[D], unit='d') // pd.to_timedelta(BASE_FREQ))
        rolled = (main_features_base.sort(['underlying_name', 'timestamp']).with_columns(pl.col(target).rolling_sum(
                    window_size=n_heads_agg, min_samples=n_heads_agg).shift(-(n_heads_agg - 1)).over('underlying_name')
                                                                                         .alias(target)))
        cols_to_drop = [col for col in (contemporanous_features[target] +
                                        [f'{c}_{D}' for c in contemporanous_features[target]]) if col in rolled.columns]
        main_features_by_D[(D, freq)] = rolled.drop(cols_to_drop)
    L_days = DAYS_CONVERTER[args.L]
    check_training_length(L_days)
    start_idx = np.searchsorted(dates, dates[0] + relativedelta(days=L_days))
    date_grid = dates[start_idx::L_days]
    date_masks = {
        (d, D, freq): (
            (main_features_by_D[(D, freq)]['timestamp'] >=
             pl.lit(d - relativedelta(days=L_days))) &
            (main_features_by_D[(D, freq)]['timestamp'] <
             pl.lit(d - relativedelta(days=embargo_days(D))))
        )
        for d, D, freq in itertools.product(date_grid, D_LIST, REPORT_FREQS)
    }
    date_grid_regressions_markets_Ds_freqs = \
        list(itertools.product(date_grid, [champion_regression_type], [0, 1], D_LIST, REPORT_FREQS))
    print(
        f'[FeatureImportance] '
        f'{len(date_grid_regressions_markets_Ds_freqs)} '
        f'(date, market, D, freq) windows to train '
        f'across {args.cpu_count} worker(s) '
        f'for target={target}, '
        f'regression_type={champion_regression_type}.'
    )

    with tqdm_joblib(tqdm(desc=f'Feature importance training windows (target={target})',
                          total=len(date_grid_regressions_markets_Ds_freqs))
    ):
        trained_models = dict(Parallel(n_jobs=args.cpu_count, backend='loky')(delayed(train_one_window)(
            date, main_features_by_D[(D, freq)].filter(date_masks[(date, D, freq)]), champion_regression_type,
            market, D, target, freq, L_days)
            for date, _, market, D, freq in date_grid_regressions_markets_Ds_freqs))
    date_grid_np = np.array(date_grid)
    end = np.datetime64(args.end_date)
    date_grid_test_by_D = dict()
    date_masks_tests_by_D = dict()
    in_sample_dates_by_D = dict()
    horizon_dates_by_D = dict()
    n_steps_ahead_scaler_by_D_freq = dict()
    for D, freq in itertools.product(D_LIST, REPORT_FREQS):
        D_days = DAYS_CONVERTER[D]
        grid_test = dates[dates.index(date_grid[0])::D_days]
        date_masks_tests_by_D[(D, freq)] = {
            d: (
                (main_features_by_D[(D, freq)]['timestamp'] <
                 pl.lit(d + relativedelta(days=D_days))) &
                (main_features_by_D[(D, freq)]['timestamp'] >=
                 pl.lit(d - relativedelta(days=D_days + D_days)))
            )
            for d in grid_test
        }
        grid_test = np.array(grid_test)
        date_grid_test_by_D[(D, freq)] = grid_test
        retrain_idx = np.clip(np.searchsorted(date_grid_np, grid_test, side='right') - 1, 0, len(date_grid_np) - 1)
        in_sample_dates_by_D[(D, freq)] = date_grid_np[retrain_idx]
        _dates = pd.to_datetime(grid_test).to_numpy(dtype='datetime64[ns]')
        horizon_dates_by_D[(D, freq)] = np.minimum(_dates + np.timedelta64(D_days, 'D'), end)
        _retrain_ts = pd.to_datetime(in_sample_dates_by_D[(D, freq)]).to_numpy(dtype='datetime64[ns]')
        _horizon_ts = pd.to_datetime(horizon_dates_by_D[(D, freq)]).to_numpy(dtype='datetime64[ns]')
        freq_delta = pd.to_timedelta(freq)
        freq_ns = np.timedelta64(freq_delta.value, 'ns')
        raw_steps = ((_horizon_ts - _retrain_ts).astype('timedelta64[ns]') / freq_ns)
        raw_steps = raw_steps + pd.Timedelta(days=embargo_days(D)) / freq_delta
        n_steps_ahead_scaler_by_D_freq[(D, freq)] = (np.clip(np.ceil(raw_steps), 1, None).astype(int))

    def window_mase(forecast_df: pd.DataFrame, avg_1s: pd.DataFrame, D: str, freq: str) -> Dict[str, float]:
        merged = forecast_df.merge(avg_1s.reset_index().rename(columns={'unique_id': 'underlying_name'}),
                                   on='underlying_name', how='left')
        mase = merged.groupby('underlying_name').apply(lambda x: (x.y-x.y_hat).abs().mean()/x.avg_1s.iloc[0])
        n_periods = pd.to_timedelta(DAYS_CONVERTER[D], unit='d') / pd.to_timedelta(BASE_FREQ)
        mase = mase / n_periods
        return mase.to_dict()

    def _run_one_window(market, D, freq, i):
        date_grid_test = date_grid_test_by_D[(D, freq)]
        date_test = date_grid_test[i]
        trained_entry = resolve_trained_entry(trained_models, date_grid, in_sample_dates_by_D[(D, freq)][i],
                                              champion_regression_type, market, D, freq)
        if trained_entry is None:
            return []
        model, avg_1s = trained_entry
        if avg_1s is None:
            return []
        main_features_i = filter_market_features(
            main_features_by_D[(D, freq)].filter(date_masks_tests_by_D[(D, freq)][date_test]), target, market)
        tmp_sum = n_steps_ahead_scaler_by_D_freq[(D, freq)][i]
        start = pd.Timestamp(date_test)
        horizon_end = horizon_dates_by_D[(D, freq)][i]

        def champion_forecast(mf: pl.DataFrame):
            fc = \
                evaluate_one_window(model=model, regression=champion_regression_type, target=target, freq=freq,
                                    n_steps_ahead_scaler=tmp_sum, main_features=mf)
            fc = fc.loc[(fc.timestamp >= start) & (fc.timestamp < horizon_end) &
                        (fc.regression == champion_regression.lower())]
            return fc
        try:
            baseline_forecast = champion_forecast(main_features_i)
        except Exception as e:
            print(
                f'WARNING: baseline evaluate_one_window failed '
                f'for feature importance window '
                f'{date_test}/{champion_regression}/market={market}/D={D}/freq={freq} '
                f'({type(e).__name__}: {e}) -- skipping window.'
            )
            return []

        if baseline_forecast.empty:
            return []
        main_features_pd = main_features_i.to_pandas()
        feature_cols = [c for c in main_features_pd.columns if c not in ('underlying_name', 'timestamp', target)]
        rng_local = np.random.default_rng(123 + i)
        records = []
        for col in feature_cols:
            col_std = main_features_pd[col].std()
            noise_scale = col_std if col_std and not np.isnan(col_std) else 1.0
            original_col = main_features_pd[col].copy()
            window_mases = []
            for _ in range(FEAT_IMP_N_ITER):
                main_features_pd[col] = original_col + rng_local.normal(scale=noise_scale, size=len(main_features_pd))
                try:
                    jittered_forecast = champion_forecast(pl.from_pandas(main_features_pd))
                except Exception:
                    continue
                if jittered_forecast.empty:
                    continue
                window_mases.append(window_mase(jittered_forecast, avg_1s, D, freq))
            main_features_pd[col] = original_col
            if not window_mases:
                continue
            records.append({'target': target, 'market': market, 'D': D, 'freq': freq,
                            'regression': champion_regression, 'timestamp': date_test, 'feature': col,
                            'MASE': float(pd.DataFrame(window_mases).mean().mean())})
        return records
    window_stride = 5
    tasks = []
    for market in [1, 0]:
        for D in D_LIST:
            for freq in REPORT_FREQS:
                date_grid_test = date_grid_test_by_D[(D, freq)]
                windows_to_run = [i for i in range(len(date_grid_test)) if i % window_stride == 0]
                tasks += [(market, D, freq, i) for i in windows_to_run]
    print(f'[FeatureImportance] {len(tasks)} (market, D, freq, window) tasks to evaluate across '
          f'{args.cpu_count} worker(s) for target={target}.')
    with tqdm_joblib(tqdm(desc=f'Feature importance windows (target={target})', total=len(tasks))):
        results = Parallel(n_jobs=args.cpu_count, backend='loky')(delayed(_run_one_window)(market, D, freq, i)
                                                                  for market, D, freq, i in tasks)
    importance_records = [record for window_records in results for record in window_records]
    importance_df = pd.DataFrame(importance_records)
    importance_df['timestamp'] = pd.to_datetime(importance_df['timestamp'], utc=True)
    importance_df = pd.merge(importance_df, volatility_regime, on=['timestamp', 'freq'], how='left')
    averaged = importance_df.groupby(['target', 'market', 'D', 'volatility_regime', 'freq', 'regression', 'feature'])
    averaged = averaged[['MASE']].mean().reset_index()
    averaged['regression'] = averaged['regression'].str.lower()
    detail_path = f'{args.target_path}/crypto_option_spot/features_importance_{target}.csv'
    averaged.to_csv(detail_path, mode='a', header=not os.path.exists(detail_path), index=False)
    print(f'[Done] target={target}: feature MASE computed for champion regression={champion_regression!r} '
          f'({champion_regression_type}, champion_mase={champion_mase:.6f}), '
          f'averaged over every {window_stride}th test window across D={D_LIST} x freq={REPORT_FREQS}, '
          f'{averaged["feature"].nunique()} features, and stored with relative_MASE.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script generating all results presented in the associated paper.\n'
                    'Self-contained: runs both targets, all markets, all reporting '
                    'frequencies, and all forecasting horizons in one execution.',
        allow_abbrev=False)
    parser.add_argument('--L', type=str, help='Parameter controlling the size of the training set size.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2023-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2023-07-01')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    parser.add_argument('--cpu_count', type=int, default=1, help='Number of processes to use for the pipeline.')
    parser.add_argument('--feature_importance', type=int, default=0, help='Feature importance part of the pipeline.')
    args = parser.parse_args()
    print(args)
    print(f'targets={TARGET_LIST}, markets=[1, 0], D={D_LIST}, report_freqs={REPORT_FREQS}')
    for target in TARGET_LIST:
        if args.feature_importance == 0:
            run_for_target(target, args)
        else:
            run_feature_importance(args.target_path, target, args)
    print('[Done] All targets, markets, D, and reporting frequencies processed.')