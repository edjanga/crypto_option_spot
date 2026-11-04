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
from generate_mase_best_models import select_best_models
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
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
FEAT_IMP_N_ITER = 5
FEAT_IMP_TOP = 5                                    # features per row of the feature-importance table
UNCONDITIONAL = 'all'                               # volatility_regime label of the unconditional importance
TABLE_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']   # pairs (rows) of the feature-importance table
BASE_FREQ = '15min'
MASE_FROM_FORECAST_ERROR = True
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


def evaluate_one_window(model, regression: str, target: str, freq: str, n_steps_ahead_scaler, main_features,
                        keep_from=None, keep_to=None):
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
    # only the rows of [keep_from, keep_to) are predicted; the features were filled on all rows before, so these
    # predictions are exactly those of a run without the range
    if keep_from is not None:
        test_X = test_X[test_X['ds'] >= keep_from]
    if keep_to is not None:
        test_X = test_X[test_X['ds'] < keep_to]
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


def window_errors(forecast_df: pd.DataFrame, naive: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Per (pair, volatility regime): number of forecasts, mean |y - y_hat| and mean |y - y_naive| on the same
    forecasts, where y_naive is the naive forecast y(t - D). Every forecast row takes the regime of its own timestamp
    (as in the forecasting run), not the regime of the start of its window."""
    merged = forecast_df.assign(timestamp=pd.to_datetime(forecast_df['timestamp']).astype('datetime64[ns]'))
    merged = merged.merge(naive, on=['underlying_name', 'timestamp'], how='inner')
    merged = merged.merge(regimes, on='timestamp', how='inner')
    merged = merged.dropna(subset=['y', 'y_hat', 'y_naive'])
    merged = merged.assign(abs_error=(merged['y'] - merged['y_hat']).abs(),
                           abs_error_naive=(merged['y'] - merged['y_naive']).abs())
    return merged.groupby(['underlying_name', 'volatility_regime'], observed=True).agg(
        n=('abs_error', 'size'), abs_error=('abs_error', 'mean'), abs_error_naive=('abs_error_naive', 'mean'))


def feature_importance_window(model, main_features_i: pl.DataFrame, target: str, market: int, D: str, freq: str,
                              date_test, horizon_end, tmp_sum: int, seed: int, regression: str,
                              regression_type: str, pairs: list, regimes: pd.DataFrame) -> list:
    """Feature importance of one test window of one (regression, freq, D, market) model, scored on the pairs whose
    best model it is (`pairs`). A module-level function that receives only its own window (features, model, regimes):
    a worker process never holds the full feature set, so memory stays flat with --cpu_count.
    The noise added to a feature is scaled by its standard deviation over all pairs of the window, so the
    perturbation of a pair does not depend on which other pairs selected the same model."""
    torch.set_num_threads(1)
    start = pd.Timestamp(date_test)
    horizon_end = pd.Timestamp(horizon_end)

    def model_forecast(mf: pl.DataFrame):
        fc = evaluate_one_window(model=model, regression=regression_type, target=target, freq=freq,
                                 n_steps_ahead_scaler=tmp_sum, main_features=mf, keep_from=start,
                                 keep_to=horizon_end)
        return fc.loc[(fc.timestamp >= start) & (fc.timestamp < horizon_end) &
                      (fc.regression == regression.lower()) & fc.underlying_name.isin(pairs)]
    try:
        baseline_forecast = model_forecast(main_features_i)
    except Exception as e:
        print(f'WARNING: baseline evaluate_one_window failed for feature importance window '
              f'{date_test}/{regression}/market={market}/D={D}/freq={freq} '
              f'({type(e).__name__}: {e}) -- skipping window.')
        return []
    if baseline_forecast.empty:
        return []
    # naive forecast of every row: the realised target of the same pair D earlier, known at t (the window's feature
    # rows start 2 D before the test date, so it exists for every forecast)
    naive = main_features_i.select(['underlying_name', 'timestamp', target]).to_pandas()
    naive = naive.rename(columns={target: 'y_naive'})
    naive['timestamp'] = (pd.to_datetime(naive['timestamp']) + pd.Timedelta(days=DAYS_CONVERTER[D])).astype(
        'datetime64[ns]')
    baseline = window_errors(baseline_forecast, naive, regimes)
    if baseline.empty:
        return []
    main_features_pd = main_features_i.to_pandas()
    feature_cols = [c for c in main_features_pd.columns if c not in ('underlying_name', 'timestamp', target)]
    rng_local = np.random.default_rng(seed)
    records = []
    for col in feature_cols:
        col_std = main_features_pd[col].std()
        noise_scale = col_std if col_std and not np.isnan(col_std) else 1.0
        original_col = main_features_pd[col].copy()
        window_scores = []
        for _ in range(FEAT_IMP_N_ITER):
            main_features_pd[col] = original_col + rng_local.normal(scale=noise_scale, size=len(main_features_pd))
            try:
                jittered_forecast = model_forecast(pl.from_pandas(main_features_pd))
            except Exception:
                continue
            if jittered_forecast.empty:
                continue
            window_scores.append(window_errors(jittered_forecast, naive, regimes)['abs_error'])
        main_features_pd[col] = original_col
        if not window_scores:
            continue
        jittered_error = pd.concat(window_scores, axis=1).mean(axis=1)        # averaged over the noise draws
        for (underlying_name, regime), row in baseline.iterrows():
            if (underlying_name, regime) not in jittered_error.index or row['abs_error_naive'] == 0:
                continue
            records.append({'target': target, 'market': market, 'D': D, 'freq': freq, 'regression': regression,
                            'timestamp': date_test, 'feature': col, 'underlying_name': underlying_name,
                            'volatility_regime': regime, 'n': int(row['n']),
                            'abs_error': float(jittered_error[(underlying_name, regime)]),
                            'abs_error_baseline': float(row['abs_error']),
                            'abs_error_naive': float(row['abs_error_naive'])})
    return records


def run_feature_importance(target_path: str, target: str, args) -> pd.DataFrame:
    """Top features of every (target, underlying_name, D, volatility regime), each (pair, D) scored with its own best
    (regression, freq) of generate_mase_best_models.py (lowest out-of-sample MASE averaged over X^S and X^(S,O), whole
    sample, as in generate_pnl.py). Returns the ranking (also written to features_importance_ranked_{target}.csv)."""
    forecasts_path = f'{target_path}/crypto_option_spot/forecasts.parquet'
    if not os.path.exists(forecasts_path):
        raise FileNotFoundError(f'feature importance selects the best (regression, freq) of every (pair, D) from '
                                f'{forecasts_path}, which does not exist: run csv_2_parquet.py first.')
    best = select_best_models(forecasts_path).filter(pl.col('target') == target)
    if best.is_empty():
        raise ValueError(f'{forecasts_path} has no forecasts of target={target}')
    print(f'[FeatureImportance] target={target}: best (regression, freq) per (pair, D):\n'
          f'{best.select(["underlying_name", "D", "regression", "freq", "MASE", "basis"])}', flush=True)
    model_names_reverse = {v: k for k, v in model_names.items()}
    regression_types = {r: 'dl' if model_names_reverse[r.lower()] == 'MLPRegressor' else 'ml'
                        for r in best['regression'].unique().to_list()}
    # the models to train and evaluate: one per (D, regression, freq) selected by at least one pair; the pairs it
    # is scored on are those that selected it (the models are trained on all pairs, as in the forecasting run)
    pairs_by_model = {(D, regression, freq): sorted(g['underlying_name'].to_list())
                      for (D, regression, freq), g in best.group_by(['D', 'regression', 'freq'])}
    models = sorted(pairs_by_model)
    model_Ds = sorted({D for D, _, _ in models}, key=D_LIST.index)
    model_freqs = sorted({freq for _, _, freq in models}, key=REPORT_FREQS.index)
    print(f'[FeatureImportance] target={target}: {len(models)} (D, regression, freq) model(s) to evaluate: '
          f'{models}', flush=True)
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
    # the forecasts carry naive UTC timestamps (the feature timestamps are made naive below)
    volatility_regime['timestamp'] = (pd.to_datetime(volatility_regime['timestamp'], utc=True)
                                      .dt.tz_localize(None).astype('datetime64[ns]'))
    volatility_regime['volatility_regime'] = volatility_regime['volatility_regime'].astype(str)
    regimes_by_freq = {freq: g[['timestamp', 'volatility_regime']].sort_values('timestamp').reset_index(drop=True)
                       for freq, g in volatility_regime.groupby('freq')}
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
    for D, freq in itertools.product(model_Ds, model_freqs):
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
        for d, (D, _, freq) in itertools.product(date_grid, models)
    }
    training_windows = [(date, D, regression_types[regression], freq, market)
                        for date, (D, regression, freq), market in itertools.product(date_grid, models, [0, 1])]
    print(f'[FeatureImportance] {len(training_windows)} (date, D, regression, freq, market) windows to train across '
          f'{args.cpu_count} worker(s) for target={target}.', flush=True)
    with tqdm_joblib(tqdm(desc=f'Feature importance training windows (target={target})',
                          total=len(training_windows))):
        trained_models = dict(Parallel(n_jobs=args.cpu_count, backend='loky')(delayed(train_one_window)(
            date, main_features_by_D[(D, freq)].filter(date_masks[(date, D, freq)]), regression_type,
            market, D, target, freq, L_days)
            for date, D, regression_type, freq, market in training_windows))
    date_grid_np = np.array(date_grid)
    end = np.datetime64(args.end_date)
    date_grid_test_by_D = dict()
    date_masks_tests_by_D = dict()
    in_sample_dates_by_D = dict()
    horizon_dates_by_D = dict()
    n_steps_ahead_scaler_by_D_freq = dict()
    for D, freq in itertools.product(model_Ds, model_freqs):
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

    window_stride = 5
    tasks = []
    for market in [1, 0]:
        for D, regression, freq in models:
            date_grid_test = date_grid_test_by_D[(D, freq)]
            windows_to_run = [i for i in range(len(date_grid_test)) if i % window_stride == 0]
            tasks += [(market, D, regression, freq, i) for i in windows_to_run]

    def window_jobs():
        """One job per window, built when joblib asks for it: only that window's features, model and regimes are
        sent to a worker, and only a few windows exist at a time."""
        for market, D, regression, freq, i in tasks:
            date_test = date_grid_test_by_D[(D, freq)][i]
            horizon_end = horizon_dates_by_D[(D, freq)][i]
            trained_entry = resolve_trained_entry(trained_models, date_grid, in_sample_dates_by_D[(D, freq)][i],
                                                  regression_types[regression], market, D, freq)
            if trained_entry is None:
                continue
            main_features_i = filter_market_features(
                main_features_by_D[(D, freq)].filter(date_masks_tests_by_D[(D, freq)][date_test]), target, market)
            regimes = regimes_by_freq[freq]
            regimes = regimes.loc[(regimes['timestamp'] >= pd.Timestamp(date_test)) &
                                  (regimes['timestamp'] < pd.Timestamp(horizon_end))]
            yield delayed(feature_importance_window)(
                trained_entry[0], main_features_i, target, market, D, freq, date_test, horizon_end,
                n_steps_ahead_scaler_by_D_freq[(D, freq)][i], 123 + i, regression, regression_types[regression],
                pairs_by_model[(D, regression, freq)], regimes)
    print(f'[FeatureImportance] {len(tasks)} (market, D, regression, freq, window) tasks to evaluate across '
          f'{args.cpu_count} worker(s) for target={target}.', flush=True)
    with tqdm_joblib(tqdm(desc=f'Feature importance windows (target={target})', total=len(tasks))):
        results = Parallel(n_jobs=args.cpu_count, backend='loky', pre_dispatch='2*n_jobs')(window_jobs())
    importance_df = pd.DataFrame([record for window_records in results for record in window_records])
    if importance_df.empty:
        raise ValueError(f'feature importance of target={target} produced no record: check the warnings above')
    ranked = aggregate_feature_importance(importance_df, args.target_path, target)
    print(f'[Done] target={target}: out-of-sample feature importance of {len(models)} best (D, regression, freq) '
          f'model(s), every {window_stride}th test window, {ranked["feature"].nunique()} features.', flush=True)
    return ranked


def aggregate_feature_importance(importance_df: pd.DataFrame, target_path: str, target: str) -> pd.DataFrame:
    """Out-of-sample MASE of every (pair, D, regime, market, feature), with and without perturbation, from the window
    records of feature_importance_window, and the ranking averaged over the markets.
    MASE = sum |y - y_hat| / sum |y - y_naive| over all the forecasts of the group (every pair has one model per D).
    The unconditional level (volatility_regime = UNCONDITIONAL) pools the forecasts of the three regimes before the
    ratio is taken: a ratio of sums, not an average of the regime ratios, so it is the MASE over the whole sample.
    The detail file keeps the three sums, so any other pooling can be computed from it without rerunning."""
    importance_df = importance_df.assign(volatility_regime=importance_df['volatility_regime'].astype(str))
    importance_df = pd.concat([importance_df, importance_df.assign(volatility_regime=UNCONDITIONAL)],
                              ignore_index=True)
    group = ['target', 'underlying_name', 'D', 'volatility_regime', 'market', 'regression', 'freq', 'feature']
    for column in ['abs_error', 'abs_error_baseline', 'abs_error_naive']:
        importance_df[f'{column}_sum'] = importance_df[column] * importance_df['n']
    per_market = importance_df.groupby(group, observed=True).agg(
        n=('n', 'sum'), abs_error_sum=('abs_error_sum', 'sum'), abs_error_baseline_sum=('abs_error_baseline_sum', 'sum'),
        abs_error_naive_sum=('abs_error_naive_sum', 'sum'))
    per_market['MASE'] = per_market['abs_error_sum'] / per_market['abs_error_naive_sum']
    per_market['MASE_baseline'] = per_market['abs_error_baseline_sum'] / per_market['abs_error_naive_sum']
    # relative_MASE > 1: perturbing the feature worsens the forecasts, i.e. the feature matters
    per_market['relative_MASE'] = per_market['MASE'] / per_market['MASE_baseline']
    per_market = per_market[['n', 'MASE', 'MASE_baseline', 'relative_MASE', 'abs_error_sum',
                             'abs_error_baseline_sum', 'abs_error_naive_sum']].reset_index()
    detail_path = f'{target_path}/crypto_option_spot/features_importance_{target}.csv'
    per_market.to_csv(detail_path, mode='w', header=True, index=False)
    # averaged over X^S and X^(S,O); an option feature exists in X^(S,O) only, so it is ranked on X^(S,O) alone
    ranked = (per_market.groupby([c for c in group if c != 'market'], observed=True)
              .agg(relative_MASE=('relative_MASE', 'mean'), n_markets=('market', 'nunique')).reset_index())
    ranked = ranked.sort_values(['target', 'underlying_name', 'D', 'volatility_regime', 'relative_MASE', 'feature'],
                                ascending=[True, True, True, True, False, True])
    ranked['rank'] = ranked.groupby(['target', 'underlying_name', 'D', 'volatility_regime']).cumcount() + 1
    ranked_path = f'{target_path}/crypto_option_spot/features_importance_ranked_{target}.csv'
    ranked.to_csv(ranked_path, mode='w', header=True, index=False)
    print(f'[FeatureImportance] target={target}: per market (with the error sums) in {detail_path}, ranking (averaged '
          f'over the markets, regimes and unconditional) in {ranked_path}.', flush=True)
    return ranked


def pair_label(pair: str) -> str:
    """Display name of a pair in the tables: BTCUSDT -> BTC/USDT."""
    return pair[:-len('USDT')] + '/USDT' if pair.endswith('USDT') and '/' not in pair else pair


def feature_importance_latex(ranked: pd.DataFrame, top: int = FEAT_IMP_TOP) -> str:
    """The top `top` features of every (target, pair, D, volatility regime) with their relative MASE, as a LaTeX
    table (needs the booktabs, multirow, graphicx and float packages)."""
    ranked = ranked[ranked['rank'] <= top]
    escape = lambda text: str(text).replace('_', r'\_')
    cell = {key: g.sort_values('rank') for key, g in
            ranked.groupby(['target', 'underlying_name', 'D', 'volatility_regime'], observed=True)}
    targets = [t for t in TARGET_LIST if t in ranked['target'].unique()][::-1]       # returns first, then RV
    regimes = [r for r in VOL_REGIMES if r in ranked['volatility_regime'].unique()]
    rows = {target: {pair: [D for D in D_LIST if any((target, pair, D, r) in cell for r in regimes)]
                     for pair in TABLE_PAIRS} for target in targets}
    rows = {target: {pair: Ds for pair, Ds in pairs.items() if Ds} for target, pairs in rows.items()}
    n_columns = 4 + 2 * top
    lines = [r'\begin{table}[H]', r'    \centering', r'    \resizebox{\textwidth}{!}{',
             r'        \begin{tabular}{llll|' + '|'.join(['ll'] * top) + '}', r'            \toprule',
             '            &  &  &  & ' + ' & '.join(
                 rf'\multicolumn{{2}}{{c{"|" if k < top else ""}}}{{Rank {k}}}' for k in range(1, top + 1)) + r' \\',
             '            ' + ' & ' * (n_columns - 1) + r'\\', r'            \midrule']
    for i_target, target in enumerate(rows):
        if i_target > 0:
            lines.append(rf'            \cline{{1-{n_columns}}}')
        n_target = sum(len(Ds) for Ds in rows[target].values()) * len(regimes)
        for i_pair, (pair, Ds) in enumerate(rows[target].items()):
            if i_pair > 0:
                lines.append(rf'            \cline{{2-{n_columns}}}')
            for i_D, D in enumerate(Ds):
                if i_D > 0:
                    lines.append(rf'            \cline{{3-{n_columns}}}')
                for i_regime, regime in enumerate(regimes):
                    first = [rf'\multirow[t]{{{n_target}}}{{*}}{{{escape(target)}}}'
                             if i_pair == 0 and i_D == 0 and i_regime == 0 else '',
                             rf'\multirow[t]{{{len(Ds) * len(regimes)}}}{{*}}{{{escape(pair_label(pair))}}}'
                             if i_D == 0 and i_regime == 0 else '',
                             rf'\multirow[t]{{{len(regimes)}}}{{*}}{{\textit{{{D.lower()}}}}}' if i_regime == 0 else '',
                             rf'\textit{{{regime.lower()}}}']
                    g = cell.get((target, pair, D, regime))
                    values = [] if g is None else [f'{escape(r.feature)} & {r.relative_MASE:.4f}'
                                                   for r in g.itertuples()]
                    values += ['-- & --'] * (top - len(values))
                    lines.append('            ' + ' & '.join(first) + '\n            & ' + ' & '.join(values) + r' \\')
    lines += [r'            \bottomrule', r'        \end{tabular}', r'    }',
              rf'    \caption{{Top {top} predictive features by target, pair, forecast horizon and volatility regime. '
              r'Each (pair, horizon) uses its best (regression, frequency), the one with the lowest out-of-sample '
              r'MASE averaged over $X^S$ and $X^{(S,O)}$. The value is the relative MASE: the out-of-sample MASE '
              r'with the feature perturbed by Gaussian noise divided by the MASE without perturbation, averaged over '
              r'$X^S$ and $X^{(S,O)}$ (over $X^{(S,O)}$ alone for option features); above 1, the feature matters.}',
              r'    \label{tab:featureimportance}', r'\end{table}']
    return '\n'.join(lines) + '\n'


FI_CAPTION = (r'Each (pair, horizon) uses its best (regression, frequency), the one with the lowest out-of-sample '
              r'MASE averaged over $X^S$ and $X^{(S,O)}$. The value is the relative MASE: the out-of-sample MASE '
              r'with the feature perturbed by Gaussian noise divided by the MASE without perturbation, averaged over '
              r'$X^S$ and $X^{(S,O)}$ (over $X^{(S,O)}$ alone for option features); above 1, the feature matters.')


def _fi_panel(ranked: pd.DataFrame, target: str, title: str, regimes: list, top: int) -> str:
    """One panel (one target) of a top-features table: rows pair, horizon and, if `regimes`, volatility regime."""
    escape = lambda text: str(text).replace('_', r'\_')
    cell = {key: g.sort_values('rank') for key, g in
            ranked.groupby(['underlying_name', 'D', 'volatility_regime'], observed=True)}
    use_regimes = regimes or [UNCONDITIONAL]
    n_label = 3 if regimes else 2
    n_columns = n_label + 2 * top
    lines = [r'        \begin{tabular}{' + 'l' * n_label + '|' + '|'.join(['ll'] * top) + '}', r'            \toprule',
             '            ' + ' & ' * (n_label - 1) + ' & ' + ' & '.join(
                 rf'\multicolumn{{2}}{{c{"|" if k < top else ""}}}{{Rank {k}}}' for k in range(1, top + 1)) + r' \\',
             r'            \midrule', rf'            \multicolumn{{{n_columns}}}{{l}}{{\textbf{{{title}}}}} \\',
             r'            \midrule']
    pairs = [p for p in TABLE_PAIRS if any((p, D, r) in cell for D in D_LIST for r in use_regimes)]
    for i_pair, pair in enumerate(pairs):
        Ds = [D for D in D_LIST if any((pair, D, r) in cell for r in use_regimes)]
        if i_pair > 0:
            lines.append(rf'            \cline{{1-{n_columns}}}')
        for i_D, D in enumerate(Ds):
            if i_D > 0 and regimes:
                lines.append(rf'            \cline{{2-{n_columns}}}')
            for i_regime, regime in enumerate(use_regimes):
                first = [rf'\multirow[t]{{{len(Ds) * len(use_regimes)}}}{{*}}{{{escape(pair_label(pair))}}}'
                         if i_D == 0 and i_regime == 0 else '',
                         (rf'\multirow[t]{{{len(use_regimes)}}}{{*}}{{\textit{{{D.lower()}}}}}' if i_regime == 0 else '')
                         if regimes else rf'\textit{{{D.lower()}}}']
                if regimes:
                    first.append(rf'\textit{{{regime.lower()}}}')
                g = cell.get((pair, D, regime))
                values = [] if g is None else [f'{escape(r.feature)} & {r.relative_MASE:.4f}'
                                               for r in g[g['rank'] <= top].itertuples()]
                values += ['-- & --'] * (top - len(values))
                lines.append('            ' + ' & '.join(first) + '\n            & ' + ' & '.join(values) + r' \\')
    lines += [r'            \bottomrule', r'        \end{tabular}']
    return '\n'.join(lines)


def feature_importance_unconditional_latex(ranked: pd.DataFrame, top: int = FEAT_IMP_TOP) -> str:
    """Top `top` features of every (target, pair, D) over the whole sample (no regime split): returns in Panel A, RV
    in Panel B, one table (needs booktabs, multirow, float and adjustbox)."""
    ranked = ranked[ranked['volatility_regime'].astype(str) == UNCONDITIONAL]
    panels = [_fi_panel(ranked[ranked['target'] == target], target, title, [], top)
              for target, title in [('returns', 'Panel A: Returns'), ('RV', 'Panel B: RV')]
              if (ranked['target'] == target).any()]
    lines = [r'\begin{table}[H]', r'    \centering',
             r'    \begin{adjustbox}{max width=\textwidth, max totalheight=0.85\textheight}', r'    \begin{tabular}{c}',
             ('\n' + r'    \\[1em]' + '\n').join(panels), r'    \end{tabular}', r'    \end{adjustbox}',
             rf'    \caption{{Top {top} predictive features by target, pair, and forecast horizon over the whole '
             rf'sample (unconditional). ' + FI_CAPTION + '}',
             r'    \label{tab:featureimportance_unconditional}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def feature_importance_regimes_latex(ranked: pd.DataFrame, top: int = FEAT_IMP_TOP) -> str:
    """Top `top` features by target, pair, D and volatility regime, in the layout of the paper: returns and RV on two
    landscape pages, the second a continuation of the first (needs pdflscape, caption, adjustbox, multirow)."""
    ranked = ranked[ranked['volatility_regime'].astype(str) != UNCONDITIONAL]
    regimes = [r for r in VOL_REGIMES if r in ranked['volatility_regime'].astype(str).unique()]
    out = []
    for k, (target, title) in enumerate([('returns', 'Panel A: Returns'), ('RV', 'Panel B: RV')]):
        if not (ranked['target'] == target).any():
            continue
        caption = (rf'    \caption{{Top {top} predictive features by target, pair, forecast horizon, and volatility '
                   rf'regime. ' + FI_CAPTION + '}' + '\n' + r'    \label{tab:featureimportance}') if k == 0 \
            else r'    \caption{(continued) Panel B: RV.}'
        out += [r'\begin{landscape}', r'\begin{table}[p]'] + ([r'    \ContinuedFloat'] if k else []) + [
            r'    \centering', r'    \begin{adjustbox}{max width=\linewidth, max totalheight=0.80\textwidth}',
            _fi_panel(ranked[ranked['target'] == target], target, title, regimes, top), r'    \end{adjustbox}',
            caption, r'\end{table}', r'\end{landscape}', '']
    return '\n'.join(out)


def write_feature_importance_table(target_path: str) -> None:
    """Pooled LaTeX table of both targets from the features_importance_ranked_{target}.csv files that exist."""
    folder = f'{target_path}/crypto_option_spot'
    paths = [f'{folder}/features_importance_ranked_{target}.csv' for target in TARGET_LIST]
    paths = [path for path in paths if os.path.exists(path)]
    if not paths:
        print('WARNING: no features_importance_ranked_*.csv found -- no feature-importance table written.')
        return
    ranked = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    conditional = ranked[ranked['volatility_regime'].astype(str) != UNCONDITIONAL]
    outputs = [('features_importance.tex', feature_importance_latex(conditional)),          # previous layout
               ('features_importance_split.tex', feature_importance_regimes_latex(ranked)),   # layout of the paper
               ('features_importance_unconditional.tex', feature_importance_unconditional_latex(ranked))]
    for name, content in outputs:
        with open(f'{folder}/{name}', 'w') as file:
            file.write(content)
        print(f'[Done] feature-importance table (top {FEAT_IMP_TOP}) written to {folder}/{name}', flush=True)


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
    if args.feature_importance != 0:
        write_feature_importance_table(args.target_path)
    print('[Done] All targets, markets, D, and reporting frequencies processed.')