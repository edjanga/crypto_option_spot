import argparse
import itertools
import os
import pdb
import polars as pl
import pytz
import sqlite3
from constants import *
import pandas as pd
from model.lab import FeatureBuilder, LassoRegressor, ElasticRegressor, BoostingRegressor, GRURegressor, LSTMRegressor,\
    PermutationFeatureImportance
from dateutil.relativedelta import relativedelta
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from sklearn.preprocessing import StandardScaler

models = [LassoRegressor(), ElasticRegressor(), BoostingRegressor(), GRURegressor(), LSTMRegressor()]
models_factory_dd = {model.name.lower(): model for model in models}
LAGS = INTRADAY_LAGS+INTERDAY_LAGS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all results presented in the associated paper.\n'
                                                 'Results are computed model by model.')
    parser.add_argument('--regression', type=str, help='Regression technique to be used for modelling.')
    parser.add_argument('--L', type=str, help='Parameter controlling the size of the training set size.')
    parser.add_argument('--freq', type=str, help='Parameter controlling forcasting horizon.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2023-07-01')
    parser.add_argument('--target', type=str, help='RV or returns.', default='returns')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='./data_centre')
    parser.add_argument('--to_csv', type=int, help='Saving results in .csv files or not', default=1)
    args = parser.parse_args()
    print(args)
    dates = list(pd.date_range(start=args.start_date, end=args.end_date, freq='1D', tz=pytz.UTC, inclusive='left'))
    dates = list(map(lambda date: date.date(), dates))
    feature_builder = FeatureBuilder()
    db_connect = sqlite3.connect(database=f'{args.target_path}/crypto_option_spot.db',
                                 check_same_thread=False)
    ####################################################################################################################
    ## Spot
    ####################################################################################################################
    SYMBOLS_QUERY = [f'\"{symbol}\"' for symbol in SYMBOLS]
    SYMBOLS_QUERY = f'({",".join(SYMBOLS_QUERY)})'
    features_query = f'WITH base AS(SELECT DISTINCT * FROM spot_features WHERE "underlying_name" IN {SYMBOLS_QUERY} ' \
    f'AND "target" = "{args.target}" AND "freq" = "{args.freq}") SELECT underlying_name, timestamp, features, ' \
                     f'features_values FROM base ' \
    f'UNION ALL SELECT underlying_name, timestamp, target AS features, target_values AS features_values ' \
    f'FROM base;'
    features = pl.read_database(query=features_query, connection=db_connect)
    features = features.with_columns([pl.col('timestamp').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%z'),
                                      pl.col('features_values').cast(pl.Float32, strict=False)])
    features = features.filter((pl.col('timestamp') >= pl.lit(pd.to_datetime(args.start_date, utc=True))) &
        (pl.col('timestamp') < pl.lit(pd.to_datetime(args.end_date, utc=True)))).to_pandas()
    features = \
        features.pivot_table(values='features_values', index=['underlying_name', 'timestamp'], columns='features')
    features = features.replace([np.inf, -np.inf],
                                [np.finfo(np.float32).max, np.finfo(np.float32).min])
    ####################################################################################################################
    ## Options
    ####################################################################################################################
    try:
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
    except Exception:
        main_features = features
    mase_score = MeanAbsoluteScaledError()
    permutation_feature_importance_obj = PermutationFeatureImportance()
    ####################################################################################################################
    ## One hot label encoding
    ####################################################################################################################
    symbol_one_hot = \
        pd.DataFrame(
            data=pd.get_dummies(main_features.index.get_level_values(0), prefix='is', prefix_sep='_')).astype(np.int8)
    symbol_one_hot.index = main_features.index
    main_features = pd.concat([main_features, symbol_one_hot], axis=1)
    main_features = main_features.groupby(by=pd.Grouper(level=0), group_keys=False).apply(lambda x: x.ffill()).dropna()
    ####################################################################################################################
    ## Modelling
    ####################################################################################################################
    model = models_factory_dd[args.regression]
    start_idx = dates.index(dates[0] + relativedelta(days=DAYS_CONVERTER[args.L]))
    permutation_feature_importance_obj = PermutationFeatureImportance()
    for D, options in itertools.product(['1D', '1W', '1M'], [0, 1]):
        models = pd.Series(data=np.nan, index=dates)
        features_importance = dict()
        features_names = LAGS[LAGS.index(args.freq):LAGS.index(D) + 1]+[args.target]+\
                         ['is_BTCUSDT', 'is_ETHUSDT', 'is_BNBUSDT', 'is_SOLUSDT', 'is_XRPUSDT']
        features_names = f'{"|".join(features_names)}'
        features = main_features.filter(regex=features_names)
        for idx, date in enumerate(dates[start_idx::DAYS_CONVERTER[args.L]]):
            start_date = date-relativedelta(days=DAYS_CONVERTER[args.L])
            train_X = features.loc[(features.index.get_level_values(1).date >= start_date) &
                                   (features.index.get_level_values(1).date < date), :]
            train_y = train_X.pop(args.target)
            cols_ignore_in_scaling = [col for col in train_X.columns if any(pat in col for pat in BINARY_PATTERN)]
            if args.regression in ['lasso', 'elastic']:
                standard_scaler = dict([(symbol, StandardScaler().set_output(transform='pandas')) for symbol in SYMBOLS])
                for symbol in train_X.index.get_level_values(0).unique():
                    train_X.loc[train_X.index.get_level_values(0) == symbol,
                    ~train_X.columns.isin(cols_ignore_in_scaling)] = \
                        standard_scaler[symbol].fit_transform(train_X.xs(symbol, drop_level=False).loc[:,
                                                            ~train_X.columns.isin(cols_ignore_in_scaling)])
            study_name = f'{model.name}_{args.L}_{args.target}_{date.strftime(DATE_FORMAT)}'
            if model.name in ['lasso', 'elastic', 'lightgbm']:
                rres_study = model.model_train(features=train_X, target=train_y, study_name=study_name)
                models.loc[date] = rres_study.user_attrs['best_estimator']
            else:
                train_dates = list(np.unique(train_X.index.get_level_values(1)))
                train_dates, valid_dates = train_dates[:int(.8 * len(train_dates))], train_dates[
                                                                                     int(.8 * len(train_dates)):]
                train_X, valid_X = train_X.loc[train_X.index.get_level_values(1).isin(train_dates), :], \
                    train_X.loc[train_X.index.get_level_values(1).isin(valid_dates), :]
                train_y, valid_y = train_y.loc[train_y.index.get_level_values(1).isin(train_dates)], \
                    train_y.loc[train_y.index.get_level_values(1).isin(valid_dates)]
                model.model_train(train_X=train_X, train_y=train_y, valid_X=valid_X, valid_y=valid_y,
                                  n_trials=N_TRIALS//20, positiveness=args.target == 'RV',
                                  cols_ignore=cols_ignore_in_scaling)
                models.loc[date] = model
                standard_scaler = model.standardscaler
        models = models.ffill()
        models = models.dropna().apply(lambda x: x.model) if args.regression in ['gru', 'lstm'] else models.dropna()
        models.index = pd.to_datetime(models.index, utc=pytz.UTC)
        ################################################################################################################
        ## Performance evaluation
        ################################################################################################################
        forecasts = list()
        mase = list()
        permutation_feature_importance_obj.freq = args.freq
        for start_date_forecast in dates[start_idx::pd.to_timedelta(DAYS_CONVERTER[D], unit='D').days]:
            end_date_forecast = \
                (start_date_forecast+relativedelta(days=pd.to_timedelta(DAYS_CONVERTER[D], unit='D').days))
            test_X = features.loc[(features.index.get_level_values(1).date >= start_date_forecast) &
                                  (features.index.get_level_values(1).date < end_date_forecast), :]
            test_y = test_X.pop(args.target)
            if (D != '1D') & (model.name in ['gru', 'lstm']):
                ########################################################################################################
                ## Fine-tuning
                ########################################################################################################
                fine_tuned_model = models.loc[models.index.date == start_date_forecast].iloc[0]
                train_fine_tune_X = \
                    features.loc[features.index.get_level_values(1).date == date - relativedelta(days=1), :].dropna()
                train_fine_tune_y = train_fine_tune_X.pop(args.target)
                for symbol in train_fine_tune_X.index.get_level_values(0).unique():
                    train_fine_tune_X.loc[train_fine_tune_X.index.get_level_values(0) == symbol,
                    ~train_fine_tune_X.columns.isin(cols_ignore_in_scaling)] = \
                        standard_scaler[symbol].transform(train_fine_tune_X.xs(symbol, drop_level=False).loc[:,
                                                          ~train_fine_tune_X.columns.isin(cols_ignore_in_scaling)])
                fine_tuned_model = model.model_fine_tune(train_fine_tune_X=train_fine_tune_X, n_trials=1,
                                                         train_fine_tune_y=train_fine_tune_y,
                                                         model=fine_tuned_model, max_epochs=EPOCH//10, date=date,
                                                         L=args.L, target=args.target)
                models.loc[pd.to_datetime(date, utc=pytz.UTC)] = fine_tuned_model
                test_X = test_X.loc[test_X.index.get_level_values(0).isin(
                    train_fine_tune_X.index.get_level_values(0).unique()), train_fine_tune_X.columns].dropna()
                test_y = test_y.loc[test_X.index]
            for symbol in test_X.index.get_level_values(0).unique():
                test_X.loc[test_X.index.get_level_values(0) == symbol,
                ~test_X.columns.isin(cols_ignore_in_scaling)] = \
                    standard_scaler[symbol].transform(test_X.xs(symbol, drop_level=False).loc[:,
                                                      ~test_X.columns.isin(cols_ignore_in_scaling)])
            if model.name in ['gru', 'lstm']:
                test_X_tensor = \
                    torch.from_numpy(test_X.values.reshape((test_X.shape[0], 1,
                                                            test_X.shape[1])).astype(np.float32))
            y_hat = \
                pd.Series(index=test_y.index,
                          data=models.loc[models.index.date == date].iloc[0](test_X_tensor).detach().numpy().reshape(
                              test_X_tensor.shape[0]) if model.name in ['gru', 'lstm'] else
                          models.loc[models.index.date == date].iloc[0].predict(test_X))
            for symbol in test_X.index.get_level_values(0).unique():
                y_hat.loc[y_hat.index.get_level_values(0) == symbol] = \
                    standard_scaler[symbol].var_[0] ** .5 * y_hat.loc[y_hat.index.get_level_values(0) == symbol] + \
                    standard_scaler[symbol].mean_[0]
            ############################################################################################################
            ## Model assessement
            ############################################################################################################
            test_y = pd.concat([test_y.rename('values'), pd.Series(y_hat, index=test_y.index, name='forecast')], axis=1)
            test_y = \
                test_y.replace(0, np.nan).groupby(by=[pd.Grouper(level=0)], group_keys=False).apply(lambda x: x.ffill())
            forecasts.append(test_y)
            mase_metric = list()
            for symbol in test_y.index.get_level_values(0).unique():
                symbol_test_y, symbol_train_y = test_y.xs(symbol, axis=0, level=0), train_y.xs(symbol, axis=0, level=0)
                symbol_test_y = symbol_test_y.dropna()
                symbol_train_y = symbol_train_y.dropna()
                mase_metric.append(mase_score(y_true=symbol_test_y.iloc[:, 0].values,
                                              y_pred=symbol_test_y.iloc[:, 1].values,
                                              y_train=symbol_train_y.values))
            mase.append([sum(mase_metric)/len(mase_metric),
                         start_date_forecast, model.name.lower(), args.target, D, options, args.freq])
            ############################################################################################################
            ## Feature importance
            ############################################################################################################
            permutation_feature_importance_obj.model = models.loc[models.index.date == date].iloc[0]
            permutation_feature_importance_obj.test = test_X
            permutation_feature_importance_obj.test_y = test_y
            permutation_feature_importance_obj.train_y = train_y
            permutation_feature_importance_obj.score[pd.to_datetime(start_date_forecast, utc=pytz.UTC)] = \
                permutation_feature_importance_obj.compute_score()
        features_importance = pd.DataFrame(permutation_feature_importance_obj.score).transpose()
        features_importance.index.name = 'timestamp'
        features_importance = \
            pd.melt(frame=features_importance, ignore_index=False, value_name='MASE', var_name='features')
        metric = \
            pd.DataFrame(mase, columns=['MASE', 'timestamp', 'regression', 'target', 'D',
                                        'market', 'freq']).set_index('timestamp')
        forecasts = pd.concat(forecasts).reset_index('underlying_name').assign(
            regression=model.name.lower(), target=args.target, D=D, market=options, freq=args.freq)
        metric.index = pd.to_datetime(metric.index, utc=True)
        metric.index.name = 'timestamp'
        underlying_rv = pd.read_sql(sql='SELECT "underlying_name", "timestamp", "target", "target_values" '
                                        'FROM spot_features WHERE target == "RV";', con=db_connect).drop('target',
                                                                                                         axis=1)
        underlying_rv['timestamp'] = pd.to_datetime(underlying_rv['timestamp'], utc=pytz.UTC)
        underlying_rv['target_values'] = pd.to_numeric(underlying_rv['target_values'], errors='coerce').astype(
            'float32')
        underlying_rv = underlying_rv.groupby(by=pd.Grouper(key='timestamp'))['target_values'].mean()
        volatility_regime = underlying_rv.resample(args.freq).sum()
        volatility_regime = \
            volatility_regime.loc[(volatility_regime.index.date >= dates[start_idx]) &
                                  (volatility_regime.index.date <= dates[-1])]
        volatility_regime = pd.qcut(volatility_regime, q=[0, .45, .9, 1], labels=VOL_REGIMES)
        volatility_regime.name = 'volatility_regime'
        metric = metric.join(volatility_regime)
        features_importance = features_importance.join(volatility_regime, how='left')
        features_importance = \
            features_importance.assign(regression=args.regression, D=D, freq=args.freq, market=options,
                                       target=args.target)
        ################################################################################################################
        ## Database insertion or exporting to .csv files
        ################################################################################################################
        if args.to_csv == 0:
            forecasts.to_sql(con=db_connect, if_exists='append', name='forecasts')
            metric.to_sql(con=db_connect, if_exists='append', name='metric')
            features_importance.to_sql(con=db_connect, if_exists='append', name='feature_importance')
            print(f'[Data Insertion]: Results for {args} have been inserted into the database.')
        else:
            forecasts.to_csv(f'{args.target_path}/forecasts.csv', mode='a',
                             header=not os.path.exists(f'{args.target_path}/forecasts.csv'))
            metric.to_csv(f'{args.target_path}/metric.csv', mode='a',
                          header=not os.path.exists(f'{args.target_path}/metric.csv'))
            features_importance.to_csv(f'{args.target_path}/features_importance.csv', mode='a',
                                       header=not os.path.exists(f'{args.target_path}/features_importance.csv'))
            print(f'[Data Export]: Results for {args} have been exported into csv files.')



