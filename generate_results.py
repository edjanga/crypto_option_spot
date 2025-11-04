import argparse
import os
import pdb
import pytz
import sqlite3
from constants import *
import pandas as pd
from model.lab import FeatureBuilder, LassoRegressor, BoostingRegressor, LSTMRegressor, PermutationFeatureImportance, \
    ElasticRegressor
from dateutil.relativedelta import relativedelta
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from sklearn.preprocessing import MinMaxScaler
import polars as pl


def return_filename(filename: str):
    return filename


models = [LassoRegressor(), ElasticRegressor(), BoostingRegressor(), LSTMRegressor()]
models_factory_dd = {model.name.lower(): model for model in models}
LAGS = INTRADAY_LAGS+INTERDAY_LAGS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all results presented in the associated paper.\n'
                                                 'Results are computed model by model.')
    parser.add_argument('--regression', type=str, help='Regression technique to be used for modelling.')
    parser.add_argument('--L', type=str, help='Parameter controlling the size of the training set size.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2023-07-01')
    parser.add_argument('--target', type=str, help='RV or returns.', default='returns')
    parser.add_argument('--freq', type=str, help='Frequency used to resample and aggregate data', default='15min')
    parser.add_argument('--options', type=int,
                        help='Tag indicating whether options dynamics are included in the models', default=0)
    parser.add_argument('--D', type=str, help='Number of days ahead to forecast values for', default='1D')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    parser.add_argument('--to_csv', type=int, help='Saving results in .csv files or not', default=1)
    args = parser.parse_args()
    print(args)
    dates = list(pd.date_range(start=args.start_date, end=args.end_date, freq='1D', tz=pytz.UTC, inclusive='left'))
    dates = list(map(lambda date: date.date(), dates))
    feature_builder = FeatureBuilder()
    db_connect = sqlite3.connect(database=f'{args.target_path}/crypto_option_spot.db', check_same_thread=False)
    ####################################################################################################################
    ## Spot
    ####################################################################################################################
    SYMBOLS_QUERY = [f'\"{symbol}\"' for symbol in SYMBOLS]
    SYMBOLS_QUERY = f'({",".join(SYMBOLS_QUERY)})'
    features_query = \
        f'SELECT "underlying_name","timestamp","target","target_values", "features", "features_values" ' \
        f'FROM spot_features WHERE "underlying_name" IN {SYMBOLS_QUERY} AND "target" == "{args.target}";'
    features = pd.read_sql(sql=features_query, con=db_connect).drop('target', axis=1)
    features['timestamp'] = pd.to_datetime(features['timestamp'], utc=pytz.UTC)
    features = features.set_index(['underlying_name', 'timestamp'])
    target = features.pop('target_values').rename(args.target)
    features = \
        pd.pivot(features, columns='features', values='features_values').join(target, on=['underlying_name',
                                                                                          'timestamp'])
    ####################################################################################################################
    ## Options
    ####################################################################################################################
    if args.options == 1:
        LAGS = LAGS[LAGS.index(args.freq):LAGS.index(args.D) + 1]
        options_features_query = \
            f'SELECT "features","features_values","underlying_name","timestamp" FROM option_features ' \
            f'WHERE "underlying_name" IN {SYMBOLS_QUERY};'
        options_features = pd.read_sql(sql=options_features_query, con=db_connect)
        options_features['timestamp'] = pd.to_datetime(options_features['timestamp'], utc=pytz.UTC)
        options_features = options_features.set_index(['underlying_name', 'timestamp', 'features'])
        options_features = options_features.loc[~options_features.index.duplicated(), :]
        options_features = options_features.unstack(-1).droplevel(axis=1, level=0)
        options_features.loc[:, ~options_features.columns.str.contains('market_participants')] = \
            options_features.loc[:, ~options_features.columns.str.contains('market_participants')].astype(float)
        options_features = \
            options_features.groupby(by=pd.Grouper(level='underlying_name'),
                                     group_keys=False).apply(lambda x: x.ffill())
        options_features = \
            pd.concat([options_features,
                       pd.get_dummies(options_features.filter(regex='market_participants'),
                                      columns=['market_participants_call', 'market_participants_put'],
                                      prefix='is', prefix_sep='_').astype(np.int8)], axis=1).drop(
                ['market_participants_call', 'market_participants_put'], axis=1)
        features = features.join(options_features, on=['underlying_name', 'timestamp'])
    #
    #
    #
    # underlying_prices = \
    #     pl.scan_parquet(f'{args.target_path}/*/spot/*-*-*').select([pl.col('symbol').alias('underlying_name'),
    #                                                                 pl.col('timestamp'), pl.col('mid_price[0]')])
    # underlying_prices = underlying_prices.collect(engine=ENGINE)
    # underlying_prices = underlying_prices.pivot(index='timestamp', on='underlying_name', values='mid_price[0]')
    # target = underlying_prices.select([pl.col('timestamp')] +
    #                                   [(pl.col(c) / pl.col(c).shift(1)).log() for c in underlying_prices.columns[1:]])
    # underlying_rv = target.sort(by='timestamp').group_by_dynamic(index_column='timestamp', every='5m').agg(
    #         [(pl.col(symbol) ** 2).sum() for symbol in underlying_prices.columns[1:]])
    # underlying_rv = underlying_rv.filter((pl.col('timestamp') >= pl.lit(args.start_date).cast(pl.Date)) &
    #                                      (pl.col('timestamp') <= pl.lit(args.end_date).cast(pl.Date)))
    # if args.target == 'returns':
    #     target = target.sort(by='timestamp').group_by_dynamic(index_column='timestamp', every='5m').agg(
    #         [pl.col(symbol).sum() for symbol in underlying_prices.columns[1:]])
    #     target = target.filter((pl.col('timestamp') >= pl.lit(args.start_date).cast(pl.Date)) &
    #                            (pl.col('timestamp') <= pl.lit(args.end_date).cast(pl.Date)))
    # else:
    #     target = underlying_rv
    # underlying_rv = underlying_rv.to_pandas()
    # target = target.unpivot(index='timestamp', value_name=args.target, variable_name='underlying_name')
    # ####################################################################################################################
    # ## Lags
    # ####################################################################################################################
    # target_lags = feature_builder.rv_lags(data=target, freq=args.freq, D=args.D) if args.target == 'RV' else \
    #     feature_builder.returns_moments(data=target, freq=args.freq)
    # features = features.join(target_lags.to_pandas().set_index(['underlying_name', 'timestamp'])) if args.options == 1 \
    #     else target_lags.to_pandas().set_index(['underlying_name', 'timestamp'])
    # target = features.pop(args.target)
    # pdb.set_trace()
    # features_lags = \
    #     features.loc[:, ~features.columns.str.contains('|'.join(INTRADAY_LAGS+INTERDAY_LAGS+BINARY_PATTERN),
    #                                                    regex=True)].groupby(
    #         by=pd.Grouper(level='underlying_name'), group_keys=True).apply(
    #         lambda x: feature_builder.lags(x, args.freq, args.D))
    # features = pd.concat([target_lags.to_pandas().set_index(['underlying_name', 'timestamp']), features_lags], axis=1)
    # features = features.filter(regex='|'.join(INTRADAY_LAGS+INTERDAY_LAGS+BINARY_PATTERN))
    mase_score = MeanAbsoluteScaledError()
    permutation_feature_importance_obj = PermutationFeatureImportance()
    ####################################################################################################################
    ## One hot label encoding
    ####################################################################################################################
    symbol_one_hot = \
        pd.DataFrame(
            data=pd.get_dummies(features.index.get_level_values(0), prefix='is', prefix_sep='_')).astype(np.int8)
    symbol_one_hot.index = features.index
    features = pd.concat([features, symbol_one_hot], axis=1)
    features = features.groupby(by=pd.Grouper(level=0), group_keys=False).apply(lambda x: x.ffill()).dropna()
    ####################################################################################################################
    ## Modelling
    ####################################################################################################################
    model = models_factory_dd[args.regression]
    start_idx = dates.index(dates[0]+relativedelta(days=DAYS_CONVERTER[args.L]))
    models = pd.Series(data=np.nan, index=dates)
    features_importance = dict()
    minmax_scaler = dict([(symbol, MinMaxScaler().set_output(transform='pandas')) for symbol in SYMBOLS])
    for idx, date in enumerate(dates[start_idx::DAYS_CONVERTER[args.L]]):
        start_date = date-relativedelta(days=DAYS_CONVERTER[args.L])
        train_X = features.loc[(features.index.get_level_values(1).date >= start_date) &
                               (features.index.get_level_values(1).date < date), :]
        train_y = train_X.pop(args.target)
        cols_ignore_in_scaling = [col for col in train_X.columns if any(pat in col for pat in BINARY_PATTERN)]
        if args.regression in ['lasso', 'elastic', 'lstm']:
            for symbol in train_X.index.get_level_values(0).unique():
                train_X.loc[train_X.index.get_level_values(0) == symbol,
                ~train_X.columns.isin(cols_ignore_in_scaling)] = \
                    minmax_scaler[symbol].fit_transform(train_X.xs(symbol, drop_level=False).loc[:,
                                                        ~train_X.columns.isin(cols_ignore_in_scaling)])
        study_name = f'{model.name}_{args.L}_{args.target}_{date.strftime("%Y-%m-%d")}'
        if model.name in ['lasso', 'elastic', 'lightgbm']:
            rres_study = model.model_train(features=train_X, target=train_y, study_name=study_name)
            models.loc[date] = rres_study.user_attrs['best_estimator'] if model.name == 'lightgbm' else rres_study
        else:
            train_dates = list(np.unique(train_X.index.get_level_values(1)))
            train_dates, valid_dates = train_dates[:int(.8 * len(train_dates))], train_dates[
                                                                                 int(.8 * len(train_dates)):]
            train_X, valid_X = train_X.loc[train_X.index.get_level_values(1).isin(train_dates), :], \
                train_X.loc[train_X.index.get_level_values(1).isin(valid_dates), :]
            train_y, valid_y = train_y.loc[train_y.index.get_level_values(1).isin(train_dates)], \
                train_y.loc[train_y.index.get_level_values(1).isin(valid_dates)]
            model.model_train(train_X=train_X, train_y=train_y, valid_X=valid_X, valid_y=valid_y,
                              n_trials=N_TRIALS//20, max_epochs=EPOCH//5, positiveness=args.target == 'RV',
                              cols_ignore_in_scaling=cols_ignore_in_scaling)
            models.loc[date] = model
    models = models.ffill()
    models = models.dropna().apply(lambda x: x.model) if args.regression == 'lstm' else models.dropna()
    models.index = pd.to_datetime(models.index, utc=pytz.UTC)
    ####################################################################################################################
    ## Performance evaluation
    ####################################################################################################################
    forecasts = list()
    mase = list()
    permutation_feature_importance_obj = PermutationFeatureImportance(freq=args.freq)
    for start_date_forecast in dates[start_idx::pd.to_timedelta(DAYS_CONVERTER[args.D], unit='D').days]:
        end_date_forecast = \
            (start_date_forecast+relativedelta(days=pd.to_timedelta(DAYS_CONVERTER[args.D], unit='D').days))
        test_X = features.loc[(features.index.get_level_values(1).date >= start_date_forecast) &
                              (features.index.get_level_values(1).date < end_date_forecast), :]
        test_X = test_X.groupby(by=pd.Grouper(level=0), group_keys=False).apply(lambda x: x.ffill())
        test_y = test_X.pop(args.target)
        if (args.D != '1D') & (model.name == 'lstm'):
            ############################################################################################################
            ## Fine-tuning
            ############################################################################################################
            try:
                fine_tuned_model = models.loc[models.index.date == start_date_forecast].iloc[0]
            except AttributeError:
                pdb.set_trace()
            train_fine_tune_X = \
                features.loc[features.index.get_level_values(1).date == date - relativedelta(days=1), :].dropna()
            train_fine_tune_y = train_fine_tune_X.pop(args.target)
            for symbol in train_fine_tune_X.index.get_level_values(0).unique():
                train_fine_tune_X.loc[train_fine_tune_X.index.get_level_values(0) == symbol,
                ~train_fine_tune_X.columns.isin(cols_ignore_in_scaling)] = \
                    minmax_scaler[symbol].transform(train_fine_tune_X.xs(symbol, drop_level=False).loc[:,
                                                    ~train_fine_tune_X.columns.isin(cols_ignore_in_scaling)])
            fine_tuned_model = model.model_fine_tune(train_fine_tune_X=train_fine_tune_X, n_trials=1,
                                                     train_fine_tune_y=train_fine_tune_y,
                                                     model=fine_tuned_model, max_epochs=EPOCH//10, date=date,
                                                     L=args.L, target=args.target)
            models.loc[pd.to_datetime(date, utc=pytz.UTC)] = fine_tuned_model
            test_X = test_X.loc[test_X.index.get_level_values(0).isin(
                train_fine_tune_X.index.get_level_values(0).unique()), train_fine_tune_X.columns].dropna()
            test_y = test_y.loc[test_X.index]
        if model.name in ['lasso', 'elastic', 'lstm']:
            for symbol in test_X.index.get_level_values(0).unique():
                test_X.loc[test_X.index.get_level_values(0) == symbol, ~test_X.columns.isin(cols_ignore_in_scaling)] = \
                    minmax_scaler[symbol].transform(test_X.xs(symbol, drop_level=False).loc[:,
                                                    ~test_X.columns.isin(cols_ignore_in_scaling)])
            if model.name == 'lstm':
                test_X_tensor = \
                    torch.from_numpy(test_X.values.reshape((test_X.shape[0], 1,
                                                            test_X.shape[1])).astype(np.float32))
        y_hat = \
            pd.Series(index=test_y.index,
                      data=models.loc[models.index.date == date].iloc[0](test_X_tensor).detach().numpy().reshape(
                          test_X_tensor.shape[0]) if model.name == 'lstm' else
                      models.loc[models.index.date == date].iloc[0].predict(test_X))
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
            mase_metric.append(mase_score(y_true=symbol_test_y.iloc[:, 0].values,
                                          y_pred=symbol_test_y.iloc[:, 1].values,
                                          y_train=symbol_train_y.values))
        mase.append([sum(mase_metric)/len(mase_metric),
                     start_date_forecast, model.name.lower(), args.target, args.D, args.options, args.freq])
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
    features_importance = pd.melt(frame=features_importance, ignore_index=False, value_name='MASE', var_name='features')
    metric = \
        pd.DataFrame(mase, columns=['MASE', 'timestamp', 'regression', 'target', 'D',
                                    'market', 'freq']).set_index('timestamp')
    forecasts = pd.concat(forecasts).reset_index('underlying_name').assign(
        regression=model.name.lower(), target=args.target, D=args.D, market=args.options, freq=args.freq)
    metric.index = pd.to_datetime(metric.index, utc=True)
    metric.index.name = 'timestamp'
    underlying_rv = pd.read_sql(sql='SELECT "underlying_name", "timestamp", "target", "target_values" '
                                    'FROM spot_features WHERE target == "RV";', con=db_connect).drop('target', axis=1)
    underlying_rv['timestamp'] = pd.to_datetime(underlying_rv['timestamp'], utc=pytz.UTC)
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
        features_importance.assign(regression=args.regression, D=args.D, freq=args.freq, market=args.options,
                                   target=args.target)
    pdb.set_trace()
    ####################################################################################################################
    ## Database insertion or exporting to .csv files
    ####################################################################################################################
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



