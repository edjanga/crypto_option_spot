import argparse
import os
import pdb
import torch
import pytz
import sqlite3
from constants import *
from data_centre.data import Reader
import pandas as pd
from model.lab import FeatureBuilder, LassoRegressor, BoostingRegressor, LSTMRegressor, PermutationFeatureImportance, \
    ElasticRegressor
from dateutil.relativedelta import relativedelta
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from sklearn.preprocessing import MinMaxScaler


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
    reader_obj = Reader()
    feature_builder = FeatureBuilder()
    db_connect = sqlite3.connect(database=f'{args.target_path}/database.db', check_same_thread=False)
    if args.options == 1:
        LAGS = LAGS[LAGS.index(args.freq):LAGS.index(args.D) + 1]
        SYMBOLS_QUERY = [f'\"{symbol}\"' for symbol in SYMBOLS]
        SYMBOLS_QUERY = f'({",".join(SYMBOLS_QUERY)})'
        features_query = \
            f'SELECT "features","values","symbol","timestamp" FROM features WHERE "symbol" IN {SYMBOLS_QUERY};'
        features = pd.read_sql(sql=features_query, con=db_connect)
        features['timestamp'] = pd.to_datetime(features['timestamp'], utc=pytz.UTC)
        features = features.set_index(['symbol', 'timestamp', 'features'])
        features = features.loc[~features.index.duplicated(), :]
        features = features.unstack(-1).droplevel(axis=1, level=0)
        features.loc[:, ~features.columns.str.contains('market_participants')] = \
            features.loc[:, ~features.columns.str.contains('market_participants')] .astype(float)
        features = features.groupby(by=pd.Grouper(level='symbol'), group_keys=False).apply(lambda x: x.ffill())
        features = \
            pd.concat([features,
                       pd.get_dummies(features['market_participants'], prefix='is', prefix_sep='_').astype(np.int8)],
                      axis=1).drop('market_participants', axis=1)
    ####################################################################################################################
    ## Spot
    ####################################################################################################################
    underlying_rv = reader_obj.rv_read(symbol=SYMBOLS)
    underlying_rv = underlying_rv.loc[(underlying_rv.index.date >= pd.to_datetime(args.start_date).date()) &
                                      (underlying_rv.index.date < pd.to_datetime(args.end_date).date()), :]
    if args.target == 'returns':
        target = reader_obj.returns_read(symbol=SYMBOLS)
        target = target.loc[(target.index.date >= pd.to_datetime(args.start_date).date()) &
                            (target.index.date < pd.to_datetime(args.end_date).date())]
    else:
        target = underlying_rv.copy()
    target.columns.name = 'symbol'
    target = target.stack().swaplevel(0, 1)
    target.name = args.target
    target = pd.DataFrame(target)
    ####################################################################################################################
    ## Lags
    ####################################################################################################################
    target_lags = feature_builder.rv_lags(data=target, freq=args.freq, D=args.D) if args.target == 'RV' else \
        feature_builder.returns_moments(data=target, freq=args.freq)
    features = features.join(target_lags, how='inner') if args.options == 1 else target_lags.copy()
    target = features.pop(args.target)
    features_lags = \
        features.loc[:, ~features.columns.str.contains('|'.join(INTRADAY_LAGS+INTERDAY_LAGS+BINARY_PATTERN),
                                                       regex=True)].groupby(
            by=pd.Grouper(level='symbol'), group_keys=True).apply(
            lambda x: feature_builder.lags(x, args.freq, args.D))
    features = pd.concat([target_lags, features_lags], axis=1)
    features = features.filter(regex='|'.join(INTRADAY_LAGS+INTERDAY_LAGS+BINARY_PATTERN))
    features = features.apply(lambda x: x.ffill()).dropna()
    mase_score = MeanAbsoluteScaledError()
    permutation_feature_importance_obj = PermutationFeatureImportance()
    reader_obj = Reader()
    ####################################################################################################################
    ## One hot label encoding
    ####################################################################################################################
    symbol_one_hot = \
        pd.DataFrame(
            data=pd.get_dummies(features.index.get_level_values(0), prefix='is', prefix_sep='_')).astype(np.int8)
    symbol_one_hot.index = features.index
    features = pd.concat([features, target, symbol_one_hot], axis=1).dropna()
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
                               (features.index.get_level_values(1).date < date), :].dropna()
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
            rres_study = model.model_train(features=train_X, target=train_y.loc[train_X.index], study_name=study_name)
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
    models.ffill(inplace=True)
    models.dropna(inplace=True)
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
                              (features.index.get_level_values(1).date < end_date_forecast), :].dropna()
        test_y = test_X.pop(args.target)
        if args.D != '1D':
            ############################################################################################################
            ## Fine-tuning
            ############################################################################################################
            if model.name == 'lstm':
                fine_tuned_model = models.loc[models.index.date == start_date_forecast].iloc[0].model
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
        test_y = test_y.replace(0, np.nan).dropna()
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
    features_importance = pd.concat(permutation_feature_importance_obj.score).swaplevel(0, 1)
    features_importance = features_importance.transpose()[SYMBOLS].transpose()
    features_importance = features_importance.groupby(by=pd.Grouper(level=1)).mean()
    metric = \
        pd.DataFrame(mase, columns=['MASE', 'timestamp', 'regression', 'target', 'D',
                                    'market', 'freq']).set_index('timestamp')
    forecasts = pd.concat(forecasts).reset_index('symbol').assign(regression=model.name.lower(), target=args.target,
                                                                  D=args.D, market=args.options, freq=args.freq)
    metric.index = pd.to_datetime(metric.index, utc=True)
    metric.index.name = 'timestamp'
    volatility_regime = reader_obj.rv_read(symbol=SYMBOLS).resample('1D').sum()
    volatility_regime = \
        volatility_regime.loc[(volatility_regime.index.date >= dates[start_idx]) &
                              (volatility_regime.index.date <= dates[-1])].mean(axis=1)
    volatility_regime = pd.qcut(volatility_regime, q=[0, .45, .9, 1], labels=VOL_REGIMES)
    volatility_regime.name = 'volatility_regime'
    metric = metric.join(volatility_regime)
    features_importance = features_importance.join(volatility_regime, how='left')
    features_importance = pd.melt(ignore_index=False, id_vars=['volatility_regime'],
                                  value_name='MASE', var_name='features', frame=features_importance)
    features_importance = \
        features_importance.assign(regression=args.regression, D=args.D, freq=args.freq, market=args.options,
                                   target=args.target)
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



