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
from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob
import itertools
import concurrent.futures
import optuna
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
from sklearn.preprocessing import MinMaxScaler


def return_filename(filename: str):
    return filename


RESAMPLING_MAT_MONEYNESS = ['gex', 'vax', 'vex', 'vox', 'volume', 'gex_exp', 'vax_exp', 'vex_exp', 'vox_exp']
RESAMPLING_MAT_MONEYNESS = {feature: 'sum' for feature in RESAMPLING_MAT_MONEYNESS}
RESAMPLING_MAT_MONEYNESS.update({feature: 'last' for feature in ['tau', 'moneyness', 'mark_iv']})
resampling = RESAMPLING.copy()
for key in ['ask_iv', 'bid_iv', 'rho', 'theta', 'mark_price']:
    resampling.pop(key)
models = [LassoRegressor(), ElasticRegressor(), BoostingRegressor(), LSTMRegressor()]
models_factory_dd = {model.name.lower(): model for model in models}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all results presented in the associated paper.\n'
                                                 'Results are computed model by model.')
    parser.add_argument('--regression', type=str, help='Regression technique to be used for modelling.')
    parser.add_argument('--L', type=str, help='Parameter controlling the size of the training set size.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2023-07-01')
    parser.add_argument('--target', type=str, help='RV or returns.', default='returns')
    parser.add_argument('--freq', type=str, help='Frequency used to resample and aggregate data')
    parser.add_argument('--options', type=int,
                        help='Tag indicating whether options dynamics are included in the models', default=0)
    parser.add_argument('--D', type=str, help='Number of days ahead to forecast values for', default='1D')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    parser.add_argument('--to_csv', type=int, help='Saving results in .csv files or not', default=1)
    args = parser.parse_args()
    print(args)
    feature_builder = FeatureBuilder()
    mase_score = MeanAbsoluteScaledError()
    permutation_feature_importance_obj = PermutationFeatureImportance()
    reader_obj = Reader()
    underlying_price = reader_obj.prices_read(symbol=SYMBOLS)
    underlying_price = underlying_price.dropna(how='all', axis=1)
    ####################################################################################################################
    ## Spot
    ####################################################################################################################
    overall_spot_dynamics = \
        feature_builder.returns_moments(data=underlying_price, freq=args.freq) if args.target == 'returns' else \
            feature_builder.rv_lags(data=reader_obj.rv_read(), freq=args.freq,
                                    D=args.D).set_index('symbol', append=True).swaplevel(1, 0)
    ####################################################################################################################
    ## Options
    ####################################################################################################################
    if args.options == 1:
        overall_option_dynamics = list()
        for year in range(int(args.start_date[:4]), int(args.end_date[:4])+1):
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                futures = \
                    [executor.submit(return_filename, filename) for filename in glob.glob(f'../data_centre/{year}/*')]
                for future in concurrent.futures.as_completed(futures):
                    filename = future.result()
                    daily_option_chain_data = pd.read_parquet(filename)
                    # if 'mid_price' not in daily_option_chain_data.columns:
                    #     daily_option_chain_data['mid_price'] = \
                    #         .5*(daily_option_chain_data['bid_price'] + daily_option_chain_data['ask_price'])
                    #     daily_option_chain_data['mid_price'] = \
                    #         daily_option_chain_data['mid_price'] * daily_option_chain_data['underlying_price']
                    #     daily_option_chain_data['spread'] = \
                    #         .5 * (daily_option_chain_data['ask_price'] - daily_option_chain_data['bid_price'])
                    daily_option_chain_data['tau'] = \
                        (daily_option_chain_data.expiration - daily_option_chain_data.timestamp)
                    daily_option_chain_data['tau'] /= pd.to_timedelta('360D')
                    daily_option_chain_data['trades'] = np.nan
                    daily_option_chain_data['moneyness_label'] = np.nan
                    daily_option_chain_data.loc[daily_option_chain_data.type == 'put', 'moneyness_label'] = \
                        pd.cut(daily_option_chain_data.loc[daily_option_chain_data.type == 'put', 'moneyness'],
                               bins=MONEYNESS_BINS, labels=LABELS)
                    daily_option_chain_data.loc[daily_option_chain_data.type == 'call', 'moneyness_label'] = \
                        pd.cut(daily_option_chain_data.loc[daily_option_chain_data.type == 'call', 'moneyness'],
                               bins=MONEYNESS_BINS, labels=LABELS[::-1])
                    daily_option_chain_data.loc[daily_option_chain_data['last_price'].gt(
                        daily_option_chain_data['mid_price']), 'trades'] = 1
                    daily_option_chain_data.loc[daily_option_chain_data['last_price'].lt(
                        daily_option_chain_data['mid_price']), 'trades'] = -1
                    daily_option_chain_data = \
                        daily_option_chain_data.loc[~daily_option_chain_data['trades'].isnull(), :]
                    daily_option_chain_data['volume'] = \
                        np.where(daily_option_chain_data['trades'] == 1,
                                 daily_option_chain_data['ask_amount'], daily_option_chain_data['bid_amount'])
                    daily_option_chain_data['volume'] = \
                        daily_option_chain_data['volume'] * daily_option_chain_data['mid_price']
                    daily_option_chain_data.dropna(inplace=True, axis=1)
                    daily_option_chain_data.set_index(['symbol', 'timestamp'], inplace=True)
                    daily_option_chain_data = feature_builder.define_market_participants(daily_option_chain_data)
                    daily_option_chain_data['netDealerExposure'] = \
                        -1 * (daily_option_chain_data['volume'].mul(daily_option_chain_data['trades']))
                    netExposure = feature_builder.compute_exposure(data=daily_option_chain_data)
                    daily_option_chain_data.drop(['trades'], axis=1, inplace=True)
                    netExposure = netExposure.groupby(by=[pd.Grouper(key='underlying_name'),
                                                          pd.Grouper(level='timestamp', freq=args.freq,
                                                                     sort=True)])[EXPOSURE].sum()
                    put_call_ratio = feature_builder.put_call_ratio(data=daily_option_chain_data[['type', 'volume',
                                                                                                  'underlying_name']],
                                                                    freq=args.freq)
                    volatility_skew = \
                        feature_builder.volatility_skew(
                            data=daily_option_chain_data[['type', 'volume', 'mark_iv', 'moneyness_label',
                                                          'underlying_name']], freq=args.freq)
                    mkt_participants = \
                        feature_builder.label_market_participants(freq=args.freq,
                            data=daily_option_chain_data[['type', 'market_participants', 'underlying_name']])
                    overall_option_dynamics.append(pd.concat([netExposure, put_call_ratio, volatility_skew,
                                                              mkt_participants], axis=1))
    if args.options == 1:
        overall_option_dynamics = pd.concat(overall_option_dynamics)
        overall_option_dynamics = \
            overall_option_dynamics.loc[~overall_option_dynamics.index.get_level_values(1).isna(), :]
        overall_option_dynamics.index.names = ['symbol', 'timestamp']
        features = pd.concat([overall_spot_dynamics, overall_option_dynamics], axis=1)
    else:
        features = overall_spot_dynamics
        features.index = features.index.set_names(['symbol', 'timestamp'])
    target = features.pop(args.target)
    ####################################################################################################################
    ## Lags
    ####################################################################################################################
    target_lags = pd.concat([target, features.filter(regex='|'.join(INTERDAY_LAGS+INTRADAY_LAGS))], axis=1)
    features_lags = \
        features.loc[:, ~features.columns.str.contains('|'.join(INTERDAY_LAGS+INTRADAY_LAGS), regex=True)].groupby(
            by=pd.Grouper(level='symbol'), group_keys=True).apply(lambda x: feature_builder.lags(x, args.freq, args.D))
    features = pd.concat([target_lags, features_lags], axis=1).reset_index(0)#.dropna()
    # features[cols_to_cast] = features[cols_to_cast].astype(bool)
    # features.dropna(inplace=True)
    ####################################################################################################################
    ## One hot label encoding
    ####################################################################################################################
    features =  \
        pd.concat([features, pd.get_dummies(features['symbol'], prefix='is', prefix_sep='_').astype(int)], axis=1)
    binary_pattern = ['retail', 'informed', 'institutions', 'is_']
    features = features.set_index('symbol', append=True).swaplevel(0, 1).sort_index()
    features = features.apply(lambda x: x.ffill())
    dates = list(np.unique(features.index.get_level_values(1).date))
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
        cols_ignore_in_scaling = [col for col in train_X.columns if any(pat in col for pat in binary_pattern)]
        if args.regression in ['lasso', 'elastic', 'lstm']:
            for symbol in train_X.index.get_level_values(0).unique():
                train_X.loc[train_X.index.get_level_values(0) == symbol,
                ~train_X.columns.isin(cols_ignore_in_scaling)] = \
                    minmax_scaler[symbol].fit_transform(train_X.xs(symbol, drop_level=False).loc[:,
                                                        ~train_X.columns.isin(cols_ignore_in_scaling)])
        study_name = f'{model.name}_{args.L}_{args.target}_{date.strftime("%Y-%m-%d")}'
        if model.name in ['lasso', 'elastic', 'lightgbm']:
            rres_study = model.model_train(features=train_X.dropna(),
                                           target=train_y.loc[train_X.index], study_name=study_name)
            models.loc[date] = rres_study.user_attrs['best_estimator']
        else:
            train_dates = list(np.unique(train_X.index.get_level_values(1)))
            train_dates, valid_dates = train_dates[:int(.8 * len(train_dates))], train_dates[
                                                                                 int(.8 * len(train_dates)):]
            train_X, valid_X = train_X.loc[train_X.index.get_level_values(1).isin(train_dates), :], \
                train_X.loc[train_X.index.get_level_values(1).isin(valid_dates), :]
            train_y, valid_y = train_y.loc[train_y.index.get_level_values(1).isin(train_dates)], \
                train_y.loc[train_y.index.get_level_values(1).isin(valid_dates)]
            # for symbol in train_X.index.get_level_values(0).unique():
            #     valid_X.loc[valid_X.index.get_level_values(0) == symbol,
            #     ~valid_X.columns.isin(cols_ignore_in_scaling)] = \
            #         minmax_scaler[symbol].transform(valid_X.xs(symbol, drop_level=False).loc[:,
            #                                         ~valid_X.columns.isin(cols_ignore_in_scaling)])
            model.model_train(train_X=train_X, train_y=train_y, valid_X=valid_X, valid_y=valid_y, n_trials=1,
                              max_epochs=EPOCH*5, cols_ignore_in_scaling=cols_ignore_in_scaling)
            models.loc[date] = model
    models.ffill(inplace=True)
    models.dropna(inplace=True)
    models.index = pd.to_datetime(models.index, utc=pytz.UTC)
    ####################################################################################################################
    ## Performance evaluation
    ####################################################################################################################
    forecasts = list()
    mase = list()
    permutation_feature_importance_obj = PermutationFeatureImportance()
    for start_date_forecast in dates[start_idx::pd.to_timedelta(DAYS_CONVERTER[args.D], unit='D').days]:
            end_date_forecast = \
                (start_date_forecast+relativedelta(days=pd.to_timedelta(DAYS_CONVERTER[args.D], unit='D').days))
            test_X = features.loc[(features.index.get_level_values(1).date >= start_date_forecast) &
                                  (features.index.get_level_values(1).date < end_date_forecast), :]
            test_y = test_X.pop(args.target)
            fine_tuned_model = models.loc[models.index.date == start_date_forecast].iloc[0] \
                if model.name in ['lasso', 'elastic', 'lightgbm'] else \
                models.loc[models.index.date == start_date_forecast].iloc[0].model
            # if model.name == 'lstm':
            #     test_X.iloc[:, ~test_X.columns.isin(cols_ignore_in_scaling)] = \
            #         models.loc[models.index.date ==
            #                    start_date_forecast].iloc[0].maxabsscaler.transform(
            #             test_X.iloc[:, :~test_X.columns.isin(cols_ignore_in_scaling)])
            ############################################################################################################
            ## Fine-tuning
            ############################################################################################################
            train_fine_tune_X = \
                features.loc[features.index.get_level_values(1).date == date-relativedelta(days=1), :].dropna()
            train_fine_tune_y = train_fine_tune_X.pop(args.target)
            if model.name in ['lasso', 'elastic', 'lstm']:
                for symbol in train_fine_tune_X.index.get_level_values(0).unique():
                    train_fine_tune_X.loc[train_fine_tune_X.index.get_level_values(0) == symbol,
                    ~train_fine_tune_X.columns.isin(cols_ignore_in_scaling)] = \
                        minmax_scaler[symbol].transform(train_fine_tune_X.xs(symbol, drop_level=False).loc[:,
                                                        ~train_fine_tune_X.columns.isin(cols_ignore_in_scaling)])
            fine_tuned_model = model.model_fine_tune(train_fine_tune_X=train_fine_tune_X, n_trials=N_TRIALS//20,
                                                     train_fine_tune_y=train_fine_tune_y,
                                                     model=fine_tuned_model, max_epochs=EPOCH, date=date,
                                                     L=args.L, target=args.target)
            test_X = test_X.loc[test_X.index.get_level_values(0).isin(
                train_fine_tune_X.index.get_level_values(0).unique()), train_fine_tune_X.columns].dropna()
            test_y = test_y.loc[test_X.index]
            if model.name in ['lasso', 'elastic', 'lstm']:
                for symbol in test_X.index.get_level_values(0).unique():
                    test_X.loc[test_X.index.get_level_values(0) == symbol,
                    ~test_X.columns.isin(cols_ignore_in_scaling)] = \
                        minmax_scaler[symbol].transform(test_X.xs(symbol, drop_level=False).loc[:,
                                                        ~test_X.columns.isin(cols_ignore_in_scaling)])
                if model.name == 'lstm':
                    test_X_tensor = \
                        torch.from_numpy(test_X.values.reshape((test_X.shape[0], 1,
                                                                test_X.shape[1])).astype(np.float32))
            y_hat = fine_tuned_model(test_X_tensor).detach().numpy() if \
                model.name == 'lstm' else fine_tuned_model.user_attrs['best_estimator'].predict(test_X)
            ############################################################################################################
            ## Model assessement
            ############################################################################################################
            mase.append([mase_score(y_true=test_y.values, y_pred=y_hat, y_train=train_y.values),
                         start_date_forecast, model.name.lower(), args.target, args.D, args.options, args.freq])
            test_y = pd.concat([test_y, pd.Series(y_hat, index=test_y.index, name='forecast')], axis=1)
            forecasts.append(test_y.rename(columns={args.target: 'values'}))
            ############################################################################################################
            ## Feature importance
            ############################################################################################################
            permutation_feature_importance_obj.mase_benchmark = mase[-1][0]
            if isinstance(fine_tuned_model, optuna.study.study.Study):
                permutation_feature_importance_obj.model = fine_tuned_model.user_attrs['best_estimator']
            else:
                permutation_feature_importance_obj.model = fine_tuned_model
            permutation_feature_importance_obj.test = test_X
            permutation_feature_importance_obj.test_y = test_y
            permutation_feature_importance_obj.train_y = train_y
            permutation_feature_importance_obj.score[start_date_forecast] = \
                permutation_feature_importance_obj.compute_score()
    features_importance = pd.DataFrame(permutation_feature_importance_obj.score).transpose()
    features_importance.index = pd.to_datetime(features_importance.index, utc=True)
    metric = \
        pd.DataFrame(mase, columns=['MASE', 'timestamp', 'regression', 'target', 'D',
                                    'market', 'freq']).set_index('timestamp')
    forecasts = pd.concat(forecasts).reset_index('symbol').assign(regression=model.name.lower(), target=args.target,
                                                                  D=args.D, market=args.options, freq=args.freq)
    metric.index = pd.to_datetime(metric.index, utc=True)
    metric.index.name = 'timestamp'
    volatility_regime = reader_obj.rv_read(symbol=underlying_price.columns.tolist()).resample('1D').sum()
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
        db_connect = sqlite3.connect(database=f'{args.target_path}/database.db', check_same_thread=False)
        forecasts.to_sql(con=db_connect, if_exists='append', name='forecasts')
        metric.to_sql(con=db_connect, if_exists='append', name='metric')
        features_importance.to_sql(con=db_connect, if_exists='append', name='feature_importance')
        print(f'[Data Insertion]: Results for {args} have been inserted into the database.')
    else:
        forecasts.to_csv(path=f'{args.target_path}/forecasts.csv', mode='a')
        metric.to_csv(path=f'{args.target_path}/metric.csv', mode='a')
        features_importance.to_csv(path=f'{args.target_path}/features_importance.csv', mode='a')
        print(f'[Data Export]: Results for {args} have been exported into csv files.')



