import argparse
import torch
import pytz
import sqlite3
from constants import *
from data_centre.data import Reader
import pandas as pd
from model.lab import FeatureBuilder, LassoRegressor, BoostingRegressor, LSTMRegressor, PermutationFeatureImportance
from datetime import datetime
from dateutil.relativedelta import relativedelta
import glob
import itertools
import concurrent.futures
import optuna


RESAMPLING_MAT_MONEYNESS = ['gex', 'vax', 'vex', 'vox', 'volume', 'gex_exp', 'vax_exp', 'vex_exp', 'vox_exp']
RESAMPLING_MAT_MONEYNESS = {feature: 'sum' for feature in RESAMPLING_MAT_MONEYNESS}
RESAMPLING_MAT_MONEYNESS.update({feature: 'last' for feature in ['tau', 'moneyness', 'mark_iv']})
resampling = RESAMPLING.copy()
for key in ['ask_iv', 'bid_iv', 'rho', 'theta']:
    resampling.pop(key)
models = [LassoRegressor(), BoostingRegressor(), LSTMRegressor()]
models_factory_dd = {model.name.lower(): model for model in models}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all results presented in the associated paper.\n'
                                                 'Results are computed model by model.')
    parser.add_argument('--regression', type=str, help='Regression technique to be used for modelling.')
    parser.add_argument('--L', type=str, help='Parameter controlling the size of the training set size.')
    parser.add_argument('--start_date', type=str, help='Start date.', default='2021-01-01')
    parser.add_argument('--end_date', type=str, help='End date.', default='2022-01-01')
    parser.add_argument('--target', type=str, help='RV or returns.', default='returns')
    parser.add_argument('--freq', type=str, help='Frequency used to resample and aggregate data', default='1h')
    parser.add_argument('--options', type=int,
                        help='Tag indicating whether options dynamics are included in the models', default=0)
    parser.add_argument('--D', type=str, help='Number of days ahead to forecast values for', default='1D')
    args = parser.parse_args()
    print(args)
    benchmark_model = {True: LassoRegressor(), False: None}[args.regression != 'lasso']
    feature_builder = FeatureBuilder()
    permutation_feature_importance_obj = PermutationFeatureImportance()
    symbols = ['BTC', 'ETH']
    symbols = list(map(lambda x: f'{x}USDT', symbols))
    reader_obj = Reader()
    underlying_price = reader_obj.prices_read(symbol=symbols)
    usecols = ['type', 'strike_price', 'expiration', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount',
               'mark_price', 'mark_iv', 'delta', 'gamma', 'vega', 'moneyness', 'last_price']
    overall_spot_dynamics = dict([(symbol, list()) for symbol in symbols])
    if args.options == 1:
        overall_moneyness_mat_exp_report = list()
        overall_agg_exposure_report = list()
        overall_put_call_ratio = dict([(symbol, list()) for symbol in symbols])
        overall_omms_exposure = dict([(symbol, list()) for symbol in symbols])
        overall_options_dynamics = dict([(symbol, list()) for symbol in symbols])
        mkt_participants_dynamics = dict([(symbol, list()) for symbol in symbols])
    files = list()
    files = [glob.glob(f'../data_centre/{symbol.replace("USDT", "")}/*/*') for symbol in symbols]
    files = list(itertools.chain(*files))
    files.sort()
    files = [file for file in files if pd.to_datetime(file.split('/')[-1], format=DATE_FORMAT).date() <=
             pd.to_datetime(args.end_date, format=DATE_FORMAT).date()]
    for file in files:
        symbol = file.split("/")[2]
        symbol = f'{symbol}USDT'
        date = datetime.strptime(file.split('/')[-1], '%Y-%m-%d')
        print(f'[Data processing]: {symbol}_{file.split("/")[-1]} is being treated now............')
        data = pd.read_parquet(file, columns=usecols + ['option_tag'])
        if 'option_tag' not in data.columns:
            data.reset_index(level=0, inplace=True)
        data.loc[:, 'symbol'] = [option_tag.split('-')[0] for option_tag in data['option_tag']]
        data = \
            data.join(underlying_price.loc[:, symbol].resample('5min').last(),
                      how='left').rename(columns={symbol: 'underlying_price'})
        symbol_underlying_price = underlying_price.filter(regex=symbol)
        symbol_underlying_price = symbol_underlying_price.loc[symbol_underlying_price.index.date == date.date()]
        ################################################################################################################
        ## Spot
        ################################################################################################################
        returns_moments = \
            feature_builder.returns_moments(data=symbol_underlying_price, freq=args.freq).assign(symbol=symbol)
        overall_spot_dynamics[symbol].append(returns_moments)
        ################################################################################################################
        ## Options
        ################################################################################################################
        if args.options == 1:
            ####################################################################################################
            ## OMMs
            ####################################################################################################
            data = data.loc[data.index.date == date.date(), :]
            data['tau'] = (data.expiration - data.index) / pd.to_timedelta('360D')
            data['mid_price'] = .5 * (data['bid_price'] + data['ask_price'])
            data['trades'] = np.nan
            data['moneyness'] = np.log(data['moneyness'])
            data.loc[data.type == 'put', 'moneyness'] = pd.cut(data.loc[data.type == 'put', 'moneyness'],
                                                               bins=MONEYNESS_BINS, labels=LABELS)
            data.loc[data.type == 'call', 'moneyness'] = pd.cut(data.loc[data.type == 'call', 'moneyness'],
                                                                bins=MONEYNESS_BINS, labels=LABELS[::-1])
            data.loc[data['last_price'].gt(data['mid_price']), 'trades'] = 1
            data.loc[data['last_price'].lt(data['mid_price']), 'trades'] = -1
            data = data.loc[~data['trades'].isnull(), :]
            data['volume'] = np.where(data['trades'] == 1, data['ask_amount'], data['bid_amount'])
            data.dropna(inplace=True)
            mkt_participants = \
                feature_builder.define_market_participants(data)[['type', 'market_participants', 'volume', 'mark_iv']]
            data['netDealerExposure'] = -1 * (data['volume'].mul(data['trades']))
            data = feature_builder.compute_exposure(data=data)
            data.drop(['trades'], axis=1, inplace=True)
            data = \
                data.groupby(by=[pd.Grouper(key='type'),
                                 pd.Grouper(key='option_tag'),
                                 pd.Grouper(level='timestamp', sort=True,
                                            freq=args.freq)]).agg(resampling).reset_index(level=['type', 'option_tag'])
            omms_exposure = data[EXPOSURE].groupby(by=pd.Grouper(level='timestamp')).sum()
            overall_omms_exposure[symbol].append(omms_exposure.assign(symbol=symbol))
            ####################################################################################################
            ## Options
            ####################################################################################################
            put_call_ratio = feature_builder.put_call_ratio(data=data[['type', 'volume']])
            volatility_skew = feature_builder.volatility_skew(data=data[['type', 'volume', 'mark_iv', 'moneyness']],
                                                              freq=args.freq)
            overall_options_dynamics[symbol].append(pd.concat(
                [put_call_ratio, volatility_skew], axis=1).assign(symbol=symbol))
            ####################################################################################################
            ## Market participants
            ####################################################################################################
            mkt_participants = mkt_participants.groupby(by=[pd.Grouper(key='type'),
                                                            pd.Grouper(key='market_participants'),
                                                            pd.Grouper(level='timestamp', sort=True, freq=args.freq)])
            mkt_participants = mkt_participants[['volume', 'mark_iv']].agg({'volume': 'sum', 'mark_iv': 'last'})
            mkt_participants.reset_index(inplace=True)
            mkt_participants = \
                pd.concat(axis=1,
                          objs=[
                              mkt_participants.query(
                                  f'type=="{pair[0]}" & '
                                  f'market_participants=="{pair[1]}"').iloc[:, 2:].set_index('timestamp').rename(
                                  columns={column: f'{column}_{pair[0]}_{pair[1]}' for column in
                                           ['volume', 'mark_iv']}) for pair in
                              list(itertools.product(['call', 'put'], ['informed_traders', 'retail', 'institutions']))])
            mkt_participants['symbol'] = symbol
            mkt_participants_dynamics[symbol].append(mkt_participants)
    features = pd.concat([pd.concat(overall_spot_dynamics[symbol]) for symbol in symbols])
    features.index.name = 'timestamp'
    features = features.set_index('symbol', append=True).swaplevel(-1, 1)
    if args.options == 1:
        overall_options_dynamics = \
            pd.concat([pd.concat(overall_options_dynamics[symbol]) for
                       symbol in symbols]).set_index('symbol', append=True).swaplevel(-1, 1)
        overall_omms_exposure = pd.concat([pd.concat(overall_omms_exposure[symbol]) for
                                           symbol in symbols]).set_index('symbol', append=True).swaplevel(-1, 1)
        mkt_participants_dynamics = pd.concat([pd.concat(mkt_participants_dynamics[symbol]) for
                                               symbol in symbols]).set_index('symbol', append=True).swaplevel(-1, 1)
        features = \
            pd.concat([overall_options_dynamics, overall_omms_exposure, mkt_participants_dynamics, features], axis=1)
        features.dropna(inplace=True)
    target = features.loc[:, ['RV', 'returns']].copy()
    features = features.groupby(by=pd.Grouper(level='symbol'),
                                group_keys=False).apply(lambda x: x.fillna(x.expanding().mean()))
    features.reset_index(level=1, inplace=True)
    ####################################################################################################################
    ## Lags
    ####################################################################################################################
    features = \
        features.groupby(by=pd.Grouper(key='symbol'),
                         group_keys=False).apply(lambda x: feature_builder.lags(x, args.freq, args.D)).dropna()
    ####################################################################################################################
    ## One hot label encoding
    ####################################################################################################################
    features =  \
        pd.concat([features, pd.get_dummies(features['symbol'], prefix='is', prefix_sep='_').astype(int)], axis=1)
    features.sort_index(inplace=True)
    ####################################################################################################################
    ## Modelling
    ####################################################################################################################
    benchmark_model = models_factory_dd['lasso']
    model = models_factory_dd[args.regression]
    dates = pd.date_range(start=args.start_date, end=args.end_date, freq='1D', inclusive='left', tz=pytz.UTC)
    dates = list(dates)
    start_idx = dates.index(dates[0]+relativedelta(days=L_TRAIN_CONVERTER_DAYS[args.L]))
    models = pd.Series(data=np.nan, index=dates)
    benchmark_models = pd.Series(data=np.nan, index=dates)
    avg_1step = dict()
    features_importance = dict()
    for idx, date in enumerate(dates[start_idx::L_TRAIN_CONVERTER_DAYS[args.L]]):
        start_date = date-relativedelta(days=L_TRAIN_CONVERTER_DAYS[args.L])
        train_X = features.loc[(features.index.date >= start_date.date()) & (features.index.date < date.date()), :]
        in_sample_symbols = train_X.pop('symbol')
        train_y = train_X[[args.target]]
        train_X.drop(['RV', 'returns'], axis=1, inplace=True)
        train_dates = train_X.index.unique()
        train_dates, valid_dates = train_dates[:int(.8 * len(train_dates))], train_dates[int(.8 * len(train_dates)):]
        train_X, valid_X = train_X.loc[train_X.index.isin(train_dates), :], \
            train_X.loc[train_X.index.isin(valid_dates), :]
        train_y, valid_y = train_y.loc[train_y.index.isin(train_dates)], \
            train_y.loc[train_y.index.isin(valid_dates)]
        study_name = f'{model.name}_{args.L}_{args.target}_{date.strftime("%Y-%m-%d")}'
        if model.name in ['lasso', 'lightgbm']:
            rres_study = model.model_train(features=train_X, target=train_y.loc[:, args.target], study_name=study_name)
            models.loc[date] = rres_study.user_attrs['best_estimator']
        else:
            model.model_train(train_X=train_X, train_y=train_y, valid_X=valid_X, valid_y=valid_y, n_trials=1,
                              max_epochs=2)
            models.loc[date] = model
        if model.name != 'lasso':
            benchmark_models.loc[date] = \
                benchmark_model.model_train(features=train_X, target=train_y.loc[:, args.target],
                                            study_name=study_name.replace(args.regression, 'lasso_benchmark'))
        in_sample = pd.DataFrame(data=models.loc[date].predict(train_X),
                                 index=train_X.index, columns=['y_hat']) if args.regression == 'lasso' else \
            pd.DataFrame(data=benchmark_models.loc[date].user_attrs['best_estimator'].predict(train_X),
                         index=train_X.index, columns=['y_hat'])
        in_sample = pd.concat([in_sample, in_sample_symbols.loc[train_dates]], axis=1)
        in_sample_shift = \
            in_sample.groupby(by=pd.Grouper(key='symbol'), group_keys=True)[['y_hat']].apply(lambda x: x.shift())
        in_sample = in_sample.set_index('symbol', append=True).swaplevel(0, 1)
        avg_1step[date] = in_sample.sub(in_sample_shift).abs().mean()
    avg_1step = pd.concat(avg_1step).sort_index(level=0).droplevel(axis=0, level=1)
    models.ffill(inplace=True)
    models.dropna(inplace=True)
    ####################################################################################################################
    ## Performance evaluation
    ####################################################################################################################
    missing_dates = list()
    forecasts = list()
    mase = list()
    permutation_feature_importance_obj = PermutationFeatureImportance()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda x: x, x=date) for date in dates[start_idx::pd.to_timedelta(args.D).days]]
        for future in concurrent.futures.as_completed(futures):
            start_date_forecast = future.result().date()
            end_date_forecast = (start_date_forecast+relativedelta(days=pd.to_timedelta(args.D).days))
            test_X = features.loc[(features.index.date >= start_date_forecast) &
                                  (features.index.date < end_date_forecast), :]
            out_sample_symbols = test_X.pop('symbol')
            test_y = test_X[[args.target]]
            test_X.drop(['RV', 'returns'], axis=1, inplace=True)
            fine_tuned_model = models.loc[models.index.date == start_date_forecast].iloc[0] \
                if model.name in ['lasso', 'lightgbm'] else \
                models.loc[models.index.date == start_date_forecast].iloc[0].model
            try:
                if model.name == 'lstm':
                    test_X.iloc[:, :-2] = \
                        models.loc[models.index.date ==
                                   start_date_forecast].iloc[0].maxabsscaler.transform(test_X.iloc[:, :-2])
                    test_X_tensor = \
                        torch.from_numpy(test_X.values.reshape((test_X.shape[0], 1,
                                                                test_X.shape[1])).astype(np.float32))
                ########################################################################################################
                ## Fine-tuning
                ########################################################################################################
                train_fine_tune_X = features.loc[(features.index.date >= (date.date()-relativedelta(
                    days=pd.to_timedelta(args.D)//pd.to_timedelta('1D')))) & (features.index.date < date.date()), :]
                train_fine_tune_symbols = train_fine_tune_X.pop('symbol')
                train_fine_tune_y = train_fine_tune_X[[args.target]]
                train_fine_tune_X.drop(['RV', 'returns'], axis=1, inplace=True)
                fine_tuned_model = model.model_fine_tune(train_fine_tune_X=train_fine_tune_X, n_trials=5,
                                                         train_fine_tune_y=train_fine_tune_y,
                                                         model=fine_tuned_model, max_epochs=EPOCH//20, date=date,
                                                         L=args.L, target=args.target)
                test_y.loc[:, f'{args.target}_hat'] = fine_tuned_model(test_X_tensor).detach().numpy() if \
                    model.name == 'lstm' else fine_tuned_model.user_attrs['best_estimator'].predict(test_X)
            except Exception:
                if test_X.empty:
                    missing_dates.append(date)
            else:
                ########################################################################################################
                ## Model assessement
                ########################################################################################################
                test_y = pd.concat([test_y, out_sample_symbols], axis=1)
                test_y.loc[:, 'diff'] = test_y[[args.target, f'{args.target}_hat']].diff(axis=1).dropna(axis=1).values
                y_diff = test_y[['diff', 'symbol']].set_index('symbol', append=True).swaplevel(-1, 0)
                test_y.drop('diff', inplace=True, axis=1)
                mase.append(y_diff.div(avg_1step.loc[date]).abs().rename(columns={'diff': 'mase'}))
                mase[-1].loc[:, 'model_tag'] = \
                    f'{model.name.upper()}<sup>({args.target}, {args.D}, {dict([(0, "S"), (1, "O")])[args.options]}, ' \
                    f'{args.freq})</sup>'
                mase[-1].loc[:, 'model'] = model.name
                mase[-1].loc[:, 'target'] = args.target
                mase[-1].loc[:, 'horizon'] = args.D
                mase[-1].loc[:, 'options'] = args.options
                mase[-1].loc[:, 'freq'] = args.freq
                forecasts.append(test_y.rename(columns={args.target: 'values',
                                                        f'{args.target}_hat': 'forecast'}).assign(target=args.target))
                ########################################################################################################
                ## Feature importance
                ########################################################################################################
                permutation_feature_importance_obj.mase_benchmark = mase[-1].loc[:, 'mase']
                if isinstance(fine_tuned_model, optuna.study.study.Study):
                    permutation_feature_importance_obj.model = fine_tuned_model.user_attrs['best_estimator']
                else:
                    permutation_feature_importance_obj.model = fine_tuned_model
                permutation_feature_importance_obj.test = test_X
                permutation_feature_importance_obj.test_y = test_y
                permutation_feature_importance_obj.avg_1step = avg_1step.dropna()
                permutation_feature_importance_obj.score[test_y.index[0].date()] = \
                    permutation_feature_importance_obj.compute_score()
    features_importance = pd.DataFrame(permutation_feature_importance_obj.score).transpose()
    features_importance.index = pd.to_datetime(features_importance.index, utc=True)
    metric = pd.concat(mase).reset_index(level=0).sort_index()
    forecasts = pd.concat(forecasts)
    metric.index = pd.to_datetime(metric.index, utc=True)
    metric.index.name = 'timestamp'
    volatility_regime = reader_obj.rv_read(symbol=symbols).resample(args.freq).sum()
    volatility_regime = \
        volatility_regime.loc[(volatility_regime.index.date >= metric.index.unique()[0].date()) &
                              (volatility_regime.index.date <= metric.index.unique()[-1].date())].mean(axis=1)
    volatility_regime = pd.qcut(volatility_regime, q=[0, .45, .9, 1], labels=VOL_REGIMES)
    volatility_regime.name = 'volatility_regime'
    metric = metric.join(volatility_regime)
    features_importance = features_importance.join(volatility_regime, how='left')
    features_importance = pd.melt(ignore_index=False, id_vars=['volatility_regime'],
                                  value_name='MASE', var_name='features', frame=features_importance)
    features_importance = features_importance.assign(model=model.name, target=args.target, horizon=args.D,
                                                     options=args.options, freq=args.freq)
    ####################################################################################################################
    ## Database insertion
    ####################################################################################################################
    db_connect = sqlite3.connect(database='../data_centre/database.db', check_same_thread=False)
    forecasts.to_sql(con=db_connect, if_exists='append', name='forecasts')
    metric.to_sql(con=db_connect, if_exists='append', name='metric')
    features_importance.to_sql(con=db_connect, if_exists='append', name='feature_importance')
    print(f'[Data Insertion]: Results for {args} have been inserted into the database.')



