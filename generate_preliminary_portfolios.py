import itertools
from model.lab import FeatureBuilder
from data_centre.data import Reader
from constants import *
import os
import argparse
import glob
from datetime import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import pytz
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

mkt_participants_resampling = {exposure: 'sum' for exposure in EXPOSURE}
mkt_participants_resampling.update({'market_participants': 'last'})


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[:, 'symbol'] = [option_tag.split('-')[0] for option_tag in data['option_tag']]
    data['tau'] = (data.expiration - data.index) / pd.to_timedelta('360D')
    data['mid_price'] = .5 * (data['bid_price'] + data['ask_price'])
    data['trades'] = np.nan
    data.loc[data['last_price'].gt(data['mid_price']), 'trades'] = 1
    data.loc[data['last_price'].lt(data['mid_price']), 'trades'] = -1
    data = data.loc[~data['trades'].isnull(), :]
    data['volume'] = np.where(data['trades'] == 1, data['ask_amount'], data['bid_amount'])
    data.dropna(inplace=True)
    return data


def put_call_ratio(data: pd.DataFrame) -> pd.Series:
    put_call_ratio = \
        data.groupby(by=[pd.Grouper(key='type'), pd.Grouper(level='timestamp')])[['volume']].sum()
    put_call_ratio.reset_index(level=0, inplace=True)
    put_call_ratio = \
        put_call_ratio.query('type == "put"').volume.div(put_call_ratio.query('type == "call"').volume)
    put_call_ratio.name = 'put_call_ratio'
    return put_call_ratio


def map_maturity_moneyness(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[:, 'moneyness'] = np.log(data.loc[:, 'moneyness'])
    data['tau'] = pd.cut(data['tau'], bins=[0, zeroDTE, SHORT, MID, np.inf], labels=['0DTE', 'short', 'mid', 'long'])
    data.loc[data.type == 'put', 'moneyness'] = pd.cut(data.loc[data.type == 'put', 'moneyness'],
                                                       bins=MONEYNESS_BINS, labels=LABELS)
    data.loc[data.type == 'call', 'moneyness'] = pd.cut(data.loc[data.type == 'call', 'moneyness'],
                                                        bins=MONEYNESS_BINS, labels=LABELS[::-1])
    return data


def resample_dominant_mkt_participant(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    data = data.dropna().groupby(by=pd.Grouper(level='timestamp', sort=True, freq=freq),
                                 group_keys=False)[EXPOSURE + ['market_participants']].apply(
        lambda x: x[x['market_participants'] == x['market_participants'].mode()[0] if not
        x.empty else pd.DataFrame()])
    data = data.groupby(
        by=pd.Grouper(level='timestamp', sort=True, freq=freq)).agg(mkt_participants_resampling)
    data.replace(0, np.nan, inplace=True)
    data.ffill(inplace=True)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating tables and figures presenting preliminary'
                                                 'portfolios analysis.')
    parser.add_argument('--universe', type=str, help='Symbols to run analysis for.', default='BTC-ETH')
    parser.add_argument('--quantile', type=int, help='Number of quantiles to run the the analysis for.',
                        default=5)
    parser.add_argument('--freq', type=str, help='Bucket size.', default='30min')
    parser.add_argument('--agg_freq', type=str, help='Frequency used to compute metrics.', default='1D')
    parser.add_argument('--signal_filter', type=float, help='Threshold used to filter signals.', default=.5)
    args = parser.parse_args()
    num_quantile = args.quantile
    SUPERSCRIPT.update({qth: 'th' for qth in range(4, num_quantile + 1)})
    feature_builder = FeatureBuilder()
    symbols = args.universe.split('-')
    symbols = list(map(lambda x: f'{x}USDT', symbols))
    if not os.path.exists(os.path.abspath('../exposureReturns.csv')):
        reader_obj = Reader()
        underlying_price = reader_obj.prices_read(symbol=symbols)
        returns = np.log(underlying_price.div(underlying_price.shift()))
        usecols = ['type', 'strike_price', 'expiration', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount',
                   'mark_price', 'mark_iv', 'delta', 'gamma', 'vega', 'moneyness', 'last_price']
        overall_mkt_participants_report_exp_report = list()
        overall_agg_exposure_report = list()
        overall_put_call_ratio = dict([(symbol, list()) for symbol in symbols])
        files = list()
        files = [glob.glob(f'../data_centre/{symbol.replace("USDT","")}/*/*') for symbol in symbols]
        files = list(itertools.chain(*files))
        files.sort()
        # btc_files, eth_files = files[:(len(files) + 1) // 2], files[(len(files) + 1) // 2:]
        # btc_files, eth_files = btc_files[:31], eth_files[:31]
        # files = btc_files + eth_files
        for file in files:
            if file.split('/')[-1] != '2021-01-01':
                symbol = file.split("/")[2]
                symbol = f'{symbol}USDT'
                print(f'[Data processing]: {symbol}_{file.split("/")[-1]} is being treated now............')
                date = datetime.strptime(file.split('/')[-1], '%Y-%m-%d')
                yesterday_date = date-relativedelta(days=1)
                symbol_underlying_price_date = underlying_price.loc[underlying_price.index.date == date.date(),
                [symbol]].resample('5min').last().rename(columns={symbol: 'underlying_price'})
                symbol_underlying_price_yesterday_date = \
                    underlying_price.loc[underlying_price.index.date == yesterday_date.date(),
                    [symbol]].resample('5min').last().rename(columns={symbol: 'underlying_price'})
                yesterday_date_file = file.split('/')
                yesterday_date_file[-2] = f'{yesterday_date.year}'
                yesterday_date_file = \
                    '/'.join(yesterday_date_file[:-1] + [yesterday_date.strftime(DATE_FORMAT)])
                try:
                    data = pd.read_parquet(file, columns=usecols + ['option_tag'])
                    yesterday_data = pd.read_parquet(yesterday_date_file, columns=usecols + ['option_tag'])
                except FileNotFoundError as e:
                    continue
                if 'option_tag' not in data.columns:
                    data.reset_index(level='option_tag', inplace=True)
                data = preprocess_data(data)
                data = data.loc[data.index.date == date.date(), :]
                if 'option_tag' not in yesterday_data.columns:
                    yesterday_data.reset_index(level='option_tag', inplace=True)
                yesterday_data = preprocess_data(yesterday_data)
                yesterday_data = yesterday_data.loc[yesterday_data.index.date == yesterday_date.date(), :]
                data = data.join(symbol_underlying_price_date)
                yesterday_data = yesterday_data.join(symbol_underlying_price_yesterday_date)
                # ####################################################################################################
                # ## Put-Call ratio
                # ####################################################################################################
                # put_call_ratio_s = put_call_ratio(data=data)
                # overall_put_call_ratio[symbol].append(put_call_ratio_s.loc[put_call_ratio_s.index.date == date.date()])
                ####################################################################################################
                ## Exposures
                ####################################################################################################
                data['netDealerExposure'] = -1 * (data['volume'].mul(data['trades']))
                data['gamma'] = data['gamma'].replace(0, np.nan)
                data['gamma'] = data['gamma'].fillna(data['gamma'].expanding().mean())
                yesterday_data['gamma'] = yesterday_data['gamma'].replace(0, np.nan)
                yesterday_data['gamma'] = yesterday_data['gamma'].fillna(yesterday_data['gamma'].expanding().mean())
                data = feature_builder.compute_exposure(data)
                yesterday_data['netDealerExposure'] = -1 * (yesterday_data['volume'].mul(yesterday_data['trades']))
                yesterday_data = feature_builder.compute_exposure(yesterday_data)
                ####################################################################################################
                ## Underlying
                ####################################################################################################
                spot = symbol_underlying_price_date.resample(args.freq).last()
                ####################################################################################################
                ## Straddles
                ####################################################################################################
                spread = data[['type', 'option_tag', 'bid_price', 'ask_price']].copy()
                spread['spread'] = spread.ask_price.sub(data.bid_price)
                spread.drop(['bid_price', 'ask_price'], axis=1, inplace=True)
                spread = spread.groupby(by=[pd.Grouper(key='option_tag'),
                                            pd.Grouper(level='timestamp', sort=True)])[['spread']].sum()
                spread = spread.groupby(by=[pd.Grouper(level='option_tag'),
                                            pd.Grouper(level='timestamp', sort=True,
                                                       freq=args.freq)])[['spread']].last()
                spread = pd.pivot(spread.reset_index(), columns='option_tag', index='timestamp', values='spread')
                straddles = data[['type', 'option_tag', 'mid_price']].copy()
                straddles = straddles.groupby(by=[pd.Grouper(key='option_tag'),
                                                  pd.Grouper(level='timestamp', sort=True, freq=args.freq)])[
                    ['mid_price']].sum().reset_index()
                straddles = pd.pivot(straddles, columns='option_tag', index='timestamp', values='mid_price')
                call = data[['type', 'option_tag', 'mid_price']].query('type == "call"').drop('type', axis=1)
                call = pd.pivot(call.reset_index(), columns='option_tag', index='timestamp', values='mid_price')
                call = call.resample(args.freq).last()
                atm_straddles = \
                    data[['type', 'option_tag', 'moneyness']].query('type == "call"').drop('type', axis=1).copy()
                atm_straddles = \
                    pd.pivot(atm_straddles.reset_index(), columns='option_tag', index='timestamp', values='moneyness')
                atm_straddles = atm_straddles.resample(args.freq).last()
                atm_straddles = atm_straddles.gt(.9) & atm_straddles.lt(1.1)
                options_fees = underlying_price.loc[underlying_price.index.date == date.date(), symbol]
                spread = spread.reindex(options_fees.resample(args.freq).last().index)
                straddles = straddles.reindex(options_fees.resample(args.freq).last().index)
                call = call.reindex(options_fees.resample(args.freq).last().index)
                spread = spread.mul(options_fees.resample(args.freq).last().values, axis=0)
                straddles = straddles.mul(options_fees.resample(args.freq).last().values, axis=0)
                call = call.mul(options_fees.resample(args.freq).last().values, axis=0)
                atm_straddles = (straddles * (straddles.gt(call)) * atm_straddles)
                options_fees = atm_straddles.shape[1]*options_fees.resample(args.freq).last().values.tolist()
                options_fees = np.array(options_fees).reshape((atm_straddles.shape[1], atm_straddles.shape[0]))
                options_fees = pd.DataFrame(columns=atm_straddles.index, index=atm_straddles.columns,
                                            data=options_fees).transpose().mul(FEES_OPTIONS)
                options_fees = options_fees.where(options_fees < straddles, straddles.mul(FEES_STRADDLE_CAP))
                atm_straddles = atm_straddles.replace(0, np.nan).dropna(axis=1, how='all')
                entry_straddles = atm_straddles.copy()
                entry_straddles_fees = options_fees.copy()
                exit_straddles = atm_straddles.shift().copy()
                exit_straddles_fees = options_fees.shift().copy()
                PnL_atm_straddles = \
                    exit_straddles.sub(entry_straddles).sub(entry_straddles_fees.add(exit_straddles_fees)).div(
                        entry_straddles.sub(entry_straddles_fees)).mean(axis=1)
                Tn_atm_straddles = (~atm_straddles.isnull()).astype(int).sum(axis=1)
                PnL_atm_straddles = PnL_atm_straddles.div(Tn_atm_straddles.replace(0, np.nan))
                # atm_straddles.sub(spread)
                # long_atm_straddles = \
                #     atm_straddles.sub(options_fees.sub(spread)).sub(atm_straddles.shift().sub(
                #         options_fees.shift().sub(spread.shift())))
                # long_atm_straddles = long_atm_straddles.div(atm_straddles).mean(axis=1)
                # short_atm_straddles = \
                #     atm_straddles.shift().sub(options_fees.shift().sub(spread.shift())).sub(
                #         atm_straddles.sub(options_fees.sub(spread)))
                # short_atm_straddles = short_atm_straddles.div(atm_straddles).mean(axis=1)
                data.drop(['expiration', 'last_price', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount',
                           'mark_price'], axis=1, inplace=True)
                yesterday_data.drop(['expiration', 'last_price', 'bid_price', 'bid_amount', 'ask_price', 'ask_amount',
                                     'mark_price'], axis=1, inplace=True)
                weighted_avg_mark_iv = data[['type', 'mark_iv', 'volume']].copy()
                weighted_avg_mark_iv['weighted_mark_iv'] = data.mark_iv.mul(data.volume)
                vol_weighted_mark_iv = \
                    weighted_avg_mark_iv.groupby(
                        [pd.Grouper(key='type'),
                         pd.Grouper(level='timestamp', sort=True, freq=args.freq)])['weighted_mark_iv'].sum()
                vol_volume = weighted_avg_mark_iv.groupby(
                    [pd.Grouper(key='type'), pd.Grouper(level='timestamp', sort=True, freq=args.freq)])['volume'].sum()
                weighted_avg_mark_iv = pd.pivot(vol_weighted_mark_iv.div(vol_volume).reset_index(),
                                                index='timestamp', values=0, columns='type').mean(axis=1)
                weighted_avg_mark_iv = \
                    pd.qcut(weighted_avg_mark_iv, q=num_quantile, duplicates='drop',
                            labels=[f'{th}<sup>{SUPERSCRIPT[th]}</sup>' for th in range(1, num_quantile + 1)])
                weighted_avg_mark_iv.name = 'mark_iv'
                ####################################################################################################
                ## Aggregated exposure based signals
                ####################################################################################################
                agg_exp = data[EXPOSURE].groupby(pd.Grouper(level='timestamp', sort=True, freq=args.freq)).sum()
                yesterday_agg_exp = \
                    yesterday_data[EXPOSURE].groupby(pd.Grouper(level='timestamp', sort=True, freq=args.freq)).sum()
                agg_exp = agg_exp.loc[agg_exp.index.date == date.date(), :]
                yesterday_agg_exp = yesterday_agg_exp.loc[yesterday_agg_exp.index.date == yesterday_date.date(), :]
                # agg_exp = agg_exp.replace(0, np.nan)
                # agg_exp = agg_exp.fillna(agg_exp.expanding().mean())
                # yesterday_agg_exp = yesterday_agg_exp.replace(0, np.nan)
                # yesterday_agg_exp = yesterday_agg_exp.fillna(agg_exp.expanding().mean())
                periods = pd.to_timedelta(args.agg_freq) // (agg_exp.index[1] - agg_exp.index[0])
                if agg_exp.shape[0] != periods:
                    agg_exp = \
                        agg_exp.reindex(pd.date_range(start=date.date(), end=date.date() + relativedelta(days=1),
                                                      periods=periods + 1,
                                                      tz=pytz.UTC, inclusive='left'))
                    agg_exp = agg_exp.fillna(agg_exp.expanding().mean()).fillna(agg_exp.mean())
                    yesterday_agg_exp = \
                        yesterday_agg_exp.reindex(pd.date_range(start=yesterday_date.date(), end=date.date(),
                                                                periods=periods + 1, tz=pytz.UTC, inclusive='left'))
                    yesterday_agg_exp = \
                        yesterday_agg_exp.fillna(agg_exp.expanding().mean()).fillna(yesterday_agg_exp.mean())
                # returns
                intraday_returns_train = \
                    returns.loc[returns.index.date == date.date() - relativedelta(days=1), symbol].copy()
                intraday_returns_train = \
                    intraday_returns_train.resample(f'{intraday_returns_train.shape[0] // periods}min').sum()
                intraday_returns = returns.loc[returns.index.date == date.date(), symbol].copy()
                intraday_returns = intraday_returns.resample(f'{intraday_returns.shape[0] // periods}min').sum()
                sampled_mean_returns = returns.loc[returns.index.date == date.date(), symbol].copy()
                sampled_mean_returns = \
                    sampled_mean_returns.resample(f'{sampled_mean_returns.shape[0] // periods}min').mean()
                # rv
                intraday_rv_train = \
                    returns.loc[returns.index.date == date.date() - relativedelta(days=1), symbol].copy()**2
                intraday_rv_train = \
                    intraday_rv_train.resample(f'{intraday_rv_train.shape[0] // periods}min').sum()
                intraday_rv = returns.loc[returns.index.date == date.date(), symbol].copy()**2
                intraday_rv = intraday_rv.resample(f'{intraday_rv.shape[0] // periods}min').sum()
                sampled_mean_rv = (returns.loc[returns.index.date == date.date(), symbol].copy()**2).copy()
                sampled_mean_rv = sampled_mean_rv.resample(f'{sampled_mean_rv.shape[0] // periods}min').mean()
                try:
                    tmp_returns = \
                        yesterday_agg_exp.apply(lambda x, y: LinearRegression().fit(X=x.values.reshape(-1, 1), y=y),
                                                y=intraday_returns_train.loc[yesterday_agg_exp.index].values)
                except ValueError:
                    pdb.set_trace()
                tmp_rv = yesterday_agg_exp.apply(lambda x, y: LinearRegression().fit(X=x.values.reshape(-1, 1), y=y),
                                                 y=intraday_rv_train.loc[yesterday_agg_exp.index].values)
                forecast_returns = \
                    pd.DataFrame(index=EXPOSURE, columns=agg_exp.index,
                                 data=np.array([tmp_returns[exp].predict(agg_exp[exp].values.reshape(-1, 1))
                                                for exp in EXPOSURE])).transpose()
                # filter_signals = \
                #     forecast_returns.sub(intraday_returns.shift().values.reshape(-1, 1)).div(
                #         sampled_mean_returns.values.reshape(-1, 1)).gt(100*args.signal_filter)
                # forecast_returns = filter_signals.mul(forecast_returns)
                signals_agg_exp_returns = -np.sign(forecast_returns)
                signals_agg_exp_returns.columns = pd.MultiIndex.from_product([['spot'], EXPOSURE])
                forecast_rv = \
                    pd.DataFrame(index=EXPOSURE, columns=agg_exp.index,
                                 data=np.array([tmp_rv[exp].predict(agg_exp[exp].values.reshape(-1, 1))
                                                for exp in EXPOSURE])).transpose()
                # filter_signals = \
                #     forecast_rv.sub(intraday_rv.shift().values.reshape(-1, 1)).div(
                #         sampled_mean_rv.values.reshape(-1, 1)).gt(100 * args.signal_filter)
                # forecast_rv = filter_signals.mul(forecast_rv)
                signals_agg_exp_rv = -np.sign(forecast_rv)
                signals_agg_exp_rv.columns = pd.MultiIndex.from_product([['options'], EXPOSURE])
                signals = pd.concat([signals_agg_exp_returns, signals_agg_exp_rv], axis=1).shift()
                agg_exposure_report = pd.DataFrame(index=signals.index, columns=signals.columns, data=np.nan)
                entry_spot = spot.copy()
                entry_fees = np.abs(signals.iloc[:, :4]).mul(spot.values.reshape(-1, 1)).mul(FEES_SPOT)
                exit_spot = spot.shift().copy()
                exit_fees = np.abs(signals.iloc[:, :4]).mul(spot.shift().values.reshape(-1, 1)).mul(FEES_SPOT)
                total_trades_fees_spot = exit_fees.add(entry_fees)
                agg_exposure_report.iloc[:, :4] = \
                    total_trades_fees_spot.add(
                        exit_spot.sub(entry_spot).values.reshape(-1, 1)).div(entry_fees.sub(
                        entry_spot.values.reshape(-1, 1)))
                agg_exposure_report.iloc[:, 4:] = signals.iloc[:, 4:].apply(lambda x, y: x.mul(y), y=PnL_atm_straddles)
                # agg_exposure_report = \
                #     pd.concat([signals.iloc[:, :4].apply(lambda x, y: x.mul(y),
                #                                          y=intraday_returns.sub(FEES_SPOT).values),
                #                signals.iloc[:, 4:].apply(lambda x, y, z: np.where(x > 0, y, z), y=long_atm_straddles,
                #                                          z=short_atm_straddles)], axis=1)
                agg_exp = agg_exp.resample(args.freq).sum()
                agg_exp = \
                    agg_exp.join(agg_exp.apply(
                        lambda x: pd.qcut(x, q=num_quantile, labels=[f'{th}<sup>{SUPERSCRIPT[th]}</sup>' for th
                                                                     in range(1, num_quantile + 1)])),
                        rsuffix='_quantile')
                agg_exposure_report.columns = [f'spot_{exposure}' for exposure in EXPOSURE] + \
                                              [f'options_{exposure}' for exposure in EXPOSURE]
                agg_exposure_report = agg_exposure_report.join(agg_exp, how='left')
                agg_exposure_report = pd.concat([agg_exposure_report, weighted_avg_mark_iv], axis=1)
                agg_exposure_report.loc[:, 'symbol'] = symbol
                overall_agg_exposure_report.append(agg_exposure_report)
                ####################################################################################################
                ## Market participant based signals
                ####################################################################################################
                mkt_participants_data = data[EXPOSURE + ['type', 'tau', 'moneyness']].copy()
                yesterday_mkt_participants_data = yesterday_data[EXPOSURE + ['type', 'tau', 'moneyness']].copy()
                mkt_participants_data = map_maturity_moneyness(mkt_participants_data)
                yesterday_mkt_participants_data = map_maturity_moneyness(yesterday_mkt_participants_data)
                mkt_participants_data = feature_builder.define_market_participants(data=mkt_participants_data).drop(
                        ['type', 'tau', 'moneyness'], axis=1)
                yesterday_mkt_participants_data = \
                    feature_builder.define_market_participants(data=yesterday_mkt_participants_data).drop(
                        ['type', 'tau', 'moneyness'], axis=1)
                mkt_participants_data = resample_dominant_mkt_participant(mkt_participants_data, args.freq)
                yesterday_mkt_participants_data = resample_dominant_mkt_participant(yesterday_mkt_participants_data,
                                                                                    args.freq)
                # mkt_participants_data = \
                #     feature_builder.define_market_participants(data=mkt_participants_data).drop(['tau', 'moneyness'],
                #                                                                                 axis=1)
                # yesterday_mkt_participants_data = \
                #     feature_builder.define_market_participants(data=yesterday_mkt_participants_data).drop(
                #         ['tau', 'moneyness'], axis=1)
                # mkt_participants_data = mkt_participants_data.replace(0, np.nan).dropna()
                # yesterday_mkt_participants_data = yesterday_mkt_participants_data.replace(0, np.nan).dropna()
                # contract_exp = data[EXPOSURE + ['type']].copy()
                # contract_exp = \
                #     contract_exp.groupby(by=[pd.Grouper(key='type'),
                #                              pd.Grouper(level='timestamp',
                #                                         sort=True,
                #                                         freq=args.freq)])[EXPOSURE].sum().reset_index(level=0)
                # contract_exp['date_lag1'] = [timestamp - relativedelta(days=1) for timestamp in contract_exp.index]
                # returns
                # intraday_returns_train = intraday_returns_train.reindex(contract_exp['date_lag1'])
                # intraday_returns = intraday_returns.reindex(contract_exp.index)
                # sampled_mean_returns = sampled_mean_returns.reindex(contract_exp.index)
                # contract_exp.drop('date_lag1', axis=1, inplace=True)
                # tmp_returns = contract_exp[EXPOSURE].apply(lambda x, y:
                #                                            LinearRegression().fit(X=x.values.reshape(-1, 1), y=y),
                #                                            y=intraday_returns_train.values)
                tmp_returns = \
                    yesterday_mkt_participants_data[EXPOSURE].apply(
                        lambda x, y: LinearRegression().fit(X=x.values.reshape(-1, 1), y=y),
                        y=intraday_returns_train.loc[yesterday_mkt_participants_data.index].values)
                forecast_returns = \
                    pd.DataFrame(index=EXPOSURE, columns=mkt_participants_data.index,
                                 data=np.array(
                                     [tmp_returns[exp].predict(mkt_participants_data[exp].values.reshape(-1, 1))
                                      for exp in EXPOSURE])).transpose()
                # filter_signals = \
                #     forecast_returns.sub(
                #         intraday_returns.loc[forecast_returns.index].shift().values.reshape(-1, 1)).div(
                #         sampled_mean_returns.loc[forecast_returns.index].values.reshape(-1, 1)).gt(
                #         100 * args.signal_filter)
                # forecast_returns = filter_signals.mul(forecast_returns)
                signals_contract_exp_returns = -np.sign(forecast_returns)
                signals_contract_exp_returns.columns = pd.MultiIndex.from_product([['spot'], EXPOSURE])
                # rv
                # intraday_rv_train = intraday_rv_train.reindex(intraday_returns_train.index)
                # intraday_rv = intraday_rv.reindex(contract_exp.index)
                # sampled_mean_rv = sampled_mean_rv.reindex(contract_exp.index)
                tmp_rv = \
                    yesterday_mkt_participants_data[EXPOSURE].apply(
                        lambda x, y: LinearRegression().fit(X=x.values.reshape(-1, 1), y=y),
                        y=intraday_rv_train.loc[yesterday_mkt_participants_data.index].values)
                forecast_rv = \
                    pd.DataFrame(index=EXPOSURE, columns=mkt_participants_data.index,
                                 data=np.array(
                                     [tmp_rv[exp].predict(mkt_participants_data[exp].values.reshape(-1, 1))
                                      for exp in EXPOSURE])).transpose()
                # filter_signals = \
                #     forecast_rv.sub(intraday_rv.loc[forecast_rv.index].shift().values.reshape(-1, 1)).div(
                #         sampled_mean_rv.loc[forecast_rv.index].values.reshape(-1, 1)).gt(100 * args.signal_filter)
                # forecast_rv = filter_signals.mul(forecast_rv)
                signals_contract_exp_rv = -np.sign(forecast_rv)
                signals_contract_exp_rv.columns = pd.MultiIndex.from_product([['options'], EXPOSURE])
                signals = pd.concat([signals_contract_exp_returns, signals_contract_exp_rv], axis=1)
                mkt_participants_report = pd.DataFrame(data=np.nan, index=signals.index, columns=signals.columns)
                spot = spot.reindex(signals.index)
                entry_spot = spot.copy()
                entry_fees = np.abs(signals.iloc[:, :4]).mul(spot.values.reshape(-1, 1)).mul(FEES_SPOT)
                exit_spot = spot.shift().copy()
                exit_fees = np.abs(signals.iloc[:, :4]).mul(spot.shift().values.reshape(-1, 1)).mul(FEES_SPOT)
                total_trades_fees_spot = exit_fees.add(entry_fees)
                mkt_participants_report.iloc[:, :4] = \
                    total_trades_fees_spot.add(
                        exit_spot.sub(entry_spot).values.reshape(-1, 1)).div(entry_fees.sub(
                        entry_spot.values.reshape(-1, 1)))
                atm_straddles = atm_straddles.reindex(signals.index)
                mkt_participants_report.iloc[:, 4:] = signals.iloc[:, 4:].apply(lambda x, y: x.mul(y),
                                                                                y=PnL_atm_straddles)
                # moneyness_mat_exp_report = \
                #     pd.concat([signals.iloc[:, :4].apply(lambda x, y: x.mul(y), y=intraday_returns.values),
                #                signals.iloc[:, 4:].apply(lambda x, y, z: np.where(x > 0, y, z),
                #                                          y=long_atm_straddles.reindex(intraday_returns.index),
                #                                          z=short_atm_straddles.reindex(intraday_returns.index))],
                #               axis=1)
                mkt_participants_report.columns = [f'spot_{exposure}' for exposure in EXPOSURE] + \
                                                  [f'options_{exposure}' for exposure in EXPOSURE]
                # data.loc[:, 'moneyness'] = np.log(data.loc[:, 'moneyness'])
                # data['tau'] = pd.cut(data['tau'], bins=[0, zeroDTE, SHORT, MID, np.inf],
                #                      labels=['0DTE', 'short', 'mid', 'long'])
                # data.loc[data.type == 'put', 'moneyness'] = pd.cut(data.loc[data.type == 'put', 'moneyness'],
                #                                                    bins=MONEYNESS_BINS, labels=LABELS)
                # data.loc[data.type == 'call', 'moneyness'] = pd.cut(data.loc[data.type == 'call', 'moneyness'],
                #                                                     bins=MONEYNESS_BINS, labels=LABELS[::-1])
                # data = feature_builder.define_market_participants(data=data).drop(['tau', 'moneyness'], axis=1)
                # mkt_participants_data = \
                #     data.groupby([pd.Grouper(key='market_participants'), pd.Grouper(key='type'),
                #                  pd.Grouper(level='timestamp', sort=True, freq=args.freq)])[['volume',
                #                                                                             'mark_iv']+EXPOSURE]
                # pdb.set_trace()
                # mkt_participants_data = mkt_participants_data.agg(mkt_participants_resampling)
                # pdb.set_trace()
                # data
                # data[['tau', 'moneyness', 'volume', 'mark_iv', 'type'] +
                #      EXPOSURE].resample(args.freq).agg({'tau': 'last', 'moneyess': 'last', 'volume': 'sum', 'mark_iv'})
                # moneyness_mat_exp_report = \
                #     moneyness_mat_exp_report.join(data[['tau', 'moneyness', 'volume', 'mark_iv', 'type'] + EXPOSURE],
                #                                   how='left', rsuffix='_exp')
                # moneyness_mat_exp_report.loc[:, 'moneyness'] = np.log(moneyness_mat_exp_report.loc[:, 'moneyness'])
                # moneyness_mat_exp_report['tau'] = pd.cut(moneyness_mat_exp_report['tau'],
                #                                          bins=[0, zeroDTE, SHORT, MID, np.inf],
                #                                          labels=['0DTE', 'short', 'mid', 'long'])
                # moneyness_mat_exp_report.loc[moneyness_mat_exp_report.type == 'put', 'moneyness'] = \
                #     pd.cut(moneyness_mat_exp_report.loc[moneyness_mat_exp_report.type == 'put', 'moneyness'],
                #            bins=MONEYNESS_BINS, labels=LABELS)
                # moneyness_mat_exp_report.loc[moneyness_mat_exp_report.type == 'call', 'moneyness'] = \
                #     pd.cut(moneyness_mat_exp_report.loc[moneyness_mat_exp_report.type == 'call', 'moneyness'],
                #            bins=MONEYNESS_BINS, labels=LABELS[::-1])
                # pdb.set_trace()
                # moneyness_mat_exp_report = feature_builder.define_market_participants(moneyness_mat_exp_report)
                # moneyness_mat_exp_report = \
                #     moneyness_mat_exp_report.groupby(by=[pd.Grouper(key='market_participants'),
                #                                          pd.Grouper(key='type'),
                #                                          pd.Grouper(level='timestamp', sort=True, freq=freq)]
                #                                      ).agg(RESAMPLING_MAT_MONEYNESS).reset_index()
                mkt_participants_report = \
                    pd.concat([mkt_participants_report, mkt_participants_data, weighted_avg_mark_iv], axis=1)
                mkt_participants_report.loc[:, 'symbol'] = symbol
                overall_mkt_participants_report_exp_report.append(mkt_participants_report)
    overall_agg_exposure_report = pd.concat(overall_agg_exposure_report)
    overall_agg_exposure_report.index.name = 'timestamp'
    overall_agg_exposure_report.set_index(['symbol', 'mark_iv'])
    overall_moneyness_mat_exp_report = pd.concat(overall_mkt_participants_report_exp_report)
    overall_agg_exposure_report = pd.concat([pd.DataFrame(
        overall_agg_exposure_report.groupby(
            by=[pd.Grouper(key='symbol'),
                pd.Grouper(key='mark_iv'),
                pd.Grouper(key=f'{exp}_quantile'),
                pd.Grouper(level='timestamp',
                           sort=True, freq=args.agg_freq)]).agg({f'spot_{exp}': 'sum',
                                                                 f'options_{exp}': 'mean',
                                                                 exp: 'sum'})).reset_index(2).rename(
        columns={f'{exp}': 'levels', f'spot_{exp}': 'spot', f'options_{exp}': 'options',
                 f'{exp}_quantile': 'exp_quantile'}).assign(exp=exp) for exp in EXPOSURE]
                                            ).set_index('exp_quantile', append=True).swaplevel(-1, -2)
    overall_moneyness_mat_exp_report = pd.concat([pd.DataFrame(
        overall_moneyness_mat_exp_report.groupby(
            by=[pd.Grouper(key='symbol'),
                pd.Grouper(key='market_participants'),
                pd.Grouper(key='mark_iv'),
                pd.Grouper(level='timestamp',
                           sort=True,
                           freq=args.agg_freq)])[[f'spot_{exp}', f'options_{exp}']].mean().rename(
            columns={f'spot_{exp}': 'spot',
                     f'options_{exp}': 'options'})).assign(exp=exp) for exp in EXPOSURE])
    overall_agg_exposure_PnL = overall_agg_exposure_report.groupby(by=[pd.Grouper(level='symbol'),
                                                                       pd.Grouper(level='mark_iv'),
                                                                       pd.Grouper(level='exp_quantile'),
                                                                       pd.Grouper(key='exp')])
    overall_moneyness_mat_exp_PnL = overall_moneyness_mat_exp_report.groupby(by=[pd.Grouper(level='symbol'),
                                                                                 pd.Grouper(level='market_participants'),
                                                                                 pd.Grouper(level='mark_iv'),
                                                                                 pd.Grouper(key='exp')])
    agg_exposure_sharpe_mean = overall_agg_exposure_PnL[['spot', 'options']].mean()
    agg_exposure_sharpe_std = overall_agg_exposure_PnL[['spot', 'options']].std()
    agg_exposure_sharpe = \
        agg_exposure_sharpe_mean.div(agg_exposure_sharpe_std).mul(
            FREQUENCY_CONVERTER[args.agg_freq] ** .5).round(2).reset_index()
    agg_exposure_sharpe = \
        pd.melt(id_vars=['symbol', 'mark_iv', 'exp_quantile', 'exp'], var_name='markets',
                value_name='PnL', frame=agg_exposure_sharpe)
    agg_exposure_PnL = \
        overall_agg_exposure_PnL[['spot',
                                  'options']].mean().mul(FREQUENCY_CONVERTER[args.agg_freq]).round(4).reset_index()
    agg_exposure_PnL = \
        pd.melt(id_vars=['symbol', 'mark_iv', 'exp_quantile', 'exp'], var_name='markets',
                value_name='PnL', frame=agg_exposure_PnL)
    moneyness_mat_exp_sharpe_mean = overall_moneyness_mat_exp_PnL[['spot', 'options']].mean()
    moneyness_mat_exp_sharpe_std = overall_moneyness_mat_exp_PnL[['spot', 'options']].std()
    moneyness_mat_exp_sharpe = \
        moneyness_mat_exp_sharpe_mean.div(moneyness_mat_exp_sharpe_std).mul(
            (FREQUENCY_CONVERTER[args.agg_freq] ** .5)).round(2).reset_index()
    moneyness_mat_exp_sharpe = pd.melt(id_vars=['symbol', 'market_participants', 'mark_iv', 'exp'], var_name='markets',
                                       value_name='PnL', frame=moneyness_mat_exp_sharpe)
    moneyness_mat_exp_PnL = \
        overall_moneyness_mat_exp_PnL[['spot', 'options']].mean().mul(
            FREQUENCY_CONVERTER[args.agg_freq]).round(4).reset_index()
    moneyness_mat_exp_PnL = pd.melt(id_vars=['symbol', 'market_participants', 'mark_iv', 'exp'], var_name='markets',
                                    value_name='PnL', frame=moneyness_mat_exp_PnL)
    moneyness_mat_exp_PnL = moneyness_mat_exp_PnL.groupby(by=[pd.Grouper(key='symbol'),
                                                              pd.Grouper(key='exp')]).apply(
        lambda x: x.sort_values('PnL', ascending=False)).drop(['symbol', 'exp'], axis=1).droplevel(axis=0, level=-1)
    ####################################################################################################################
    ## Data visualisation
    ####################################################################################################################
    returns.index.name = 'timestamp'
    exposure = overall_agg_exposure_report.copy().reset_index(level=['mark_iv', 'exp_quantile'])
    rv = pd.melt((returns ** 2).resample(args.freq).sum().shift(-1).reset_index(), value_name='rv',
                 var_name='symbol', id_vars='timestamp').set_index(['symbol', 'timestamp']).loc[exposure.index, :]
    returns = pd.melt(returns.resample(args.freq).sum().shift(-1).reset_index(), value_name='returns',
                      var_name='symbol', id_vars='timestamp').set_index(['symbol',
                                                                        'timestamp']).loc[exposure.index, :]
    exposure = pd.concat([exposure, returns, rv], axis=1).reset_index().replace(0, np.nan).dropna()
    fig = \
        px.scatter(exposure, x='levels', y='returns', trendline='ols', trendline_color_override='red',
                   facet_row='symbol', facet_col='exp', facet_col_spacing=.05, facet_row_spacing=.1,
                   title='OMMs exposures versus returns')
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].upper()}),
                            selector=lambda x: 'exp' in x['text'])
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1]}),
                            selector=lambda x: 'symbol' in x['text'])
    fig.update_xaxes(tickangle=45)
    fig.show()
    fig.for_each_xaxis(lambda x: x.update({'title': ''}))
    fig = \
        px.scatter(exposure, x='levels', y='rv', trendline='ols', trendline_color_override='red',
                   facet_row='symbol', facet_col='exp', facet_col_spacing=.05, facet_row_spacing=.1,
                   title='OMMs exposures versus RV')
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].upper()}),
                            selector=lambda x: 'exp' in x['text'])
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1]}),
                            selector=lambda x: 'symbol' in x['text'])
    fig.update_xaxes(tickangle=45)
    fig.for_each_xaxis(lambda x: x.update({'title': ''}))
    fig.show()
    for market in ['spot', 'options']:
        heatmap_pnl_quantile_plot = \
            make_subplots(cols=len(EXPOSURE), rows=len(symbols), column_titles=EXPOSURE,
                          row_titles=[f'IV<sup>({symbol.replace("USDT", "")})</sup>' for symbol in symbols],
                          shared_yaxes=True)
        heatmap_sharpe_quantile_plot = \
            make_subplots(cols=len(EXPOSURE), rows=len(symbols), column_titles=EXPOSURE,
                          row_titles=[f'IV<sup>({symbol.replace("USDT", "")})</sup>' for symbol in symbols],
                          shared_yaxes=True)
        heatmap_pnl_tau_moneyness_plot = \
            make_subplots(cols=len(EXPOSURE), rows=len(symbols), column_titles=EXPOSURE,
                          row_titles=[f'IV<sup>({symbol.replace("USDT", "")})</sup>' for symbol in symbols],
                          shared_yaxes=True)
        heatmap_sharpe_tau_moneyness_plot = \
            make_subplots(cols=len(EXPOSURE), rows=len(symbols), column_titles=EXPOSURE,
                          row_titles=[f'IV<sup>({symbol.replace("USDT", "")})</sup>' for symbol in symbols],
                          shared_yaxes=True)
        for col, exp in enumerate(EXPOSURE):
            for row, symbol in enumerate(symbols):
                tmp = agg_exposure_PnL.query(f'exp == "{exp}" & symbol == "{symbol}" & markets == "{market}"')
                tmp = tmp.rename(columns={'exp_quantile': exp}).drop('exp', axis=1)
                heatmap_pnl_quantile_plot.add_trace(go.Heatmap(z=tmp['PnL'].values, x=tmp.mark_iv.values,
                                                               y=tmp[exp].values, colorscale='Viridis',
                                                               zmin=agg_exposure_PnL.query(f'markets == "{market}"')
                                                               ['PnL'].min(),
                                                               zmax=agg_exposure_PnL.query(f'markets == "{market}"')
                                                               ['PnL'].max()), col=col + 1,
                                                    row=row + 1)
                tmp = agg_exposure_sharpe.query(f'exp == "{exp}" & symbol == "{symbol}" & markets == "{market}"')
                tmp = tmp.rename(columns={'exp_quantile': exp}).drop('exp', axis=1)
                heatmap_sharpe_quantile_plot.add_trace(
                    go.Heatmap(z=tmp['PnL'].values, x=tmp.mark_iv.values, y=tmp[exp].values, colorscale='Viridis',
                               zmin=agg_exposure_sharpe.query(f'markets == "{market}"')['PnL'].min(),
                               zmax=agg_exposure_sharpe.query(f'markets == "{market}"')['PnL'].max()),
                               col=col + 1, row=row + 1)
                tmp = \
                    moneyness_mat_exp_PnL.reset_index(level=-1).query(f'exp == "{exp}" & '
                                                                      f'symbol == "{symbol}" &'
                                                                      f'markets == "{market}"').drop('exp', axis=1)
                heatmap_pnl_tau_moneyness_plot.add_trace(
                    go.Heatmap(z=tmp['PnL'].values, x=tmp.market_participants.values, y=tmp.mark_iv.values,
                               colorscale='Viridis',
                               zmin=moneyness_mat_exp_PnL.query(f'markets == "{market}"')['PnL'].min(),
                               zmax=moneyness_mat_exp_PnL.query(f'markets == "{market}"')['PnL'].max()),
                    col=col + 1, row=row + 1)
                tmp = \
                    moneyness_mat_exp_sharpe.reset_index(level=-1).query(f'exp == "{exp}" & '
                                                                         f'symbol == "{symbol}" &'
                                                                         f'markets == "{market}"').drop('exp', axis=1)
                heatmap_sharpe_tau_moneyness_plot.add_trace(
                    go.Heatmap(z=tmp['PnL'].values, x=tmp.market_participants.values, y=tmp.mark_iv.values,
                               colorscale='Viridis',
                               zmin=moneyness_mat_exp_sharpe.query(f'markets == "{market}"')['PnL'].min(),
                               zmax=moneyness_mat_exp_sharpe.query(f'markets == "{market}"')['PnL'].max()),
                    col=col + 1, row=row + 1)
        heatmap_pnl_quantile_plot.update_xaxes(categoryorder='array',
                                               categoryarray=[f'{qth}<sup>{SUPERSCRIPT[qth]}</sup>' for
                                                              qth in range(1, num_quantile + 1)])
        heatmap_pnl_quantile_plot.update_yaxes(categoryorder='array',
                                               categoryarray=[f'{qth}<sup>{SUPERSCRIPT[qth]}</sup>' for
                                                              qth in range(1, num_quantile + 1)])
        heatmap_sharpe_quantile_plot.update_xaxes(categoryorder='array',
                                                  categoryarray=[f'{qth}<sup>{SUPERSCRIPT[qth]}</sup>' for
                                                                 qth in range(1, num_quantile + 1)])
        heatmap_sharpe_quantile_plot.update_yaxes(categoryorder='array',
                                                  categoryarray=[f'{qth}<sup>{SUPERSCRIPT[qth]}</sup>' for
                                                                 qth in range(1, num_quantile + 1)])
        # heatmap_pnl_tau_moneyness_plot.update_xaxes(categoryorder='array',
        #                                             categoryarray=['DITM', 'ITM', 'ATM', 'OTM', 'DOTM'],
        #                                             gridcolor='black')
        # heatmap_pnl_tau_moneyness_plot.update_yaxes(categoryorder='array',
        #                                             categoryarray=['0DTE', 'short', 'mid', 'long'],
        #                                             gridcolor='black')
        # heatmap_sharpe_tau_moneyness_plot.update_xaxes(categoryorder='array',
        #                                                categoryarray=['DITM', 'ITM', 'ATM', 'OTM', 'DOTM'],
        #                                                gridcolor='black')
        # heatmap_sharpe_tau_moneyness_plot.update_yaxes(categoryorder='array',
        #                                                categoryarray=['0DTE', 'short', 'mid', 'long'],
        #                                                gridcolor='black')
        heatmap_pnl_tau_moneyness_plot.update_xaxes(categoryorder='array',
                                                    categoryarray=['retail', 'institutions', 'informed_traders'],
                                                    gridcolor='black')
        heatmap_pnl_tau_moneyness_plot.update_yaxes(categoryorder='array',
                                                    categoryarray=[f'{qth}<sup>{SUPERSCRIPT[qth]}</sup>' for
                                                                   qth in range(1, num_quantile + 1)],
                                                    gridcolor='black')
        heatmap_sharpe_tau_moneyness_plot.update_xaxes(categoryorder='array',
                                                       categoryarray=['retail', 'institutions', 'informed_traders'],
                                                       gridcolor='black')
        heatmap_sharpe_tau_moneyness_plot.update_yaxes(categoryorder='array',
                                                       categoryarray=[f'{qth}<sup>{SUPERSCRIPT[qth]}</sup>' for
                                                                      qth in range(1, num_quantile + 1)],
                                                       gridcolor='black')

        heatmap_pnl_quantile_plot.update_layout(title=dict(font=dict(size=30),
                                                           text=f'Annualised P&L - IV and exposure '
                                                                f'quantiles: {market.title()}'))
        heatmap_sharpe_quantile_plot.update_layout(title=dict(font=dict(size=30),
                                                              text=f'Annualised Sharpe - IV and exposure '
                                                                   f'quantiles: {market.title()}'))
        heatmap_pnl_tau_moneyness_plot.update_layout(title=dict(font=dict(size=30),
                                                                text=f'Annualised P&L per trade - IV and '
                                                                     f'market participants: {market.title()}'),
                                                     plot_bgcolor='black')
        heatmap_sharpe_tau_moneyness_plot.update_layout(title=dict(font=dict(size=30),
                                                                   text=f'Annualised Sharpe - IV and '
                                                                        f'market participants: {market.title()}'),
                                                        plot_bgcolor='black')
        heatmap_pnl_quantile_plot.show()
        heatmap_sharpe_quantile_plot.show()
        heatmap_pnl_tau_moneyness_plot.show()
        heatmap_sharpe_tau_moneyness_plot.show()
    ####################################################################################################################
    ## Top performing portfolio per exposure
    ####################################################################################################################
    top_performing_strategy_per_exp_agg_PnL = \
        agg_exposure_PnL.loc[agg_exposure_PnL.groupby([pd.Grouper(key='markets'),
                                                       pd.Grouper(key='symbol'),
                                                       pd.Grouper(key='exp')])['PnL'].transform('max') ==
                             agg_exposure_PnL['PnL']]
    top_performing_strategy_per_exp_agg_sharpe = \
        agg_exposure_sharpe.loc[agg_exposure_sharpe.groupby([pd.Grouper(key='markets'),
                                                             pd.Grouper(key='symbol'),
                                                             pd.Grouper(key='exp')])['PnL'].transform('max') ==
                                agg_exposure_sharpe['PnL']]
    top_performing_strategy_per_moneyness_mat_exp_PnL = \
        moneyness_mat_exp_PnL.loc[moneyness_mat_exp_PnL.groupby([pd.Grouper(key='markets'),
                                                                 pd.Grouper(level='symbol'),
                                                                 pd.Grouper(level='exp')])['PnL'].transform('max') ==
                                  moneyness_mat_exp_PnL['PnL']]
    top_performing_strategy_per_moneyness_mat_exp_sharpe = \
        moneyness_mat_exp_sharpe.loc[moneyness_mat_exp_sharpe.groupby(
            [pd.Grouper(key='market_participants'),
             pd.Grouper(key='symbol'),
             pd.Grouper(key='exp')])['PnL'].transform('max') == moneyness_mat_exp_sharpe['PnL']]
    overall_agg_exposure_report = \
        pd.melt(overall_agg_exposure_report.reset_index(),
                id_vars=['symbol', 'mark_iv', 'exp_quantile', 'timestamp', 'exp', 'levels'],
                var_name='markets', value_name='PnL')
    overall_moneyness_mat_exp_report = \
        pd.melt(overall_moneyness_mat_exp_report.reset_index(),
                id_vars=['symbol', 'market_participants', 'mark_iv', 'timestamp', 'exp'],
                var_name='markets', value_name='PnL')
    tmp = \
        pd.concat([overall_agg_exposure_report.query(f'exp == "{row["exp"]}" & '
                                                     f'mark_iv == "{row["mark_iv"]}" & '
                                                     f'exp_quantile == "{row["exp_quantile"]}" &'
                                                     f'symbol == "{row["symbol"]}" & '
                                                     f'markets == "{row["markets"]}"')
                   for _, row in top_performing_strategy_per_exp_agg_PnL.iterrows()])
    tmp['exp'] = tmp['exp'].str.upper()
    tmp.reset_index(inplace=True)
    tmp['portfolio'] = [f'{row["exp"]}<sup>({row["mark_iv"]},{row["exp_quantile"]})</sup>'
                        for _, row in tmp[['exp', 'exp_quantile', 'mark_iv']].iterrows()]
    tmp.drop(['exp', 'mark_iv', 'exp_quantile', 'exp', 'levels'], axis=1, inplace=True)
    tmp = pd.pivot(tmp, index=['symbol', 'markets', 'timestamp'], columns='portfolio', values='PnL')
    first_day = pd.DataFrame(index=pd.MultiIndex.from_product([symbols, ['spot', 'options'], pd.date_range(
        start=tmp.index.get_level_values(2).unique()[0].date()-relativedelta(days=1),
        end=tmp.index.get_level_values(2).unique()[0].date(), inclusive='left', tz=pytz.UTC)],
                                                              names=['symbol', 'markets', 'timestamp']),
                             columns=tmp.columns, data=np.nan)
    tmp = pd.concat([tmp, first_day]).sort_index(axis=0, level=[0, 1]).fillna(0)
    tmp = tmp.groupby(by=[pd.Grouper(level='symbol'),
                          pd.Grouper(level='markets')]).cumsum().replace(0, np.nan).dropna(how='all', axis=1)
    tmp.loc[tmp.index.get_level_values(2) == datetime(2021, 1, 1, 0, 0, 0, tzinfo=pytz.UTC), :] = 0
    tmp = tmp.stack().reset_index().rename(columns={0: 'PnL'})
    fig = px.line(tmp, y='PnL', x='timestamp', color='portfolio', facet_row='symbol', facet_col='markets',
                  title='Best performing annualised implied volatility and exposure level quantiles per type of exp',
                  facet_col_spacing=.1)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].title()}),
                            selector=lambda x: 'markets' in x['text'])
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1]}),
                            selector=lambda x: 'symbol' in x['text'])
    fig.update_xaxes(tickangle=45)
    fig.for_each_xaxis(lambda x: x.update({'title': ''}))
    fig.show()
    ####################################################################################################################
    ## Top performing portfolio per moneyness and maturity
    ####################################################################################################################
    tmp = pd.concat([overall_moneyness_mat_exp_report.query(f'exp == "{row["exp"]}" & '
                                                            f'market_participants == "{row["market_participants"]}" & '
                                                            f'mark_iv == "{row["mark_iv"]}" & '
                                                            f'symbol == "{row["symbol"]}" & '
                                                            f'markets == "{row["markets"]}"')
                     for _, row in top_performing_strategy_per_moneyness_mat_exp_PnL.reset_index().iterrows()])
    tmp['exp'] = tmp['exp'].str.upper()
    tmp['portfolio'] = [f'{row["exp"]}<sup>({row["market_participants"]},{row["mark_iv"]})</sup>'
                        for _, row in tmp.reset_index()[['exp', 'market_participants', 'mark_iv']].iterrows()]
    tmp.reset_index(inplace=True)
    tmp.drop(['exp', 'market_participants', 'mark_iv'], axis=1, inplace=True)# ['volume', 'exp']
    tmp = pd.pivot(tmp, index=['symbol', 'markets', 'timestamp'], columns='portfolio', values='PnL')
    first_day = pd.DataFrame(index=pd.MultiIndex.from_product([symbols, ['spot', 'options'], pd.date_range(
        start=tmp.index.get_level_values(2).unique()[0].date() - relativedelta(days=1),
        end=tmp.index.get_level_values(2).unique()[0].date(), inclusive='left', tz=pytz.UTC)],
                                                              names=['symbol', 'markets', 'timestamp']),
                             columns=tmp.columns, data=np.nan)
    tmp = pd.concat([tmp, first_day]).sort_index(axis=0, level=[0, 1]).fillna(0)
    tmp = tmp.groupby(by=[pd.Grouper(level='symbol'),
                          pd.Grouper(level='markets')]).cumsum().replace(0, np.nan).dropna(how='all', axis=1)
    tmp = tmp.stack().reset_index().rename(columns={0: 'PnL'})
    fig = px.line(tmp, y='PnL', x='timestamp', color='portfolio', facet_row='symbol', facet_col='markets',
                  title='Best performing portfolios annualised implied volatility and market participants per markets',
                  facet_col_spacing=.1)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1].title()}),
                            selector=lambda x: 'markets' in x['text'])
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1]}),
                            selector=lambda x: 'symbol' in x['text'])
    fig.update_xaxes(tickangle=45)
    fig.for_each_xaxis(lambda x: x.update({'title': ''}))
    fig.show()
    pdb.set_trace()