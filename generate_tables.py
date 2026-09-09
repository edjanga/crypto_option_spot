import datetime
import itertools
import os
import pdb
import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from constants import DATE_FORMAT, DAYS_CONVERTER, SYMBOLS, VOL_REGIMES, ATM_MONEYNESSES
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from scipy.stats.distributions import norm
from pandas.api.types import CategoricalDtype
import polars as pl
from itertools import product
import argparse
import concurrent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all performance tables.')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='./data_centre')
    parser.add_argument('--type', type=str, help='Type of table to generate', default='mase_reporting')
    args = parser.parse_args()
    print(args)
    ####################################################################################################################
    ## Order
    ####################################################################################################################
    pairs_order = CategoricalDtype(categories=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT'], ordered=True)
    d_order = CategoricalDtype(categories=['1d', '1w', '1m'], ordered=True)
    volatility_order = CategoricalDtype(categories=['Low', 'Normal', 'High'], ordered=True)
    market_order = CategoricalDtype(categories=[0, 1], ordered=True)
    ####################################################################################################################
    ## Model selection
    ####################################################################################################################
    mase = pd.read_csv(f'{args.target_path}/crypto_option_spot/metric.csv')
    mase = mase.set_index('D')
    model_ranking = mase.groupby(by=[pd.Grouper(key='target'),
                                     pd.Grouper(key='regression'),
                                     pd.Grouper(key='freq')])[['MASE']].mean()
    mase_comparison = \
        model_ranking.groupby(['target', 'regression', 'freq'])['MASE'].mean().groupby(level='target').idxmin()
    best_model_tags = model_ranking.groupby('target').idxmin().to_dict()['MASE']
    ####################################################################################################################
    ## MASE Reporting
    ####################################################################################################################
    if args.type == 'mase_reporting':
        for target in mase_comparison.keys():
            tmp = model_ranking.xs(target, level=0, axis=0, drop_level=True)
            tmp = tmp['MASE'].unstack(level='regression')
            tmp = tmp.reindex(['15min', '30min', '1h'])
            tmp.index = \
                [f'${freq[:2]}\\mathrm{{{freq[2:]}}}$' if 'min' in freq else f'${freq[0]}\\mathrm{{{freq[1]}}}$'
                 for freq in tmp.index]
            tmp.columns = ['LightGBM', 'SOFTS']
            mase_min = tmp.min().min()
            tmp = tmp.map(lambda x, y: f'$\\mathbf{{{x:.4}}}$' if x == y else x, y=mase_min)
            tmp = tmp.style.format(decimal='.', thousands=',', precision=4)
            with open(f'{args.target_path}/crypto_option_spot/mase_per_regression_freq_pair_{target}.txt', 'w') as \
                    mase_per_regression_pair:
                mase_per_regression_pair.writelines(tmp.to_latex())
        metric_per_target = dict()
        tmp = pd.read_csv(f'{args.target_path}/crypto_option_spot/metric.csv')
        for target, values in mase_comparison.items():
            metric_per_target[target] = \
                tmp.query('target == @values[0] and regression == @values[1] and freq == @values[-1]')
        mase_comparison = pd.concat(metric_per_target.values())
        for target, values in best_model_tags.items():
            regression = best_model_tags[target][1]
            freq = best_model_tags[target][-1]
            best_model = mase_comparison.query('regression == @regression and freq == @freq and target == @target')
            mase_global = \
                best_model.groupby(by=[pd.Grouper(key='D'),
                                       pd.Grouper(key='market')])[['MASE']].mean().reset_index()
            mase_global['D'] = mase_global['D'].transform(lambda x: f'<i>{x.lower()}</i>')
            mase_global['market'] = \
                np.where(mase_global['market'] == 1, '<i>X<sup>(S,O)</sup></i>', '<i>X<sup>S</sup></i>')
            mase_conditional = \
                best_model.groupby(by=[pd.Grouper(key='D'),
                                       pd.Grouper(key='volatility_regime'), pd.Grouper(key='market')])[['MASE']].mean()
            mase_conditional_copy = mase_conditional.copy()
            mase_conditional_copy = \
                mase_conditional.groupby(by=[pd.Grouper(level='D'),
                                             pd.Grouper(level='volatility_regime')],
                                         group_keys=False).apply(lambda x: 100 * x.pct_change()).round(4)
            mase_conditional = mase_conditional_copy.fillna(mase_conditional)
            mase_conditional = mase_conditional.unstack([1, -1])
            mase_conditional = mase_conditional.loc[:, [('MASE', 'Low', 0), ('MASE', 'Low', 1),
                                                        ('MASE', 'Normal', 0), ('MASE', 'Normal', 1),
                                                        ('MASE', 'High', 0), ('MASE', 'High', 1)]]
            mase_conditional.index = mase_conditional.index.str.lower()
            mase_conditional = mase_conditional.loc[d_order.categories, :]
            mase_conditional.loc[:, mase_conditional.columns.get_level_values(2) == 1] = \
                mase_conditional.loc[:, mase_conditional.columns.get_level_values(2) == 1].round(2)
            mase_conditional.columns = mase_conditional.columns.set_names('Market Regimes', level=0)
            mase_unconditional = best_model.groupby(by=[pd.Grouper(key='D'), pd.Grouper(key='market')])[['MASE']].mean()
            mase_unconditional_copy = mase_unconditional.copy()
            mase_unconditional = mase_unconditional.groupby(by=pd.Grouper(level='D'), group_keys=False).apply(
                lambda x: 100 * x.pct_change()).round(4)
            mase_unconditional = mase_unconditional.fillna(mase_unconditional_copy)
            mase_unconditional = mase_unconditional.unstack(-1)
            mase_unconditional.index = mase_unconditional.index.str.lower()
            mase_unconditional = mase_unconditional.loc[d_order.categories, :]
            mase_unconditional.columns = \
                pd.MultiIndex.from_product([['MASE'], [''], [0, 1]],
                                           names=['Market Regimes', 'volatility_regime', 'market'])
            mase_performance = pd.concat([mase_unconditional, mase_conditional], axis=1)
            mase_performance.columns.names = 3*['']
            mase_performance.index.name = ''
            mase_performance.index = [f'\\textit{{{v}}}' for v in mase_performance.index]
            mase_performance.columns = mase_performance.columns.set_levels(
                [f'\\textit{{{v.lower()}}}' if v else v for v in mase_performance.columns.levels[1]], level=1)
            option_features = pd.IndexSlice
            mase_performance.loc[:, option_features[:, :, 1]] = \
                mase_performance.loc[:, option_features[:, :, 1]].map(
                    lambda x: f'$\\mathbf{{{x:.2f}}}$\\%' if x < 0 else f'${x:.2f}$\\%')
            mase_performance.columns = mase_performance.columns.set_levels(
                ['$X^S$' if market == 0 else '$X^\{(S,O)}$' for market in mase_performance.columns.levels[2]],
                level=2)
            mase_performance = mase_performance.style.format(decimal='.', thousands=',', precision=4)
            with open(f'{args.target_path}/crypto_option_spot/mase_performance_{target}.txt', 'w') as \
                    mase_performance_file:
                mase_performance_file.writelines(mase_performance.to_latex())
    ####################################################################################################################
    ## Feature importance
    ####################################################################################################################
    if args.type == 'features_importance':
        feature_importance = pd.read_csv(f'{args.target_path}/crypto_option_spot/features_importance.csv')
        for target, values in best_model_tags.items():
            regression, freq = values[1:]
            target_feature_importance = feature_importance.query('(regression == @regression) and (freq == @freq)')
            target_feature_importance['D'] = target_feature_importance['D'].str.lower()
            target_feature_importance = target_feature_importance.drop(['freq', 'market'], axis=1)
            target_feature_importance = target_feature_importance.set_index('D')
            feature_importance_ranking = target_feature_importance.groupby(by=[pd.Grouper(key='target'),
                                                                               pd.Grouper(key='volatility_regime'),
                                                                               pd.Grouper(key='feature'),
                                                                               pd.Grouper(level='D')])[['MASE']].mean()
            feature_importance_mean = feature_importance_ranking.groupby(
                by=[pd.Grouper(level=0), pd.Grouper(level=1), pd.Grouper(level=-1)]).mean().rename(
                columns={'MASE': 'benchmark'})
            feature_importance_ranking = (
                feature_importance_ranking.join(feature_importance_mean, on=['target', 'volatility_regime', 'D']))
            feature_importance_ranking = \
                feature_importance_ranking.groupby(
                    by=[pd.Grouper(level='target'), pd.Grouper(level='volatility_regime'),
                        pd.Grouper(level='D')],
                    group_keys=False).apply(
                    lambda g: g.nlargest(5, 'MASE')).reset_index()
            feature_importance_ranking['MASE'] = feature_importance_ranking['MASE'].round(4)
            feature_importance_ranking = \
                feature_importance_ranking.sort_values('MASE', ascending=False).groupby(['target', 'D',
                                                                                         'volatility_regime'],
                                                                                        group_keys=False).apply(
                    lambda g: g.assign(rank=g['MASE'].rank(method='first', ascending=False).astype(int).astype(str)))
            feature_importance_ranking['feature'] = \
                feature_importance_ranking.feature.str.replace(r'_1([DWM])', lambda m: f"_1{m.group(1).lower()}",
                                                               regex=True)
            feature_importance_ranking = \
                feature_importance_ranking.assign(cell=lambda x: x['feature'] + ': ' + x['MASE'].round(4).astype(str))
            feature_importance_ranking = feature_importance_ranking.pivot(index=['target', 'D', 'volatility_regime'],
                                                                          columns='rank', values='cell')
            feature_importance_ranking = feature_importance_ranking.rename(columns=lambda c: f'Rank {c}')
            feature_importance_ranking = \
                feature_importance_ranking.reindex(pd.MultiIndex.from_product(
                    [['returns', 'RV'], ['1d', '1w', '1m'], ['Low', 'Normal', 'High']],
                    names=['target', 'D', 'volatility_regime']))
            feature_importance_ranking.columns.name = ''
            feature_importance_ranking.index.names = 3*['']
            feature_importance_ranking.index = feature_importance_ranking.index.set_levels(
                [f'\\textit{{{v}}}' for v in feature_importance_ranking.index.levels[1]], level=1
            ).set_levels(
                [f'\\textit{{{v.lower()}}}' for v in feature_importance_ranking.index.levels[2]], level=2
            )
            with open(f'{args.target_path}/crypto_option_spot/features_importance.txt', 'w') as \
                    features_importance_file:
                features_importance_file.writelines(feature_importance_ranking.to_latex())
    ####################################################################################################################
    ## SVIX
    ####################################################################################################################
    if args.type == 'svix_rv_vrp_effect':
        regression = mase_comparison['RV'][1]
        freq = mase_comparison['RV'][-1]
        chunks = pd.read_csv(f'{args.target_path}/crypto_option_spot/forecasts.csv', chunksize=1_000_000)
        data = \
            pd.concat(chunk.query('target == RV and regression == @regression and freq == @freq') for chunk in chunks)
        data = data.rename(columns={'y_hat': 'forecast', 'y': 'values'})
        data = data.set_index('timestamp')
        data.index = pd.to_datetime(data.index, utc=True)
        data = pd.pivot_table(data=data, columns=['underlying_name', 'target', 'D'], values=['values', 'forecast'],
                              index='timestamp')
        svix = pd.read_csv(f'{args.target_path}/crypto_option_spot/svix.csv', index_col='index')
        svix['svix'] = svix['svix']/100
        svix['timestamp'] = pd.to_datetime(svix['timestamp'], utc=True)
        svix = svix.set_index(['aggregated', 'underlying_name', 'D', 'timestamp'])
        svix_agg = \
            pd.concat([svix.xs(1, axis=0, level=0,
                               drop_level=False).droplevel(axis=0, level=1).assign(underlying_name=underlying_name)
                       for underlying_name in SYMBOLS]).set_index('underlying_name', append=True)
        svix_agg = svix_agg.swaplevel(-1, 1).swaplevel(-1, 2)
        svix_non_agg = svix.xs(0, axis=0, level=0, drop_level=False)
        svix = pd.concat([svix_agg, svix_non_agg]).reset_index()
        ################################################################################################################
        ## VRP
        ################################################################################################################
        rv = data.xs('forecast', axis=1, level=0).xs('RV', axis=1, level=1)
        rv = rv.resample(best_model_tags[-1]).sum().replace(0, np.nan)
        rv.columns = rv.columns.set_levels(rv.columns.levels[-1].str.lower(), level=-1).dropna()
        if os.path.exists(f'{args.target_path}/crypto_option_spot/vrp.csv'):
            vrp = \
                pd.merge(svix, rv.unstack().rename('rv').dropna().reset_index(),
                         on=['underlying_name', 'D', 'timestamp'], how='right')
            vrp['vrp'] = vrp['svix'] - vrp['rv']
            vrp = vrp.set_index(['aggregated', 'underlying_name', 'D', 'timestamp'])
        else:
            svix['timestamp'] = pd.to_datetime(svix['timestamp'], utc=pytz.UTC)
            svix = svix.set_index(['aggregated', 'timestamp'])
            vrp = dict()
            for aggregated, underlying_name in list(product([False, True], list(pairs_order.categories))):
                tmp_svix = svix.xs(int(aggregated), level=0, axis=0)
                if aggregated:
                    if 'agg_svix' not in vars():
                        agg_svix = tmp_svix.drop('underlying_name', axis=1)
                        agg_svix = pd.pivot_table(agg_svix.reset_index(), columns='D', values='svix', index='timestamp')
                        agg_svix.index = pd.to_datetime(agg_svix.index, utc=pytz.UTC)
                    vrp[(int(aggregated), underlying_name)] = \
                        agg_svix.reindex(rv.index).subtract(rv[underlying_name].values).dropna()
                else:
                    tmp_svix = pd.pivot_table(tmp_svix.reset_index(), columns=['underlying_name', 'D'], values='svix',
                                              index='timestamp')
                    tmp_svix.index = pd.to_datetime(tmp_svix.index, utc=pytz.UTC)
                    vrp[(int(aggregated), underlying_name)] = \
                        tmp_svix.xs(underlying_name, axis=1,
                                    level=0).reindex(rv.index).subtract(rv[underlying_name].values)
                vrp[(int(aggregated), underlying_name)] = \
                    vrp[(int(aggregated), underlying_name)].dropna(axis=1, how='all').dropna()
                vrp[(int(aggregated), underlying_name)] = \
                    pd.DataFrame(vrp[(int(aggregated), underlying_name)].unstack()).rename(columns={0: 'vrp'})
                vrp[(int(aggregated), underlying_name)] = \
                    vrp[(int(aggregated), underlying_name)].assign(underlying_name=underlying_name,
                                                                   aggregated=int(aggregated))
            vrp_0 = pd.concat([vrp[(0, symbol)] for symbol in pairs_order.categories])
            vrp_1 = pd.concat([vrp[(1, symbol)] for symbol in pairs_order.categories])
            vrp = pd.concat([vrp_0, vrp_1])
            vrp.to_csv(f'{args.target_path}/crypto_option_spot/vrp.csv')
