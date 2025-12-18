import itertools
import pdb
from pathlib import Path
import pandas as pd
import numpy as np
from constants import MONEYNESS_BINS, LABELS, SYMBOLS
import argparse
import polars as pl
from scipy.stats.mstats import winsorize


class EDA:

    def __init__(self, target_path: str):
        self._target_path = target_path

    def summary(self, options: bool = True) -> None:
        summary_eda = dict()
        if options:
            moneyness_eda, iv_eda, strikes_eda, time2mat_eda, trading_days, n_obs_eda = \
                dict(), dict(), dict(), dict(), dict(), dict()
            for symbol, year in itertools.product(SYMBOLS, list(range(2023, 2026))):
                paths = list(Path(f'{self._target_path}/{symbol}/{year}/options').glob('*-*-*'))
                if len(paths) == 0:
                    continue
                chunksize = len(paths)//13
                for chunk in list(range(0, chunksize, len(paths)+1)):
                    option_chain = pl.scan_parquet(paths[chunk:chunk+chunksize])
                    option_chain = \
                        option_chain.select(['symbol', 'underlying_name', 'timestamp', 'type', 'strike_price',
                                             'expiration', 'moneyness', 'mid_price', 'mark_iv'])
                    option_chain = option_chain.with_columns(
                        ((pl.col('expiration')-pl.col('timestamp')).dt.total_days()/360).alias('tau'))
                    option_chain = \
                        option_chain.with_columns(pl.when(pl.col('type') == 'call').then(pl.col('moneyness').cut(
                        breaks=MONEYNESS_BINS, labels=LABELS)).when(pl.col('type') == 'put').then(pl.col('moneyness')
                        .cut(breaks=MONEYNESS_BINS, labels=LABELS[::-1])).otherwise(None).alias('moneyness_label'))
                    ####################################################################################################
                    ## MONEYNESS
                    ####################################################################################################
                    moneyness_label = option_chain.select(['underlying_name', 'type', 'moneyness_label', 'moneyness'])
                    moneyness_label = \
                        moneyness_label.sort(by=[pl.col('underlying_name'), pl.col('type'), pl.col('moneyness_label')])
                    moneyness = moneyness_label.group_by(['underlying_name', 'type', 'moneyness_label']).agg(
                        pl.len().alias(''), pl.col('moneyness').min().alias('min'),
                        pl.col('moneyness').mean().alias('mean'), pl.col('moneyness').max().alias('max')
                    ).collect().to_pandas()
                    tmp_moneyness_label = pd.pivot(moneyness.iloc[:, :4].dropna(),
                                                   columns=['underlying_name', 'type'], index='moneyness_label',
                                                   values='')
                    tmp_moneyness_label = tmp_moneyness_label.loc[['DOTM', 'OTM', 'ATM', 'ITM', 'DITM'], :]
                    tmp_moneyness_values = \
                        moneyness.drop(['', 'moneyness_label'], axis=1).groupby(['underlying_name', 'type'])
                    tmp_moneyness_values = \
                        tmp_moneyness_values.agg({'min': 'min', 'mean': 'mean', 'max': 'max'}).transpose()
                    tmp_moneyness_values = tmp_moneyness_values.loc[['max', 'mean', 'min'], :]
                    tmp_moneyness = pd.concat([tmp_moneyness_label, tmp_moneyness_values])
                    tmp_moneyness.index = pd.MultiIndex.from_product([['log m'], ['DOTM', 'OTM', 'ATM', 'ITM', 'DITM',
                                                                                  'max', 'mean', 'min']])
                    if moneyness_eda.get(symbol) is None:
                        moneyness_eda[symbol] = tmp_moneyness
                    else:
                        moneyness_eda[symbol].iloc[:-3, :] = \
                            moneyness_eda[symbol].iloc[:-3, :].add(tmp_moneyness.iloc[:-3, :])
                        ############################################### CALL ###########################################
                        moneyness_eda[symbol].iloc[-3, 0] = \
                            tmp_moneyness.iloc[-3, 0] if tmp_moneyness.iloc[-3, 0] > moneyness_eda[symbol].iloc[-3, 0] \
                                else moneyness_eda[symbol].iloc[-3, 0]
                        moneyness_eda[symbol].iloc[-2, 0] = \
                            .5*(tmp_moneyness.iloc[-2, 0]+moneyness_eda[symbol].iloc[-2, 0])
                        moneyness_eda[symbol].iloc[-3, 0] = \
                            tmp_moneyness.iloc[-1, 0] if tmp_moneyness.iloc[-1, 0] < moneyness_eda[symbol].iloc[-1, 0] \
                                else moneyness_eda[symbol].iloc[-1, 0]
                        ############################################### PUT ############################################
                        moneyness_eda[symbol].iloc[-3, 1] = \
                            tmp_moneyness.iloc[-3, 1] if tmp_moneyness.iloc[-3, 1] > moneyness_eda[symbol].iloc[-3, 1] \
                                else moneyness_eda[symbol].iloc[-3, 1]
                        moneyness_eda[symbol].iloc[-2, 1] = .5 * (
                                    tmp_moneyness.iloc[-2, 1] + moneyness_eda[symbol].iloc[-2, 1])
                        moneyness_eda[symbol].iloc[-3, 1] = \
                            tmp_moneyness.iloc[-1, 1] if tmp_moneyness.iloc[-1, 1] < moneyness_eda[symbol].iloc[-1, 1] \
                                else moneyness_eda[symbol].iloc[-1, 1]
                    ####################################################################################################
                    ## IV
                    ####################################################################################################
                    iv = option_chain.select(['underlying_name', 'type', 'mark_iv'])
                    iv = iv.sort(by=[pl.col('underlying_name'), pl.col('type')])
                    iv = iv.group_by(['underlying_name', 'type']).agg(pl.col('mark_iv').min().alias('min'),
                                                                      pl.col('mark_iv').mean().alias('mean'),
                                                                      pl.col('mark_iv').max().alias('max')).collect()
                    iv = \
                        iv.to_pandas().set_index(['underlying_name', 'type']).transpose().loc[['max', 'mean', 'min'], :]
                    iv.index = pd.MultiIndex.from_product([['IV'], ['max', 'mean', 'min']])
                    if iv_eda.get(symbol) is None:
                        iv_eda[symbol] = iv
                    else:
                        ############################################### CALL ###########################################
                        iv_eda[symbol].iloc[0, 0] = \
                            iv.iloc[0, 0] if iv.iloc[0, 0] > iv_eda[symbol].iloc[0, 0] else iv_eda[symbol].iloc[0, 0]
                        iv_eda[symbol].iloc[1, 0] = .5*(iv.iloc[1, 0]+iv_eda[symbol].iloc[1, 0])
                        iv_eda[symbol].iloc[2, 0] = \
                            iv.iloc[2, 0] if iv.iloc[2, 0] < iv_eda[symbol].iloc[2, 0] else iv_eda[symbol].iloc[2, 0]
                        ############################################### PUT ############################################
                        iv_eda[symbol].iloc[0, 1] = \
                            iv.iloc[0, 1] if iv.iloc[0, 1] > iv_eda[symbol].iloc[0, 1] else iv_eda[symbol].iloc[0, 1]
                        iv_eda[symbol].iloc[1, 1] = .5 * (iv.iloc[1, 1] + iv_eda[symbol].iloc[1, 1])
                        iv_eda[symbol].iloc[2, 1] = \
                            iv.iloc[2, 1] if iv.iloc[2, 1] < iv_eda[symbol].iloc[2, 1] else iv_eda[symbol].iloc[2, 1]
                    ####################################################################################################
                    ## Strikes
                    ####################################################################################################
                    strikes = option_chain.select(['underlying_name', 'type', 'strike_price'])
                    strikes = strikes.sort(by=[pl.col('underlying_name'), pl.col('type')])
                    strikes = strikes.group_by(['underlying_name', 'type'])
                    strikes = strikes.agg(pl.col('strike_price').n_unique().alias('strike_count')).collect().to_pandas()
                    strikes = strikes.set_index(['underlying_name', 'type']).transpose()
                    strikes.index = pd.MultiIndex.from_product([['Total strikes'], ['']])
                    strikes_eda[symbol] = \
                        strikes if strikes_eda.get(symbol) is None else strikes_eda[symbol].add(strikes)
                    ####################################################################################################
                    ## Time to expiration
                    ####################################################################################################
                    time2mat = option_chain.select(['underlying_name', 'type', 'tau'])
                    time2mat = time2mat.sort(by=[pl.col('underlying_name'), pl.col('type')])
                    time2mat = time2mat.group_by(['underlying_name', 'type']).agg(pl.col('tau').min().alias('min'),
                        pl.col('tau').mean().alias('mean'), pl.col('tau').max().alias('max')).collect().to_pandas()
                    time2mat = \
                        time2mat.set_index(['underlying_name', 'type']).transpose().loc[['max', 'mean', 'min'], :]
                    time2mat.index = pd.MultiIndex.from_product([['Time to maturity'], ['max', 'mean', 'min']])
                    if time2mat_eda.get(symbol) is None:
                        time2mat_eda[symbol] = time2mat
                    else:
                        ############################################### CALL ###########################################
                        time2mat_eda[symbol].iloc[0, 0] = \
                            time2mat.iloc[0, 0] if time2mat.iloc[0, 0] > time2mat_eda[symbol].iloc[0, 0] \
                                else time2mat_eda[symbol].iloc[0, 0]
                        time2mat_eda[symbol].iloc[1, 0] = .5 * (time2mat_eda[symbol].iloc[1, 0] + time2mat.iloc[1, 0])
                        time2mat_eda[symbol].iloc[2, 0] = \
                            time2mat[symbol].iloc[2, 0] if time2mat.iloc[2, 0] < time2mat_eda[symbol].iloc[2, 0] \
                                else time2mat_eda[symbol].iloc[2, 0]
                        ############################################### PUT ############################################
                        time2mat_eda[symbol].iloc[0, 1] = \
                            time2mat.iloc[0, 1] if time2mat.iloc[0, 1] > time2mat_eda[symbol].iloc[0, 1] else \
                                time2mat_eda[symbol].iloc[0, 1]
                        time2mat_eda[symbol].iloc[1, 1] = .5 * (time2mat.iloc[1, 1] + time2mat_eda[symbol].iloc[1, 1])
                        time2mat_eda[symbol].iloc[2, 1] = \
                            time2mat.iloc[2, 1] if time2mat.iloc[2, 1] < time2mat_eda[symbol].iloc[2, 1] else \
                                time2mat_eda[symbol].iloc[2, 1]
                    ####################################################################################################
                    ## Trading days + Total observations
                    ####################################################################################################
                    tmp_symbol = \
                        option_chain.filter(pl.col('underlying_name') == symbol).select(['underlying_name', 'type',
                                                                                         'timestamp'])
                    tmp_symbol = tmp_symbol.with_columns(pl.col('timestamp').dt.date().alias('date'))
                    tmp_symbol = tmp_symbol.sort(by=[pl.col('underlying_name'), pl.col('type')])
                    tmp_trading_days = \
                        tmp_symbol.group_by(['underlying_name',
                                             'type']).agg(pl.col('date').n_unique().alias('Trading days')).collect()
                    tmp_n_obs = \
                        tmp_symbol.group_by(['underlying_name',
                                             'type']).agg(pl.col('date').len().alias('Total observations')).collect()
                    tmp_trading_days = tmp_trading_days.to_pandas().set_index(['underlying_name', 'type']).transpose()
                    tmp_trading_days.index = pd.MultiIndex.from_product([['Trading days'], ['']])
                    tmp_n_obs = tmp_n_obs.to_pandas().set_index(['underlying_name', 'type']).transpose()
                    tmp_n_obs.index = pd.MultiIndex.from_product([['Total observations'], ['']])
                    trading_days[symbol] = tmp_trading_days if trading_days.get(symbol) is None else \
                        trading_days[symbol].add(tmp_trading_days)
                    n_obs_eda[symbol] = tmp_n_obs if n_obs_eda.get(symbol) is None else n_obs_eda[symbol].add(tmp_n_obs)
            summary_eda = pd.concat([pd.concat(moneyness_eda.values(), axis=1), pd.concat(iv_eda.values(), axis=1),
                                     pd.concat(strikes_eda.values(), axis=1), pd.concat(time2mat_eda.values(), axis=1),
                                     pd.concat(trading_days.values(), axis=1), pd.concat(n_obs_eda.values(), axis=1)])
        else:
            underlying_price = pl.scan_parquet(f'{self._target_path}/*/*/spot/*-*-*')
            underlying_price = underlying_price.select(['symbol', 'timestamp', 'mid_price[0]'])
            underlying_price = \
                underlying_price.collect().pivot(index='timestamp', values='mid_price[0]', on='symbol').to_pandas()
            underlying_price = underlying_price.set_index('timestamp').sort_index()
            returns = np.log(underlying_price.div(underlying_price.shift())).resample('1D').sum()
            returns = returns.apply(lambda x: winsorize(x, [.005, .005]))
            rv = (np.log(underlying_price.div(underlying_price.shift()))).apply(lambda x: winsorize(x, [.005, .005]))
            rv = rv.pow(2).resample('1D').sum()
            freq = returns.index[1] - returns.index[0]
            rv = rv.mul((pd.to_timedelta(360, unit='d')//freq) ** .5)
            summary_eda['Returns'] = returns.agg(['max', 'mean', 'min'])
            summary_eda['Annualised RV'] = rv.agg(['max', 'mean', 'min'])
            summary_eda = pd.concat(summary_eda)[SYMBOLS]
        summary_eda = summary_eda.style.format(decimal='.', thousands=',',
                                               precision={True: 2, False: 6}[options])
        with open(f'{self._target_path}'
                  f'/crypto_option_spot/summary{dict([(True, "_options"), (False, "")])[options]}.txt', 'w') as \
                summary_file:
            summary_file.writelines(summary_eda.to_latex())
            print(f'[EXPORT]: '
                  f'Summary EDA table for {dict([(True, "_options"), (False, "")])[options]} has been exported.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all performance tables.')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    args = parser.parse_args()
    eda_obj = EDA(target_path=args.target_path)
    eda_obj.summary(options=False)
    eda_obj.summary()
