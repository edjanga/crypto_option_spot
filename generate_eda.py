import concurrent.futures
import pandas as pd
import numpy as np
import glob
import os


class EDA:
    # _figures_dir = os.path.abspath(__file__).replace('/executables/maker.py', '/.')
    files = [file for file in glob.glob('./data_centre/options/*-*-*')]

    def __init__(self, test: bool = True):
        self.data = pd.concat([pd.read_parquet(option_chain) for option_chain in
                               EDA.files[:{False: 5, True: len(EDA.files)}[test]]])
        self.data.reset_index(inplace=True)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], utc=True)
        self.data['time2mat'] = (self.data['expiration']-self.data['timestamp'])/(pd.to_timedelta('1D')*365)
        self.data['moneyness'] = \
            self.data['strike_price'].div(self.data['underlying_price'].mul(np.exp(-self.data['time2mat'])))

    def summary(self) -> None:
        info = dict()
        data = \
            self.data[['option_tag', 'timestamp', 'strike_price', 'underlying_price', 'time2mat', 'type', 'moneyness',
                       'mark_iv']]
        data = data.assign(symbol=[symbol[0] for symbol in data.option_tag.str.split('-')])
        data.set_index(['option_tag', 'timestamp'], inplace=True)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda x:x, x=symbol) for symbol in data.symbol.unique()]
            for future in concurrent.futures.as_completed(futures):
                symbol = future.result()
                print(f'[Summary statistics]: Process to generate summary statistics for {symbol} has just started.')
                for option_type in data.query(f'symbol == "{symbol}"').type.unique():
                    info_symbol = dict()
                    tmp_data = \
                        data.query(f'type == "{option_type}" & symbol == "{symbol}"').drop(['type', 'symbol'],
                                                                                           axis=1).droplevel(axis=0,
                                                                                                             level=0)
                    for feature in ['time2mat', 'strike_price', 'moneyness', 'mark_iv']:
                        if feature == 'strike_price':
                            info_symbol[(symbol, option_type, feature)] = tmp_data.resample('1D')[[feature]].apply(
                                lambda x: x.value_counts()).agg(['mean', 'max', 'min']).round(4)
                        elif feature == 'total_obs':
                            info_symbol[(symbol, option_type, feature)] = tmp_data.shape[0]
                        elif feature == 'trading_days':
                            info_symbol[(symbol, option_type, 'trading_days')] = len(tmp_data.resample('1D').groups)
                        elif feature == 'moneyness':
                            moneyness_data = tmp_data[['moneyness']].rename(columns={'moneyness': 'values'}).round(4)
                            moneyness_data = \
                                moneyness_data.assign(
                                    moneyness=
                                    pd.cut(moneyness_data['values'],
                                           bins={'call': [-1_000, .8, .99, 1.01, 1.2, 1_000],
                                                 'put': [-1_000, .6, .8, .99, 1.01, 1_000]}[option_type],
                                           include_lowest=True,
                                           labels={'call': ['DITM', 'ITM', 'ATM', 'OTM', 'DOTM'],
                                                   'put': ['VDOTM', 'DOTM', 'OTM', 'ATM', 'ITM']}[option_type]))
                            moneyness_data = pd.concat([moneyness_data['values'].agg(['mean', 'max', 'min']),
                                                        moneyness_data.groupby('moneyness', observed=True).size()])
                            moneyness_data = \
                                pd.DataFrame(
                                    index=pd.MultiIndex.from_product([moneyness_data.index, ['Log Moneyness']]),
                                    data=moneyness_data.values
                                )
                        else:
                            info_symbol[(symbol, option_type, feature)] = tmp_data[feature].agg(['mean', 'max', 'min'])
                    tmp_data_option_type = pd.DataFrame(info_symbol).stack(future_stack=True)
                    tmp_data_option_type.loc[('', 'Trading days'), :] = len(tmp_data.resample('1D').groups)
                    tmp_data_option_type.loc[('', 'Total observations'), :] = tmp_data.shape[0]
                    moneyness_data.columns = tmp_data_option_type.columns
                    tmp_data_option_type = pd.concat([tmp_data_option_type, moneyness_data])
                    info[(symbol, option_type)] = tmp_data_option_type
                info = pd.concat(info.values(), axis=1).swaplevel(i=1, j=0).sort_index(level=0)
                print(f'[Summary statistics]: Process to generate summary statistics for {symbol} is now completed.')
        info.reset_index(inplace=True)
        info['level_0'] = info['level_0'].str.replace('strike_price', 'Strikes per day')
        info['level_0'] = info['level_0'].str.replace('time2mat', 'Time to expiration')
        info['level_0'] = info['level_0'].str.replace('mark_iv', 'IV')
        info.set_index(['level_0', 'level_1'], inplace=True)
        info.fillna('', inplace=True)
        info.index.names = [None, None]
        print(info.to_latex())


if __name__ == '__main__':
    eda_obj = EDA()
    eda_obj.summary()
