import plotly.express as px
import sqlite3
import numpy as np
import pandas as pd
import pdb

if __name__ == '__main__':
    db_connect = sqlite3.connect(database='./data_centre/database.db', check_same_thread=False)
    metric = pd.read_sql(sql='SELECT * FROM metric;', con=db_connect, index_col='index')
    metric['volatility_regime'] = metric['volatility_regime'].str.lower()
    metric.index = pd.to_datetime(metric.index)
    metric = \
        metric.groupby(by=[pd.Grouper(level=0, sort=True, freq='1D'), pd.Grouper(key='model'), pd.Grouper(key='symbol'),
                           pd.Grouper(key='target'), pd.Grouper(key='horizon'), pd.Grouper(key='options'),
                           pd.Grouper(key='freq')])[['mase', 'model_tag',
                                                     'volatility_regime', 'features']].agg(
            {'mase': 'mean', 'model_tag': 'last', 'volatility_regime': 'last', 'features': 'last'})
    metric.reset_index(inplace=True)
    metric['model_tag'] = metric['model_tag'].str.replace('LIGHTGBM', 'LightGBM')
    # metric = metric.loc[metric.options == 0, :]
    ####################################################################################################################
    ## Best performing models
    ####################################################################################################################
    ranking = metric.groupby(by=[pd.Grouper(key='model'), pd.Grouper(key='model_tag'), pd.Grouper(key='target'),
                                 pd.Grouper(key='horizon')])[['mase']].mean()
    ranking = ranking.reset_index().set_index('model_tag')
    ranking = \
        ranking.groupby(by=[pd.Grouper(key='model'), pd.Grouper(key='target'),
                            pd.Grouper(key='horizon')])[['mase']].agg(['min', lambda x: x.idxmin()])
    ranking = \
        ranking.droplevel(axis=1, level=0).rename(columns={'<lambda_0>': 'model_tag', 'min': 'mase'}).reset_index()
    ####################################################################################################################
    ## Data visualisation: Boxplot
    ####################################################################################################################
    fig = \
        px.box(data_frame=metric.loc[metric.model_tag.isin(ranking.model_tag), :].sort_values('model_tag'),
               x='model_tag', y='mase',
               facet_row='horizon', facet_col='target', facet_row_spacing=.25, facet_col_spacing=.1,
               category_orders=dict([('horizon', ['1D', '2D', '7D']), ('target', ['returns', 'RV'])]),
               title='MASE: Boxplot per models, time buckets, and forecasting horizons', color='model')
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1]}),
                            selector=lambda x: 'target' in x['text'])
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1]}),
                            selector=lambda x: 'horizon' in x['text'])
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].title()}),
                            selector=lambda x: 'returns' in x['text'])
    fig.update_yaxes(matches=None, showticklabels=True, title_text='MASE')
    fig.update_xaxes(matches=None, showticklabels=True, title='')
    fig.update_layout(showlegend=False)
    fig.show()
    ###################################################################################################################
    ## Data visualisation: Curves
    ####################################################################################################################
    timeseries = \
        metric.loc[metric.model_tag.isin(ranking.model_tag), :][['symbol', 'mase', 'model_tag', 'horizon', 'target',
                                                                 'index']].set_index('index')
    timeseries = timeseries.groupby(by=[pd.Grouper(key='model_tag'), pd.Grouper(key='horizon'),
                                        pd.Grouper(key='target'), pd.Grouper(level=0)])[['mase']].mean().reset_index()
    fig = \
        px.line(data_frame=timeseries, x='index', y='mase', facet_row='horizon', facet_col='target',
                facet_row_spacing=.15, category_orders=dict([('horizon', ['1D', '2D', '7D']),
                                                             ('target', ['returns', 'RV'])]),
                facet_col_spacing=.1, title='MASE per models, time buckets, and forecasting horizons',
                color='model_tag')
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].split('=')[-1]}),
                            selector=lambda x: ('horizon' in x['text']) | ('target' in x['text']))
    fig.for_each_annotation(lambda x: x.update({'text': x['text'].title()}),
                            selector=lambda x: 'returns' in x['text'])
    fig.update_xaxes(tickangle=45, title_text='')
    fig.update_yaxes(title_text='MASE', matches=None, showticklabels=True)
    fig.update_layout(legend=dict(title=''))
    fig.show()
    ####################################################################################################################
    ## Best performing models per volatility regime
    ####################################################################################################################
    ranking_vol_regime = metric.groupby(by=[pd.Grouper(key='volatility_regime'), pd.Grouper(key='model'),
                                            pd.Grouper(key='model_tag'), pd.Grouper(key='target'),
                                            pd.Grouper(key='horizon'), pd.Grouper(key='options')])[['mase']].mean()
    ranking_vol_regime = ranking_vol_regime.reset_index().set_index('model_tag')
    ranking_vol_regime = \
        ranking_vol_regime.groupby(
            by=[pd.Grouper(key='volatility_regime'), pd.Grouper(key='target'), pd.Grouper(key='horizon'),
                pd.Grouper(key='options')])[['mase']].agg(['min', lambda x: x.idxmin()])
    ranking_vol_regime = \
        ranking_vol_regime.droplevel(axis=1, level=0).rename(columns={'<lambda_0>': 'model_tag',
                                                                      'min': 'mase'}).reset_index()
    ####################################################################################################################
    ## Data visualisation: Best performing model per volatility regime (LaTeX table)
    ####################################################################################################################
    ranking_vol_regime.loc[:, 'model_mase'] = \
        [f'{row["model_tag"]}: {round(row["mase"], 4)}' if row["options"] == 0 else
         f'{row["model_tag"]}: '
         f'{round((ranking_vol_regime.iloc[idx].mase / ranking_vol_regime.iloc[idx - 1].mase - 1) * 100, 2)}\%'
         for idx, row in ranking_vol_regime.iterrows()]
    table_volatility_regime = \
        pd.pivot_table(ranking_vol_regime, columns=['volatility_regime', 'options'], index=['target', 'horizon'],
                       values='model_mase', aggfunc='first')
    table_volatility_regime = table_volatility_regime.reindex(columns=['low', 'normal', 'high'], level=0)
    table_volatility_regime = table_volatility_regime.reindex(index=['returns', 'RV'], level=0)
    table_volatility_regime = table_volatility_regime.droplevel(axis=1, level=1)
    print(table_volatility_regime.to_latex().replace('<sup>(', '$^{\\text{(').replace(')</sup>', ')}}$'))
    ####################################################################################################################
    ## Data visualisation: Most used features for each best performing model per volatility regime (LaTeX table)
    ####################################################################################################################
    # ranking_vol_regime = ranking_vol_regime.set_index('model_tag').groupby(
    #     by=[pd.Grouper(key='volatility_regime'),
    #         pd.Grouper(key='horizon'),
    #         pd.Grouper(key='target')])[['mase']].idxmin().rename(columns={'mase': 'model_tag'}).droplevel(axis=0,
    #                                                                                                       level=1)
    # ranking_vol_regime.reset_index(inplace=True)
    features = metric.loc[metric.model_tag.isin(ranking_vol_regime.model_tag), :]
    features = \
        pd.concat([features.loc[(features.volatility_regime == row.volatility_regime) &
                                (features.model_tag == row.model_tag) & (features.target == row.target)]
                   for _, row in ranking_vol_regime.iterrows()])
    features = \
        features.groupby(by=[pd.Grouper(key='volatility_regime'),
                             pd.Grouper(key='horizon'),
                             pd.Grouper(key='target')])[['features']].apply(
            lambda x: x.value_counts().index[0][0]).replace('', np.nan).dropna().reset_index()
    features = pd.pivot(features, index=['horizon', 'target'], columns='volatility_regime', values=0)
    print(features.to_latex())
    pdb.set_trace()