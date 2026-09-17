import pdb

import plotly.express as px
import pandas as pd
import argparse
from constants import TITLE_FONT_SIZE, LABEL_AXIS_FONT_SIZE, HEIGHT, WIDTH, SYMBOLS, D_HORIZONS
from pandas.api.types import CategoricalDtype
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from itertools import product
from pathlib import Path

########################################################################################################################
## Order
########################################################################################################################
pairs_order = CategoricalDtype(categories=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT'], ordered=True)
d_order = CategoricalDtype(categories=['1d', '1w', '1m'], ordered=True)
volatility_order = CategoricalDtype(categories=['Low', 'Normal', 'High'], ordered=True)
market_order = CategoricalDtype(categories=[0, 1], ordered=True)


def plot_svix(crypto_svix: pd.DataFrame, aggregated: bool, args) -> None:

    fig = make_subplots(rows=1 if aggregated else len(SYMBOLS), cols=1, vertical_spacing=.1, shared_xaxes=False,
                        row_titles=None if aggregated else SYMBOLS,
                        row_heights=None if aggregated else [.18] * len(SYMBOLS))
    tmp_crypto_svix = crypto_svix.xs(int(aggregated), axis=0, level=0, drop_level=True)

    def add_backwardation_structure_rectangles(fig, tmp_svix, row=1):
        tmp_svix = tmp_svix.loc[:, d_order.categories]
        svix_1d = tmp_svix.iloc[:, 0]
        svix_1w = tmp_svix.iloc[:, 1]
        svix_1m = tmp_svix.iloc[:, 2]
        valid = (svix_1d < svix_1w) & (svix_1w < svix_1m)
        broken = ~valid
        if not broken.any():
            return
        groups = broken.ne(broken.shift()).cumsum()
        xref = 'x' if row == 1 else f'x{row}'
        yref = 'y domain' if row == 1 else f'y{row} domain'
        for _, group in broken.groupby(groups):
            if not group.iloc[0]:
                continue
            start = group.index[0]
            end = group.index[-1]
            idx = tmp_svix.index
            end_pos = idx.get_loc(end)
            if end_pos < len(idx) - 1:
                end = idx[end_pos + 1]
            if isinstance(start, pd.Timestamp):
                if start.tzinfo is not None:
                    start = start.tz_convert(None)
                start = start.to_pydatetime()
            if isinstance(end, pd.Timestamp):
                if end.tzinfo is not None:
                    end = end.tz_convert(None)
                end = end.to_pydatetime()
            fig.add_shape(type='rect', x0=start, x1=end, xref=xref, y0=0, y1=1, yref=yref, fillcolor='yellow',
                          opacity=.35, line=dict(width=0), layer='below')
    if aggregated:
        tmp_crypto_svix = tmp_crypto_svix.drop('underlying_name', axis=1)
        aggregated_svix = []
        for D in d_order.categories:
            tmp_svix = tmp_crypto_svix.xs(D, level=0, axis=0, drop_level=True)
            tmp_svix = tmp_svix.resample('7D').mean()
            aggregated_svix.append(tmp_svix['svix'].rename(D))
        aggregated_svix = pd.concat(aggregated_svix, axis=1)
        aggregated_svix = aggregated_svix.loc[:, d_order.categories]
        add_backwardation_structure_rectangles(fig, aggregated_svix, row=1)
        for idx, D in enumerate(d_order.categories):
            tmp_svix = aggregated_svix[D]
            fig.add_trace(go.Scatter(x=tmp_svix.index.tz_convert(None), y=tmp_svix.values,
                                     line=dict(color=px.colors.qualitative.Plotly[idx], width=3),
                                     name=f'Crypto SVIX<sub>0<span>&#10230;</span>{D}</sub>', showlegend=True),
                          row=1, col=1)
        fig.update_xaxes(tickangle=45)
        fig.update_layout(title=dict(text='<b>Term Structure of Crypto SVIX</b>', font=dict(size=TITLE_FONT_SIZE)),
                          font=dict(size=LABEL_AXIS_FONT_SIZE), width=1.25 * WIDTH, height=.75 * HEIGHT,
                          legend=dict(font=dict(size=LABEL_AXIS_FONT_SIZE)))
        fig.update_annotations(font=dict(size=LABEL_AXIS_FONT_SIZE))
    else:
        for pair in pairs_order.categories:
            tmp_svix = crypto_svix.dropna().query('underlying_name == @pair').drop('underlying_name', axis=1)
            tmp_svix = tmp_svix.droplevel(level=0, axis=0).reset_index(0)
            tmp_svix = pd.pivot_table(tmp_svix, columns='D', values='svix', index='timestamp')
            tmp_svix = tmp_svix.resample('7D').mean()
            tmp_svix = tmp_svix.loc[:, d_order.categories]
            row = list(pairs_order.categories).index(pair) + 1
            add_backwardation_structure_rectangles(fig, tmp_svix, row=row)
            for idx, D in enumerate(d_order.categories):
                fig.add_trace(go.Scatter(x=tmp_svix.index.tz_convert(None), y=tmp_svix.iloc[:, idx].values,
                                         line=dict(color=px.colors.qualitative.Plotly[idx], width=3),
                                         name=f'Crypto SVIX<sub>0<span>&#10230;</span>{D}</sub>',
                                         showlegend=pair == 'BTCUSDT'), row=row, col=1)
        fig.update_xaxes(tickangle=45)
        fig.for_each_annotation(lambda x: x.update({'text': f'{x["text"].split("=")[-1]}'.replace("USDT", "/USDT")}),
                                selector=lambda x: 'USDT' in x['text'])
        fig.update_layout(title=dict(text='<b>Term Structure of Crypto SVIX across Crypto Pairs</b>',
                                     font=dict(size=2 * TITLE_FONT_SIZE)), font=dict(size=2 * LABEL_AXIS_FONT_SIZE),
                          width=2 * WIDTH, height=len(SYMBOLS) * HEIGHT,
                          legend=dict(font=dict(size=2 * LABEL_AXIS_FONT_SIZE)))
        fig.update_annotations(font=dict(size=2 * LABEL_AXIS_FONT_SIZE))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False, showgrid=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False, showgrid=False)
    fig.update_traces(line=dict(width=5))

    fig.update_layout(title=dict(x=0.5, y=1, xanchor='center', yanchor='top', pad=dict(b=150, t=10)),
                      legend=dict(orientation='h', yanchor='top', y=dict([(True, -0.35), (False, -0.075)])[aggregated],
                                  xanchor='center', x=0.5, itemwidth=100),
                      paper_bgcolor='white', plot_bgcolor='white')
    fig.write_image(f'{args.target_figures}/svix{dict([(True, ""), (False, "_non_aggregated")])[aggregated]}.pdf',
                    format='pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all the plots.')
    parser.add_argument('--type', type=str, help='Type of plot to display', default='svix_rv_vrp')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    parser.add_argument('--target_figures', type=str, help='Directory where database is located',
                        default='../data_centre')
    args = parser.parse_args()
    print(args)
    mase = pd.read_csv(f'{args.target_path}/crypto_option_spot/metric.csv')
    mase = mase.set_index('D')
    model_ranking = mase.groupby(by=[pd.Grouper(key='target'),
                                     pd.Grouper(key='regression'),
                                     pd.Grouper(key='freq')])[['MASE']].mean()
    mase_comparison = \
        model_ranking.groupby(['target', 'regression', 'freq'])['MASE'].mean().groupby(level='target').idxmin()
    best_model_tags = model_ranking.groupby('target').idxmin().to_dict()['MASE']
    ####################################################################################################################
    ## Crypto SVIX, RV, and VRP
    ####################################################################################################################
    if args.type == 'svix_rv_vrp':
        svix_parquet = Path(f'{args.target_path}/crypto_option_spot/svix.parquet')
        crypto_svix = pl.scan_parquet(str(svix_parquet)).select(['underlying_name', 'timestamp', 'svix', 'aggregated',
                                                                 'D']).collect().to_pandas()
        crypto_svix = crypto_svix.set_index(['aggregated', 'D', 'timestamp'])
        crypto_svix['svix'] = crypto_svix['svix'].mul(100)
        ####################################################################
        ## Crypto SVIX
        ####################################################################
        plot_svix(crypto_svix, True, args)
        plot_svix(crypto_svix, False, args)
        ####################################################################
        ## Crypto SVIX, RV, and VRP
        ####################################################################
        forecasts_parquet = Path(f'{args.target_path}/crypto_option_spot/forecasts.parquet')
        rv_cols = ['timestamp', 'underlying_name', 'y_hat', 'y', 'regression', 'target', 'D', 'market', 'freq']
        regression, freq = best_model_tags['RV'][1:]
        if forecasts_parquet.exists():
            rv = pl.scan_parquet(str(forecasts_parquet)).select(rv_cols).filter((pl.col('target') == 'RV')
                                                                                & (pl.col('freq') == freq)
                                                                                & (pl.col('regression') == regression))
            rv = rv.drop(['freq']).with_columns([pl.col('timestamp').cast(pl.Datetime('ms')),
                                                 pl.col('y_hat').alias('forecast'), pl.col('y').alias('rv')])
        rv = rv.sort(['underlying_name', 'market', 'D', 'timestamp'])

        rv = rv.group_by_dynamic(index_column='timestamp', every=freq.replace('in', ''),
                                 group_by=['underlying_name', 'market', 'D'])
        rv = rv.agg([pl.col('regression').last(), pl.col('forecast').sum(), pl.col('rv').sum()])
        ####################################################################
        ## Plot SVIX, RV, and VRP
        ####################################################################
        for aggregated in [False, True]:
            n_rows = len(pairs_order.categories)
            n_cols = len(d_order.categories)
            fig = make_subplots(rows=n_rows, cols=n_cols, column_titles=list(d_order.categories),
                                row_titles=list(pairs_order.categories), shared_yaxes=False, shared_xaxes=False,
                                vertical_spacing=0.1, horizontal_spacing=0.06)
            tmp_crypto_svix = crypto_svix.xs(int(aggregated), axis=0, level=0, drop_level=True)
            if aggregated:
                tmp_crypto_svix = tmp_crypto_svix.drop('underlying_name', axis=1)
            for D, pair in product(list(d_order.categories), list(pairs_order.categories)):
                if aggregated:
                    tmp_pair = tmp_crypto_svix.xs(D, axis=0, level=0, drop_level=False)
                else:
                    tmp_pair = tmp_crypto_svix.xs(D, axis=0, level=0, drop_level=False)
                    tmp_pair = tmp_pair.query(f'underlying_name == "{pair}"')
                rv_pair = rv.filter((pl.col('underlying_name') == pair) & (pl.col('D') == D.upper()))
                rv_pair = rv.collect().to_pandas().drop('regression', axis=1)
                rv_pair = rv_pair.groupby([pd.Grouper(key='market'), pd.Grouper(key='timestamp', freq=freq)])
                rv_pair = rv_pair[['forecast', 'rv']].last()
                D_days = D_HORIZONS[D.lower()]
                annualisation = 360.0 / D_days
                rv_pair = rv_pair.pow(0.5).mul(annualisation ** 0.5)
                rv_pair = \
                    pd.pivot(rv_pair.reset_index(), index='timestamp', columns='market', values=['forecast', 'rv'])
                valid = rv_pair.notna().all(axis=1)
                rv_pair = rv_pair['forecast']
                rv_pair.columns = ['rv_0', 'rv_1']
                rv_pair = rv_pair.loc[valid].mul(100)
                try:
                    svix = tmp_pair.drop('underlying_name', axis=1).droplevel(axis=0, level=0)
                except KeyError:
                    svix = tmp_pair.droplevel(axis=0, level=0)
                svix.index = pd.to_datetime(svix.index, utc=True)
                rv_pair.index = pd.to_datetime(rv_pair.index, utc=True)
                if svix.index.has_duplicates:
                    if aggregated:
                        svix = svix.groupby(level=0).mean()
                    else:
                        duplicate_timestamps = svix.index[svix.index.duplicated(keep=False)].unique()
                        raise ValueError(f'Duplicate SVIX timestamps for {pair}, {D}: {duplicate_timestamps}')
                svix = svix.sort_index()
                svix = svix.reindex(rv_pair.index, method='nearest')
                vrp = -rv_pair.apply(lambda x, y: x.subtract(y.values.reshape(y.shape[0])), y=svix)
                vrp.columns = ['vrp_0', 'vrp_1']
                tmp_pair = pd.concat([vrp, rv_pair, svix], axis=1)
                tmp_pair = tmp_pair.resample('7D').mean()
                name_svix = 'Crypto SVIX<sub>0<span>&#10230;</span>D</sub>'
                name_rv0 = 'RV<sub>0<span>&#10230;</span>D</sub><sup>X<sup>(S)</sup></sup>'
                name_rv1 = 'RV<sub>0<span>&#10230;</span>D</sub><sup>X<sup>(S,O)</sup></sup>'
                name_vrp0 = 'VRP<sub>0<span>&#10230;</span>D</sub><sup>X<sup>(S)</sup></sup>'
                name_vrp1 = 'VRP<sub>0<span>&#10230;</span>D</sub><sup>X<sup>(S,O)</sup></sup>'
                col_idx = list(d_order.categories).index(D) + 1
                row_idx = list(pairs_order.categories).index(pair) + 1
                fig.add_trace(go.Scatter(x=tmp_pair.index.to_pydatetime(), y=tmp_pair['svix'].values, name=name_svix,
                                         line=dict(color=px.colors.qualitative.Plotly[0], width=1.5),
                                         showlegend=(pair == 'BTCUSDT' and D == '1d')),
                              col=col_idx, row=row_idx, secondary_y=False)
                fig.add_trace(go.Scatter(x=tmp_pair.index.to_pydatetime(), y=tmp_pair['rv_0'].values, name=name_rv0,
                                         line=dict(color=px.colors.qualitative.Plotly[1], width=1.5),
                                         showlegend=(pair == 'BTCUSDT' and D == '1d')),
                              col=col_idx, row=row_idx, secondary_y=False)
                fig.add_trace(go.Scatter(x=tmp_pair.index.to_pydatetime(), y=tmp_pair['rv_1'].values, name=name_rv1,
                                         line=dict(color=px.colors.qualitative.Plotly[3], width=1.5, dash='dot'),
                                         showlegend=pair == 'BTCUSDT' and D == '1d'),
                              col=col_idx, row=row_idx, secondary_y=False)
                fig.add_trace(go.Scatter(x=tmp_pair.index.to_pydatetime(), y=tmp_pair['vrp_0'].values, name=name_vrp0,
                                         line=dict(color=px.colors.qualitative.Plotly[2], width=1.5),
                                         showlegend=pair == 'BTCUSDT' and D == '1d'),
                              col=col_idx, row=row_idx, secondary_y=False)
                fig.add_trace(go.Scatter(x=tmp_pair.index.to_pydatetime(), y=tmp_pair['vrp_1'].values, name=name_vrp1,
                                         line=dict(color=px.colors.qualitative.Plotly[4], width=1.5, dash='dot'),
                                         showlegend=pair == 'BTCUSDT' and D == '1d'),
                              col=col_idx, row=row_idx, secondary_y=False)
            suffix = '(Non-Aggregated)' if not aggregated else ''
            title = f'Dynamics of Crypto SVIX, RV, and VRP {suffix}'
            fig.update_layout(title=dict(text=f'<b>{title}</b>', x=0.5, xanchor='center', y=0.995, yanchor='top',
                                         font=dict(size=18), automargin=True, yref='container'),
                              font=dict(size=.66 * LABEL_AXIS_FONT_SIZE), width=430 * n_cols, height=330 * n_rows,
                              paper_bgcolor='white', plot_bgcolor='white', margin=dict(b=190, r=90, t=90))
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=False, showgrid=False, tickangle=45,
                             nticks=6)
            # fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=False, showgrid=False, nticks=6,
            #                  secondary_y=False)
            # fig.update_yaxes(showline=True, linewidth=1, linecolor='rgba(150,50,50,0.6)', mirror=False, showgrid=False,
            #                  nticks=4)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='rgba(150,50,50,0.6)', mirror=False, showgrid=False,
                             nticks=4)
            fig.for_each_annotation(lambda x: x.update({'text': f'<i>{x["text"].split("=")[-1].lower()}</i>'}),
                                    selector=lambda x: x['text'].split('=')[-1].lower() in d_order.categories)
            fig.for_each_annotation(lambda x:
                                    x.update({'text': f'{x["text"].split("=")[-1].replace("USDT", "/USDT")}',
                                              'x': 1.045}), selector=lambda x: 'USDT' in x['text'])
            fig.update_annotations(font=dict(size=20))
            fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.09, xanchor='center', x=0.5, font=dict(
                size=LABEL_AXIS_FONT_SIZE), itemwidth=40, tracegroupgap=25, bgcolor='rgba(255,255,255,0.95)',
                                          bordercolor='rgba(0,0,0,0.3)', borderwidth=1))
            fig.write_image(
                f'{args.target_figures}/rv_svix_vrp {dict([(True, ""), (False, "_non_aggregated")])[aggregated]}.pdf',
                format='pdf')



