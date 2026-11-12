import argparse
import pandas as pd
import polars as pl
from generate_mase_best_models import select_best_models

DAYS = {'1D': 1, '1W': 7, '1M': 30}
HORIZON_LABELS = {'1D': r'\textit{1d}', '1W': r'\textit{1w}', '1M': r'\textit{1m}'}
VOL_REGIMES = ['Low', 'Normal', 'High']
REGIME_LABELS = {'All': '', 'Low': r'\textit{low}', 'Normal': r'\textit{normal}', 'High': r'\textit{high}'}
MARKET_LABELS = {0: r'$X^S$', 1: r'$X^{(S,O)}$'}
PAIR_ORDER = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']      # the pairs of the paper: any other pair is left out
PAIR_LABELS = {'BTCUSDT': 'BTC/USDT', 'ETHUSDT': 'ETH/USDT', 'SOLUSDT': 'SOL/USDT', 'BNBUSDT': 'BNB/USDT'}
REGRESSION_LABELS = {'lightgbm': 'LightGBM', 'mlp': 'MLP', 'softs': 'SOFTS'}
TARGET_PANELS = [('returns', 'Panel A: MASE for returns', 'Returns'), ('RV', 'Panel B: MASE for RV', 'RV')]
FREQS = ['15min', '30min', '1h']
KEYS = ['underlying_name', 'target', 'regression', 'freq', 'D']


def scored_rows(fc: pl.LazyFrame) -> pl.LazyFrame:
    """Absolute errors of X^S (e_0), X^(S,O) (e_1) and of the naive forecast (e_naive), on the timestamps where both
    markets have a forecast."""
    naive = (fc.select(['underlying_name', 'target', 'freq', 'D', 'timestamp', 'y'])
             .unique(subset=['underlying_name', 'target', 'freq', 'D', 'timestamp'])
             .with_columns((pl.col('timestamp')
                            + pl.duration(days=pl.col('D').replace_strict(DAYS, return_dtype=pl.Int64)))
                           .alias('timestamp'))
             .rename({'y': 'y_naive'}))
    return (fc.filter(pl.col('market') == 0).select(KEYS + ['timestamp', 'y', pl.col('y_hat').alias('y_hat_0')])
            .join(fc.filter(pl.col('market') == 1).select(KEYS + ['timestamp', pl.col('y_hat').alias('y_hat_1')]),
                  on=KEYS + ['timestamp'], how='inner')
            .join(naive, on=['underlying_name', 'target', 'freq', 'D', 'timestamp'], how='inner')
            .drop_nulls(['y', 'y_naive', 'y_hat_0', 'y_hat_1'])
            .select(KEYS + ['timestamp', (pl.col('y') - pl.col('y_hat_0')).abs().alias('e_0'),
                            (pl.col('y') - pl.col('y_hat_1')).abs().alias('e_1'),
                            (pl.col('y') - pl.col('y_naive')).abs().alias('e_naive')]))


def chunks(scan: pl.LazyFrame):
    """The forecasts one (target, regression, freq) at a time, to bound memory."""
    for target, regression, freq in sorted(scan.select(['target', 'regression', 'freq']).unique().collect().rows()):
        yield (target, regression, freq), scan.filter((pl.col('target') == target)
                                                      & (pl.col('regression') == regression)
                                                      & (pl.col('freq') == freq))


def volatility_regimes(spot_features_path: str) -> pl.DataFrame:
    """Regime of every bar, per frequency, exactly as in generate_results.py (market RV: the cross-sectional average
    RV of the pairs of the paper)."""
    rv = (pl.scan_parquet(spot_features_path).filter((pl.col('target') == 'RV') & (pl.col('freq') == '15min')
                                                     & pl.col('underlying_name').is_in(PAIR_ORDER))
          .select(['underlying_name', 'timestamp', 'target_values']).unique(maintain_order=True).collect().to_pandas())
    rv = rv.groupby('timestamp')['target_values'].mean()
    regimes = [pd.DataFrame({'timestamp': summed.index,
                             'regime': pd.qcut(summed, q=[0, .33, .66, 1], labels=VOL_REGIMES).astype(str).values,
                             'freq': freq})
               for freq in FREQS for summed in [rv.resample(freq).sum()]]
    regimes = pl.from_pandas(pd.concat(regimes, ignore_index=True))
    return regimes.with_columns(pl.col('timestamp').cast(pl.Datetime('us', 'UTC')))


def latex_table(table: pl.DataFrame) -> str:
    horizons = [D for D in DAYS if D in table['D'].unique().to_list()]
    regimes = ['All'] + VOL_REGIMES
    pairs = [p for p in PAIR_ORDER if p in table['underlying_name'].unique().to_list()]
    cell = {(r['underlying_name'], r['D'], r['regime']): r for r in table.iter_rows(named=True)}
    n = 2 * len(regimes)
    lines = [r'\begin{tabular}{ll' + 'rl' * len(regimes) + '}',
             rf' &  & \multicolumn{{{n}}}{{r}}{{MASE}} \\',
             ' &  & ' + ' & '.join(rf'\multicolumn{{2}}{{r}}{{{REGIME_LABELS[g]}}}' for g in regimes) + r' \\',
             ' &  & ' + ' & '.join(f'{MARKET_LABELS[0]} & {MARKET_LABELS[1]}' for _ in regimes) + r' \\']
    for pair in pairs:
        for i, D in enumerate(horizons):
            values = []
            for regime in regimes:
                row = cell.get((pair, D, regime))
                if row is None:
                    values += ['--', '--']
                    continue
                change = 100 * row['change']
                change = rf'$\mathbf{{{change:.2f}}}$\%' if change < 0 else rf'${change:.2f}$\%'
                values += [f"{row['MASE_0']:.4f}", change]
            label = pair.replace('_', r'\_') if i == 0 else ''
            lines.append(f'{label} & {HORIZON_LABELS[D]} & ' + ' & '.join(values) + r' \\')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'


def paper_mase_table(table: pl.DataFrame) -> str:
    """Table mase_performance.tex of the paper: returns in Panel A, RV in Panel B; X^S as a MASE level, X^(S,O) as the
    change in % relative to X^S (bold when negative), unconditional then by volatility regime (needs booktabs)."""
    cell = {(r['target'], r['underlying_name'], r['D'], r['regime']): r for r in table.iter_rows(named=True)}
    regimes = ['All'] + VOL_REGIMES
    blocks = []
    for target, title, _ in TARGET_PANELS:
        lines = [r'    \begin{tabular}{ll|rr|rr|rr|rr}', r'\toprule', rf'\multicolumn{{10}}{{c}}{{{title}}} \\',
                 r'\midrule',
                 r' & & \multicolumn{2}{c|}{Unconditional} & \multicolumn{6}{c}{Conditional}\\',
                 r' &  & \multicolumn{2}{c|}{} & \multicolumn{2}{c|}{\textit{low}} & '
                 r'\multicolumn{2}{c|}{\textit{normal}} & \multicolumn{2}{c}{\textit{high}} \\',
                 r' &  & ' + ' & '.join([rf'{MARKET_LABELS[0]} & {MARKET_LABELS[1]}'] * len(regimes)) + r' \\',
                 r'\midrule']
        for i_pair, pair in enumerate(PAIR_ORDER):
            if i_pair:
                lines.append(r' \cmidrule{2-10}')
            for i_D, D in enumerate(DAYS):
                values = []
                for regime in regimes:
                    row = cell.get((target, pair, D, regime))
                    if row is None or row['MASE_0'] is None or row['change'] is None:
                        values += ['NaN', 'NaN']
                        continue
                    change = 100 * row['change']
                    values += [f"{row['MASE_0']:.4f}",
                               rf'$\mathbf{{{change:.2f}}}$\%' if change < 0 else rf'${change:.2f}$\%']
                label = PAIR_LABELS[pair] if i_D == 0 else ''
                lines.append(f'{label} & {HORIZON_LABELS[D]} & ' + ' & '.join(values) + r' \\')
        lines += [r' \bottomrule', r'\end{tabular}']
        blocks.append('\n'.join(lines))
    return '\n'.join([
        r'\begin{table}[!htbp]', r'    \centering', r'    \small',
        r'    \setlength{\tabcolsep}{4.3pt}\setlength{\aboverulesep}{0pt}\setlength{\belowrulesep}{0pt}'
        r'\renewcommand{\arraystretch}{1.15}',
        ('\n' + r'    \vspace{1em}' + '\n\n').join(blocks),
        r'    \caption{Out-of-sample MASE for return (Panel A) and RV (Panel B) forecasting using the spot-derived '
        r'feature set $X^S$ and the augmented feature set $X^{(S,O)}$, which incorporates option-based information, '
        r'across cryptocurrency underlyings and forecast horizons. Each underlying--horizon pair is evaluated using its '
        r'selected regression model and sampling frequency (Table~\ref{tab:mase_per_regression_freq_returns}). The '
        r'$X^S$ columns report MASE levels, whereas the $X^{(S,O)}$ columns report percentage changes relative to the '
        r'$X^S$ benchmark. Negative changes indicate improved predictive accuracy from incorporating option-based '
        r'features and are highlighted in bold. The first column block reports unconditional results, while the '
        r'remaining blocks present results conditional on low-, normal-, and high-volatility regimes.}',
        r'    \label{tab:mase_perfomance_returns}\label{tab:mase_perfomance_rv}', r'\end{table}']) + '\n'


def paper_selected_models_table(best: pl.DataFrame) -> str:
    """Table selected_models.tex of the paper: the (regression, freq) of every (pair, target, horizon) (needs booktabs
    and adjustbox)."""
    cell = {(r['target'], r['underlying_name'], r['D']): r for r in best.iter_rows(named=True)}

    def name(target, pair, D):
        row = cell.get((target, pair, D))
        if row is None:
            return '--'
        return rf"{REGRESSION_LABELS.get(row['regression'], row['regression'])}$_{{\mathrm{{{row['freq']}}}}}$"
    lines = [r'\begin{table}[!htbp]', r'    \centering', r'    \small', r'    \setlength{\tabcolsep}{4pt}',
             r'    \begin{adjustbox}{max width=\textwidth}', r'    \begin{tabular}{l|ccc|ccc}', r'        \toprule',
             r'        & \multicolumn{3}{c|}{Returns} & \multicolumn{3}{c}{RV} \\',
             r'        & ' + ' & '.join([HORIZON_LABELS[D] for D in DAYS] * 2) + r' \\', r'        \midrule']
    for pair in PAIR_ORDER:
        lines.append(f'        {PAIR_LABELS[pair]} & '
                     + ' & '.join(name(target, pair, D) for target, _, _ in TARGET_PANELS for D in DAYS) + r' \\')
    lines += [r'        \bottomrule', r'    \end{tabular}', r'    \end{adjustbox}',
              r'    \caption{Selected forecasting models and sampling frequencies by target, cryptocurrency pair, and '
              r'forecast horizon. For each target, pair, and horizon, the (regression, sampling frequency) combination '
              r'is selected according to the criterion in Equation~\eqref{eqn:model_selection} of '
              r'Section~\ref{subsec:performance_evaluation}, that is, the lowest out-of-sample MASE averaged over $X^S$ '
              r'and $X^{(S,O)}$, computed over the full out-of-sample period. Model parameters are estimated on rolling '
              r'windows of past data only (Section~\ref{subsec:training_procedure}). Subscripts denote the sampling '
              r'frequency.}',
              r'    \label{tab:mase_per_regression_freq_returns}\label{tab:mase_per_regression_freq_rv}', r'\end{table}']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MASE by pair, horizon and volatility regime, as LaTeX.')
    parser.add_argument('--target_path', type=str, default='./data_centre',
                        help='Directory where the database is located')
    args = parser.parse_args()
    folder = f'{args.target_path}/crypto_option_spot'
    scan = pl.scan_parquet(f'{folder}/forecasts.parquet').filter(pl.col('underlying_name').is_in(PAIR_ORDER))
    regimes = volatility_regimes(f'{args.target_path}/spot_features.parquet')

    results, model_scores = [], []
    for (target, regression, freq), fc in chunks(scan):
        rows = scored_rows(fc).join(regimes.lazy(), on=['freq', 'timestamp'], how='left').collect()
        n_missing = rows['regime'].null_count()
        if n_missing:
            print(f'WARNING: {n_missing} forecasts of {target}/{regression}/{freq} have no regime; '
                  f'they count in the whole sample only.')
        whole = rows.with_columns(pl.lit('All').alias('regime'))
        by_regime = rows.drop_nulls('regime')
        results.append(pl.concat([whole, by_regime])
                       .group_by(KEYS + ['regime']).agg(pl.len().alias('n'), pl.col('e_0').mean(),
                                                        pl.col('e_1').mean(), pl.col('e_naive').mean()))
    scores = (pl.concat(results)
              .with_columns((pl.col('e_0') / pl.col('e_naive')).alias('MASE_0'),
                            (pl.col('e_1') / pl.col('e_naive')).alias('MASE_1'),
                            (pl.col('e_1') / pl.col('e_0') - 1).alias('change')))
    # best (regression, freq) of every (target, pair, D) on the whole sample: the selection of
    # generate_mase_best_models.py, also used by generate_pnl.py and the feature importance of generate_results.py
    # (lowest MASE averaged over the two markets; ties broken by name for reproducibility)
    best = (select_best_models(f'{folder}/forecasts.parquet')
            .filter(pl.col('underlying_name').is_in(PAIR_ORDER))
            .select(['target', 'underlying_name', 'D', 'regression', 'freq']))
    table = scores.join(best, on=['target', 'underlying_name', 'D', 'regression', 'freq'], how='inner')
    for target in sorted(table['target'].unique().to_list()):
        path = f'{folder}/mase_regimes_{target}.tex'
        with open(path, 'w') as file:
            file.write(latex_table(table.filter(pl.col('target') == target)))
        print(f'{target}: {path}')
    # the two tables of the paper, ready to \input
    for name, text in [('mase_performance.tex', paper_mase_table(table)),
                       ('selected_models.tex', paper_selected_models_table(best))]:
        with open(f'{folder}/{name}', 'w') as file:
            file.write(text)
        print(f'{name}: {folder}/{name}')