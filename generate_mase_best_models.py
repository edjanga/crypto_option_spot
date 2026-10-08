import argparse
import os
import polars as pl

DAYS = {'1D': 1, '1W': 7, '1M': 30}
HORIZON_LABELS = {'1D': r'\textit{1d}', '1W': r'\textit{1w}', '1M': r'\textit{1m}'}
REGRESSION_LABELS = {'lightgbm': 'LightGBM', 'mlp': 'MLP'}
PAIR_ORDER = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
KEYS = ['underlying_name', 'target', 'regression', 'freq', 'D']


def mase_one_chunk(fc: pl.LazyFrame) -> pl.DataFrame:
    """MASE of one (target, regression, freq), per (underlying_name, D), averaged over the two markets. Both markets
    are scored on the timestamps where both have a forecast, against the same naive denominator."""
    naive = (fc.select(['underlying_name', 'target', 'freq', 'D', 'timestamp', 'y'])
             .unique(subset=['underlying_name', 'target', 'freq', 'D', 'timestamp'])
             .with_columns((pl.col('timestamp') + pl.duration(days=pl.col('D').replace_strict(DAYS, return_dtype=pl.Int64)))
                           .alias('timestamp'))
             .rename({'y': 'y_naive'}))
    both = (fc.filter(pl.col('market') == 0).select(KEYS + ['timestamp', 'y', pl.col('y_hat').alias('y_hat_0')])
            .join(fc.filter(pl.col('market') == 1).select(KEYS + ['timestamp', pl.col('y_hat').alias('y_hat_1')]),
                  on=KEYS + ['timestamp'], how='inner')
            .join(naive, on=['underlying_name', 'target', 'freq', 'D', 'timestamp'], how='inner')
            .drop_nulls(['y', 'y_naive', 'y_hat_0', 'y_hat_1']))
    scores = (both.group_by(KEYS).agg((pl.col('y') - pl.col('y_hat_0')).abs().mean().alias('mae_0'),
                                      (pl.col('y') - pl.col('y_hat_1')).abs().mean().alias('mae_1'),
                                      (pl.col('y') - pl.col('y_naive')).abs().mean().alias('mae_naive'))
              .collect())
    return scores.select(*KEYS, ((pl.col('mae_0') + pl.col('mae_1')) / 2 / pl.col('mae_naive')).alias('MASE'))


def latex_table(best: pl.DataFrame) -> str:
    horizons = [D for D in DAYS if D in best['D'].unique().to_list()]
    pairs = [p for p in PAIR_ORDER if p in best['underlying_name'].unique().to_list()]
    pairs += sorted(set(best['underlying_name'].unique().to_list()) - set(pairs))
    cell = {(r['underlying_name'], r['D']): r for r in best.iter_rows(named=True)}
    lines = [r'\begin{tabular}{l' + 'c' * len(horizons) + '}', r'\toprule',
             'Pair & ' + ' & '.join(HORIZON_LABELS[D] for D in horizons) + r' \\', r'\midrule']
    for pair in pairs:
        values = []
        for D in horizons:
            row = cell.get((pair, D))
            values.append('--' if row is None else
                          f"{REGRESSION_LABELS.get(row['regression'], row['regression'])} {row['freq']}")
        lines.append(pair.replace('_', r'\_') + ' & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Best (regression, freq) per pair and horizon, as LaTeX.')
    parser.add_argument('--target_path', type=str, default='./data_centre',
                        help='Directory where the database is located')
    args = parser.parse_args()
    folder = f'{args.target_path}/crypto_option_spot'
    scan = pl.scan_parquet(f'{folder}/forecasts.parquet')
    # one (target, regression, freq) at a time, to bound memory
    chunks = sorted(scan.select(['target', 'regression', 'freq']).unique().collect().rows())
    mase = pl.concat([mase_one_chunk(scan.filter((pl.col('target') == t) & (pl.col('regression') == r)
                                                 & (pl.col('freq') == f))) for t, r, f in chunks])
    # best (regression, freq) of every (target, underlying_name, D); ties broken by name for reproducibility
    best = (mase.sort(['MASE', 'regression', 'freq'])
            .group_by(['target', 'underlying_name', 'D'], maintain_order=True).first())
    for target in sorted(best['target'].unique().to_list()):
        path = f'{folder}/mase_per_freq_regression_{target}.tex'
        with open(path, 'w') as file:
            file.write(latex_table(best.filter(pl.col('target') == target)))
        print(f'{target}: {path}')
    print(r'The tables use \toprule and \midrule: add \usepackage{booktabs} to the preamble.')