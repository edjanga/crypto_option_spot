import argparse
import pandas as pd
import polars as pl

DAYS = {'1D': 1, '1W': 7, '1M': 30}
HORIZON_LABELS = {'1D': r'\textit{1d}', '1W': r'\textit{1w}', '1M': r'\textit{1m}'}
REGRESSION_LABELS = {'lightgbm': 'LightGBM', 'mlp': 'MLP'}
PAIR_ORDER = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
KEYS = ['underlying_name', 'target', 'regression', 'freq', 'D']


def mase_one_chunk(fc: pl.LazyFrame, selection_end=None) -> pl.DataFrame:
    """Scores of one (target, regression, freq), per (underlying_name, D).
    MASE_both: MASE averaged over the two markets, both scored on the timestamps where both have a forecast, against the
               same naive denominator; n_both is the number of those timestamps.
    MASE_S   : MASE of X^S alone, on every timestamp where X^S has a forecast (n_S timestamps); used only when a
               (pair, D) has no timestamp with both markets.
    selection_end: when given, only forecasts whose outcome is fully realised by then are scored (t + D <= end)."""
    naive = (fc.select(['underlying_name', 'target', 'freq', 'D', 'timestamp', 'y'])
             .unique(subset=['underlying_name', 'target', 'freq', 'D', 'timestamp'])
             .with_columns((pl.col('timestamp') + pl.duration(days=pl.col('D').replace_strict(DAYS, return_dtype=pl.Int64)))
                           .alias('timestamp'))
             .rename({'y': 'y_naive'}))
    market_0 = fc.filter(pl.col('market') == 0).select(KEYS + ['timestamp', 'y', pl.col('y_hat').alias('y_hat_0')])
    if selection_end is not None:
        # the outcome of a forecast made at t is realised over [t, t + D]: it is known at t + D
        horizon_end = pl.col('timestamp') + pl.duration(days=pl.col('D').replace_strict(DAYS, return_dtype=pl.Int64))
        market_0 = market_0.filter(horizon_end <= pl.lit(selection_end))
    market_0 = market_0.join(naive, on=['underlying_name', 'target', 'freq', 'D', 'timestamp'], how='inner') \
        .drop_nulls(['y', 'y_naive', 'y_hat_0'])
    both = (market_0
            .join(fc.filter(pl.col('market') == 1).select(KEYS + ['timestamp', pl.col('y_hat').alias('y_hat_1')]),
                  on=KEYS + ['timestamp'], how='inner')
            .drop_nulls(['y_hat_1']))
    scores_both = (both.group_by(KEYS).agg(pl.len().alias('n_both'),
                                           (pl.col('y') - pl.col('y_hat_0')).abs().mean().alias('mae_0'),
                                           (pl.col('y') - pl.col('y_hat_1')).abs().mean().alias('mae_1'),
                                           (pl.col('y') - pl.col('y_naive')).abs().mean().alias('mae_naive'))
                   .select(*KEYS, 'n_both',
                           ((pl.col('mae_0') + pl.col('mae_1')) / 2 / pl.col('mae_naive')).alias('MASE_both')))
    scores_s = (market_0.group_by(KEYS).agg(pl.len().alias('n_S'),
                                            ((pl.col('y') - pl.col('y_hat_0')).abs().mean()
                                             / (pl.col('y') - pl.col('y_naive')).abs().mean()).alias('MASE_S')))
    return scores_s.join(scores_both, on=KEYS, how='left').with_columns(pl.col('n_both').fill_null(0)).collect()


def select_best_models(forecasts_path: str, selection_end=None) -> pl.DataFrame:
    """Best (regression, freq) of every (target, underlying_name, D): lowest MASE_both, or lowest MASE_S when the
    (pair, D) has no timestamp where both markets have a forecast. Ties are broken by name for reproducibility.
    Columns: target, underlying_name, D, regression, freq, MASE, basis, n."""
    scan = pl.scan_parquet(forecasts_path)
    if selection_end is not None:
        selection_end = pd.Timestamp(selection_end)
        selection_end = (selection_end.tz_localize('UTC') if selection_end.tzinfo is None
                         else selection_end.tz_convert('UTC')).to_pydatetime()
    # one (target, regression, freq) at a time, to bound memory
    chunks = sorted(scan.select(['target', 'regression', 'freq']).unique().collect().rows())
    scores = pl.concat([mase_one_chunk(scan.filter((pl.col('target') == t) & (pl.col('regression') == r)
                                                   & (pl.col('freq') == f)), selection_end) for t, r, f in chunks])
    group = ['target', 'underlying_name', 'D']
    scores = scores.with_columns((pl.col('n_both').max().over(group) > 0).alias('has_both'))
    scores = scores.filter(~pl.col('has_both') | (pl.col('n_both') > 0)).with_columns(
        pl.when(pl.col('has_both')).then(pl.col('MASE_both')).otherwise(pl.col('MASE_S')).alias('MASE'),
        pl.when(pl.col('has_both')).then(pl.lit('both markets')).otherwise(pl.lit('X^S only')).alias('basis'),
        pl.when(pl.col('has_both')).then(pl.col('n_both')).otherwise(pl.col('n_S')).alias('n'))
    return (scores.drop_nulls('MASE').sort(['MASE', 'regression', 'freq'])
            .group_by(group, maintain_order=True).first()
            .select(group + ['regression', 'freq', 'MASE', 'basis', 'n']).sort(group))


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
    parser.add_argument('--selection_end', type=str, default=None,
                        help='Only score forecasts whose outcome is realised by this date (e.g. 2025-01-01, the start '
                             'of the backtest, as generate_pnl.py does). Default: the whole sample.')
    args = parser.parse_args()
    folder = f'{args.target_path}/crypto_option_spot'
    best = select_best_models(f'{folder}/forecasts.parquet', selection_end=args.selection_end)
    for target in sorted(best['target'].unique().to_list()):
        path = f'{folder}/mase_per_freq_regression_{target}.tex'
        with open(path, 'w') as file:
            file.write(latex_table(best.filter(pl.col('target') == target)))
        print(f'{target}: {path}')
    print(r'The tables use \toprule and \midrule: add \usepackage{booktabs} to the preamble.')