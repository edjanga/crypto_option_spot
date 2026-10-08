import argparse
import pandas as pd
import polars as pl

DAYS = {'1D': 1, '1W': 7, '1M': 30}
HORIZON_LABELS = {'1D': r'\textit{1d}', '1W': r'\textit{1w}', '1M': r'\textit{1m}'}
VOL_REGIMES = ['Low', 'Normal', 'High']
REGIME_LABELS = {'All': '', 'Low': r'\textit{low}', 'Normal': r'\textit{normal}', 'High': r'\textit{high}'}
MARKET_LABELS = {0: r'$X^S$', 1: r'$X^{(S,O)}$'}
PAIR_ORDER = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
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
    """Regime of every bar, per frequency, exactly as in generate_results.py."""
    rv = (pl.scan_parquet(spot_features_path).filter((pl.col('target') == 'RV') & (pl.col('freq') == '15min'))
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
    pairs += sorted(set(table['underlying_name'].unique().to_list()) - set(pairs))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MASE by pair, horizon and volatility regime, as LaTeX.')
    parser.add_argument('--target_path', type=str, default='./data_centre',
                        help='Directory where the database is located')
    args = parser.parse_args()
    folder = f'{args.target_path}/crypto_option_spot'
    scan = pl.scan_parquet(f'{folder}/forecasts.parquet')
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
    # best (regression, freq) of every (target, pair, D) on the whole sample, as in generate_mase_best_models.py:
    # lowest MASE averaged over the two markets; ties broken by name for reproducibility
    best = (scores.filter(pl.col('regime') == 'All')
            .with_columns(((pl.col('MASE_0') + pl.col('MASE_1')) / 2).alias('MASE'))
            .sort(['MASE', 'regression', 'freq'])
            .group_by(['target', 'underlying_name', 'D'], maintain_order=True).first()
            .select(['target', 'underlying_name', 'D', 'regression', 'freq']))
    table = scores.join(best, on=['target', 'underlying_name', 'D', 'regression', 'freq'], how='inner')
    for target in sorted(table['target'].unique().to_list()):
        path = f'{folder}/mase_regimes_{target}.tex'
        with open(path, 'w') as file:
            file.write(latex_table(table.filter(pl.col('target') == target)))
        print(f'{target}: {path}')