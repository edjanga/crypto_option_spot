import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from constants import DOTM, OTM, ITM, DITM, LABELS, SYMBOLS
import argparse
import polars as pl
from scipy.stats.mstats import winsorize

# pairs (columns) of the option summary table, in this order
OPTION_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
YEARS = list(range(2023, 2026))
# option files aggregated per query: one query per batch bounds the size of a query plan; the streaming engine
# bounds memory within a batch
FILES_PER_BATCH = 200
# moneyness in the option files is log(S/K): the category thresholds are the logs of the S/K ratios in constants.py
# (constants.MONEYNESS_BINS only logs the two middle ones, and polars sorts unsorted breaks silently)
MONEYNESS_LOG_BINS = list(np.log([DOTM, OTM, ITM, DITM]))
TYPES = ['call', 'put']
STATS = ['max', 'mean', 'min']


def pair_label(pair: str) -> str:
    return pair[:-len('USDT')] + '/USDT' if pair.endswith('USDT') else pair


def option_aggregation(paths: list) -> pl.LazyFrame:
    """One aggregation of a set of option files per (option type, moneyness category): number of rows, sum, count,
    min and max of log-moneyness, mark IV and time to maturity, and the distinct strikes."""
    chain = pl.scan_parquet(paths).select(['type', 'timestamp', 'strike_price', 'expiration', 'moneyness', 'mark_iv'])
    # exact time to maturity in years of 360 days
    ttm = (pl.col('expiration') - pl.col('timestamp')).dt.total_milliseconds() / (86_400_000 * 360)
    # moneyness category 0..4 (DOTM..DITM for a call), intervals closed on the right as in polars' cut
    category = pl.sum_horizontal([(pl.col('moneyness') > b).cast(pl.Int8) for b in MONEYNESS_LOG_BINS])
    category = pl.when(pl.col('moneyness').is_null()).then(None).otherwise(category)
    chain = chain.select('type', 'strike_price', category.alias('category'), pl.col('moneyness'),
                         pl.col('mark_iv').alias('iv'), ttm.alias('ttm'))
    return chain.group_by(['type', 'category']).agg(
        [pl.len().alias('n'), pl.col('strike_price').unique().alias('strikes')] +
        [expr for variable in ['moneyness', 'iv', 'ttm'] for expr in (
            pl.col(variable).sum().alias(f'{variable}_sum'), pl.col(variable).count().alias(f'{variable}_n'),
            pl.col(variable).min().alias(f'{variable}_min'), pl.col(variable).max().alias(f'{variable}_max'))])


def collect_fast(query: pl.LazyFrame) -> pl.DataFrame:
    """Runs a query on polars' streaming engine (bounded memory, all cores), or the default engine on old polars."""
    try:
        return query.collect(engine='streaming')
    except TypeError:
        return query.collect()


class EDA:

    def __init__(self, target_path: str):
        self._target_path = target_path

    def option_statistics(self) -> dict:
        """Exact summary statistics of the option chains of every pair and option type, over every option file of
        YEARS: number of observations per moneyness category and in total, (max, mean, min) of log-moneyness, mark IV
        and time to maturity (means over all observations), and number of distinct strikes.
        Every file is read once: a single aggregation query per batch of files, run on polars' streaming engine."""
        out = dict()
        for pair in OPTION_PAIRS:
            paths = sorted(str(path) for year in YEARS
                           for path in Path(f'{self._target_path}/{pair}/{year}/options').glob('*-*-*'))
            if not paths:
                print(f'WARNING: no option file for {pair} in {YEARS} -- its columns are left as NA.')
                continue
            parts = [collect_fast(option_aggregation(paths[start:start + FILES_PER_BATCH]))
                     for start in range(0, len(paths), FILES_PER_BATCH)]
            parts = pl.concat(parts)
            print(f'[EDA] {pair}: {len(paths)} option files read', flush=True)
            for option_type in TYPES:
                rows = parts.filter(pl.col('type') == option_type)
                labels = LABELS if option_type == 'call' else LABELS[::-1]   # a deep OTM call is a deep ITM put
                categories = dict((label, 0) for label in LABELS)
                for category_index, n in rows.filter(pl.col('category').is_not_null()).group_by('category').agg(
                        pl.col('n').sum()).iter_rows():
                    categories[labels[category_index]] += n
                stats = dict(categories=categories, total=int(rows['n'].sum()),
                             strikes=rows.select(pl.col('strikes').explode().drop_nulls().n_unique()).item())
                for variable in ['moneyness', 'iv', 'ttm']:
                    n = rows[f'{variable}_n'].sum()
                    stats[variable] = dict(max=rows[f'{variable}_max'].max(), mean=rows[f'{variable}_sum'].sum() / n,
                                           min=rows[f'{variable}_min'].min()) if n else None
                out[(pair, option_type)] = stats
        return out

    @staticmethod
    def option_latex(stats: dict) -> str:
        """The option summary table, in the layout of the paper."""
        columns = [(pair, option_type) for pair in OPTION_PAIRS for option_type in TYPES]

        def count(value):
            return 'NA' if value is None or value == 0 else f'{value:,}'

        def number(value):
            return 'NA' if value is None or not np.isfinite(value) else f'{value:.2f}'

        def row(label, name, values):
            return f' {label} & {name} & ' + ' & '.join(values) + r' \\'

        def get(column, *keys):
            value = stats.get(column)
            for key in keys:
                value = None if value is None else value.get(key) if isinstance(value, dict) else None
            return value

        lines = [r'\begin{table}[H]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{1.5pt}',
                 r'\begin{tabular}{l l ' + 'r' * len(columns) + '}', r'\toprule',
                 ' &  & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{pair_label(pair)}}}' for pair in OPTION_PAIRS)
                 + r' \\',
                 ' '.join(rf'\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(OPTION_PAIRS))),
                 ' & No. obs. & ' + ' & '.join(option_type.capitalize() for _, option_type in columns) + r' \\',
                 r'\midrule']
        for label in LABELS:
            lines.append(row('', label, [count(get(c, 'categories', label)) for c in columns]))
        lines += [r'\midrule', row('', 'Total', [count(get(c, 'total')) for c in columns]), r'\midrule']
        for variable, label in [('moneyness', r'\textit{log m}'), ('strikes', None), ('iv', 'IV'), ('ttm', 'TTM')]:
            if variable == 'strikes':
                lines += [row('', 'Strikes', [count(get(c, 'strikes')) for c in columns]), r'\midrule']
                continue
            for i, stat in enumerate(STATS):
                lines.append(row(label if i == 0 else '', stat, [number(get(c, variable, stat)) for c in columns]))
            lines.append(r'\midrule' if variable != 'ttm' else r'\bottomrule')
        lines += [r'\end{tabular}',
                  r'\caption{Summary statistics of option chain data. We report the total number of observations '
                  r'\mc{across each moneyness category}. The (max, mean, min) triplets represent the maximum, average, '
                  r'and minimum values of the variables \mc{across observations} in the dataset. The figures reported '
                  r'across the different moneyness categories indicate the number of contracts classified into each '
                  r'category.}',
                  r'\label{tab:data_summary_option}', r'\label{tab:moneyness_obs}', r'\end{table}']
        return '\n'.join(lines) + '\n'

    def summary(self, options: bool = True) -> None:
        summary_eda = dict()
        if options:
            path = f'{self._target_path}/crypto_option_spot/data_summary_option.tex'
            with open(path, 'w') as summary_file:
                summary_file.write(self.option_latex(self.option_statistics()))
            print(f'[EXPORT]: Summary EDA table for options has been exported to {path}.')
            return
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
        summary_eda['Returns'] = \
            returns.agg(['max', 'mean', 'min', lambda x: x.quantile(.75), lambda x: x.quantile(.5),
                         lambda x: x.quantile(.25)])
        summary_eda['Returns'].index = ['max', 'mean', 'min', '75%', '50%', '25%']
        summary_eda['Annualised RV'] = rv.agg(['max', 'mean', 'min', lambda x: x.quantile(.75),
                                               lambda x: x.quantile(.5), lambda x: x.quantile(.25)])
        summary_eda['Annualised RV'].index = ['max', 'mean', 'min', '75%', '50%', '25%']
        summary_eda = pd.concat(summary_eda)[SYMBOLS]
        summary_eda = summary_eda.style.format(decimal='.', thousands=',', precision=4)
        with open(f'{self._target_path}/crypto_option_spot/summary.txt', 'w') as summary_file:
            summary_file.writelines(summary_eda.to_latex())
            print('[EXPORT]: Summary EDA table for the spot market has been exported.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all performance tables.')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    args = parser.parse_args()
    eda_obj = EDA(target_path=args.target_path)
    eda_obj.summary(options=True)