import os
import argparse
# --cpu_count must be known before polars is imported: polars sizes its thread pool once, at import
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--cpu_count', type=int, default=None)
_cpu_count = _pre_parser.parse_known_args()[0].cpu_count
if _cpu_count:
    os.environ['POLARS_MAX_THREADS'] = str(_cpu_count)
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from constants import DOTM, OTM, ITM, DITM, LABELS, SYMBOLS
import polars as pl

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
# spot summary: returns are built from the last mid price of every SPOT_FREQ bar
SPOT_FREQ = '5m'
BARS_PER_DAY = 288
# days with fewer 5-minute returns than this are left out (incomplete days would understate the realised variance)
MIN_BARS_PER_DAY = BARS_PER_DAY // 2
# daily realised variance is printed in units of 10^-4 (multiplied by this factor), returns as they are
RV_SCALE = 1e4
SPOT_STATS = [('max', 'max'), ('mean', 'mean'), ('min', 'min'), ('75', r'75$^{\text{th}}$'),
              ('50', r'50$^{\text{th}}$'), ('25', r'25$^{\text{th}}$')]
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


def spot_bars(paths: list) -> pl.LazyFrame:
    """Last mid price of every SPOT_FREQ bar of a set of spot files (bars labelled by their start)."""
    spot = pl.scan_parquet(paths).select([pl.col('timestamp').cast(pl.Datetime('ms')), pl.col('mid_price[0]').alias('mid')])
    spot = spot.filter(pl.col('mid') > 0)
    return spot.group_by(pl.col('timestamp').dt.truncate(SPOT_FREQ).alias('bar')).agg(
        pl.col('mid').sort_by('timestamp').last(), pl.col('timestamp').max().alias('last_update'))

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
                             strikes=len(set(strike for strikes in rows['strikes'].to_list() for strike in strikes
                                               if strike is not None)))
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
                  r'category. Calls and puts share the same log-moneyness, strike, implied volatility and time-to-maturity '
                  r'statistics because \deribit{} lists a call and a put for every strike and expiry, and both legs carry the '
                  r'same mark implied volatility. For the same reason, the moneyness counts of calls and puts mirror each other: '
                  r'a strike that is deep out of the money for the call is deep in the money for the put.}',
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
        path = f'{self._target_path}/crypto_option_spot/data_summary_spot.tex'
        with open(path, 'w') as summary_file:
            summary_file.write(self.spot_latex(self.spot_statistics()))
        print(f'[EXPORT]: Summary EDA table for the spot market has been exported to {path}.')

    def spot_statistics(self) -> dict:
        """Daily log returns and daily realised variance of every pair, from 5-minute log returns of the mid price:
        the daily log return is the sum of the day's 5-minute log returns (close to close), the daily realised variance
        the sum of their squares. Returns {pair: {'returns': {stat: value}, 'rv': {stat: value}, 'days': n}}."""
        out = dict()
        for pair in OPTION_PAIRS:
            paths = sorted(str(path) for year in YEARS
                           for path in Path(f'{self._target_path}/{pair}/{year}/spot').glob('*-*-*'))
            if not paths:
                print(f'WARNING: no spot file for {pair} in {YEARS} -- its column is left as NA.')
                continue
            bars = pl.concat([collect_fast(spot_bars(paths[start:start + FILES_PER_BATCH]))
                              for start in range(0, len(paths), FILES_PER_BATCH)])
            # a bar split across two batches (files overlapping midnight) keeps its latest price
            bars = bars.sort(['bar', 'last_update']).group_by('bar', maintain_order=True).last()
            returns = bars.select('bar', pl.col('mid').log().diff().alias('r')).drop_nulls()
            daily = returns.group_by(pl.col('bar').dt.date().alias('date')).agg(
                pl.col('r').sum().alias('returns'), pl.col('r').pow(2).sum().alias('rv'), pl.len().alias('n_bars'))
            kept = daily.filter(pl.col('n_bars') >= MIN_BARS_PER_DAY)
            print(f'[EDA] {pair}: {len(paths)} spot files, {daily.height} days, {daily.height - kept.height} day(s) '
                  f'with fewer than {MIN_BARS_PER_DAY} 5-minute returns left out', flush=True)
            out[pair] = dict(days=kept.height)
            for variable in ['returns', 'rv']:
                x = kept[variable]
                out[pair][variable] = dict(max=x.max(), mean=x.mean(), min=x.min(),
                                           **dict((q, x.quantile(int(q) / 100, interpolation='linear'))
                                                  for q in ['75', '50', '25']))
        return out

    @staticmethod
    def spot_latex(stats: dict) -> str:
        """The spot summary table, in the layout of the paper."""
        def number(value, scale=1., decimals=4):
            return 'NA' if value is None or not np.isfinite(value) else f'{value * scale:.{decimals}f}'

        lines = [r'\begin{table}[H]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
                 r'\begin{tabular}{l l ' + 'r' * len(OPTION_PAIRS) + '}', r'\toprule',
                 ' &  & ' + ' & '.join(pair_label(pair) for pair in OPTION_PAIRS) + r' \\', r'\midrule']
        for i_variable, (variable, label, scale, decimals) in enumerate(
                [('returns', 'Returns', 1., 4), ('rv', r'RV ($\times 10^{-4}$)', RV_SCALE, 2)]):
            if i_variable:
                lines.append(r'\midrule')
            for i, (stat, name) in enumerate(SPOT_STATS):
                first = rf'\multirow[c]{{{len(SPOT_STATS)}}}{{*}}{{{label}}}' if i == 0 else ''
                values = [number(stats.get(pair, {}).get(variable, {}).get(stat), scale, decimals)
                          for pair in OPTION_PAIRS]
                lines.append(f'{first} & {name} & ' + ' & '.join(values) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}',
                  r'\caption{Summary statistics for daily \mc{log-}returns and daily realised variance (RV). '
                  r'\edd{Both are computed from non-overlapping 5-minute log-returns of the mid price, 288 per day: the '
                  r'daily log-return is their sum and the daily RV the sum of their squares. RV is reported in units of '
                  r'$10^{-4}$.}}',
                  r'\label{tab:data_summary_spot}', r'\end{table}']
        return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all performance tables.')
    parser.add_argument('--target_path', type=str, help='Directory where database is located',
                        default='../data_centre')
    parser.add_argument('--cpu_count', type=int, default=None,
                        help='Number of CPUs polars may use (default: all the CPUs it can see).')
    parser.add_argument('--table', type=str, default='both', choices=['option', 'spot', 'both'],
                        help='Summary table(s) to export.')
    args = parser.parse_args()
    print(f'polars uses {pl.thread_pool_size()} thread(s)', flush=True)
    eda_obj = EDA(target_path=args.target_path)
    if args.table in ('option', 'both'):
        eda_obj.summary(options=True)
    if args.table in ('spot', 'both'):
        eda_obj.summary(options=False)