from constants import *
from itertools import product
import pandas as pd
import torch
from typing import List, Tuple
import polars as pl
import statsmodels.api as sm

torch.set_float32_matmul_precision('high')
DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'


def run_pnl_ols(x, data, targets, pnl) -> pd.Series|pd.DataFrame:
    list_factors = x.features.values.tolist()
    target = x.target.unique()[0]
    # targets = targets.xs(target, axis=1, level=-1).filter(regex='|'.join(list_factors)).unstack()
    # targets = targets.loc[targets.index.get_level_values(0).isin(list_factors)]
    X = data.filter(regex='|'.join(list_factors))
    pnl = pnl.xs(x.name[1], axis=1, level=1).xs(x.name[-1], axis=0, level=1).stack()
    pnl.name = 'P&L'
    X = pnl.reset_index(0).swaplevel(0, 1).join(X, on=['underlying_name', 'timestamp']).set_index('aggregated',
                                                                                                  append=True)
    X['P&L'] = X['P&L'] + np.random.uniform(0, 1e-8, X.shape[0])
    X = X.apply(lambda x: pd.to_numeric(x, errors='coerce').astype('float32'))
    X = (X - X.min()).div(X.max()-X.min()).replace([-np.inf, np.inf], np.nan).dropna()
    params = sm.OLS(X['P&L'], sm.add_constant(X.drop(columns=['P&L']))).fit().params
    params.index = ['α'] + [f'β<sub>{i}</sub>' for i in range(1, params.shape[0])]
    return params


class FeatureBuilder:
    def __init__(self):
        pass

    @staticmethod
    def returns_moments(data: pl.DataFrame | pl.LazyFrame, freq: str, upperbound: int = 6) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        pl_freq = freq.replace('in', '')
        data = data.sort(['underlying_name', 'timestamp'])
        aggs = [pl.sum('returns').alias('returns')]
        K = (pd.to_timedelta('1D') // pd.to_timedelta(freq))
        for m in range(2, upperbound + 1):
            aggs.append((pl.col('returns') ** m).sum().alias(f'm{m}'))
            if m == 2:
                aggs.append((pl.col('returns') ** 2).sum().alias('RV'))
        moments = data.group_by_dynamic(index_column='timestamp', every=pl_freq, by='underlying_name').agg(aggs)
        for m in range(3, upperbound + 1):
            scale = float(K ** (m / 2 - 1))
            exp = float(m / 2)
            moments = \
                moments.with_columns((pl.col(f'm{m}') * scale / pl.col('m2').cast(pl.Float64).pow(exp))
                                     .cast(pl.Float32).alias(f'm{m}'))
            moments = moments.with_columns(pl.when(pl.col(f'm{m}').is_infinite())
                                           .then(None).otherwise(pl.col(f'm{m}'))
                                           .forward_fill().over('underlying_name').alias(f'm{m}'))
        return moments.collect()

    @staticmethod
    def lags(data: pl.DataFrame, features: List[str] = ['RV']+[f'm{i}' for i in range(2, 7)]) -> pl.LazyFrame:
        intraday_lags_exprs = [pl.col(feature) for feature in features]
        for lag in INTRADAY_LAGS:
            window_size = int(pd.to_timedelta(lag) // pd.to_timedelta('5m'))
            for feature in features:
                intraday_lags_exprs.append((pl.col(f'{feature}').rolling_mean(window_size)).shift().over
                                           ('underlying_name').alias(f'{feature}_{lag}'))
        for lag in INTERDAY_LAGS:
            feature_lag = list(product(features, [lag]))
            pl_lag = f'{lag[:-1]}mo' if 'M' in lag else lag.lower()
            if 'feature_lags' not in vars():
                feature_lags = data.with_columns(intraday_lags_exprs)
            feature_interday_lag = \
                (data.group_by_dynamic(index_column='timestamp', every=pl_lag.lower(),  by='underlying_name')
                 .agg([(pl.col(f'{feature}').mean()).alias(f'{feature}_{lag}') for feature, lag in feature_lag]))
            feature_interday_lag = feature_interday_lag.select(
                [pl.col('underlying_name'), pl.col('timestamp')]+
                [(pl.col(f'{feature}_{lag}').shift(1).over('underlying_name')) for feature, lag in feature_lag])
        feature_lags = feature_lags.join(feature_interday_lag, on=['underlying_name', 'timestamp'], how='left')
        feature_lags = \
            feature_lags.with_columns([pl.col(f'{feature}_{lag}').fill_null(strategy='forward').over('underlying_name')
                 for feature, lag in feature_lag])
        return feature_lags

    @staticmethod
    def define_market_participants(data: pl.DataFrame) -> pd.DataFrame:
        data = data.with_columns(
            pl.when(pl.col('moneyness_label').is_in(['ATM', 'ITM', 'DITM']))
            .then(pl.lit('informed_traders')).when(pl.col('moneyness_label') == 'OTM')
            .then(pl.lit('institutions')).otherwise(pl.lit('retail')).alias('market_participants'))
        return data

    @staticmethod
    def compute_exposure(data: pl.DataFrame, freq: str, collect: bool = True) -> pd.DataFrame:
        exposure = data
        pl_freq = freq.replace('min', 'm')
        for exp, greek in GREEKS_EXPOSURE.items():
            if exp == 'dex':
                expr = (pl.col('netDealerExposure') * pl.col(greek) * pl.col('underlying_price'))
            elif exp == 'gex':
                expr = (pl.col('netDealerExposure') * pl.col(greek) * (pl.col('underlying_price') ** 2))
            elif exp == "vex":
                expr = (pl.col('gex') * pl.col('tau') * (pl.col('mark_iv') ** 2))
            elif exp == "vax":
                expr = (pl.col('gex') * (pl.col('moneyness') + 0.5 * pl.col('tau') * (pl.col('mark_iv') ** 2)))
            elif exp == "vox":
                expr = (pl.col('vax') * (-pl.col('moneyness') - 0.5 * pl.col('tau') * (pl.col('mark_iv') ** 2)))
            exposure = exposure.with_columns(expr.alias(exp))
        exposure = exposure.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by='underlying_name').agg([pl.col(exp).sum() for exp in
                                                                                ['gex', 'vax', 'vex', 'vox']])
        if collect:
            exposure = exposure.collect()
        return exposure

    @staticmethod
    def put_call_ratio(data: pl.DataFrame, freq: str, collect: bool = True) -> pl.DataFrame|pl.LazyFrame:
        pl_freq = freq.replace('min', 'm')
        data.filter(pl.col('type') == 'put')
        pcr = data.filter(pl.col('type') == 'put').sort(by=[pl.col('underlying_name'), pl.col('type'),
                                                            pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('volume').sum())
        call = data.filter(pl.col('type') == 'call').sort(by=[pl.col('underlying_name'), pl.col('type'),
                                                              pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('volume').sum())
        pcr = pcr.join(call, on=['underlying_name', 'timestamp'], how='left', suffix='_call')
        pcr = pcr.with_columns((pl.col('volume')/pl.col('volume_call')).alias('pcr')).drop(
            ['type', 'volume', 'volume_call', 'type_call'])
        if collect:
            pcr = pcr.collect()
        return pcr

    @staticmethod
    def spread(data: pl.DataFrame, freq: str, collect: bool = True) -> pl.DataFrame:
        pl_freq = freq.replace('min', 'm')
        spread = data
        if collect:
            spread = spread.collect()
        spread = spread.select(['underlying_name', 'timestamp', 'type', 'spread'])
        spread = spread.sort(by=[pl.col('underlying_name'), pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(pl.col('spread').mean())
        spread_put = spread.filter(pl.col('type') == 'put').rename({'spread': 'spread_put'}).drop('type')
        spread = spread.filter(pl.col('type') == 'call').rename({'spread': 'spread_call'}).drop('type')
        spread = spread.join(spread_put, on=['underlying_name', 'timestamp'])
        return spread

    @staticmethod
    def label_market_participants(data: pl.DataFrame, freq: str) -> pl.DataFrame:
        pl_freq = freq.replace('min', 'm')
        mkt_participants = data.sort(by=[pl.col('underlying_name'), pl.col('type'),
                                         pl.col('timestamp')]).group_by_dynamic(
            index_column='timestamp', every=pl_freq, by=['underlying_name', 'type']).agg(
            pl.col('moneyness_label').mode())
        mkt_participants = mkt_participants.collect(engine=ENGINE).to_pandas()
        mkt_participants.moneyness_label = [participants[0] for participants in mkt_participants.moneyness_label]
        mkt_participants = pl.from_dataframe(mkt_participants)
        mkt_participants_call = mkt_participants.filter(pl.col('type') == 'call').drop('type')
        mkt_participants_put = mkt_participants.filter(pl.col('type') == 'put').drop('type')
        mkt_participants = \
            mkt_participants_call.with_columns(pl.when(pl.col('moneyness_label').is_in(['ATM', 'ITM']))
                                               .then(pl.lit('informed_traders_call'))
                                               .when(pl.col('moneyness_label') == 'OTM')
                                               .then(pl.lit('institutional_call'))
                                               .otherwise(pl.lit('retail_call')).alias('market_participants_call'))
        mkt_participants = mkt_participants.drop('moneyness_label')
        mkt_participants_put = \
            mkt_participants_put.with_columns(pl.when(pl.col("moneyness_label").is_in(['ATM', 'ITM']))
                                              .then(pl.lit('informed_traders_put'))
                                              .when(pl.col('moneyness_label') == 'OTM')
                                              .then(pl.lit('institutional_put'))
                                              .otherwise(pl.lit('retail_put')).alias('market_participants_put'))
        mkt_participants_put = mkt_participants_put.drop('moneyness_label')
        mkt_participants = mkt_participants.join(mkt_participants_put, on=['underlying_name', 'timestamp'])
        return mkt_participants


class PnLExplained:

    def __init__(self, underlying_returns: pd.DataFrame, underlying_rv: pd.DataFrame):
        self._returns = underlying_returns.set_index('timestamp')
        self._rv = underlying_rv.set_index('timestamp')

    def resample_moments(self, freq: str) -> None:
        self._returns = self._returns.resample(freq).sum()
        self._rv = self._rv.resample(freq).sum()

    def pnl_analysis(self, pnl: pd.DataFrame, ranking: pd.DataFrame,
                     features: pd.DataFrame, targets: pd.DataFrame, freq: str) -> Tuple[pd.DataFrame]:
        self.resample_moments(freq)
        rv_mkt = pd.qcut(self._rv.mean(axis=1), q=[0, .45, .9, 1], labels=VOL_REGIMES)
        rv_mkt.name = 'volatility_regime'
        ranking['features'] = ranking.features.str.replace(r'^is_(BTCUSDT|ETHUSDT|XRPUSDT|BNBUSDT|SOLUSDT)$', r'\1',
                                                           regex=True)
        ranking['features'] = \
            ['_'.join(name.split('_')[:-1]+[name.split('_')[-1].upper()])
             if ('_1m' in name) | ('_1w' in name) | ('_1d' in name) else name for name in ranking['features']]
        factor_analysis = ranking.groupby(by=[pd.Grouper(key='volatility_regime'), pd.Grouper(key='D'),
                                              pd.Grouper(key='target')],
                                          group_keys=True).apply(
            lambda x, data, targets, pnl: run_pnl_ols(x, data, targets, pnl), data=features,
            targets=targets.xs('1D', axis=1, level=-1), pnl=pnl).unstack()
        return factor_analysis





