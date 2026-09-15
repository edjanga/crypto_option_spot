import argparse
import pdb
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import brentq
from scipy.stats import norm

MIN_SIGMA, MAX_SIGMA = 0.01, 3.0

def _bs76_price(forward: float, strike: float, tau: float, sigma: float, r: float,
                option_type: str) -> float:
    if tau <= 0 or sigma <= 0:
        intrinsic = max(forward - strike, 0) if option_type == 'call' else max(strike - forward, 0)
        return intrinsic * np.exp(-r * tau)
    d1 = (np.log(forward / strike) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if option_type == 'call':
        return np.exp(-r * tau) * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return np.exp(-r * tau) * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def _implied_vol(price: float, forward: float, strike: float, tau: float, r: float,
                 option_type: str) -> float:
    intrinsic = max(forward - strike, 0) if option_type == 'call' else max(strike - forward, 0)
    intrinsic *= np.exp(-r * tau)
    if tau <= 0 or price <= intrinsic:
        return np.nan
    try:
        return brentq(lambda s: _bs76_price(forward, strike, tau, s, r, option_type) - price, 1e-4, 10.0, xtol=1e-6)
    except ValueError:
        return np.nan


def _call_put_by_strike(group: pd.DataFrame) -> tuple:
    """Returns (calls, puts) as strike-indexed Series, with duplicate strikes collapsed via
    mean() first -- a plain .set_index('strike') allows a non-unique index, which turns a
    later .loc[k] lookup into a Series instead of a scalar the moment two rows share a
    strike (e.g. resampling landing two observations on the same contract)."""
    calls = group.loc[group['type'] == 'call'].groupby('strike')['mid_price'].mean()
    puts = group.loc[group['type'] == 'put'].groupby('strike')['mid_price'].mean()
    return calls, puts


def _forward_price(group: pd.DataFrame, spot: float, tau: float, r: float) -> float:
    calls, puts = _call_put_by_strike(group)
    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        return spot * np.exp(r * tau)
    diff = (calls.loc[common_strikes] - puts.loc[common_strikes]).abs()
    k_star = diff.idxmin()
    return float(k_star + (calls.loc[k_star] - puts.loc[k_star]) * np.exp(r * tau))


def _fit_dumas_surface(snapshot: pd.DataFrame, forwards_by_tau: Dict[float, float], r: float,
                       iv_column: Optional[str]) -> Optional[np.ndarray]:
    rows = []
    for _, row in snapshot.iterrows():
        forward = forwards_by_tau[row['tau']]
        strike = row['strike']
        if not (np.isfinite(forward) and forward > 0 and np.isfinite(strike) and strike > 0):
            continue  # degenerate forward/strike would otherwise poison log(strike/forward)
        is_otm = (row['type'] == 'put' and strike < forward) or \
                 (row['type'] == 'call' and strike >= forward)
        if not is_otm:
            continue
        if iv_column is not None and pd.notna(row.get(iv_column)):
            iv = float(row[iv_column])
        else:
            iv = _implied_vol(row['mid_price'], forward, strike, row['tau'], r, row['type'])
        if pd.isna(iv) or not (MIN_SIGMA <= iv <= MAX_SIGMA):
            continue
        m = np.log(strike / forward)
        if not np.isfinite(m):
            continue
        rows.append((m, row['tau'], iv))
    if len(rows) < 8:
        return None
    m_arr, tau_arr, iv_arr = (np.array(x) for x in zip(*rows))
    X = np.column_stack([np.ones_like(m_arr), m_arr, m_arr ** 2, tau_arr, tau_arr ** 2, m_arr * tau_arr])
    finite_rows = np.isfinite(X).all(axis=1) & np.isfinite(iv_arr)
    if finite_rows.sum() < 8:
        return None
    X, iv_arr = X[finite_rows], iv_arr[finite_rows]
    try:
        coeffs, *_ = np.linalg.lstsq(X, iv_arr, rcond=None)
    except np.linalg.LinAlgError:
        # None (not a NaN-filled array) -- the caller checks `if coeffs is not None` to
        # decide whether to fall back to raw integration; a NaN array looks like success
        # and silently produces all-NaN output instead of degrading gracefully.
        return None
    if not np.all(np.isfinite(coeffs)):
        return None
    return coeffs


def _fitted_iv(coeffs: np.ndarray, m, tau) -> np.ndarray:
    X = np.column_stack([np.ones_like(m), m, m ** 2, np.full_like(m, tau), np.full_like(m, tau) ** 2, m * tau])
    return np.clip(X @ coeffs, MIN_SIGMA, MAX_SIGMA)


def _variance_swap_squared_smoothed(tau: float, forward: float, spot: float, r: float, coeffs: np.ndarray,
                                    observed_m: np.ndarray, n_grid_points: int, grid_multiplier: float) -> tuple:
    """Returns (svix_sq, vix_sq) from the same fitted-smile grid. The two measures only
    differ in how the OTM prices are weighted before integrating:
      SVIX (Martin 2017): weight = 1 / S_t^2   (CONSTANT across strikes)
      VIX  (CBOE-style):  weight = 1 / K^2     (varies per strike)
    Computing them together avoids re-fitting the smile or re-pricing the grid twice."""
    if tau <= 0 or len(observed_m) == 0:
        return np.nan, np.nan
    half_width = grid_multiplier * max(np.abs(observed_m).max(), 1e-3)
    half_width = min(half_width, 3.0)  # cap the extrapolation -- see module docstring
    m_grid = np.linspace(-half_width, half_width, n_grid_points)
    sigma_grid = _fitted_iv(coeffs, m_grid, tau)
    strikes = forward * np.exp(m_grid)
    otm = np.array([
        _bs76_price(forward, k, tau, s, r, 'put' if k < forward else 'call')
        for k, s in zip(strikes, sigma_grid)
    ])
    d_k = np.empty(len(strikes))
    d_k[1:-1] = (strikes[2:] - strikes[:-2]) / 2
    d_k[0] = strikes[1] - strikes[0]
    d_k[-1] = strikes[-1] - strikes[-2]
    svix_integral = float(np.sum(d_k * otm))
    vix_integral = float(np.sum(d_k * otm / strikes ** 2))
    svix_sq = (2 * np.exp(r * tau) / (spot ** 2 * tau)) * svix_integral
    vix_sq = (2 * np.exp(r * tau) / tau) * vix_integral
    return svix_sq, vix_sq


def _variance_swap_squared_raw(group: pd.DataFrame, spot: float, tau: float, forward: float, r: float) -> tuple:
    """Fallback (no smile smoothing): integrate raw listed-strike prices directly. Returns
    (svix_sq, vix_sq), same as the smoothed version."""
    calls, puts = _call_put_by_strike(group)
    strikes = np.sort(pd.Index(calls.index).union(puts.index).unique())
    if len(strikes) < 2:
        return np.nan, np.nan
    otm = np.full(len(strikes), np.nan)
    for i, k in enumerate(strikes):
        if k < forward:
            otm[i] = puts.get(k, np.nan)
        elif k > forward:
            otm[i] = calls.get(k, np.nan)
        else:
            vals = [v for v in (calls.get(k, np.nan), puts.get(k, np.nan)) if pd.notna(v)]
            otm[i] = np.mean(vals) if vals else np.nan
    valid = ~np.isnan(otm)
    strikes, otm = strikes[valid], otm[valid]
    if len(strikes) < 2:
        return np.nan, np.nan
    d_k = np.empty(len(strikes))
    d_k[1:-1] = (strikes[2:] - strikes[:-2]) / 2
    d_k[0] = strikes[1] - strikes[0]
    d_k[-1] = strikes[-1] - strikes[-2]
    svix_integral = float(np.sum(d_k * otm))
    vix_integral = float(np.sum(d_k * otm / strikes ** 2))
    svix_sq = (2 * np.exp(r * tau) / (spot ** 2 * tau)) * svix_integral
    vix_sq = (2 * np.exp(r * tau) / tau) * vix_integral
    return svix_sq, vix_sq


def _interpolate_to_target(tau_data: pd.DataFrame, target_tau: float, value_col: str) -> float:
    tau_data = tau_data.dropna(subset=[value_col]).sort_values('tau')
    if tau_data.empty:
        return np.nan
    taus = tau_data['tau'].to_numpy()
    sq_vals = tau_data[value_col].to_numpy()
    if target_tau <= taus[0]:
        return float(np.sqrt(sq_vals[0]))
    if target_tau >= taus[-1]:
        return float(np.sqrt(sq_vals[-1]))
    i = int(np.searchsorted(taus, target_tau))
    tau1, tau2 = taus[i - 1], taus[i]
    var1, var2 = sq_vals[i - 1] * tau1, sq_vals[i] * tau2
    w = (target_tau - tau1) / (tau2 - tau1)
    total_var = (1 - w) * var1 + w * var2
    return float(np.sqrt(total_var / target_tau))


def compute_svix(option_chain_data: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame],
                 target_maturities: Dict[str, int] = None,
                 risk_free_rate: float = 0.0, day_count: float = 360.0,
                 smile_smoothing: bool = True, iv_column: Optional[str] = 'mark_iv',
                 n_grid_points: int = 200, grid_multiplier: float = 1.5) -> pl.DataFrame:
    target_maturities = target_maturities or {'1D': 1, '1W': 7, '1M': 30}
    if isinstance(option_chain_data, pl.LazyFrame):
        df = option_chain_data.collect().to_pandas()
    elif isinstance(option_chain_data, pl.DataFrame):
        df = option_chain_data.to_pandas()
    else:
        df = option_chain_data.copy()
    if 'strike' not in df.columns:
        df = df.assign(strike=df['underlying_price'] * np.exp(-df['moneyness']))
    if iv_column is not None and iv_column not in df.columns:
        iv_column = None
    per_expiry = []
    for (underlying_name, timestamp), snapshot in df.groupby(['underlying_name', 'timestamp']):
        spot = float(snapshot['underlying_price'].iloc[0])
        taus = sorted(snapshot['tau'].unique())
        forwards_by_tau = {tau: _forward_price(snapshot[snapshot['tau'] == tau], spot, tau, risk_free_rate)
                           for tau in taus}
        coeffs = _fit_dumas_surface(snapshot, forwards_by_tau, risk_free_rate, iv_column) \
            if smile_smoothing else None
        for tau in taus:
            group = snapshot[snapshot['tau'] == tau]
            forward = forwards_by_tau[tau]
            if coeffs is not None:
                observed_m = np.log(group['strike'].to_numpy() / forward)
                svix_sq, vix_sq = _variance_swap_squared_smoothed(tau, forward, spot, risk_free_rate, coeffs,
                                                                  observed_m, n_grid_points, grid_multiplier)
            else:
                svix_sq, vix_sq = _variance_swap_squared_raw(group, spot, tau, forward, risk_free_rate)
            per_expiry.append((underlying_name, timestamp, tau, svix_sq, vix_sq))
    per_expiry = pd.DataFrame(per_expiry, columns=['underlying_name', 'timestamp', 'tau', 'svix_sq', 'vix_sq'])
    rows = []
    for (underlying_name, timestamp), group in per_expiry.groupby(['underlying_name', 'timestamp']):
        for label, days in target_maturities.items():
            target_tau = days / day_count
            svix = _interpolate_to_target(group, target_tau, 'svix_sq')
            vix = _interpolate_to_target(group, target_tau, 'vix_sq')
            rows.append((label, timestamp, underlying_name, svix, vix))
    result = pd.DataFrame(rows, columns=['D', 'timestamp', 'underlying_name', 'svix', 'vix'])
    result['aggregated'] = 0
    agg = result.copy()
    agg['svix'] = agg.groupby(['D', 'timestamp'])['svix'].transform('mean')
    agg['vix'] = agg.groupby(['D', 'timestamp'])['vix'].transform('mean')
    agg['aggregated'] = 1
    out = pd.concat([result, agg], ignore_index=True)[['D', 'timestamp', 'svix', 'vix', 'underlying_name', 'aggregated']]
    return pl.from_pandas(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute SVIX from raw option chain files.')
    parser.add_argument('--target_path', type=str, default='./data_centre')
    parser.add_argument('--underlying_names', type=str, nargs='+',
                        default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT'])
    parser.add_argument('--start_date', type=str, required=True)
    parser.add_argument('--end_date', type=str, required=True)
    parser.add_argument('--resample_freq', type=str, default='15m')
    args = parser.parse_args()
    REQUIRED_COLUMNS = ['symbol', 'timestamp', 'type', 'strike_price', 'expiration', 'underlying_price', 'mid_price']
    IV_COLUMN = 'mark_iv'
    date_range = pd.date_range(args.start_date, args.end_date, freq='1D', inclusive='left')
    frames = []
    for underlying_name in args.underlying_names:
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            file_path = f'{args.target_path}/{underlying_name}/{date.year}/options/{date_str}'
            if not Path(file_path).exists():
                print(f"WARNING: no option chain file for {underlying_name} on {date_str} ({file_path}) -- skipping.")
                continue
            raw = pl.scan_parquet(file_path)
            schema_names = raw.collect_schema().names()
            missing = [c for c in REQUIRED_COLUMNS if c not in schema_names]
            if missing:
                print(f"WARNING: {file_path} is missing required columns {missing} -- skipping.")
                continue
            has_iv = IV_COLUMN in schema_names
            raw = raw.filter(pl.col('mid_price').is_not_null())
            raw = raw.with_columns([
                pl.col('strike_price').alias('strike'),
                ((pl.col('expiration') - pl.col('timestamp')).dt.total_seconds() / (86400 * 360)).alias('tau'),
            ])
            agg_exprs = [pl.col('type').last(), pl.col('strike').last(), pl.col('tau').last(),
                         pl.col('underlying_price').last(), pl.col('mid_price').last()]
            if has_iv:
                agg_exprs.append(pl.col(IV_COLUMN).last())
            raw = raw.sort(['symbol', 'timestamp'])
            raw = raw.group_by_dynamic(index_column='timestamp', every=args.resample_freq,
                                       group_by='symbol').agg(agg_exprs)
            raw = raw.filter(pl.col('tau') > 0)
            raw = raw.with_columns(pl.lit(underlying_name).alias('underlying_name'))
            if not has_iv:
                raw = raw.with_columns(pl.lit(None, dtype=pl.Float64).alias(IV_COLUMN))
            frames.append(raw)
    if not frames:
        raise FileNotFoundError(
            f'No usable option chain files found for underlying_names={args.underlying_names} '
            f'between {args.start_date} and {args.end_date} under {args.target_path}')
    option_chain_data = pl.concat(frames)
    n_quotes = option_chain_data.select(pl.len()).collect().item()
    print(f'Loaded {n_quotes} resampled quotes across {len(frames)} (underlying_name, date) file(s).')
    svix = compute_svix(option_chain_data, target_maturities={'1D': 1, '1W': 7, '1M': 30}, iv_column=IV_COLUMN)
    svix = svix.with_columns([pl.when(pl.col('aggregated') == 1).then(pl.col('svix').mean().over(['D', 'timestamp']))
                             .otherwise(pl.col('svix')).alias('svix'),
                              pl.when(pl.col('aggregated') == 1).then(pl.col('vix').mean().over(['D', 'timestamp']))
                             .otherwise(pl.col('vix')).alias('vix'),
                              pl.col('D').str.to_lowercase()])
    svix = svix.to_pandas()
    out_path = f'{args.target_path}/crypto_option_spot/svix.parquet'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    svix.to_parquet(out_path)
    print(f'\nSaved to {out_path}')