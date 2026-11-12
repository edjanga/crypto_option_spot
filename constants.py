import numpy as np
import torch
NUM_BOOST_ROUND = 1_000
STOPPING_ROUNDS = 3
MAX_ITER = 250
N_TRIALS = 100
N_JOBS = 1
EPOCH = 50
EARLY_STOPPING_PATIENCE = 10
DATE_FORMAT = '%Y-%m-%d'
DAYS_CONVERTER = {'1d': 1, '1D': 1, '1w': 7, '1W': 7, '1m': 30, '1M': 30, '3M': 90,}
FREQUENCY_CONVERTER = {'1D': 360, '1W': 52}
zeroDTE, SHORT, MID = 1/360, 7/360, 30/360
MATURITY_BINS = [-np.inf, zeroDTE, SHORT, MID, np.inf]
MATURITY_LABELS = ['0DTE', 'Short', 'Mid', 'Long']
DOTM, OTM, ITM, DITM = .85, .95, 1.05, 1.15
# cut points of the log-moneyness k = log(K/F), F = S e^{r tau} = S since r = 0 (S: underlying spot price); every
# threshold is a log ratio (the two outer ones used not to be logged). The option files store log(S/K): readers use
# k = -moneyness.
MONEYNESS_BINS = list(np.log([DOTM, OTM, ITM, DITM]))
LABELS = ['DOTM', 'OTM', 'ATM', 'ITM', 'DITM']
SUPERSCRIPT = {1: 'st', 2: 'nd', 3: 'rd'}
EXPOSURE = ['gex', 'vax', 'vex', 'vox']
GREEKS_EXPOSURE = {exp: greek for exp, greek in zip(['gex'], ['gamma'])}
GREEKS_EXPOSURE.update({exp: None for exp in EXPOSURE[1:]})
FEES_SPOT = .0003
FEES_OPTIONS = .125
INTRADAY_LAGS = ['15min', '30min', '1h', '1D']
INTERDAY_LAGS = ['1W', '1M']
VOL_REGIMES = ['Low', 'Normal', 'High']
MARKETS = {'returns': 'spot', 'RV': 'options'}
TITLE_FONT_SIZE = 40
LABEL_AXIS_FONT_SIZE = 30
LEGEND_FONT_SIZE = 20
HEIGHT = 900
WIDTH = 1_300
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
LOW_ATM_MONEYNESS = .85
HIGH_ATM_MONEYNESS = 1.15
ATM_MONEYNESSES = \
    np.log(np.linspace(LOW_ATM_MONEYNESS, HIGH_ATM_MONEYNESS, int(100*(HIGH_ATM_MONEYNESS-LOW_ATM_MONEYNESS)+1)))
BINARY_PATTERN = ['retail', 'informed', 'institutions', 'is_']
DEVICE = 'cpu'
if torch.cuda.is_available():
    ENGINE = 'cuda'
elif torch.backends.mps.is_available():
    ENGINE = 'mps'
model_names = {'LGBMRegressor': 'lightgbm', 'SOFTS': 'softs'}
OPTION_PREFIXES = ('gex', 'vax', 'vex', 'vox', 'pcr', 'spread_', 'is_retail', 'is_informed', 'is_institutional')
D_HORIZONS = {'1d': 1, '1w': 7, '1m': 30}