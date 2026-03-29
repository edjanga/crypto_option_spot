import numpy as np
import torch
NUM_BOOST_ROUND = 1_000
STOPPING_ROUNDS = 3
MAX_ITER = 250
N_TRIALS = 100
N_JOBS = 1
EPOCH = 50
EARLY_STOPPING_PATIENCE = 5
DATE_FORMAT = '%Y-%m-%d'
DAYS_CONVERTER = {'1D': 1, '1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 360}
FREQUENCY_CONVERTER = {'1D': 360, '1W': 52}
# Maturity tags
zeroDTE, SHORT, MID = 1/360, 7/360, 30/360
MATURITY_BINS = [-np.inf, zeroDTE, SHORT, MID, np.inf]
MATURITY_LABELS = ['0DTE', 'Short', 'Mid', 'Long']
DOTM, OTM, ITM, DITM = .85, .95, 1.05, 1.15
MONEYNESS_BINS = [DOTM, OTM, ITM, DITM]
MONEYNESS_BINS[1:-1] = list(map(lambda x: np.log(x), MONEYNESS_BINS[1:-1]))
LABELS = ['DOTM', 'OTM', 'ATM', 'ITM', 'DITM']
SUPERSCRIPT = {1: 'st', 2: 'nd', 3: 'rd'}
EXPOSURE = ['gex', 'vax', 'vex', 'vox']
GREEKS_EXPOSURE = {exp: greek for exp, greek in zip(['gex'], ['gamma'])}
GREEKS_EXPOSURE.update({exp: None for exp in EXPOSURE[1:]})
# Transaction costs
FEES_SPOT = .0003
FEES_OPTIONS = .125
# Core lags
INTRADAY_LAGS = ['15min', '30min', '1h', '1D']
INTERDAY_LAGS = ['1W', '1M']
# Volatility regimes
VOL_REGIMES = ['Low', 'Normal', 'High']
# Number of times features get shuffled
N_SHUFFLES = 50
# Params for DL architecture
PARAMS_DL = {'hidden_size': [128, 256, 512], 'num_layers': [1, 2],
             'learning_rate': [1e-3, 5e-3], 'beta': list(np.linspace(1, 10, 10)), 'dropout': [.1, .2],
             'batch_size': [512]}
MARKETS = {'returns': 'spot', 'RV': 'options'}
# Font sizes for plots
TITLE_FONT_SIZE = 40
LABEL_AXIS_FONT_SIZE = 30
LEGEND_FONT_SIZE = 20
HEIGHT = 900
WIDTH = 1_300
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT']
LOW_ATM_MONEYNESS = .85
HIGH_ATM_MONEYNESS = 1.15
ATM_MONEYNESSES = \
    np.log(np.linspace(LOW_ATM_MONEYNESS, HIGH_ATM_MONEYNESS, int(100*(HIGH_ATM_MONEYNESS-LOW_ATM_MONEYNESS)+1)))
BINARY_PATTERN = ['retail', 'informed', 'institutions', 'is_']
ENGINE = 'cuda' if torch.cuda.is_available() else 'cpu'