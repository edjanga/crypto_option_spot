import numpy as np
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
DOTM, OTM, ITM, DITM = .85, .95, 1.05, 1.15
MONEYNESS_BINS = [-np.inf, DOTM, OTM, ITM, DITM, np.inf]
MONEYNESS_BINS[1:-1] = list(map(lambda x: np.log(x), MONEYNESS_BINS[1:-1]))
LABELS = ['DOTM', 'OTM', 'ATM', 'ITM', 'DITM']
SUPERSCRIPT = {1: 'st', 2: 'nd', 3: 'rd'}
EXPOSURE = ['gex', 'vax', 'vex', 'vox']
COLUMNS = ['symbol', 'type', 'strike_price', 'open_interest', 'last_price', 'bid_price', 'bid_amount', 'bid_iv',
           'ask_price', 'ask_amount', 'ask_iv', 'mark_price', 'mark_iv', 'delta', 'gamma', 'vega',
           'theta', 'rho', 'expiration']
GREEKS = ['gamma', 'vega', 'theta', 'rho']
MKT_PARTICIPANTS = ['informed_traders', 'retail', 'institutions']
GREEKS_EXPOSURE = {exp: greek for exp, greek in zip(['gex'], ['gamma'])}
GREEKS_EXPOSURE.update({exp: None for exp in EXPOSURE[1:]})
RESAMPLING = {feature: 'last' for feature in ['strike_price', 'expiration', 'last_price', 'bid_price',
                                              'bid_amount', 'bid_iv', 'ask_price', 'ask_amount', 'ask_iv',
                                              'mark_price', 'mark_iv', 'moneyness']+GREEKS}
RESAMPLING.update({feature: 'sum' for feature in ['volume']+EXPOSURE})
# Transaction costs
FEES_SPOT = .0003
FEES_OPTIONS = .125
# Core lags
INTRADAY_LAGS = ['15min', '30min', '1h', '1D'] #, '3h', '6h', '12h', '1D
INTERDAY_LAGS = ['1W', '1M']
# Volatility regimes
VOL_REGIMES = ['Low', 'Normal', 'High']
# Number of times features get shuffled
N_SHUFFLES = 50
# Params for LSTM
PARAMS_LSTM = {'hidden_size': [2**n for n in range(2, 8)], 'num_layers': list(range(1, 5)),
               'learning_rate': [0.001, 0.005, 0.01]}
MARKETS = {'returns': 'spot', 'RV': 'options'}
OPTION_CHAIN_DATA_COLUMNS = ['option_tag', 'type', 'bid_price', 'ask_price', 'moneyness', 'mark_iv', 'expiration',
                             'bid_amount', 'ask_amount', 'strike_price']
YEARS = [2021, 2022, 2023]
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