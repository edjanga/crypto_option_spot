import pdb

import numpy as np
NUM_BOOST_ROUND = 50
STOPPING_ROUNDS = 3
MAX_ITER = 10
N_TRIALS = 3
N_JOBS = 1
EPOCH = 100
EARLY_STOPPING_PATIENCE = 5
BUCKET = 24
DATE_FORMAT = '%Y-%m-%d'
L_TRAIN_CONVERTER_DAYS = {'1W': 7, '1M': 30, '6M': 180}
# Maturity tags
zeroDTE, SHORT, MID = 1/360, 7/360, 30/360
DOTM, OTM, ITM, DITM = .925, .975, 1.025, 1.075#-.3, -.1, .1, .3
MONEYNESS_BINS = [-np.inf, DOTM, OTM, ITM, DITM, np.inf]
MONEYNESS_BINS[1:-1] = list(map(lambda x: np.log(x), MONEYNESS_BINS[1:-1]))
LABELS = ['DOTM', 'OTM', 'ATM', 'ITM', 'DITM']
SUPERSCRIPT = {1: 'st', 2: 'nd', 3: 'rd'}
EXPOSURE = ['dex', 'gex', 'vax', 'vex', 'vox']