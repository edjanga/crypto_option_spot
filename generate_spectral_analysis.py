import argparse
from model.lab import ForecastIVS
from data_centre.data import Reader
import glob
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating plot of spectral analysis.')
    reader_obj = Reader()
    underlying_price = reader_obj.prices_read().resample('5min').last()
    files = [file for file in glob.glob('../data_centre/tmp/options/*-*-*')]
    files.sort()
    data = pd.concat([pd.read_parquet(option_chain) for option_chain in files])
    data.reset_index(inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
    data = data.set_index('timestamp').join(underlying_price, how='left')
    data.reset_index(inplace=True)
    data.drop('underlying_price', axis=1, inplace=True)
    data = data.rename(columns={'ETHUSDT': 'underlying_price'})
    data.loc[:, 'tau'] = data['expiration'].subtract(data['timestamp']).div(pd.to_timedelta('1D'))
    latent_features_obj = ForecastIVS(L='1W', h='1H', symbol='ETHUSDT', option_type='call')
    for normalise in [False, True]:
        latent_features_obj.spectral_analysis(mkt_data=data, normalise=normalise)