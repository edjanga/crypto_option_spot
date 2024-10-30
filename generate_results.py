from model.lab import Heston
import glob
import pandas as pd
import argparse


models = [Heston]
models_factory_dd = {model.__name__.lower(): model for model in models}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script generating all results presented in the associated paper.\n'
                                                 'Results are computed model by model.')
    parser.add_argument('--model', type=str, help='Model to be trained.')
    parser.add_argument('--L', type=str, help='Rolling window to be used in the training procedure.')
    args = parser.parse_args()
    files = [file for file in glob.glob('../data_centre/tmp/options/*-*-*')]
    files.sort()
    data = pd.concat([pd.read_parquet(option_chain) for option_chain in files[:10]])
    data.reset_index(inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
    data.loc[:, 'tau'] = data['expiration'].subtract(data['timestamp']).div(pd.to_timedelta('360D'))
    model_obj = models_factory_dd[args](L=args.L)