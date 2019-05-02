"""
This is the script to load experimental records from the pickled log
"""
import argparse
import pickle
from pprint import pprint
parser = argparse.ArgumentParser(description="Load pickled log")
parser.add_argument('--fn')
args = parser.parse_args()

with open(args.fn, 'rb') as fin:
    obj = pickle.load(fin)

pprint('train_acc:{}'.format(obj['train_acc'][-1]))
pprint('val_acc:{}'.format(obj['val_acc'][-1]))
