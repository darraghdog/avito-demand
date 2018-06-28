# https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lgbm-clone?scriptVersionId=3540839
# Models Packages
import time, gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import dask
import dask.dataframe as dd
import csv


#path = '../input/'
path = "/home/darragh/avito/data/"
path = '/Users/dhanley2/Documents/avito/data/'

# path = '/home/ubuntu/avito/data/'
start_time = time.time()

validation = False

usecols = 'user_id,title'.split(',')

print('[{}] Load Train/Test'.format(time.time() - start_time))
trnadf = pd.read_csv(path + "train_active.csv", usecols = usecols)
tstadf = pd.read_csv(path + "test_active.csv", usecols = usecols)
trndf = pd.read_csv(path + "train.csv", usecols = usecols)
tstdf = pd.read_csv(path + "test.csv", usecols = usecols)

print('[{}] Concat dataframes and filter'.format(time.time() - start_time))
alladf = pd.concat([trnadf, tstadf], axis = 0)
alldf  = pd.concat([trndf , tstdf], axis = 0)
del trnadf, tstadf, trndf, tstdf
gc.collect()
alladffilt = alladf[alladf['user_id'].isin(alldf['user_id'])] # Filter on users in train or test
df = pd.concat([alldf, alladffilt], axis = 0)
del alldf, alladf, alladffilt
gc.collect()

print('[{}] Group title by user'.format(time.time() - start_time))
df['title'].fillna('', inplace = True)
ttldf = df.groupby(['user_id'])['title'].apply(lambda x: ' '.join(x))
ctdf = df.groupby(['user_id']).size().to_frame('user_ad_ct')
usrdf = pd.concat([ttldf, ctdf], axis=1)

print('[{}] Write out to features'.format(time.time() - start_time))
usrdf.to_csv(path + '../features/user_titles.csv.gz', compression = 'gzip')