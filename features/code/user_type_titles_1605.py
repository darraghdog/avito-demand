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

usecols = 'user_id,title,user_type'.split(',')

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

print('[{}] Add counts per user_id'.format(time.time() - start_time))
gp =  df[['user_id', 'user_type', 'title']].groupby(['user_id', 'user_type']).count().reset_index().rename(columns={'title':'user_id_type_ct'})
# Keep only the privates with more than one add
gp = gp[(gp['user_type']=='Private') & (gp['user_id_type_ct']>1)].sort_values('user_id_type_ct')
gp.head()

print('[{}] Group title by user'.format(time.time() - start_time))
df['title'].fillna('', inplace = True)
ttldf = pd.DataFrame(df[df['user_id'].isin(gp['user_id'])].groupby(['user_id'])['title'].apply(lambda x: ' '.join(x)))

print('[{}] Write out to features'.format(time.time() - start_time))
ttldf.to_csv(path + '../features/user_private_titles.csv.gz', compression = 'gzip')



print('[{}] Count of cities and regions by user'.format(time.time() - start_time))
del df
gc.collect()
usecols = 'user_id,region,city'.split(',')

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

print('[{}] Add counts per region, city'.format(time.time() - start_time))
gpreg =  df[['user_id', 'region']].groupby(['user_id'])['region'].nunique().reset_index().rename(columns={'region':'user_unq_region'})
gpcty =  df[['user_id', 'city'  ]].groupby(['user_id'])['city'  ].nunique().reset_index().rename(columns={'city'  :'user_unq_city'})
gp    = gpreg.merge(gpcty, on = 'user_id').set_index('user_id')
gp.head()
gp.to_csv(path + '../features/user_geo_counts.csv.gz', compression = 'gzip')
