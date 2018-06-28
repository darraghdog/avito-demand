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

usecols = 'user_id,title,parent_category_name,category_name,activation_date,price,item_seq_number'.split(',')

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
df['user_categories'] = df['parent_category_name'].fillna('') + df['category_name'].fillna('')
df.drop(['category_name'], 1, inplace = True)
df['title'].fillna('', inplace = True)

print('[{}] User aggregation features'.format(time.time() - start_time))
ttldf = df.groupby(['user_id'])['title'].apply(lambda x: ' '.join(x))
catdf = df.groupby(['user_id'])['user_categories'].apply(lambda x: ' '.join(x))
ctdf  = df.groupby(['user_id']).size().to_frame('user_ad_ct')
avgdf = df.groupby(['user_id'])['price'].mean().to_frame('user_avg_price')
print('[{}] Write out to features'.format(time.time() - start_time))
usrdf = pd.concat([ttldf, ctdf, catdf, avgdf], axis=1)
usrdf.head()
usrdf.to_csv(path + '../features/user_agg.csv.gz', compression = 'gzip')
del ttldf, catdf, ctdf, avgdf, usrdf
gc.collect()

print('[{}] User title category aggregation features'.format(time.time() - start_time))
ttldf = df.groupby(['user_id', 'parent_category_name'])['title'].apply(lambda x: ' '.join(x))
catdf = df.groupby(['user_id', 'parent_category_name'])['user_categories'].apply(lambda x: ' '.join(x))
ctdf  = df.groupby(['user_id', 'parent_category_name']).size().to_frame('user_ad_ct')
avgdf = df.groupby(['user_id', 'parent_category_name'])['price'].mean().to_frame('user_avg_price')
print('[{}] Write out to features'.format(time.time() - start_time))
usrtdf = pd.concat([ttldf, ctdf, catdf, avgdf], axis=1)
usrtdf.rename(columns={'user_categories'    :'usercat_categories',
                       'title'              :'usercat_title',
                       'user_ad_ct'         :'usercat_ad_ct',
                       'user_avg_price'     :'usercat_avg_price'}, inplace = True)
usrtdf.to_csv(path + '../features/usercat_agg.csv.gz', compression = 'gzip')
usrtdf.head()
del ttldf, ctdf, catdf, avgdf
gc.collect()


print('[{}] User title aggregation features'.format(time.time() - start_time))
ttldf = df.groupby(['user_id', 'activation_date'])['title'].apply(lambda x: ' '.join(x))
catdf = df.groupby(['user_id', 'activation_date'])['user_categories'].apply(lambda x: ' '.join(x))
ctdf  = df.groupby(['user_id', 'activation_date']).size().to_frame('user_ad_ct')
avgdf = df.groupby(['user_id', 'activation_date'])['price'].mean().to_frame('user_avg_price')
print('[{}] Write out to features'.format(time.time() - start_time))
usrddf = pd.concat([ttldf, ctdf, catdf, avgdf], axis=1)
usrddf.rename(columns={'user_categories'    :'userday_categories',
                       'title'              :'userday_title',
                       'user_ad_ct'         :'userday_ad_ct',
                       'user_avg_price'     :'userday_avg_price'}, inplace = True)
usrddf.to_csv(path + '../features/userday_agg.csv.gz', compression = 'gzip')
usrddf.head()
del ttldf, ctdf, catdf, avgdf
gc.collect()
