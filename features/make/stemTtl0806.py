#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 12:48:20 2018

@author: darragh
"""

# https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lgbm-clone?scriptVersionId=3540839

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import time, gc
import pandas as pd
import numpy as np
from sklearn import preprocessing
from nltk.corpus import stopwords 
from nltk import SnowballStemmer

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import pymorphy2
import nltk, re
from nltk.tokenize import ToktokTokenizer
from multiprocessing import cpu_count, Pool


#path = '../input/'
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'
#path = '/home/ubuntu/avito/data/'
start_time = time.time()
full = False

print('[{}] Load Train/Test'.format(time.time() - start_time))
use_cols = ['title', 'item_id']
traindf = pd.read_csv(path + 'train.csv.zip', index_col = "item_id", usecols = use_cols, compression = 'zip')
testdf = pd.read_csv(path + 'test.csv.zip', index_col = "item_id", usecols = use_cols)
trainadf = pd.read_csv(path + 'train_active.csv.zip', index_col = "item_id", usecols = use_cols, compression = 'zip')
testadf = pd.read_csv(path + 'test_active.csv.zip', index_col = "item_id", usecols = use_cols)
print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
dfa = pd.concat([trainadf,testadf],axis=0)
del traindf,testdf
del trainadf,testadf
gc.collect()


stopw_r = [word for word in nltk.corpus.stopwords.words("russian") if word!="не"]
stopw_e = [word for word in nltk.corpus.stopwords.words("english")]
stopw = stopw_r + stopw_e
stem_r = SnowballStemmer('russian')
stem_e = SnowballStemmer('english')

def str_stem(s):
    s = re.sub(u'[^a-zа-я0-9]', ' ', str(s).lower())
    s = (' ').join([stem_r.stem(w) if not re.search("[0-9a-z]", w) else stem_e.stem(w) for w in s.split() if len(w)>1 and w not in stopw])
    return s

partitions = 8
def parallelize(data, func):
    data_split = np.array_split(data.values, partitions)
    pool = Pool(partitions)
    data = pd.concat([pd.Series(l) for l in pool.map(func, data_split)]).values
    pool.close()
    pool.join()
    return data

# Stem the whole lot
all_titles = np.concatenate((df['title'].unique(),dfa['title'].unique()))
partitions = 8
pool = Pool(partitions)
all_titles_stem = pool.map(str_stem, all_titles)
pool.close()
pool.join()

# Concatenate the titles and stemmed titles
dfttl = pd.DataFrame(all_titles)
dfttl.columns = ['title']
dfttl['title_stem'] = all_titles_stem
dfttl.head()

# Join the stemmed title
df['idx'] = range(df.shape[0])
dfa['idx'] = range(dfa.shape[0])
df = df.reset_index('item_id').merge(dfttl, on = ['title'], how = 'left').set_index('item_id').drop_duplicates()
dfa = dfa.reset_index('item_id').merge(dfttl, on = ['title'], how = 'left').set_index('item_id').drop_duplicates()
gc.collect()

df.sort_values('idx', inplace = True)
dfa.sort_values('idx', inplace = True)

df[['title_stem']].to_csv(path + '../features/title_stem.csv')
dfa[['title_stem']].to_csv(path + '../features/title_stem_active.csv')
df.head()

