# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import time, gc
import pandas as pd
import numpy as np
from sklearn import preprocessing
from nltk.corpus import stopwords 
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
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

print('[{}] Load Train/Test'.format(time.time() - start_time))
date_cols = ['activation_date', 'date_from', 'date_to']
trnpdf = pd.read_csv(path + "periods_train.csv.zip", compression = 'zip', parse_dates = date_cols)
tstpdf = pd.read_csv(path + "periods_test.csv.zip", compression = 'zip', parse_dates = date_cols)

print('[{}] Load Train/Test'.format(time.time() - start_time))
traindf = pd.read_csv(path + 'train.csv.zip', index_col = "item_id", parse_dates = ["activation_date"], compression = 'zip')
traindex = traindf.index
testdf = pd.read_csv(path + 'test.csv.zip', index_col = "item_id", parse_dates = ["activation_date"])
testdex = testdf.index

traindf.loc[trnpdf['item_id'][:100].values]

pd.merge(traindf.reset_index(), trnpdf, on = ['item_id'])

pdf    = pd.concat([trnpdf, tstpdf], axis = 0)
del trnpdf, tstpdf
gc.collect()
pdf.dtypes
pdf.sort_values(['item_id', 'activation_date'], inplace = True)


pdf.head()