
# https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lgbm-clone?scriptVersionId=3540839

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

validation = False

usecols = 'item_id,user_id,region,city,parent_category_name,category_name,param_1,param_2,param_3'.split(',')

print('[{}] Load Train/Test'.format(time.time() - start_time))
trndf = dd.read_csv(path + "train_active.csv", usecols = usecols, quoting=csv.QUOTE_NONE)
tstdf = dd.read_csv(path + "test_active.csv", usecols = usecols, quoting=csv.QUOTE_NONE)

pd.DataFrame(trndf['user_id'].value_counts())

