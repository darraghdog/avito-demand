# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
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
from sklearn.linear_model import Ridge
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K

from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated

#path = '../input/'
path = "/home/darragh/avito/data/"
path = '/Users/dhanley2/Documents/avito/data/'
path = '/home/ubuntu/avito/data/'


start_time = time.time()
full = False
validation = False

print('[{}] Load Train/Test'.format(time.time() - start_time))
usecols = ['item_id', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description']
traindf = pd.read_csv(path + 'train.csv.zip', index_col = "item_id", usecols = usecols, compression = 'zip') # , parse_dates = ["activation_date"]
traindex = traindf.index
testdf = pd.read_csv(path + 'test.csv.zip', index_col = "item_id", usecols = usecols[:-1]) #, parse_dates = ["activation_date"]
testdex = testdf.index
print('Train shape: {} Rows, {} Columns'.format(*traindf.shape))
print('Test shape: {} Rows, {} Columns'.format(*testdf.shape))

print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = traindf # pd.concat([traindf,testdf],axis=0)
del traindf,testdf
gc.collect()
df['idx'] = range(df.shape[0])
df.fillna('', inplace = True)
df.head()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

def expand_me(ls):
    return np.expand_dims(np.array(ls), axis = 1)

def translate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    text = TextBlob(comment)
    try:
        text = text.translate(to=language)
        text = text.translate(to="ru")
    except NotTranslated:
        pass

    return str(text)

parallel = Parallel(16, backend="threading", verbose=5)

col = 'category_name'
def translate_col(df, col, language = 'en'):
    comments_list = df[col].unique()
    translated_data = parallel(delayed(translate)(comment, language) for comment in comments_list)
    trdf = pd.DataFrame(np.concatenate((expand_me(comments_list), expand_me(translated_data)), axis = 1), columns = [col, col+'_translated'])
    df = df.reset_index().merge(trdf, on = col).sort_values('idx').set_index('item_id')
    return df

# df = df[:100]
for col in ['parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title', 'description']:
    print('-'*50)
    print('[{}]'.format(time.time() - start_time))
    print('Translating %s'%(col))
    print('-'*50)
    df = translate_col(df, col, language = 'en')
    df[[c for c in df.columns if 'translate' in c]].to_pickle(path+'../features/translate_en1.pkl')

df[[c for c in df.columns if 'translate' in c]].to_csv(path+'../features/translate_en1.csv')



