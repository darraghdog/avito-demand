# https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time, gc
from typing import List, Dict
from sklearn import metrics

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
start_time = time.time()
#path = '../input/'
path = "/home/darragh/avito/data/"
path = '/Users/dhanley2/Documents/avito/data/'

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
def load_data():
    print('[{}] Load Train/Test'.format(time.time() - start_time))
    traindf = pd.read_csv(path + 'train.csv.zip', index_col = "item_id", parse_dates = ["activation_date"], compression = 'zip')
    traindex = traindf.index
    testdf = pd.read_csv(path + 'test.csv.zip', index_col = "item_id", parse_dates = ["activation_date"])
    testdex = testdf.index
    y = traindf.deal_probability.copy()
    traindf.drop("deal_probability",axis=1, inplace=True)
    print('Train shape: {} Rows, {} Columns'.format(*traindf.shape))
    print('Test shape: {} Rows, {} Columns'.format(*testdf.shape))
    traindf['activation_date'].value_counts()
    traindf.head()
    
    print('[{}] Create Validation Index'.format(time.time() - start_time))
    trnidx = (traindf.activation_date<=pd.to_datetime('2017-03-26')).values
    validx = (traindf.activation_date>=pd.to_datetime('2017-03-27')).values
    
    print('[{}] Combine Train and Test'.format(time.time() - start_time))
    df = pd.concat([traindf,testdf],axis=0)
    df['idx'] = range(df.shape[0])
    del traindf,testdf
    gc.collect()
    print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))  
    
    dtrain = df.loc[traindex,:][trnidx].reset_index()
    dvalid = df.loc[traindex,:][validx].reset_index()
    return dtrain, dvalid, y[trnidx], y[validx]


def common_users(varin, col, cutoff = 5):
    dft         = pd.DataFrame(varin.values, columns = [col])
    gp          = dft[col].value_counts().reset_index().rename(columns = {'index':col, col:col+'_ct'})
    var         = dft[[col]].merge(gp, on = col, how = 'left')[col+'_ct']
    idx         = var>cutoff
    var[idx]    = (varin.values)[idx]
    var[~idx]    = ''
    return var.astype(str).values

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name']      = df['title'].fillna('') 
    df['text']      = (df['description'].fillna('') + ' ' + df['title'] + ' ' \
      + df['parent_category_name'].fillna('') + ' ' + df['category_name'].fillna('') \
      + df['param_1'].fillna('') + df['param_2'].fillna('') + df['param_3'].fillna(''))
    df['price_cut'] = (np.log1p(df['price'])*5).fillna(150).astype(np.int32)
    df['item_seq_number_cut'] = (np.log1p(df['item_seq_number'])*5).fillna(150).astype(np.int32)
    df['user']   = common_users(df['user_id'], 'user_id')
    df['image_top_1'] = df['image_top_1'].fillna(3100).astype(str)
    return df[['name', 'text', 'user', 'region', 'city', 'item_seq_number_cut' \
               , 'user_type', 'price_cut', 'image_top_1']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=1e-3))#(lr=3e-3))
        for i in range(1):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        return model.predict(X_test)[:, 0]

def main():
    vectorizer = make_union(
        on_field('name', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('user', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['region', 'city', 'user_type', 'price_cut', 'item_seq_number_cut', 'image_top_1'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
    y_scaler = StandardScaler()
    with timer('process train'):
        train, valid, y_train, y_valid = load_data()
        y_train = y_train.values
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        del train
    with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    with ThreadPool(processes=4) as pool:
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        y_pred = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)
    print('Valid RMSE: {:.4f}'.format(np.sqrt(metrics.mean_squared_error(y_valid.values, y_pred))) )

if __name__ == '__main__':
    main()
    
# Valid RMSE: 0.2212 - 1 epochs (lr=2e-3))
# [fit_predict] done in 1240 s