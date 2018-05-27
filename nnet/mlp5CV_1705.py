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
    featusrttl = pd.read_csv(path + '../features/user_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
    featusrcat = pd.read_csv(path + '../features/usercat_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
    featusrcat.head()
    y = traindf.deal_probability.copy()
    traindf.drop("deal_probability",axis=1, inplace=True)
    print('Train shape: {} Rows, {} Columns'.format(*traindf.shape))
    print('Test shape: {} Rows, {} Columns'.format(*testdf.shape))
    
    print('[{}] Combine Train and Test'.format(time.time() - start_time))
    df = pd.concat([traindf,testdf],axis=0)
    df['idx'] = range(df.shape[0])
    featusrttl.rename(columns={'title': 'all_titles'}, inplace = True)
    keep = ['user_id', 'all_titles', 'user_avg_price', 'user_ad_ct']
    df = df.reset_index().merge(featusrttl[keep], on = 'user_id').set_index('item_id')
    keep = ['user_id', 'parent_category_name', 'usercat_avg_price', 'usercat_ad_ct']
    gc.collect()
    df = df.reset_index().merge(featusrcat[keep], on = ['user_id', 'parent_category_name']).set_index('item_id')
    df.head(2)
    print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))  
    
    dtrain = df.loc[traindex,:].reset_index()
    dtest  = df.loc[testdex,:].reset_index()
    
    return dtrain, dtest, y, traindex, testdex

def common_users(varin, col, cutoff = 3):
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
    df['price_cut'] = 'pc_'+(np.log1p(df['price'])*5).fillna(150).astype(np.int32).astype(str)
    df['user_avg_price_cut'] = 'apc_'+(np.log1p(df['user_avg_price'])*5).fillna(10000).astype(np.int32).astype(str)
    df['usercat_avg_price_cut'] = 'capc_'+(np.log1p(df['usercat_avg_price'])*5).fillna(10000).astype(np.int32).astype(str)
    df['user_ad_ct'] = 'uac_'+(np.log1p(df['user_ad_ct'])*5).fillna(10000).astype(np.int32).astype(str)
    df['usercat_ad_ct'] = 'ucac_'+(np.log1p(df['usercat_ad_ct'])*5).fillna(10000).astype(np.int32).astype(str)
    df['item_seq_number_cut'] = 'is_'+(np.log1p(df['item_seq_number'])*5).fillna(150).astype(np.int32).astype(str)
    df['user']    = common_users(df['user_id'], 'user_id')
    df['image_top_1'] = 'img_'+ df['image_top_1'].fillna(3100).astype(str)
    df['region']  = 'rgn_'+ df['region'].fillna('').astype(str)
    df['city']    = df['city'].fillna('').astype(str)
    df['param_1'] = 'p1_'+ df['param_1'].fillna('').astype(str)
    df['param_2'] = 'p2_'+ df['param_2'].fillna('').astype(str)
    df['param_3'] = 'p3_'+ df['param_3'].fillna('').astype(str)
    df['parent_category_name'] = 'pc1_'+ df['parent_category_name'].fillna('').astype(str)
    df['category_name'] = 'c2_'+ df['category_name'].fillna('').astype(str)
    return df[['name', 'text', 'user', 'region', 'city', 'item_seq_number_cut', 'all_titles', 'user_avg_price_cut'\
               , 'user_type', 'price_cut', 'image_top_1', 'param_1', 'param_3', 'param_3', 'user_ad_ct'\
               , 'usercat_avg_price_cut', 'usercat_ad_ct']]

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
        #out = ks.layers.Dense(512, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=1e-3))#(lr=1e-3))
        for i in range(1):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=1)
        return model.predict(X_test)[:, 0]

def main(folds):
    folds = trndf['fold'].values
    def init_preprocesses():
        vectorizer = make_union(
            on_field('name',       Tfidf(max_features=15000 , token_pattern='\w+')), #100000
            on_field('all_titles', Tfidf(max_features=80000 , token_pattern='\w+')), #100000
            #on_field('user_categories', Tfidf(max_features=10000 , token_pattern='\w+')), #100000
            on_field('text',       Tfidf(max_features=60000, token_pattern='\w+', ngram_range=(1, 2))), #100000
            on_field(['region', 'city', 'price_cut', 'item_seq_number_cut', 'image_top_1', 'user_avg_price_cut', \
                      'param_1', 'param_3', 'param_3', 'user_type', 'user', 'user_ad_ct', 'usercat_avg_price_cut', 'usercat_ad_ct'],
                     FunctionTransformer(to_records, validate=False), DictVectorizer()),
            n_jobs=4)
        y_scaler = StandardScaler()
        return vectorizer, y_scaler    
    train, test, y, trndex, tstdex = load_data()
    train.set_index('item_id', inplace = True)
    test.set_index('item_id', inplace = True)
        #y_train = y.values
    with timer('modelling'):
        #y_pred_trn = pd.Series(-np.zeros(len(trndex)))
        train.head()
        #y_pred_tst = pd.Series(-np.zeros(len(tstdex)))
        y_pred_trn = pd.Series(-np.zeros(trndf.shape[0]), index = traindex)
        y_pred_tst = pd.Series(-np.zeros(tstdf.shape[0]), index = testdex)
        for f in range(6):
            print('Fold %s'%(f) + ' [{}] Modeling Stage'.format(time.time() - start_time))
            vectorizer, y_scaler = init_preprocesses()
            with timer('process train'):
                trnidx = traindex[(trndf['fold'] != f).values]
                X_trn = vectorizer.fit_transform(preprocess(train.loc[trnidx,:])).astype(np.float32)
                y_train = y[trnidx].values
            with timer('process test'):
                if f == 5:
                    X_tst = vectorizer.transform(preprocess(test)).astype(np.float32)
                else:
                    tstidx = traindex[(trndf['fold'] == f).values]
                    X_tst = vectorizer.transform(preprocess(train.loc[tstidx,:])).astype(np.float32)
                    y_test = y[tstidx].values
            print(f'X_train: {X_trn.shape} of {X_trn.dtype}')                      
            with ThreadPool(processes=4) as pool:
                Xb_trn, Xb_tst = [x.astype(np.bool).astype(np.float32) for x in [X_trn, X_tst]]
                xs = [[Xb_trn, Xb_tst], [X_trn, X_tst]] * 2
                y_pred = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)
            print('Fold %s'%(f) + ' [{}] Modeling Evaluation'.format(time.time() - start_time))
            if f == 5:
                y_pred_tst[:] = y_pred
            else:
                y_pred_trn[tstidx] = y_pred
                print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_trn[tstidx])))
            del X_trn, X_tst, y_train, y_test
            gc.collect()
    return y_pred_trn, y_pred_tst

if __name__ == '__main__':
    print('[{}] Create folds'.format(time.time() - start_time))
    foldls = [["2017-03-15", "2017-03-16", "2017-03-17"], \
           ["2017-03-18", "2017-03-19", "2017-03-20"], \
           ["2017-03-21", "2017-03-22", "2017-03-23"], \
           ["2017-03-24", "2017-03-25", "2017-03-26"], \
            ["2017-03-27", "2017-03-28", "2017-03-29", \
                "2017-03-30", "2017-03-31", "2017-04-01", \
                "2017-04-02", "2017-04-03","2017-04-07"]]
    uc = [ "item_id", "activation_date"]
    trndf = pd.read_csv(path + 'train.csv.zip', index_col = "item_id", usecols = uc, parse_dates = ["activation_date"], compression = 'zip')
    traindex = trndf.index
    tstdf = pd.read_csv(path + 'test.csv.zip', index_col = "item_id", usecols = uc, parse_dates = ["activation_date"], compression = 'zip')
    testdex = tstdf.index
    foldls = [[pd.to_datetime(d) for d in f] for f in foldls]
    trndf['fold'] = -1
    for t, fold in enumerate(foldls):
        trndf['fold'][trndf.activation_date.isin(fold)] = t
    trndf['fold'].value_counts()
    y_pred_trn, y_pred_tst = main(trndf['fold'].values)
    
    mlpsub = pd.concat([y_pred_trn, y_pred_tst]).reset_index()
    mlpsub.rename(columns = {0 : 'deal_probability'}, inplace=True)
    mlpsub['deal_probability'].clip(0.0, 1.0, inplace=True)
    mlpsub.set_index('item_id', inplace = True)
    print('RMSE for all :', np.sqrt(metrics.mean_squared_error(y, mlpsub.loc[traindex])))
    # RMSE for all : 0.21875
    mlpsub.to_csv("../sub/mlpCV_2505.csv.gz",index=True,header=True, compression = 'gzip')
    
# Valid RMSE: 0.2166 ... all_titles bigram 60K
# Valid RMSE: 0.2165 ... avg price per user
# Valid RMSE: 0.2163 ... user Ad ct
# Valid RMSE: 0.2162 ... Parent category average price, count