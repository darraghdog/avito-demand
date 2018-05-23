
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
from sklearn.linear_model import Ridge
from math import sqrt
from functools import partial
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline

#path = '../input/'
path = "/home/darragh/avito/data/"
path = '/Users/dhanley2/Documents/avito/data/'


# path = '/home/ubuntu/avito/data/'
start_time = time.time()
full = True

print('[{}] Load Train/Test'.format(time.time() - start_time))
traindf = pd.read_csv(path + 'train.csv.zip', index_col = "item_id", parse_dates = ["activation_date"], compression = 'zip')#, nrows = 100000)
traindex = traindf.index
testdf = pd.read_csv(path + 'test.csv.zip', index_col = "item_id", parse_dates = ["activation_date"])#, nrows = 100000)
testdex = testdf.index
y = traindf.deal_probability.copy()
traindf.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*traindf.shape))
print('Test shape: {} Rows, {} Columns'.format(*testdf.shape))
traindf['activation_date'].value_counts()

print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
active_dates = df['activation_date']

del traindf,testdf
gc.collect()
df['idx'] = range(df.shape[0])
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print('[{}] Load engineered features'.format(time.time() - start_time))
featusrttl = pd.read_csv(path + '../features/user_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdg    = pd.read_csv(path + '../features/ridgeText5CV.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py

featusrttl.rename(columns={'title': 'all_titles'}, inplace = True)
keep = ['user_id', 'all_titles', 'user_avg_price', 'user_ad_ct']
df = df.reset_index().merge(featusrttl[keep], on = 'user_id').set_index('item_id')
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))  

print('[{}] Resort data correctly'.format(time.time() - start_time))
df.sort_values('idx', inplace = True)
df.drop(['idx'], axis=1,inplace=True)
df.reset_index(inplace = True)
df.head()
gc.collect()

for col in df.columns:
    if 'user_activ' in col:
        print(f'fill {col}')
        df[col].fillna(-9,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print('[{}] Text Features'.format(time.time() - start_time))
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features

print('[{}] Text Features'.format(time.time() - start_time))
df['description'].fillna('unknowndescription', inplace=True)
df['title'].fillna('unknowntitle', inplace=True)
df['text']      = (df['description'].fillna('') + ' ' + df['title'] + ' ' + 
  df['parent_category_name'].fillna('').astype(str) + ' ' + df['category_name'].fillna('').astype(str) )

print('[{}] Make Item Seq number as contiuous also'.format(time.time() - start_time))
df["item_seq_number_cont"] = df["item_seq_number"]
df['city'] = df['region'].fillna('').astype(str) + '_' + df['city'].fillna('').astype(str)
df.columns
print('[{}] Encode Variables'.format(time.time() - start_time))
categorical = ["region","parent_category_name","user_type", 'city', 'category_name', \
               "item_seq_number", 'image_top_1', 'user_id', 'price']
print("Encoding :",categorical)
dfcat = df[categorical].copy()
dfcat.head()

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))

print('[{}] [TF-IDF] Term Frequency Inverse Document Frequency Stage'.format(time.time() - start_time))
russian_stop = set(stopwords.words('russian'))
tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "lowercase": True,
    "dtype": np.float32,
    "norm": 'l2',
    "smooth_idf":False
}
countv_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "lowercase": True,
    "min_df": 5 #False
}
def get_col(col_name): return lambda x: x[col_name]
vectorizer1 = FeatureUnion([
        ('text',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=40000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('text_feat',CountVectorizer(
            **countv_para,
            preprocessor=get_col('text_feat'))),
        ('title',CountVectorizer(
            **countv_para,
            preprocessor=get_col('title')))
    ])
        
vectorizer2 = FeatureUnion([
        ('text',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('text_feat',CountVectorizer(
            **countv_para,
            preprocessor=get_col('text_feat'))),
        ('title',CountVectorizer(
            **countv_para,
            preprocessor=get_col('title'))),
        ('all_titles',TfidfVectorizer(
            max_features=20000,
            **tfidf_para,
            preprocessor=get_col('all_titles'))),
    ])

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from typing import List, Dict
def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

dictvectorizer = make_union(
        on_field(['region', 'city', 'price_cut', 'item_seq_number_cut', 'image_top_1', \
                   'user_type', 'user'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)

def common_users(varin, col, cutoff = 3):
    dft         = pd.DataFrame(varin.values, columns = [col])
    gp          = dft[col].value_counts().reset_index().rename(columns = {'index':col, col:col+'_ct'})
    var         = dft[[col]].merge(gp, on = col, how = 'left')[col+'_ct']
    idx         = var>cutoff
    var[idx]    = (varin.values)[idx]
    var[~idx]    = ''
    return var.astype(str).values
    
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['price_cut'] = 'pc_'+(np.log1p(df['price'])*5).fillna(150).astype(np.int32).astype(str)
    df['item_seq_number_cut'] = 'is_'+(np.log1p(df['item_seq_number'])*5).fillna(150).astype(np.int32).astype(str)
    df['user']    = common_users(df['user_id'], 'user_id')
    df['image_top_1'] = 'img_'+ df['image_top_1'].fillna(3100).astype(str)
    df['region']  = 'rgn_'+ df['region'].fillna('').astype(str)
    df['city']    = df['city'].fillna('').astype(str)
    df['parent_category_name'] = 'pc1_'+ df['parent_category_name'].fillna('').astype(str)
    df['category_name'] = 'c2_'+ df['category_name'].fillna('').astype(str)
    return df[['user', 'region', 'city', 'user_type', 'image_top_1', 'price_cut', \
               'item_seq_number_cut']]

df.set_index('item_id', inplace = True)
dfcat['item_id'] = df.reset_index()['item_id']
dfcat.set_index('item_id', inplace = True)
dfcat.head()
start_vect=time.time()
print('[{}] [TF-IDF] Dict Vectorisation 1'.format(time.time() - start_time))
dictvectorizer.fit(preprocess(dfcat.loc[traindex,:]))
ready_dfd = dictvectorizer.transform(preprocess(dfcat)).astype(np.float32)
gc.collect()
print('[{}] [TF-IDF] Term Frequency Vectorisation 2'.format(time.time() - start_time))
vectorizer1.fit(df.loc[traindex,:].to_dict('records'))
ready_df1 = vectorizer1.transform(df.to_dict('records'))
gc.collect()
print('[{}] [TF-IDF] Term Frequency Vectorisation 3'.format(time.time() - start_time))
vectorizer2.fit(df.loc[traindex,:].to_dict('records'))
ready_df2 = vectorizer2.transform(df.to_dict('records'))
print('[{}] Vectorisation completed'.format(time.time() - start_time))
gc.collect()
print(ready_dfd.shape)
print(ready_df1.shape)
print(ready_df2.shape)

print('[{}] Modeling Stage'.format(time.time() - start_time))
# Combine Dense Features with Sparse Text Bag of Words Features
#df.drop(textfeats+['text', 'all_titles'], axis=1,inplace=True)
#dfnorm = preprocessing.normalise(df.values)
X_train1 = hstack([ready_dfd[0:traindex.shape[0]]       ,ready_df1[0:traindex.shape[0]]])
X_test1 = hstack([ready_dfd[traindex.shape[0]:]         ,ready_df1[traindex.shape[0]:]])
X_train2 = hstack([ready_dfd[0:traindex.shape[0]]       ,ready_df2[0:traindex.shape[0]]])
X_test2 = hstack([ready_dfd[traindex.shape[0]:]         ,ready_df2[traindex.shape[0]:]])
X_train1 = X_train1.tocsr()
X_test1  = X_test1.tocsr()
X_train2 = X_train2.tocsr()
X_test2  = X_test2.tocsr()
gc.collect();

print('[{}] Set up folds'.format(time.time() - start_time))
foldls = [["2017-03-15", "2017-03-16", "2017-03-17"], \
           ["2017-03-18", "2017-03-19", "2017-03-20"], \
           ["2017-03-21", "2017-03-22", "2017-03-23"], \
           ["2017-03-24", "2017-03-25", "2017-03-26"], \
            ["2017-03-27", "2017-03-28", "2017-03-29", \
                "2017-03-30", "2017-03-31", "2017-04-01", \
                "2017-04-02", "2017-04-03","2017-04-07"]]
#folds = [df.loc[traindex,:]['activation_date'].isin(f) for f in foldls]
folds = [active_dates.loc[traindex].isin(f) for f in foldls]
[sum(f) for f in folds]

print('[{}] Modified sklearn wrapper'.format(time.time() - start_time))
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, f in enumerate(folds):
        train_index, test_index = np.where(f==False), np.where(f)
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y.values[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        gc.collect()

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

print('[{}] Start ridge'.format(time.time() - start_time))
SEED    = 200
NFOLDS  = 5
ntrain  = df.loc[traindex,:].shape[0]
ntest   = df.loc[testdex,:].shape[0]
ridge_params = { 'alpha':22.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED }

ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train1, ridge_oof_test1 = get_oof(ridge, ready_df1[:ntrain], y, ready_df1[ntrain:])
print('Ridge OOF RMSE: {}'.format(sqrt(mean_squared_error(y, ridge_oof_train1))))
ridge_oof_train2, ridge_oof_test2 = get_oof(ridge, ready_df2[:ntrain], y, ready_df2[ntrain:])
print('Ridge OOF RMSE: {}'.format(sqrt(mean_squared_error(y, ridge_oof_train2))))
#Ridge OOF RMSE: 0.2285614218020972
#Ridge OOF RMSE: 0.2276622560727391


ridge_params = { 'alpha':22.0, 'fit_intercept':True, 'normalize':True, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED }

ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train3, ridge_oof_test3 = get_oof(ridge, X_train1, y, X_test1)
print('Ridge OOF RMSE: {}'.format(sqrt(mean_squared_error(y, ridge_oof_train3))))
ridge_oof_train4, ridge_oof_test4 = get_oof(ridge, X_train2, y, X_test2)
print('Ridge OOF RMSE: {}'.format(sqrt(mean_squared_error(y, ridge_oof_train4))))


df['ridge_preds_txt1'] = np.concatenate([ridge_oof_train1, ridge_oof_test1])
df['ridge_preds_txt2'] = np.concatenate([ridge_oof_train2, ridge_oof_test2])
df['ridge_preds_strtxt1'] = np.concatenate([ridge_oof_train3, ridge_oof_test3])
df['ridge_preds_strtxt2'] = np.concatenate([ridge_oof_train4, ridge_oof_test4])

cols = [c for c in df.columns if 'ridge_' in c]
df[cols].to_csv(path + '../features/ridgeTextStr5CV.csv.gz', compression = 'gzip', index = False)
