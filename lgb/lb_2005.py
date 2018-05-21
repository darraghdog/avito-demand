
# https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lgbm-clone?scriptVersionId=3540839

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import time, gc
import pandas as pd
import numpy as np
from sklearn import preprocessing
from nltk.corpus import stopwords 
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from typing import List, Dict
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt


#path = '../input/'
path = "/home/darragh/avito/data/"
path = '/Users/dhanley2/Documents/avito/data/'

# path = '/home/ubuntu/avito/data/'
start_time = time.time()

validation = False

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

print('[{}] Create Validation Index'.format(time.time() - start_time))
trnidx = (traindf.activation_date<=pd.to_datetime('2017-03-26')).values
validx = (traindf.activation_date>=pd.to_datetime('2017-03-27')).values

print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
del traindf,testdf
gc.collect()
df['idx'] = range(df.shape[0])
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

'''
print('[{}] Image dhash'.format(time.time() - start_time))
imdhash = pd.read_csv(path + '../features/image_dhash.csv.gz', compression = 'gzip')
imdhash[['image_id', 'image_hash', 'image_hash_ct']].head()
dfhash = df[['image']].merge(imdhash[['image_id', 'image_hash', 'image_hash_ct']], how='left', left_on='image', right_on='image_id')
df['image_hash'] = dfhash['image_hash'].fillna('missing').values
df['image_hash_ct'] = dfhash['image_hash_ct'].fillna(0).astype(np.int16).values
del dfhash, imdhash
gc.collect()
'''
print('[{}] Load engineered features'.format(time.time() - start_time))
featusrttl = pd.read_csv(path + '../features/user_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrcat = pd.read_csv(path + '../features/usercat_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrprd = pd.read_csv(path + '../features/user_activ_period_stats.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrcat.head()
featusrttl.rename(columns={'title': 'all_titles'}, inplace = True)
keep = ['user_id', 'all_titles', 'user_avg_price', 'user_ad_ct']
df = df.reset_index().merge(featusrttl[keep], on = 'user_id').set_index('item_id')
keep = ['user_id', 'parent_category_name', 'usercat_avg_price', 'usercat_ad_ct']
gc.collect()
df = df.reset_index().merge(featusrcat[keep], on = ['user_id', 'parent_category_name']).set_index('item_id')
keep = ['user_id', 'user_activ_sum', 'user_activ_mean', 'user_activ_var']
gc.collect()
df = df.reset_index().merge(featusrprd[keep], on = ['user_id'], how = 'left').set_index('item_id')
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))  

print('[{}] Resort data correctly'.format(time.time() - start_time))
df.sort_values('idx', inplace = True)
df.drop(['idx'], axis=1,inplace=True)
df.head(10)
df.columns
df.isnull().sum()

print('[{}] Quantiles'.format(time.time() - start_time))
df['prc_isn_cat'] =     pd.qcut(df['price'].fillna(-1), 30, duplicates='drop').astype(str) + \
    pd.qcut(df['item_seq_number'].fillna(-1), 20, duplicates='drop').astype(str)

print('[{}] Feature Engineering'.format(time.time() - start_time))
for col in df.columns:
    if 'price' in col:
        print(f'fill {col}')
        df[col].fillna(-999,inplace=True)
for col in df.columns:
    if 'user_activ' in col:
        print(f'fill {col}')
        df[col].fillna(-9,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print('[{}] Create Time Variables'.format(time.time() - start_time))
df["Weekday"] = df['activation_date'].dt.weekday
df.drop(["activation_date","image"],axis=1,inplace=True)

print('[{}] Make Item Seq number as contiuous also'.format(time.time() - start_time))
df["item_seq_number_cont"] = df["item_seq_number"]
df['city'] = df['region'].fillna('').astype(str) + '_' + df['city'].fillna('').astype(str)
df.columns
print('[{}] Encode Variables'.format(time.time() - start_time))
categorical = ["user_id","region","city","parent_category_name", 'prc_isn_cat', \
               "category_name","item_seq_number","user_type","image_top_1", 'city']
print("Encoding :",categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))

print('[{}] Text Features'.format(time.time() - start_time))
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features

print('[{}] Text Features'.format(time.time() - start_time))
df['description'].fillna('unknowndescription', inplace=True)
df['title'].fillna('unknowntitle', inplace=True)
df['text']      = (df['description'].fillna('') + ' ' + df['title'] + ' ' + 
  df['parent_category_name'].fillna('').astype(str) + ' ' + df['category_name'].fillna('').astype(str))

df['param_1'] = 'p1_'+ df['param_1'].fillna('').astype(str)
df['param_2'] = 'p2_'+ df['param_2'].fillna('').astype(str)
df['param_3'] = 'p3_'+ df['param_3'].fillna('').astype(str)
  
print('[{}] Meta Text Features'.format(time.time() - start_time))
textfeats = ["description","text_feat", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

print('[{}] [TF-IDF] Term Frequency Inverse Document Frequency Stage'.format(time.time() - start_time))
russian_stop = set(stopwords.words('russian'))
tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}
def get_col(col_name): return lambda x: x[col_name]
from sklearn.pipeline import FeatureUnion

vectorizer = FeatureUnion([
        ('text',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=80000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('all_titles',TfidfVectorizer(
            **tfidf_para,
            max_features=40000,
            preprocessor=get_col('title'))),
        ('text_feat',CountVectorizer(
            ngram_range=1,
            #max_features=7000,
            preprocessor=get_col('text_feat'))),
        ('title',TfidfVectorizer(
            **tfidf_para,
            max_features=15000,
            preprocessor=get_col('title')))
    ])
    

start_vect=time.time()
vectorizer.fit(df.loc[traindex,:].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print('[{}] Vectorisation completed'.format(time.time() - start_time))
# Drop Text Cols
df.drop(textfeats, axis=1,inplace=True)
gc.collect()
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)
df.drop(["text","all_titles"],axis=1,inplace=True)
gc.collect()
df.dtypes

print('[{}] Modeling Setup Stage'.format(time.time() - start_time))
# Combine Dense Features with Sparse Text Bag of Words Features
X_train = hstack([csr_matrix(df.loc[traindex,:][trnidx].values),ready_df[0:traindex.shape[0]][trnidx]])
X_valid = hstack([csr_matrix(df.loc[traindex,:][validx].values),ready_df[0:traindex.shape[0]][validx]])
y_train = y[trnidx]
y_valid = y[validx]
testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X_train, X_valid,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();


# Training and Validation Set
lgbm_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective' : 'regression',
    'metric' : 'rmse',
    'num_leaves' : 32,
    'max_depth': 15,
    'learning_rate' : 0.02,
    'feature_fraction' : 0.6,
    'verbosity' : 0
}

# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab,
                categorical_feature = categorical)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab,
                categorical_feature = categorical)

print('[{}] Modeling Go Go Go'.format(time.time() - start_time))

modelstart = time.time()
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=15000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=500,
    verbose_eval=200
)

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig(path + '../plots/feature_import_2005.png')

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("../sub/lgsub_2005.csv.gz",index=True,header=True, compression = 'gzip')
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

'''
[200]   train's rmse: 0.227104  valid's rmse: 0.225609
[400]   train's rmse: 0.223217  valid's rmse: 0.223012
[600]   train's rmse: 0.221068  valid's rmse: 0.221786
[800]   train's rmse: 0.219545  valid's rmse: 0.221021
[1000]  train's rmse: 0.218376  valid's rmse: 0.220504
[1200]  train's rmse: 0.217338  valid's rmse: 0.220091
[1400]  train's rmse: 0.216442  valid's rmse: 0.21977
[1600]  train's rmse: 0.215659  valid's rmse: 0.219518
[1800]  train's rmse: 0.214944  valid's rmse: 0.21931
[2000]  train's rmse: 0.214295  valid's rmse: 0.21913
[2200]  train's rmse: 0.213685  valid's rmse: 0.218969
[2400]  train's rmse: 0.213117  valid's rmse: 0.218839
[2600]  train's rmse: 0.212594  valid's rmse: 0.218727
[2800]  train's rmse: 0.212096  valid's rmse: 0.218624
[3000]  train's rmse: 0.211602  valid's rmse: 0.218538
[3200]  train's rmse: 0.211136  valid's rmse: 0.218459
[3400]  train's rmse: 0.21069   valid's rmse: 0.218382
[3600]  train's rmse: 0.210262  valid's rmse: 0.218316
[3800]  train's rmse: 0.209834  valid's rmse: 0.218249
[4000]  train's rmse: 0.209427  valid's rmse: 0.218187
[4200]  train's rmse: 0.20903   valid's rmse: 0.218132
[4400]  train's rmse: 0.208644  valid's rmse: 0.218077
[4600]  train's rmse: 0.208268  valid's rmse: 0.218024
[4800]  train's rmse: 0.207885  valid's rmse: 0.217983
[5000]  train's rmse: 0.207529  valid's rmse: 0.217946
[5200]  train's rmse: 0.207151  valid's rmse: 0.217899
[5400]  train's rmse: 0.206801  valid's rmse: 0.217857
[5600]  train's rmse: 0.206469  valid's rmse: 0.217834
[5800]  train's rmse: 0.206131  valid's rmse: 0.217797
[6000]  train's rmse: 0.205802  valid's rmse: 0.217765
[6200]  train's rmse: 0.205463  valid's rmse: 0.217736
[6400]  train's rmse: 0.205139  valid's rmse: 0.217709
[6600]  train's rmse: 0.20481   valid's rmse: 0.217682
[6800]  train's rmse: 0.204507  valid's rmse: 0.217665
'''

'''
1305 Results
[200]   train's rmse: 0.227594  valid's rmse: 0.226211
[400]   train's rmse: 0.223908  valid's rmse: 0.223681
[600]   train's rmse: 0.221883  valid's rmse: 0.222544
[800]   train's rmse: 0.220386  valid's rmse: 0.221804
[1000]  train's rmse: 0.219137  valid's rmse: 0.221278
[2000]  train's rmse: 0.214879  valid's rmse: 0.219926
[3000]  train's rmse: 0.211994  valid's rmse: 0.219326
[4000]  train's rmse: 0.209664  valid's rmse: 0.218973
[5000]  train's rmse: 0.207654  valid's rmse: 0.218754
[6000]  train's rmse: 0.205822  valid's rmse: 0.218591
[7000]  train's rmse: 0.204192  valid's rmse: 0.218478
[8000]  train's rmse: 0.202673  valid's rmse: 0.218397
[9000]  train's rmse: 0.201233  valid's rmse: 0.218344
[10000] train's rmse: 0.199855  valid's rmse: 0.218305
[11000] train's rmse: 0.198494  valid's rmse: 0.218281
[11200] train's rmse: 0.198219  valid's rmse: 0.218275
[11400] train's rmse: 0.197957  valid's rmse: 0.218278
Early stopping, best iteration is:
[11238] train's rmse: 0.198175  valid's rmse: 0.218273
'''


