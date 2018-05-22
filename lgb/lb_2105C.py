
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


#path = '../input/'
path = "/home/darragh/avito/data/"
path = '/Users/dhanley2/Documents/avito/data/'

# path = '/home/ubuntu/avito/data/'
start_time = time.time()
full = True

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
if full:
    trnidx = (traindf.activation_date<=pd.to_datetime('2017-03-28')).values
    validx = (traindf.activation_date>=pd.to_datetime('2017-03-29')).values
else:
    trnidx = (traindf.activation_date<=pd.to_datetime('2017-03-26')).values
    validx = (traindf.activation_date>=pd.to_datetime('2017-03-27')).values

print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
del traindf,testdf
gc.collect()
df['idx'] = range(df.shape[0])
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print('[{}] Load engineered features'.format(time.time() - start_time))
featusrttl = pd.read_csv(path + '../features/user_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrcat = pd.read_csv(path + '../features/usercat_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrprd = pd.read_csv(path + '../features/user_activ_period_stats.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featenc = pd.read_csv(path + '../features/alldf_bayes_mean.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featct  = pd.read_csv(path + '../features/alldf_count.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featct.columns
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

'''
Index(['user_id_ct', 'region_ct', 'city_ct', 'category_name_ct', 'param_1_ct',
       'param_2_ct', 'param_3_ct', 'image_top_1_ct', 'item_seq_number_ct'],
      dtype='object')
'''

print('[{}] Resort data correctly'.format(time.time() - start_time))
df.sort_values('idx', inplace = True)
df.drop(['idx'], axis=1,inplace=True)
df.reset_index(inplace = True)
df.head()
df = pd.concat([df.reset_index(),featenc, featct],axis=1)
df = df.set_index('item_id')
df.drop(['index'], axis=1,inplace=True)
df.columns
del featusrttl, featusrcat, featusrprd, featenc
gc.collect()

print('[{}] Feature Engineering'.format(time.time() - start_time))
for col in df.columns:
    if 'price' in col:
        print(f'Fill {col}')
        df[col].fillna(-999,inplace=True)

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
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

print('[{}] Text Features'.format(time.time() - start_time))
df['description'].fillna('unknowndescription', inplace=True)
df['title'].fillna('unknowntitle', inplace=True)
df['text']      = (df['description'].fillna('') + ' ' + df['title'] + ' ' + 
  df['parent_category_name'].fillna('').astype(str) + ' ' + df['category_name'].fillna('').astype(str) )

print('[{}] Create Time Variables'.format(time.time() - start_time))
df["Weekday"] = df['activation_date'].dt.weekday
df.drop(["activation_date","image"],axis=1,inplace=True)

print('[{}] Make Item Seq number as contiuous also'.format(time.time() - start_time))
df["item_seq_number_cont"] = df["item_seq_number"]
df['city'] = df['region'].fillna('').astype(str) + '_' + df['city'].fillna('').astype(str)
df.columns
print('[{}] Encode Variables'.format(time.time() - start_time))
df.drop(['user_id'], 1, inplace = True)
categorical = ["region","parent_category_name","user_type", 'city', 'category_name', "item_seq_number", 'image_top_1']
print("Encoding :",categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))
  
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
vectorizer = FeatureUnion([
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
        ('all_titles',CountVectorizer(
            **countv_para,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()
vectorizer.fit(df.loc[traindex,:].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
tfvocab[:50]
print('[{}] Vectorisation completed'.format(time.time() - start_time))
# Drop Text Cols
df.drop(textfeats+['text', 'all_titles'], axis=1,inplace=True)
gc.collect()

print('[{}] Modeling Stage'.format(time.time() - start_time))
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
    'feature_fraction' : 0.5,
    'verbosity' : 0
}

# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab,
                categorical_feature = categorical)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab,
                categorical_feature = categorical)

# Go Go Go
modelstart = time.time()
if full:
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=14686,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        #early_stopping_rounds=500,
        verbose_eval=200)    
else:
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=15000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=500,
        verbose_eval=200)

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig(path + '../plots/feature_import_2105.png')

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("../sub/lgsub_2105C.csv.gz",index=True,header=True, compression = 'gzip')
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

# [200]   train's rmse: 0.22708   valid's rmse: 0.225798

'''
No encoding
Early stopping, best iteration is:
[13686] train's rmse: 0.195553  valid's rmse: 0.217396
'''

'''
[200]   train's rmse: 0.224349  valid's rmse: 0.223056
[400]   train's rmse: 0.220863  valid's rmse: 0.221124
[600]   train's rmse: 0.218924  valid's rmse: 0.220071
[800]   train's rmse: 0.217614  valid's rmse: 0.219446
[1000]  train's rmse: 0.21655   valid's rmse: 0.218989
[1200]  train's rmse: 0.215645  valid's rmse: 0.218685
[1400]  train's rmse: 0.214868  valid's rmse: 0.218432
[1600]  train's rmse: 0.214176  valid's rmse: 0.218216
[1800]  train's rmse: 0.213525  valid's rmse: 0.218035
[2000]  train's rmse: 0.21291   valid's rmse: 0.217881
[2200]  train's rmse: 0.212349  valid's rmse: 0.217758
[2400]  train's rmse: 0.211823  valid's rmse: 0.217641
[2600]  train's rmse: 0.211325  valid's rmse: 0.217535
[2800]  train's rmse: 0.210841  valid's rmse: 0.21746
[3000]  train's rmse: 0.210391  valid's rmse: 0.21738
[3200]  train's rmse: 0.209932  valid's rmse: 0.217308
[3400]  train's rmse: 0.209497  valid's rmse: 0.217239
[3600]  train's rmse: 0.209062  valid's rmse: 0.21718
[3800]  train's rmse: 0.208667  valid's rmse: 0.217125
[4000]  train's rmse: 0.208276  valid's rmse: 0.217065
[4200]  train's rmse: 0.20788   valid's rmse: 0.217009
[4400]  train's rmse: 0.207497  valid's rmse: 0.216965
[4600]  train's rmse: 0.207142  valid's rmse: 0.21693
[4800]  train's rmse: 0.20678   valid's rmse: 0.21689
[5000]  train's rmse: 0.206422  valid's rmse: 0.216848
[5200]  train's rmse: 0.206072  valid's rmse: 0.216819
[5400]  train's rmse: 0.205726  valid's rmse: 0.216786
[5600]  train's rmse: 0.205373  valid's rmse: 0.216748
[5800]  train's rmse: 0.205035  valid's rmse: 0.216718
[6000]  train's rmse: 0.204714  valid's rmse: 0.216689
[6200]  train's rmse: 0.204384  valid's rmse: 0.216677
[6400]  train's rmse: 0.204066  valid's rmse: 0.216653
[6600]  train's rmse: 0.203755  valid's rmse: 0.216632
[6800]  train's rmse: 0.203431  valid's rmse: 0.216607
[7000]  train's rmse: 0.203123  valid's rmse: 0.216588
[7200]  train's rmse: 0.202809  valid's rmse: 0.21657
[7400]  train's rmse: 0.202507  valid's rmse: 0.216552
[7600]  train's rmse: 0.202207  valid's rmse: 0.216524
[7800]  train's rmse: 0.20191   valid's rmse: 0.216506
[8000]  train's rmse: 0.201632  valid's rmse: 0.216497
[8200]  train's rmse: 0.20134   valid's rmse: 0.216479
[8400]  train's rmse: 0.201061  valid's rmse: 0.216457
[8600]  train's rmse: 0.200774  valid's rmse: 0.216445
[8800]  train's rmse: 0.200485  valid's rmse: 0.216434
[9000]  train's rmse: 0.200206  valid's rmse: 0.216421
[9200]  train's rmse: 0.199919  valid's rmse: 0.216406
[9400]  train's rmse: 0.199663  valid's rmse: 0.216395
[9600]  train's rmse: 0.199404  valid's rmse: 0.216386
[9800]  train's rmse: 0.199142  valid's rmse: 0.216371
[10000] train's rmse: 0.198872  valid's rmse: 0.216357
[10200] train's rmse: 0.198603  valid's rmse: 0.216353
[10400] train's rmse: 0.198341  valid's rmse: 0.216343
[10600] train's rmse: 0.198095  valid's rmse: 0.21634
[10800] train's rmse: 0.197826  valid's rmse: 0.216331
[11000] train's rmse: 0.197587  valid's rmse: 0.21633
[11200] train's rmse: 0.197331  valid's rmse: 0.216323
[11400] train's rmse: 0.197094  valid's rmse: 0.216313
[11600] train's rmse: 0.196826  valid's rmse: 0.216305
[11800] train's rmse: 0.196572  valid's rmse: 0.216299
[12000] train's rmse: 0.19633   valid's rmse: 0.216298
[12200] train's rmse: 0.196089  valid's rmse: 0.216293
[12400] train's rmse: 0.195855  valid's rmse: 0.216293
[12600] train's rmse: 0.195611  valid's rmse: 0.21629
[12800] train's rmse: 0.195362  valid's rmse: 0.216281
[13000] train's rmse: 0.195128  valid's rmse: 0.216274
[13200] train's rmse: 0.194898  valid's rmse: 0.216268
[13400] train's rmse: 0.194651  valid's rmse: 0.216264
[13600] train's rmse: 0.194418  valid's rmse: 0.216268
[13800] train's rmse: 0.194187  valid's rmse: 0.216262
[14000] train's rmse: 0.193947  valid's rmse: 0.216261
[14200] train's rmse: 0.19372   valid's rmse: 0.216257
[14400] train's rmse: 0.193471  valid's rmse: 0.21625
[14600] train's rmse: 0.193251  valid's rmse: 0.216253
[14800] train's rmse: 0.193038  valid's rmse: 0.216251
Early stopping, best iteration is:
[14380] train's rmse: 0.193493  valid's rmse: 0.216247
'''


