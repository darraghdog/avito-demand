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
import pymorphy2
import nltk, re
from nltk.tokenize import ToktokTokenizer
from multiprocessing import cpu_count, Pool


#path = '../input/'
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'

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
featrdgtxt = pd.read_csv(path + '../features/ridgeText5CV.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
#featrdgtxts = pd.read_csv(path + '../features/ridgeTextStr5CV.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgimg = pd.read_csv(path + '../features/ridgeImg5CV.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
#featrdgprc = pd.read_csv(path + '../features/price_category_ratios.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgprc = pd.read_csv(path + '../features/price_seq_category_ratios.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgprc.fillna(-1, inplace = True)
featimgprc = pd.read_csv(path + '../features/price_imagetop1_ratios.gz', compression = 'gzip') # created with features/make/priceImgRatios2705.R
featenc = pd.read_csv(path + '../features/alldf_bayes_mean.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featct  = pd.read_csv(path + '../features/alldf_count.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featct.columns
featimgprc.head()
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
df.reset_index(inplace = True)

df.head()
df = pd.concat([df.reset_index(),featenc, featct, featrdgtxt, featrdgprc, featimgprc],axis=1)
#df['ridge_txt'] = featrdgtxt['ridge_preds'].values
#df = pd.concat([df.reset_index(),featenc, featct, ],axis=1)

df['ridge_img'] = featrdgimg['ridge_img_preds'].values
df = df.set_index('item_id')
df.drop(['index'], axis=1,inplace=True)
df.columns
del featusrttl, featusrcat, featusrprd, featenc, featrdgprc, featimgprc
# del featusrttl, featusrcat, featusrprd, featenc, featrdgtxts
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

print('[{}] Clean text and tokenize'.format(time.time() - start_time))
toktok = ToktokTokenizer()
tokSentMap = {}
morpher = pymorphy2.MorphAnalyzer()
def tokSent(sent):
    sent = sent.replace('/', ' ')
    return " ".join(morpher.parse(word)[0].normal_form for word in toktok.tokenize(rgx.sub(' ', sent)))
def tokCol(var):
    return [tokSent(s) for s in var.tolist()]
rgx = re.compile('[%s]' % '!"#%&()*,-./:;<=>?@[\\]^_`{|}~\t\n')   

partitions = 4 
def parallelize(data, func):
    data_split = np.array_split(data.values, partitions)
    pool = Pool(partitions)
    data = pd.concat([pd.Series(l) for l in pool.map(tokCol, data_split)]).values
    pool.close()
    pool.join()
    return data

load_text = True
text_cols = ['description', 'text', 'text_feat', 'title']
if load_text:
    dftxt = pd.read_csv(path + '../features/text_features_morphed.csv.gz', compression = 'gzip')
    for col in text_cols:
        print(col + ' load tokenised [{}]'.format(time.time() - start_time))
        df[col] = dftxt[col].values
        df.fillna(' ', inplace = True)
    del dftxt
else:
    for col in text_cols:
        print(col + ' tokenise [{}]'.format(time.time() - start_time))
        df[col] = parallelize(df[col], tokCol)
    df[text_cols].to_csv(path + '../features/text_features_morphed.csv.gz', compression = 'gzip')
gc.collect()

print('[{}] Finished tokenizing text...'.format(time.time() - start_time))
df.head()
print('[{}] [TF-IDF] Term Frequency Inverse Document Frequency Stage'.format(time.time() - start_time))
russian_stop = set(stopwords.words('russian'))
tfidf_para = {
    "stop_words": russian_stop,
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
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
        #('all_titles',CountVectorizer(
        #    **countv_para,
        #    preprocessor=get_col('title')))
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

print('[{}] Drop all the categorical'.format(time.time() - start_time))
df.drop(categorical, axis=1,inplace=True)


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
    'num_leaves' : 250,
    #'max_depth': 15,
    'learning_rate' : 0.02,
    'feature_fraction' : 0.5,
    'verbosity' : 0
}

# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab)
'''
# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab,
                categorical_feature = categorical)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab,
                categorical_feature = categorical)
'''
# Go Go Go
modelstart = time.time()
if full:
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=1300, #14686,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        #early_stopping_rounds=500,
        verbose_eval=20)    
else:
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=15000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=30,
        verbose_eval=20)

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,20])
lgb.plot_importance(lgb_clf, max_num_features=100, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig(path + '../plots/feature_import_2505.png')

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv(path + "../sub/lgsubval_2705A.csv.gz",index=True,header=True, compression = 'gzip')
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

'''
[20]    train's rmse: 0.240824  valid's rmse: 0.238499
[40]    train's rmse: 0.230632  valid's rmse: 0.228476
[60]    train's rmse: 0.225015  valid's rmse: 0.223209
[80]    train's rmse: 0.221667  valid's rmse: 0.220212
[100]   train's rmse: 0.21963   valid's rmse: 0.218557
[120]   train's rmse: 0.218262  valid's rmse: 0.217574
[140]   train's rmse: 0.217186  valid's rmse: 0.216885
[160]   train's rmse: 0.216302  valid's rmse: 0.216396
[180]   train's rmse: 0.215579  valid's rmse: 0.216073
[200]   train's rmse: 0.2149    valid's rmse: 0.215808
[220]   train's rmse: 0.214288  valid's rmse: 0.215588
[240]   train's rmse: 0.213726  valid's rmse: 0.215415
[260]   train's rmse: 0.213198  valid's rmse: 0.215268
[280]   train's rmse: 0.212693  valid's rmse: 0.215135
[300]   train's rmse: 0.212208  valid's rmse: 0.215011
[320]   train's rmse: 0.211738  valid's rmse: 0.214908
[340]   train's rmse: 0.211292  valid's rmse: 0.214842
[360]   train's rmse: 0.210845  valid's rmse: 0.214741
[380]   train's rmse: 0.210419  valid's rmse: 0.214652
[400]   train's rmse: 0.210008  valid's rmse: 0.21457
[420]   train's rmse: 0.2096    valid's rmse: 0.214502
[440]   train's rmse: 0.209211  valid's rmse: 0.214435
[460]   train's rmse: 0.208838  valid's rmse: 0.214374
[480]   train's rmse: 0.208485  valid's rmse: 0.214326
[500]   train's rmse: 0.208134  valid's rmse: 0.214279
[520]   train's rmse: 0.207809  valid's rmse: 0.214243
[540]   train's rmse: 0.207486  valid's rmse: 0.214206
[560]   train's rmse: 0.207167  valid's rmse: 0.214176
[580]   train's rmse: 0.206875  valid's rmse: 0.214151
[600]   train's rmse: 0.206593  valid's rmse: 0.214139
[620]   train's rmse: 0.206314  valid's rmse: 0.214123
[640]   train's rmse: 0.206051  valid's rmse: 0.214112
[660]   train's rmse: 0.205789  valid's rmse: 0.214101
[680]   train's rmse: 0.20551   valid's rmse: 0.214079
[700]   train's rmse: 0.205261  valid's rmse: 0.214071
[720]   train's rmse: 0.205011  valid's rmse: 0.214051
[740]   train's rmse: 0.20475   valid's rmse: 0.214036
[760]   train's rmse: 0.204501  valid's rmse: 0.214023
[780]   train's rmse: 0.20426   valid's rmse: 0.214018
[800]   train's rmse: 0.204018  valid's rmse: 0.214003
[820]   train's rmse: 0.203808  valid's rmse: 0.213995
[840]   train's rmse: 0.203556  valid's rmse: 0.213986
[860]   train's rmse: 0.203329  valid's rmse: 0.213985
[880]   train's rmse: 0.203099  valid's rmse: 0.213979
[900]   train's rmse: 0.202855  valid's rmse: 0.21397
[920]   train's rmse: 0.202608  valid's rmse: 0.213955
[940]   train's rmse: 0.202394  valid's rmse: 0.213955
[960]   train's rmse: 0.20217   valid's rmse: 0.213949
[980]   train's rmse: 0.201964  valid's rmse: 0.213938
[1000]  train's rmse: 0.201753  valid's rmse: 0.213933
[1020]  train's rmse: 0.201538  valid's rmse: 0.213928
[1040]  train's rmse: 0.201342  valid's rmse: 0.213925
[1060]  train's rmse: 0.201124  valid's rmse: 0.213915
[1080]  train's rmse: 0.200916  valid's rmse: 0.213913
[1100]  train's rmse: 0.200724  valid's rmse: 0.21391
[1120]  train's rmse: 0.200511  valid's rmse: 0.213902
[1140]  train's rmse: 0.200258  valid's rmse: 0.213889
[1160]  train's rmse: 0.20005   valid's rmse: 0.213888
[1180]  train's rmse: 0.199851  valid's rmse: 0.213885
[1200]  train's rmse: 0.19966   valid's rmse: 0.213881
[1220]  train's rmse: 0.199449  valid's rmse: 0.213874
[1240]  train's rmse: 0.199247  valid's rmse: 0.213872
[1260]  train's rmse: 0.199053  valid's rmse: 0.213872
[1280]  train's rmse: 0.198852  valid's rmse: 0.213871
[1300]  train's rmse: 0.198665  valid's rmse: 0.213874
Early stopping, best iteration is:
[1281]  train's rmse: 0.198844  valid's rmse: 0.21387
'''
