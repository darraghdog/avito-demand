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

#print('[{}] Count NA row wise'.format(time.time() - start_time))
#df['NA_count_rows'] = df.isnull().sum(axis=1)

print('[{}] Load engineered features'.format(time.time() - start_time))
featlatlon = pd.read_csv(path + '../features/avito_region_city_features.csv') # https://www.kaggle.com/frankherfert/region-and-city-details-with-lat-lon-and-clusters
featlatlon.drop(['city_region', 'city_region_id', 'region_id'], 1, inplace = True)
featpop    = pd.read_csv(path + '../features/city_population_wiki_v3.csv') # https://www.kaggle.com/stecasasso/russian-city-population-from-wikipedia/comments
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
df = df.reset_index().merge(featpop, on = 'city', how = 'left')
df = df.merge(featlatlon, on = ['city', 'region'], how = 'left')
df['population'].fillna(-1, inplace = True)
df.head()
df = df.set_index('item_id')
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
        num_boost_round=1676, #14686,
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
        early_stopping_rounds=60,
        verbose_eval=20)

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig(path + '../plots/feature_import_2705B.png')

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv(path + "../sub/lgsub_2705B.csv.gz",index=True,header=True, compression = 'gzip')
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

'''
[20]    train's rmse: 0.241067  valid's rmse: 0.238745
[40]    train's rmse: 0.231096  valid's rmse: 0.228946
[60]    train's rmse: 0.225209  valid's rmse: 0.223373
[80]    train's rmse: 0.221705  valid's rmse: 0.220237
[100]   train's rmse: 0.219561  valid's rmse: 0.218494
[120]   train's rmse: 0.218128  valid's rmse: 0.217437
[140]   train's rmse: 0.217073  valid's rmse: 0.216777
[160]   train's rmse: 0.216199  valid's rmse: 0.216298
[180]   train's rmse: 0.215447  valid's rmse: 0.215955
[200]   train's rmse: 0.214788  valid's rmse: 0.215687
[220]   train's rmse: 0.214184  valid's rmse: 0.215482
[240]   train's rmse: 0.213626  valid's rmse: 0.215307
[260]   train's rmse: 0.213091  valid's rmse: 0.215162
[280]   train's rmse: 0.21258   valid's rmse: 0.21506
[300]   train's rmse: 0.21209   valid's rmse: 0.214951
[320]   train's rmse: 0.211617  valid's rmse: 0.214841
[340]   train's rmse: 0.211161  valid's rmse: 0.214736
[360]   train's rmse: 0.210718  valid's rmse: 0.214633
[380]   train's rmse: 0.210292  valid's rmse: 0.214546
[400]   train's rmse: 0.209875  valid's rmse: 0.214472
[420]   train's rmse: 0.20947   valid's rmse: 0.214397
[440]   train's rmse: 0.209083  valid's rmse: 0.214327
[460]   train's rmse: 0.208707  valid's rmse: 0.214275
[480]   train's rmse: 0.208356  valid's rmse: 0.214234
[500]   train's rmse: 0.208004  valid's rmse: 0.214178
[520]   train's rmse: 0.207672  valid's rmse: 0.214145
[540]   train's rmse: 0.20735   valid's rmse: 0.2141
[560]   train's rmse: 0.207049  valid's rmse: 0.214077
[580]   train's rmse: 0.20676   valid's rmse: 0.214061
[600]   train's rmse: 0.20648   valid's rmse: 0.214045
[620]   train's rmse: 0.206205  valid's rmse: 0.214026
[640]   train's rmse: 0.205933  valid's rmse: 0.214006
[660]   train's rmse: 0.205665  valid's rmse: 0.21399
[680]   train's rmse: 0.205415  valid's rmse: 0.213976
[700]   train's rmse: 0.205148  valid's rmse: 0.213959
[720]   train's rmse: 0.204869  valid's rmse: 0.213932
[740]   train's rmse: 0.204637  valid's rmse: 0.21392
[760]   train's rmse: 0.20441   valid's rmse: 0.213907
[780]   train's rmse: 0.204186  valid's rmse: 0.213899
[800]   train's rmse: 0.203952  valid's rmse: 0.213894
[820]   train's rmse: 0.203721  valid's rmse: 0.213885
[840]   train's rmse: 0.203491  valid's rmse: 0.213879
[860]   train's rmse: 0.203222  valid's rmse: 0.213863
[880]   train's rmse: 0.202981  valid's rmse: 0.213854
[900]   train's rmse: 0.202755  valid's rmse: 0.213848
[920]   train's rmse: 0.202523  valid's rmse: 0.213841
[940]   train's rmse: 0.202312  valid's rmse: 0.213834
[960]   train's rmse: 0.202092  valid's rmse: 0.213824
[980]   train's rmse: 0.201883  valid's rmse: 0.213824
[1000]  train's rmse: 0.201678  valid's rmse: 0.213818
[1020]  train's rmse: 0.201451  valid's rmse: 0.213809
[1040]  train's rmse: 0.201234  valid's rmse: 0.213804
[1060]  train's rmse: 0.201035  valid's rmse: 0.213799
[1080]  train's rmse: 0.200809  valid's rmse: 0.213795
[1100]  train's rmse: 0.200605  valid's rmse: 0.213792
[1120]  train's rmse: 0.200397  valid's rmse: 0.213787
[1140]  train's rmse: 0.200205  valid's rmse: 0.213789
[1160]  train's rmse: 0.199969  valid's rmse: 0.213773
[1180]  train's rmse: 0.199753  valid's rmse: 0.213769
[1200]  train's rmse: 0.199551  valid's rmse: 0.213765
[1220]  train's rmse: 0.199356  valid's rmse: 0.213765
[1240]  train's rmse: 0.199155  valid's rmse: 0.213761
[1260]  train's rmse: 0.198966  valid's rmse: 0.213757
[1280]  train's rmse: 0.198764  valid's rmse: 0.213755
[1300]  train's rmse: 0.198554  valid's rmse: 0.213752
[1320]  train's rmse: 0.198348  valid's rmse: 0.213755
[1340]  train's rmse: 0.198134  valid's rmse: 0.213745
[1360]  train's rmse: 0.197927  valid's rmse: 0.213741
[1380]  train's rmse: 0.197738  valid's rmse: 0.213739
[1400]  train's rmse: 0.197536  valid's rmse: 0.213734
[1420]  train's rmse: 0.197358  valid's rmse: 0.213733
[1440]  train's rmse: 0.197156  valid's rmse: 0.213726
[1460]  train's rmse: 0.196978  valid's rmse: 0.213731
[1480]  train's rmse: 0.196777  valid's rmse: 0.213724
[1500]  train's rmse: 0.196571  valid's rmse: 0.213719
[1520]  train's rmse: 0.196372  valid's rmse: 0.213716
[1540]  train's rmse: 0.196192  valid's rmse: 0.213716
[1560]  train's rmse: 0.195982  valid's rmse: 0.213715
[1580]  train's rmse: 0.195796  valid's rmse: 0.213713
[1600]  train's rmse: 0.195604  valid's rmse: 0.213707
[1620]  train's rmse: 0.195409  valid's rmse: 0.213704
[1640]  train's rmse: 0.195222  valid's rmse: 0.213703
[1660]  train's rmse: 0.195047  valid's rmse: 0.213705
[1680]  train's rmse: 0.194863  valid's rmse: 0.213703
[1700]  train's rmse: 0.194684  valid's rmse: 0.213704
[1720]  train's rmse: 0.194511  valid's rmse: 0.213702
Early stopping, best iteration is:
[1676]  train's rmse: 0.194902  valid's rmse: 0.213701
'''
