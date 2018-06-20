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
path = '/home/ubuntu/avito/data/'
start_time = time.time()
full = False

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

(traindf['image_top_1'] == traindf['image_top_1']).value_counts()
(testdf['image_top_1'] == testdf['image_top_1']).value_counts()



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

print('[{}] Load meta image engineered features'.format(time.time() - start_time))
featimgmeta = pd.concat([pd.read_csv(path + '../features/img_features_%s.csv.gz'%(i)) for i in range(6)])
featimgmeta.rename(columns = {'name':'image'}, inplace = True)
featimgmeta['image'] = featimgmeta['image'].str.replace('.jpg', '')
df = df.reset_index('item_id').merge(featimgmeta, on = ['image'], how = 'left').set_index('item_id')
for col in featimgmeta.columns.values[1:]:
    df[col].fillna(-1, inplace = True)
    df[col].astype(np.float32, inplace = True)
    
print('[{}] Load translated image engineered features'.format(time.time() - start_time))
feattrlten = pd.concat([pd.read_pickle(path + '../features/translate_en.pkl'),
                       pd.read_pickle(path + '../features/translate_tst_en.pkl')])
feattrlten['translation'] = feattrlten['title_translated'] + ' ' + feattrlten['param_1_translated'] + ' ' \
            + feattrlten['param_2_translated'] + ' ' + feattrlten['param_3_translated'] + ' '  \
            + feattrlten['category_name_translated'] + ' ' + feattrlten['parent_category_name_translated']
feattrlten = feattrlten[['translation']]
df = pd.merge(df, feattrlten, left_index=True, right_index=True, how='left')
del feattrlten
df['translation'].fillna('', inplace = True)
gc.collect()
 
print('[{}] Load other engineered features'.format(time.time() - start_time))
featlatlon = pd.read_csv(path + '../features/avito_region_city_features.csv') # https://www.kaggle.com/frankherfert/region-and-city-details-with-lat-lon-and-clusters
featlatlon.drop(['city_region', 'city_region_id', 'region_id'], 1, inplace = True)
featpop    = pd.read_csv(path + '../features/city_population_wiki_v3.csv') # https://www.kaggle.com/stecasasso/russian-city-population-from-wikipedia/comments
featusrttl = pd.read_csv(path + '../features/user_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrcat = pd.read_csv(path + '../features/usercat_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrprd = pd.read_csv(path + '../features/user_activ_period_stats.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgtxt = pd.read_csv(path + '../features/ridgeText5CV.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
#featrdgtxts = pd.read_csv(path + '../features/ridgeTextStr5CV.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgimg = pd.read_csv(path + '../features/ridgeImg5CV.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featnnvgg  = pd.read_csv(path + '../features/nnetImgV5CV.csv.gz', compression = 'gzip') 
#featrdgprc = pd.read_csv(path + '../features/price_category_ratios.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgprc = pd.read_csv(path + '../features/price_seq_category_ratios.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgprc.fillna(-1, inplace = True)
featrdgrnk = pd.read_csv(path + '../features/price_rank_ratios0906.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgrnk.isnull().sum()
featimgprc = pd.read_csv(path + '../features/price_imagetop1_ratios.gz', compression = 'gzip') # created with features/make/priceImgRatios2705.R
featenc = pd.read_csv(path + '../features/alldf_bayes_mean.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featnumf = pd.read_csv(path + '../features/numericFeats.gz', compression = 'gzip') 
featprmenc = pd.read_csv(path + '../features/alldf_bayes_mean_param_1006.gz', compression = 'gzip') 
featprmtro = pd.read_csv(path + '../features/price_param_ratios1006.gz', compression = 'gzip') 

featnumf.fillna(0, inplace = True)
featencfst = pd.read_csv(path + '../features/alldf_bayes_fest_1206.gz', compression = 'gzip') 
featprtfst = pd.read_csv(path + '../features/prawratios_fest_1606.gz', compression = 'gzip')

featct  = pd.read_csv(path + '../features/alldf_count.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrttl.rename(columns={'title': 'all_titles'}, inplace = True)
df = df.reset_index().merge(featpop, on = 'city', how = 'left')
df = df.merge(featlatlon, on = ['city', 'region'], how = 'left')
df['population'].fillna(-1, inplace = True)
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
df = pd.concat([df.reset_index(),featenc, featct, featrdgtxt, featrdgprc, featimgprc, \
                featrdgrnk, featnumf, featprmenc, featprmtro, featencfst, featprtfst],axis=1)
df['ridge_txt'] = featrdgtxt['ridge_preds'].values
#df = pd.concat([df.reset_index(),featenc, featct, ],axis=1)
df['ridge_img'] = featrdgimg['ridge_img_preds'].values
#df['ridge_imgv'] = featnnvgg['deal_probability'].values
df = df.set_index('item_id')
df.drop(['index'], axis=1,inplace=True)
df.columns
del featusrttl, featusrcat, featusrprd, featenc, featrdgprc, featimgprc, featrdgrnk, featnumf, featprmenc, featprmtro, featencfst, featprtfst
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

del featct, featlatlon, featimgmeta, featpop, featrdgimg, featrdgtxt
gc.collect()

print('[{}] Manage Memory'.format(time.time() - start_time))
for col in df.columns:
    if np.float64 == df[col].dtype:
        df[col] = df[col].astype(np.float32)
    if np.int64 == df[col].dtype:
        df[col] = df[col].astype(np.int32)
    gc.collect()
df.dtypes


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
    gc.collect()
df.info()
for cols in ['translation']:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently

    
print('[{}] Manage Memory'.format(time.time() - start_time))
for col in df.columns:
    if np.float64 == df[col].dtype:
        df[col] = df[col].astype(np.float32)
    if np.int64 == df[col].dtype:
        df[col] = df[col].astype(np.int32)
    gc.collect()
df.info()


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
text_cols = ['description', 'text', 'text_feat', 'title', 'translation']
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
            preprocessor=get_col('text'))),
        ('text_feat',CountVectorizer(
            **countv_para,
            preprocessor=get_col('text_feat'))),
        ('title',CountVectorizer(
            **countv_para,
            preprocessor=get_col('title'))),
        ('translation',TfidfVectorizer(
            #ngram_range=(1, 2),
            max_features=40000,
            **tfidf_para,
            preprocessor=get_col('translation'))),
    ])
    
start_vect=time.time()
vectorizer.fit(df.loc[traindex,:].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
tfvocab[:50]
print('[{}] Vectorisation completed'.format(time.time() - start_time))
# Drop Text Cols
df.drop(textfeats+['text', 'all_titles', 'translation'], axis=1,inplace=True)
gc.collect()

print('[{}] Drop all the categorical'.format(time.time() - start_time))
df.drop(categorical, axis=1,inplace=True)

ready_df.shape

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
    'num_leaves' : 1000,
    'learning_rate' : 0.02,
    'feature_fraction' : 0.5,
    'verbosity' : 0
}
for i in range(5):
    lgbm_params['seed'] = i+1
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab)

    # Go Go Go
    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=15000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=60,
        verbose_eval=20)
'''
# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig(path + '../plots/feature_import_1006A.png')

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
#lgsub.to_csv(path + "../sub/lgsub_0206A.csv.gz",index=True,header=True, compression = 'gzip')
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
'''


'''
[20]    train's rmse: 0.240862  valid's rmse: 0.238605
[40]    train's rmse: 0.230173  valid's rmse: 0.2282
[60]    train's rmse: 0.224224  valid's rmse: 0.222602
[80]    train's rmse: 0.220627  valid's rmse: 0.2194
[100]   train's rmse: 0.218441  valid's rmse: 0.217659
[120]   train's rmse: 0.216918  valid's rmse: 0.216533
[140]   train's rmse: 0.215751  valid's rmse: 0.215787
[160]   train's rmse: 0.21477   valid's rmse: 0.215233
[180]   train's rmse: 0.213934  valid's rmse: 0.214855
[200]   train's rmse: 0.213196  valid's rmse: 0.214541
[220]   train's rmse: 0.212515  valid's rmse: 0.214307
[240]   train's rmse: 0.211876  valid's rmse: 0.214083
[260]   train's rmse: 0.21129   valid's rmse: 0.213935
[280]   train's rmse: 0.210735  valid's rmse: 0.213808
[300]   train's rmse: 0.210196  valid's rmse: 0.213673
[320]   train's rmse: 0.209674  valid's rmse: 0.213531
[340]   train's rmse: 0.20917   valid's rmse: 0.213408
[360]   train's rmse: 0.208678  valid's rmse: 0.213295
[380]   train's rmse: 0.208198  valid's rmse: 0.213186
[400]   train's rmse: 0.20773   valid's rmse: 0.213081
[420]   train's rmse: 0.207269  valid's rmse: 0.212985
[440]   train's rmse: 0.206819  valid's rmse: 0.212898
[460]   train's rmse: 0.206367  valid's rmse: 0.212795
[480]   train's rmse: 0.205926  valid's rmse: 0.212715
[500]   train's rmse: 0.205499  valid's rmse: 0.212636
[520]   train's rmse: 0.205087  valid's rmse: 0.212564
[540]   train's rmse: 0.204673  valid's rmse: 0.212485
[560]   train's rmse: 0.204278  valid's rmse: 0.212433
[580]   train's rmse: 0.203884  valid's rmse: 0.212379
[600]   train's rmse: 0.203508  valid's rmse: 0.212336
[620]   train's rmse: 0.203147  valid's rmse: 0.212294
[640]   train's rmse: 0.202781  valid's rmse: 0.212252
[660]   train's rmse: 0.20243   valid's rmse: 0.212213
[680]   train's rmse: 0.202087  valid's rmse: 0.212178
[700]   train's rmse: 0.201757  valid's rmse: 0.212147
[720]   train's rmse: 0.201413  valid's rmse: 0.212122
[740]   train's rmse: 0.201077  valid's rmse: 0.212096
[760]   train's rmse: 0.200771  valid's rmse: 0.212073
[780]   train's rmse: 0.200447  valid's rmse: 0.212045
[800]   train's rmse: 0.200124  valid's rmse: 0.212028
[820]   train's rmse: 0.199809  valid's rmse: 0.212002
[840]   train's rmse: 0.199508  valid's rmse: 0.211988
[860]   train's rmse: 0.199219  valid's rmse: 0.21197
[880]   train's rmse: 0.198915  valid's rmse: 0.211946
[900]   train's rmse: 0.198614  valid's rmse: 0.211925
[920]   train's rmse: 0.198313  valid's rmse: 0.211903
[940]   train's rmse: 0.198014  valid's rmse: 0.211894
[960]   train's rmse: 0.197726  valid's rmse: 0.21188
[980]   train's rmse: 0.197449  valid's rmse: 0.211866
[1000]  train's rmse: 0.197183  valid's rmse: 0.211853
[1020]  train's rmse: 0.196904  valid's rmse: 0.211839
[1040]  train's rmse: 0.196641  valid's rmse: 0.21183
[1060]  train's rmse: 0.196349  valid's rmse: 0.211812
[1080]  train's rmse: 0.196074  valid's rmse: 0.211804
[1100]  train's rmse: 0.19582   valid's rmse: 0.211798
[1120]  train's rmse: 0.195561  valid's rmse: 0.21178
[1140]  train's rmse: 0.195305  valid's rmse: 0.21177
[1160]  train's rmse: 0.195047  valid's rmse: 0.211758
[1180]  train's rmse: 0.194793  valid's rmse: 0.211753
[1200]  train's rmse: 0.194531  valid's rmse: 0.211744
[1220]  train's rmse: 0.19428   valid's rmse: 0.211736
[1240]  train's rmse: 0.194027  valid's rmse: 0.21173
[1260]  train's rmse: 0.193775  valid's rmse: 0.211717
[1280]  train's rmse: 0.193542  valid's rmse: 0.211705
[1300]  train's rmse: 0.193307  valid's rmse: 0.211698
[1320]  train's rmse: 0.193073  valid's rmse: 0.211693
[1340]  train's rmse: 0.192845  valid's rmse: 0.211688
[1360]  train's rmse: 0.192632  valid's rmse: 0.211683
[1380]  train's rmse: 0.192397  valid's rmse: 0.211671
[1400]  train's rmse: 0.192163  valid's rmse: 0.211666
[1420]  train's rmse: 0.191938  valid's rmse: 0.211664
[1440]  train's rmse: 0.191712  valid's rmse: 0.211668
[1460]  train's rmse: 0.191481  valid's rmse: 0.211666
[1480]  train's rmse: 0.191259  valid's rmse: 0.21166
[1500]  train's rmse: 0.191012  valid's rmse: 0.211658
[1520]  train's rmse: 0.190785  valid's rmse: 0.211656
[1540]  train's rmse: 0.190572  valid's rmse: 0.211653
[1560]  train's rmse: 0.190345  valid's rmse: 0.211646
[1580]  train's rmse: 0.190111  valid's rmse: 0.211637
[1600]  train's rmse: 0.189867  valid's rmse: 0.211623
[1620]  train's rmse: 0.189641  valid's rmse: 0.211618
[1640]  train's rmse: 0.189417  valid's rmse: 0.211618
[1660]  train's rmse: 0.189188  valid's rmse: 0.211614
[1680]  train's rmse: 0.188968  valid's rmse: 0.21161
[1700]  train's rmse: 0.188747  valid's rmse: 0.211603
[1720]  train's rmse: 0.188538  valid's rmse: 0.211597
[1740]  train's rmse: 0.188325  valid's rmse: 0.211598
[1760]  train's rmse: 0.188121  valid's rmse: 0.211592
[1780]  train's rmse: 0.187888  valid's rmse: 0.211585
[1800]  train's rmse: 0.187676  valid's rmse: 0.211583
[1820]  train's rmse: 0.187453  valid's rmse: 0.211576
[1840]  train's rmse: 0.187237  valid's rmse: 0.211575
[1860]  train's rmse: 0.187036  valid's rmse: 0.21157
[1880]  train's rmse: 0.186821  valid's rmse: 0.211568
[1900]  train's rmse: 0.186635  valid's rmse: 0.211565
[1920]  train's rmse: 0.186427  valid's rmse: 0.211561
[1940]  train's rmse: 0.186222  valid's rmse: 0.211557
[1960]  train's rmse: 0.186027  valid's rmse: 0.211555
[1980]  train's rmse: 0.185835  valid's rmse: 0.211548
[2000]  train's rmse: 0.185639  valid's rmse: 0.211549
[2020]  train's rmse: 0.185433  valid's rmse: 0.211545
[2040]  train's rmse: 0.185245  valid's rmse: 0.211546
[2060]  train's rmse: 0.185059  valid's rmse: 0.21155
Early stopping, best iteration is:
[2018]  train's rmse: 0.18545   valid's rmse: 0.211544

'''

