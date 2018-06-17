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
feattrlten = pd.concat([pd.read_csv(path + '../features/translate_trn_en.csv.gz', compression = 'gzip'),
                        pd.read_csv(path + '../features/translate_tst_en.csv.gz', compression = 'gzip')])
# feattrlten = pd.concat([pd.read_pickle(path + '../features/translate_trn_en.pkl'),
#                        pd.read_pickle(path + '../features/translate_tst_en.pkl')])
feattrlten.fillna('', inplace = True)
feattrlten['translation'] = feattrlten['title_translated'] + ' ' + feattrlten['param_1_translated'] + ' ' \
            + feattrlten['param_2_translated'] + ' ' + feattrlten['param_3_translated'] + ' '  \
            + feattrlten['category_name_translated'] + ' ' + feattrlten['parent_category_name_translated']
feattrlten = feattrlten.set_index('item_id')[['translation']]
feattrlten.head()
df = pd.merge(df, feattrlten, left_index=True, right_index=True, how='left')
del feattrlten
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
#featrdgprc = pd.read_csv(path + '../features/price_category_ratios.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgprc = pd.read_csv(path + '../features/price_seq_category_ratios.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featrdgprc.fillna(-1, inplace = True)
featrdgrnk = pd.read_csv(path + '../features/price_rank_ratios0906.gz', compression = 'gzip') # created with R script and stemmer
featrdgrnk.isnull().sum()
featnumf = pd.read_csv(path + '../features/numericFeats.gz', compression = 'gzip') 
featnumf.fillna(0, inplace = True)
featencfst = pd.read_csv(path + '../features/alldf_bayes_fest_1206.gz', compression = 'gzip')
featprtfst = pd.read_csv(path + '../features/pratios_fest_1206.gz', compression = 'gzip')

featprmenc = pd.read_csv(path + '../features/alldf_bayes_mean_param_1006.gz', compression = 'gzip') 
featprmtro = pd.read_csv(path + '../features/price_param_ratios1006.gz', compression = 'gzip') 

featimgprc = pd.read_csv(path + '../features/price_imagetop1_ratios.gz', compression = 'gzip') # created with features/make/priceImgRatios2705.R
featenc = pd.read_csv(path + '../features/alldf_bayes_mean.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
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
df = pd.concat([df.reset_index(),featenc, featct, featrdgtxt, featrdgprc, featimgprc, featrdgrnk, featnumf, featprmenc, featprmtro, featencfst, featprtfst],axis=1)
#df['ridge_txt'] = featrdgtxt['ridge_preds'].values
#df = pd.concat([df.reset_index(),featenc, featct, ],axis=1)

print('[{}] Create folds'.format(time.time() - start_time))
foldls = [["2017-03-15", "2017-03-16", "2017-03-17"], \
       ["2017-03-18", "2017-03-19", "2017-03-20"], \
       ["2017-03-21", "2017-03-22", "2017-03-23"], \
       ["2017-03-24", "2017-03-25", "2017-03-26"], \
        ["2017-03-27", "2017-03-28", "2017-03-29", \
            "2017-03-30", "2017-03-31", "2017-04-01", \
            "2017-04-02", "2017-04-03","2017-04-07"]]
foldls = [[pd.to_datetime(d) for d in f] for f in foldls]
df['fold'] = -1
for t, fold in enumerate(foldls):
    df['fold'][df.activation_date.isin(fold)] = t
df['fold'].value_counts()
df.head()



df['ridge_img'] = featrdgimg['ridge_img_preds'].values
df = df.set_index('item_id')
df.drop(['index'], axis=1,inplace=True)
df.columns
del featusrttl, featusrcat, featusrprd, featenc, featrdgprc, featimgprc, featnumf, featprmenc, featprmtro, featencfst, featprtfst
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

# Training and Validation Set
lgbm_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective' : 'regression',
    'metric' : 'rmse',
    'num_leaves' : 1500,
    'nthread': 16,
    'learning_rate' : 0.02,
    'feature_fraction' : 0.5,
    'verbosity' : 0,
    'seed'      : 11
}    
    # Placeholder for predictions
df['fold'].value_counts()
y_pred_trn = pd.Series(-np.zeros(df.loc[traindex,:].shape[0]), index = traindex)
y_pred_tst = pd.Series(-np.zeros(df.loc[testdex ,:].shape[0]), index = testdex)
best_iters = []
bags       = 2
for bag in range(bags):
    for f in range(6):
        print('Fold %s'%(f) + ' [{}] Modeling Stage'.format(time.time() - start_time))
        trnidx = (df['fold'].loc[traindex] != f).values
        X_train = hstack([csr_matrix(df.drop('fold', 1).loc[traindex,:][trnidx].values),ready_df[0:traindex.shape[0]][trnidx]])
        y_train = y[trnidx]
        # 5 is the test fold
        if f == 5:
            X_test = hstack([csr_matrix(df.drop('fold', 1).loc[testdex,:].values),ready_df[traindex.shape[0]:]])
        else:
            X_test = hstack([csr_matrix(df.drop('fold', 1).loc[traindex,:][~trnidx].values),ready_df[0:traindex.shape[0]][~trnidx]])
            y_test  = y[~trnidx]
        tfvocab = df.drop('fold', 1).columns.tolist() + vectorizer.get_feature_names()
        for shape in [X_train, X_test]:
            print("Fold {} : {} Rows and {} Cols".format(f, *shape.shape))
        gc.collect();
        # LGBM Dataset Formatting 
        lgtrain = lgb.Dataset(X_train, y_train,
                        feature_name=tfvocab)
        if f != 5:
            lgvalid = lgb.Dataset(X_test, y_test,
                            feature_name=tfvocab)
        del X_train, y_train
        gc.collect()
    
        if f==5:
            best_iter = int(np.mean(best_iters))
            print('avg best iter: %s'%(best_iter))
            lgb_clf = lgb.train(
                lgbm_params,
                lgtrain,    
                num_boost_round = best_iter,
                verbose_eval=200)    
        else:
            lgb_clf = lgb.train(
                lgbm_params,
                lgtrain,    
                num_boost_round = 15000,
                valid_sets=[lgtrain, lgvalid],
                valid_names=['train','valid'],
                early_stopping_rounds=60,
                verbose_eval=100)    
            best_iters.append(lgb_clf.best_iteration)
        print("Model Evaluation Stage")
        if f == 5:
            y_pred_tst[:] += lgb_clf.predict(X_test)
        else:
            y_pred_trn[~trnidx] += lgb_clf.predict(X_test)
            print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_trn[~trnidx])))
        del X_test
        gc.collect()
        y_pred_trn.to_csv("lgbCV_1706_trn.csv",index=True)
        y_pred_tst.to_csv("lgbCV_1706_tst.csv",index=True)    

lgsub = pd.concat([y_pred_trn, y_pred_tst]).reset_index()
lgsub.rename(columns = {0 : 'deal_probability'}, inplace=True)
lgsub['deal_probability'] = lgsub['deal_probability']/bag
lgsub.set_index('item_id', inplace = True)
print('RMSE for all :', np.sqrt(metrics.mean_squared_error(y, lgsub.loc[traindex])))
# RMSE for all : 0.2168
lgsub.to_csv("lgCV_1706.csv.gz",index=True,header=True, compression = 'gzip')

lgsub.to_csv(path + "../sub/lgCV_1706.csv.gz",index=True,header=True, compression = 'gzip')

