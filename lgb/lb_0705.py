
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
#path = '/Users/dhanley2/Documents/avito/data/'

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
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print('[{}] Feature Engineering'.format(time.time() - start_time))
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print('[{}] Create Time Variables'.format(time.time() - start_time))
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day
df.drop(["activation_date","image"],axis=1,inplace=True)

print('[{}] Encode Variables'.format(time.time() - start_time))
categorical = ["user_id","region","city","parent_category_name","category_name","item_seq_number","user_type","image_top_1"]
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
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

# Meta Text Features
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
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=18000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('text_feat',CountVectorizer(
            ngram_range=(1, 2),
            #max_features=7000,
            preprocessor=get_col('text_feat'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            #max_features=7000,
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
"""
Using Randomized train/valid split doesn't seem to generalize LB score, so I will try time cutoff
"""
    
print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 15,
    'bagging_fraction': 0.8,
    'learning_rate': 0.02,
    'verbose': 0
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
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=15000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=200,
    verbose_eval=200
)

# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig(path + '../plots/feature_import.png')

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgsub_1105.csv",index=True,header=True)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


'''
[200]   train's rmse: 0.227434  valid's rmse: 0.226827
[400]   train's rmse: 0.223803  valid's rmse: 0.22351
[600]   train's rmse: 0.221773  valid's rmse: 0.221732
[800]   train's rmse: 0.220291  valid's rmse: 0.220495
[1000]  train's rmse: 0.219051  valid's rmse: 0.219493
[1200]  train's rmse: 0.218031  valid's rmse: 0.218683
[1400]  train's rmse: 0.217154  valid's rmse: 0.217975
[1600]  train's rmse: 0.216353  valid's rmse: 0.217348
[1800]  train's rmse: 0.215652  valid's rmse: 0.216815
[2000]  train's rmse: 0.214952  valid's rmse: 0.216257
[2200]  train's rmse: 0.214342  valid's rmse: 0.215792
[2400]  train's rmse: 0.213752  valid's rmse: 0.215332
[2600]  train's rmse: 0.213216  valid's rmse: 0.214934
[2800]  train's rmse: 0.212687  valid's rmse: 0.214538
[3000]  train's rmse: 0.21218   valid's rmse: 0.214157
[3200]  train's rmse: 0.211712  valid's rmse: 0.213813
[3400]  train's rmse: 0.211246  valid's rmse: 0.213476
[3600]  train's rmse: 0.210799  valid's rmse: 0.213146
[3800]  train's rmse: 0.210372  valid's rmse: 0.212834
[4000]  train's rmse: 0.209945  valid's rmse: 0.212522
[4200]  train's rmse: 0.209531  valid's rmse: 0.212234
[4400]  train's rmse: 0.209144  valid's rmse: 0.211968
[4600]  train's rmse: 0.208738  valid's rmse: 0.211674
[4800]  train's rmse: 0.20837   valid's rmse: 0.211417
[5000]  train's rmse: 0.208015  valid's rmse: 0.211159
[5200]  train's rmse: 0.207656  valid's rmse: 0.2109
[5400]  train's rmse: 0.207291  valid's rmse: 0.21065
[5600]  train's rmse: 0.206948  valid's rmse: 0.210418
[5800]  train's rmse: 0.206604  valid's rmse: 0.210178
[6000]  train's rmse: 0.206251  valid's rmse: 0.209935
[6200]  train's rmse: 0.205927  valid's rmse: 0.209705
[6400]  train's rmse: 0.205609  valid's rmse: 0.209489
[6600]  train's rmse: 0.205294  valid's rmse: 0.209274
[6800]  train's rmse: 0.204972  valid's rmse: 0.209062
[7000]  train's rmse: 0.204656  valid's rmse: 0.208846
[7200]  train's rmse: 0.204355  valid's rmse: 0.20863
[7400]  train's rmse: 0.204049  valid's rmse: 0.20842
[7600]  train's rmse: 0.203761  valid's rmse: 0.208231
[7800]  train's rmse: 0.203467  valid's rmse: 0.208025
[8000]  train's rmse: 0.203166  valid's rmse: 0.207822
[8200]  train's rmse: 0.202872  valid's rmse: 0.207628
[8400]  train's rmse: 0.202588  valid's rmse: 0.20743
[8600]  train's rmse: 0.202292  valid's rmse: 0.207238
[8800]  train's rmse: 0.202     valid's rmse: 0.207048
[9000]  train's rmse: 0.201728  valid's rmse: 0.206861
[9200]  train's rmse: 0.201452  valid's rmse: 0.20667
[9400]  train's rmse: 0.201174  valid's rmse: 0.206491
[9600]  train's rmse: 0.200894  valid's rmse: 0.206294
[9800]  train's rmse: 0.200638  valid's rmse: 0.206128
[10000] train's rmse: 0.200366  valid's rmse: 0.205949
[10200] train's rmse: 0.200102  valid's rmse: 0.205776
[10400] train's rmse: 0.199846  valid's rmse: 0.205597
[10600] train's rmse: 0.199599  valid's rmse: 0.205437
[10800] train's rmse: 0.199342  valid's rmse: 0.20527
[11000] train's rmse: 0.199095  valid's rmse: 0.205115
[11200] train's rmse: 0.198839  valid's rmse: 0.204952
[11400] train's rmse: 0.198601  valid's rmse: 0.204801
[11600] train's rmse: 0.198352  valid's rmse: 0.204632
[11800] train's rmse: 0.198109  valid's rmse: 0.204473
[12000] train's rmse: 0.197867  valid's rmse: 0.204317
[12200] train's rmse: 0.197622  valid's rmse: 0.204155
[12400] train's rmse: 0.197398  valid's rmse: 0.204011
[12600] train's rmse: 0.197169  valid's rmse: 0.203866
[12800] train's rmse: 0.196942  valid's rmse: 0.203717
[13000] train's rmse: 0.196708  valid's rmse: 0.203565
[13200] train's rmse: 0.196486  valid's rmse: 0.203416
[13400] train's rmse: 0.196248  valid's rmse: 0.203261
[13600] train's rmse: 0.196021  valid's rmse: 0.203115
[13800] train's rmse: 0.195784  valid's rmse: 0.202968
[14000] train's rmse: 0.195555  valid's rmse: 0.202821
[14200] train's rmse: 0.195331  valid's rmse: 0.202667
[14400] train's rmse: 0.195115  valid's rmse: 0.20253
[14600] train's rmse: 0.194889  valid's rmse: 0.202392
[14800] train's rmse: 0.194674  valid's rmse: 0.202245
[15000] train's rmse: 0.194457  valid's rmse: 0.202105
Did not meet early stopping. Best iteration is:
[15000] train's rmse: 0.194457  valid's rmse: 0.202105
'''
