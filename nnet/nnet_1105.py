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
from numba import jit
import nltk, re
from nltk.tokenize import ToktokTokenizer
from nltk.stem import PorterStemmer
import multiprocessing as mp
from collections import Counter



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
trnidx = (traindf.activation_date<=pd.to_datetime('2017-03-27')).values
validx = (traindf.activation_date>=pd.to_datetime('2017-03-26')).values


print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
df['idx'] = range(df.shape[0])
del traindf,testdf
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))    

print('[{}] Feature Engineering Price'.format(time.time() - start_time))
col = "price"
bins_ = 1000
df['emb_' + col] = pd.qcut(df[col], q = bins_, labels = False, duplicates = 'drop')
common_prices = df[col].value_counts()[(df[col].value_counts())>600].index.tolist()
idx_ = df[col].isin(common_prices)
df['emb_' + col ][idx_] = df[col][idx_]
df['emb_' + col].fillna(444.4,inplace=True)
df['emb_' + col].value_counts()
df["cont_price_log"] = np.log(df["price"]+0.001)
df["cont_price_log"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print('[{}] Create Time Variables'.format(time.time() - start_time))
df["emb_weekday"] = df['activation_date'].dt.weekday
df["emb_week_of_year"] = df['activation_date'].dt.week
df.drop(["activation_date","image"],axis=1,inplace=True)

print('[{}] Categoricals with some low counts'.format(time.time() - start_time))
def lowCtCat(col, cutoff = 20):
    dft         = pd.DataFrame(df[col].values, columns = [col])
    gp          = dft[col].value_counts().reset_index().rename(columns = {'index':col, col:col+'_ct'})
    var         = dft[[col]].merge(gp, on = col, how = 'left')[col+'_ct']
    idx         = var>cutoff
    var[idx]    = (df[col].values)[idx]
    var.fillna('missing', inplace = True)
    return var
for col_, cut_ in [("user_id", 10), ("image_top_1", 10), ("item_seq_number", 100)]: 
    df[col_] = lowCtCat(col_, cutoff = cut_)
for col_, cut_ in [('param_'+str(i+1), 20) for i in range(3)]: 
    df['cat_' + col_] = lowCtCat(col_, cutoff = cut_)

print('[{}] Encode Variables'.format(time.time() - start_time))
embde_me = ["item_seq_number", "user_id","image_top_1", "region", 'city', "emb_week_of_year", 'emb_price', \
            "parent_category_name", "category_name", "user_type", "emb_weekday"] + ['cat_param_'+str(i+1) for i in range(3)]
for col in embde_me:
    print('Label encode %s'%(col))
    lbl = preprocessing.LabelEncoder()
    df[col] = lbl.fit_transform(df[col].astype(str))

'''
print('[{}] Text Features'.format(time.time() - start_time))
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)
'''

print('[{}] Encode Sentences'.format(time.time() - start_time))
df['description'] = df['title'].fillna('misst') + ' ' + df['param_1'].fillna('missp') + ' ' + df['description'].fillna('missd')

for col in ['title', 'description']:
    df[col] = df[col].str.lower()
    
df[['title', 'description']].head()

toktok = ToktokTokenizer()
porter = PorterStemmer()
tokSentMap = {}
def tokSent(sent):
    sent = sent.replace('/', ' ')
    return " ".join(toktok.tokenize(rgx.sub(' ', sent)))

rgx = re.compile('[%s]' % '!"#%&()*,-./:;<=>?@[\\]^_`{|}~\t\n')   
pool = mp.Pool(processes=4)
df['description']    = pool.map(tokSent, df['description'].str.lower().tolist())
df['title']          = pool.map(tokSent, df['title'].str.lower().tolist())
print('[{}] Finished Tokenizing text...'.format(time.time() - start_time))
pool.close

@jit
def myTokenizerFitJit(strls, max_words, filt = True):
    list_=[]
    for sent in strls:
        if filt:
            sent = rgx.sub(' ', sent)
        for s in sent.split(' '):
            if s!= '':
                list_.append(s)
    return Counter(list_).most_common(max_words)

def myTokenizerFit(strls, max_words = 25000):
    mc = myTokenizerFitJit(strls, max_words)
    return dict((i, c+1) for (c, (i, ii)) in enumerate(mc))  

@jit
def fit_sequence(str_, tkn_, filt = True):
    labels = []
    for sent in str_:
        if filt:
            sent = rgx.sub(' ', sent)
        tk = []
        for i in sent.split(' '):
            if i in tkn_:
                if i != '':
                    tk.append(tkn_[i])
        labels.append(tk)
    return labels

tok_raw_dsc = myTokenizerFit(df['description'].loc[traindex].values, max_words = 60000)
tok_raw_ttl = myTokenizerFit(df['title'].loc[traindex].values, max_words = 15000)
