# https://www.kaggle.com/darraghdog/mem-check-1002-ftrlrnnverified
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
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
import plotly.offline as plt
import plotly.graph_objs as go
from numba import jit
import nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
import pymorphy2
import multiprocessing as mp
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import random
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras.layers import GlobalMaxPooling1D
from keras import initializers
from keras.utils import plot_model
from keras.regularizers import l2
from keras.constraints import non_neg, Constraint
from keras.utils.data_utils import Sequence
from sklearn import preprocessing
import math

path = "../"
#path = '../input/'
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'
path = '/home/ubuntu/avito/data/'

start_time = time.time()

validation = False
full       = True
CV   = True

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
traindf.head()

print('[{}] Create Validation Index'.format(time.time() - start_time))
if full:
    trnidx = (traindf.activation_date<=pd.to_datetime('2017-03-28')).values
    validx = (traindf.activation_date>=pd.to_datetime('2017-03-29')).values
else:
    trnidx = (traindf.activation_date<=pd.to_datetime('2017-03-26')).values
    validx = (traindf.activation_date>=pd.to_datetime('2017-03-27')).values

print('[{}] Load Densenet image features'.format(time.time() - start_time))
dnimgtrn = np.load(path+'../features/densenet_pool_array_train.npy')
dnimgtrn = dnimgtrn
scaler = preprocessing.StandardScaler()
dnimgtrn = scaler.fit_transform(dnimgtrn)
gc.collect()
dnimgtst = np.load(path+'../features/densenet_pool_array_test.npy')
dnimgtst = scaler.transform(dnimgtst)
gc.collect()


print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
df['idx'] = range(df.shape[0])
del traindf,testdf
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

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

'''
print('SHAPES....')
print(dnimgtrn.shape)
print(trnidx.shape)

print(dnimgtst.shape)
print(tstidx.shape)


print(dnimgval.shape)
print(validx.shape)
f = 0
print('Fold %s'%(f) + ' [{}] Modeling Stage'.format(time.time() - start_time))
trnidx = (df['fold'].loc[traindex] != f).values
dnfimgtrn = dnimgtrn[trnidx]
'''


'''
print('[{}] Load engineered user features'.format(time.time() - start_time))
featusrttl = pd.read_csv(path + '../features/user_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrcat = pd.read_csv(path + '../features/usercat_agg.csv.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrprd = pd.read_csv(path + '../features/user_activ_period_stats.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
featusrttl.rename(columns={'title': 'all_titles'}, inplace = True)
keep = ['user_id', 'all_titles', 'user_avg_price', 'user_ad_ct']
df = df.reset_index().merge(featusrttl[keep], on = 'user_id').set_index('item_id')
keep = ['user_id', 'parent_category_name', 'usercat_avg_price', 'usercat_ad_ct']
gc.collect()
df = df.reset_index().merge(featusrcat[keep], on = ['user_id', 'parent_category_name']).set_index('item_id')
keep = ['user_id', 'user_activ_sum', 'user_activ_mean', 'user_activ_var']
gc.collect()
df = df.reset_index().merge(featusrprd[keep], on = ['user_id'], how = 'left').set_index('item_id')
usr_cols = ['user_avg_price', 'user_ad_ct', 'usercat_avg_price', 'usercat_ad_ct']  
usr_cols += ['user_activ_sum', 'user_activ_mean', 'user_activ_var']
for col in usr_cols:
    #df["bin_no_" + col] = (df[col].isna()).astype(np.int32)
    df[col].fillna(0, inplace = True)
    df[col] = np.log1p(df[col]+0.111).values
    df.rename(columns = { col : col+'_cont'}, inplace = True)
'''

print('[{}] Load bayes mean features'.format(time.time() - start_time))
featenc = pd.read_csv(path + '../features/alldf_bayes_mean.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
for col in featenc.columns:
    featenc.rename(columns = { col : 'cont_'+col}, inplace = True)
df = pd.concat([df.reset_index(),featenc],axis=1)
df = df.set_index('item_id')
df.head()

df.sort_values('idx', inplace = True)


print('[{}] Load engineered price ratio features'.format(time.time() - start_time))

featrdgprc = pd.read_csv(path + '../features/price_category_ratios.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
for col in featrdgprc.columns:
    featrdgprc[col].fillna(featrdgprc[col].median(), inplace = True)
featrdgprc.isnull().sum()
for col in [c for c in featrdgprc.columns if '_log' not in c]:
    featrdgprc[col] = np.log1p(featrdgprc[col]+.001111)
featrdgprc.columns = [c+'_cont' for c in featrdgprc.columns]
featrdgprc.index = df.index
df = pd.concat([df, featrdgprc], axis = 1)
del featrdgprc
gc.collect()

print('[{}] Manage Memory'.format(time.time() - start_time))
for col in df.columns:
    if np.float64 == df[col].dtype:
        df[col] = df[col].astype(np.float32)
    if np.int64 == df[col].dtype:
        df[col] = df[col].astype(np.int32)
    gc.collect()
df.dtypes

'''
print('[{}] Load translated features'.format(time.time() - start_time))
usecols = [c + '_translated' for c in ['title', 'description', 'param_1', 'param_2', 'param_3']]+['item_id']
dftrl = pd.concat([pd.read_csv(path + '../features/translate_trn_en.csv.gz', usecols = usecols, compression = 'gzip').set_index('item_id'),
            pd.read_csv(path + '../features/translate_tst_en.csv.gz', usecols = usecols, compression = 'gzip').set_index('item_id')])
dftrl.columns = [c.replace('_translated', '') for c in dftrl.columns]
gc.collect()
'''

print('[{}] Missing values'.format(time.time() - start_time))
for col in ['param_1', 'param_2', 'param_3', 'description', 'price', 'image']:
    df["bin_no_" + col] = (df[col]!=df[col]).astype(np.int32)  #(df[col].isna()).astype(np.int32)
cols = [c for c in df.columns if 'bin_no_' in c]
df[cols].head()

print('[{}] Feature Engineering Price'.format(time.time() - start_time))
col = "price"
bins_ = 100
for col in ['price', 'item_seq_number']:
    df['emb_' + col] = pd.qcut(df[col], q = bins_, labels = False, duplicates = 'drop')
    df['emb_' + col].fillna(-1,inplace=True)
    df['emb_' + col].value_counts()
    df["cont_log_%s"%(col)] = np.log(df[col]+0.001)
    df["cont_log_%s"%(col)].fillna(-1,inplace=True)
df["image_top_1"].fillna(-1,inplace=True)


print('[{}] Create Time Variables'.format(time.time() - start_time))
df["emb_weekday"] = df['activation_date'].dt.weekday
#df["cont_week_of_year"] = df['activation_date'].dt.week    <- Too different between train and test
df.drop(["activation_date","image"],axis=1,inplace=True)

print('[{}] Text Features'.format(time.time() - start_time))
def expand_description(df_, category = True):
    df_['text_feat'] = df_.apply(lambda row: ' '.join([
        str(row['param_1']),
        str(row['param_2']),
        str(row['param_3'])]),axis=1) # Group Param Features
    for col in ['title', 'description', 'text_feat']:
        df_[col] = df_[col].str.lower()
    if category:
        df_['description'] = df_['title'].fillna('missd') + ' ' + df_["parent_category_name"].fillna('') + ' ' \
                            + df_["category_name"].fillna('') + ' ' + df_['text_feat'].fillna('') + \
                            ' ' + df_['description'].fillna('')
    else:
        df_['description'] = df_['title'].fillna('missd') + ' ' + df_['text_feat'].fillna('') + \
                            ' ' + df_['description'].fillna('')
        df_.drop(['param_1', 'param_2', 'param_3', 'text_feat'], axis = 1, inplace = True)
    return df_
df =  expand_description(df)
# dftrl =  expand_description(dftrl, category = False)

print('[{}] Categoricals with some low counts'.format(time.time() - start_time))
def lowCtCat(col, cutoff = 20):
    dft         = pd.DataFrame(df[col].values, columns = [col])
    gp          = dft[col].value_counts().reset_index().rename(columns = {'index':col, col:col+'_ct'})
    var         = dft[[col]].merge(gp, on = col, how = 'left')[col+'_ct']
    idx         = var>cutoff
    var[idx]    = (df[col].values)[idx]
    var[~idx]    = 'locount'
    var.fillna('missing', inplace = True)
    return var.astype(str).values
for col_, cut_ in [("user_id", 5), ("image_top_1", 30), ("item_seq_number", 100)]:
    df[col_] = lowCtCat(col_, cutoff = cut_)
for col_, cut_ in [('param_'+str(i+1), 20) for i in range(3)]:
    df['cat_' + col_] = lowCtCat(col_, cutoff = cut_)


print('[{}] Encode Variables'.format(time.time() - start_time))
embed_me = ["emb_item_seq_number", "user_id","image_top_1", "region", 'city', 'emb_price', \
            "parent_category_name", "category_name", "user_type", "emb_weekday", 'text_feat'] \
            + ['cat_param_'+str(i+1) for i in range(3)]
for col in embed_me:
    print('Label encode %s'%(col))
    lbl = preprocessing.LabelEncoder()
    df[col] = lbl.fit_transform(df[col].astype(str))

print('[{}] Scale Variables'.format(time.time() - start_time))
scl = StandardScaler()
for col in df.columns:
    if 'cont_' in col:
        print('Scale %s'%(col))
        df[col] = scl.fit_transform(df[col].values.reshape(-1, 1)).flatten()

print('[{}] Embedding dimensions'.format(time.time() - start_time))
col_szs = dict((col, df[col].unique().shape[0]) for col in embed_me)
embed_szs = dict((col, int(np.ceil(np.log(col_szs[col])))+1) for col in embed_me)
print(embed_szs)

print('[{}] Clean text and tokenize'.format(time.time() - start_time))
toktok = ToktokTokenizer()
tokSentMap = {}
stopwords_set = set(stopwords.words('russian'))
morpher = pymorphy2.MorphAnalyzer()
def tokSent(sent):
   sent = sent.replace('/', ' ')
   return " ".join(morpher.parse(word)[0].normal_form for word in toktok.tokenize(rgx.sub(' ', sent)) if word not in stopwords_set)

def tokCol(var):
   return [tokSent(s) for s in var.tolist()]
rgx = re.compile('[%s]' % '!"#%&()*,-./:;<=>?@[\\]^_`{|}~\t\n')

from multiprocessing import cpu_count, Pool

partitions = min(cpu_count(), 8) #Define as many partitions as you want
cores=4
def parallelize(data, func):
   data_split = np.array_split(data.values, partitions)
   pool = Pool(cores)
   data = pd.concat([pd.Series(l) for l in pool.map(tokCol, data_split)]).values
   pool.close()
   pool.join()
   return data

for col in ['description', 'title',]:
   print('Tokenise %s'%(col))
   df[col] = parallelize(df[col], tokCol)
   # dftrl[col] = parallelize(dftrl[col], tokCol)
print('[{}] Finished tokenizing text...'.format(time.time() - start_time))

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

print('[{}] Finished FITTING TEXT DATA...'.format(time.time() - start_time))
tok_raw = myTokenizerFit(df['description'].loc[traindex].values.tolist()+ \
                         df['title'].loc[traindex].values.tolist(), max_words = 80000)
                         #dftrl['title'].loc[traindex].values.tolist()+ \
                         #dftrl['description'].loc[traindex].values.tolist(), max_words = 80000)
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))
df["title"]       = fit_sequence(df.title, tok_raw)
df["description"] = fit_sequence(df.description, tok_raw)
#df["title_translated"]       = fit_sequence(dftrl.title, tok_raw)
#df["description_translated"] = fit_sequence(dftrl.description, tok_raw)
df["title"]       = [l if len(l)>0 else [0] for l in df["title"]]
gc.collect()
#del dftrl
gc.collect()


MAX_DSC = max(tok_raw.values())+1
MAX_TTL = max(tok_raw.values())+1

bin_cols = [c for c in df.columns if 'bin_no' in c]
cont_cols = [c for c in df.columns if ('cont_' in c) or ('_cont' in c)]
print('Continuous Columns : %s'%(' '.join(cont_cols) ))


#TEST DOESNT HAVE ANY 1s
bin_cols=[x for x in bin_cols if x!='bin_no_description']


print('[{}] Finished FEATURE CREATION'.format(time.time() - start_time))



def map_sort(seq1, seq2):
	return sorted(range(len(seq1)), key=lambda x: max(len(seq1[x]),len(seq2[x])))


class Seq_generator(Sequence):
    def __init__(self, dt, dn, bsize, sort_vals, target_out=True):
        self.batch_size = bsize
        self.dt = dt.iloc[sort_vals].reset_index(drop=True)
        self.dn = dn[sort_vals]
        if target_out:
            self.y = self.dt.target.values
        else:
            self.y = None

    def get_keras_data(self, dataset, dndataset):
        X = {
            'title': pad_sequences(dataset.title,
                                  maxlen=max([len(l) for l in dataset.title]))
            ,'description': pad_sequences(dataset.description,
                                  maxlen=max([len(l) for l in dataset.description]))
            #,'title_translated': pad_sequences(dataset.title_translated,
            #                      maxlen=max([len(l) for l in dataset.title_translated]))
            #,'description_translated': pad_sequences(dataset.description_translated,
            #                      maxlen=max([len(l) for l in dataset.description_translated]))
            }
        for col in embed_szs.keys():
            X[col] = dataset[col].values
        X['bin_vars'] = dataset[bin_cols].values
        X['cont_vars'] = dataset[cont_cols].values
        X['img_layer'] = dndataset
        return X

    def __len__(self):
        return int(np.ceil(self.dt.shape[0]*1./self.batch_size))

    def __getitem__(self, i):
        slc = slice(i*self.batch_size, min((i+1)*self.batch_size, self.dt.shape[0]))
        X = self.get_keras_data(self.dt.iloc[slc], self.dn[slc])
        if self.y is not None:
            return (X, self.y[slc])
        return X

'''
dtrain = df.loc[traindex,:][trnidx].reset_index()
dvalid = df.loc[traindex,:][validx].reset_index()
dtest  = df.loc[testdex,:].reset_index()
dtrain['target'] = y[trnidx].values
dvalid['target'] = y[validx].values


train_sorted_ix = np.array(map_sort(dtrain["title"].tolist(), dtrain["description"].tolist()))
val_sorted_ix = np.array(map_sort(dvalid["title"].tolist(), dvalid["description"].tolist()))
tst_sorted_ix = np.array(map_sort(dtest ["title"].tolist(), dtest ["description"].tolist()))

y_pred_epochs = []
'''

def get_model(emb_size = 32, dr = 0.1, l2_val = 0.0001):

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


    def mean_error(y_true, y_pred):
        return K.mean(y_true-y_pred)

    class FreezePadding(Constraint):
        """Freezes the last weight to be near 0."""
        def __call__(self, w):
            other_weights = K.cast(K.ones(K.shape(w))[1:], K.floatx())
            last_weight = K.cast(K.equal(K.reshape(w[0, :], (1, K.shape(w)[1])), 0.), K.floatx())
            appended = K.concatenate([other_weights, last_weight], axis=0)
            w *= appended
            return w

    ##Inputs
    title = Input(shape=[None], name="title")
    description = Input(shape=[None], name="description")
    #title_translated = Input(shape=[None], name="title_translated")
    #description_translated = Input(shape=[None], name="description_translated")
    img_layer = Input(shape=[dnimgtrn.shape[1]], name="img_layer")
    

    # Categorical embeddings
    emb_inputs = dict((col, Input(shape=[1], name = col))  for col in embed_szs.keys())
    emb_model  = dict((col, Embedding(col_szs[col]+1, emb_n, embeddings_regularizer=l2(l2_val))(emb_inputs[col])) for (col, emb_n) in embed_szs.items())
    fe = concatenate([(emb_) for emb_ in emb_model.values()])
    #fe = SpatialDropout1D(dr)(fe)
    
    dense_l = Dropout(0.1) (Dense(256,activation='relu', kernel_regularizer=l2(l2_val)) (img_layer))
    dense_l = Dropout(0.1) (Dense(32,activation='relu', kernel_regularizer=l2(l2_val)) (dense_l))

    # Binary Inputs
    bin_vars = Input(shape= [len(bin_cols)], name = 'bin_vars')
    ## Continuous Inputs
    cont_vars = Input(shape= [len(cont_cols)], name = 'cont_vars')

    #Embeddings layers
    #emb_size = 64
    embs_text = Embedding(MAX_DSC, emb_size, embeddings_regularizer=l2(l2_val), embeddings_constraint=FreezePadding())
    emb_dsc = embs_text(description)
    emb_ttl = embs_text(title)    
    #emb_dsct = embs_text(description_translated)
    #emb_ttlt = embs_text(title_translated)

    # GRU Layer
    rnn_dsc = (CuDNNGRU(emb_size))(emb_dsc)
    rnn_ttl = (CuDNNGRU(emb_size)) (emb_ttl)
    #rnn_dsct = (CuDNNGRU(emb_size))(emb_dsct)
    #rnn_ttlt = (CuDNNGRU(emb_size)) (emb_ttlt)
    #rnn_dsc = (GRU(emb_size))(emb_dsc)
    #rnn_ttl = (GRU(emb_size)) (emb_ttl)

    #main layer
    main_l = concatenate([
        rnn_dsc
        , rnn_ttl
        #, rnn_dsct
        #, rnn_ttlt
        , Flatten()(fe)
        , bin_vars
        , cont_vars
        , dense_l
    ])  
    #main_l = BatchNormalization()(main_l)
    main_l = Dense(256, kernel_regularizer=l2(l2_val)) (main_l)
    main_l = PReLU()(main_l)
    main_l = Dropout(dr)(main_l)
    #main_l = Dense(256, kernel_regularizer=l2(l2_val)) (main_l)
    #main_l = PReLU()(main_l)
    #main_l = BatchNormalization()(main_l)
    #main_l = Dropout(dr)(main_l)
    main_l = Dense(32, kernel_regularizer=l2(l2_val)) (main_l)
    main_l = PReLU()(main_l)
    #main_l = BatchNormalization()(main_l)
    main_l = Dropout(dr/2)(main_l)

    #output
    output = Dense(1,activation="linear", kernel_regularizer=l2(l2_val)) (main_l)

    #model
    model = Model([title, description] + \
                  [inp for inp in emb_inputs.values()] + [bin_vars] + [cont_vars] + [img_layer], output)
                    # , title_translated, description_translated
    #optimizer = optimizers.Adam(clipnorm=10)
    optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(loss=root_mean_squared_error,
                  optimizer=optimizer, metrics=['mae'])

    return model

gc.collect()


        
def to_logit(ls):
    ls=np.array(ls)
    ls=np.clip(ls,.0001,.9999)
    return np.log(ls/(1-ls))

def to_proba(ls):
    return 1/(1+np.exp(-ls))

# Placeholder for predictions
df['fold'].value_counts()
y_pred_trn = pd.Series(-np.zeros(df.loc[traindex,:].shape[0]), index = traindex)
y_pred_tst = pd.Series(-np.zeros(df.loc[testdex ,:].shape[0]), index = testdex)
for f in range(5, 6):
    print('Fold %s'%(f) + ' [{}] Modeling Stage'.format(time.time() - start_time))
    trnidx = (df['fold'].loc[traindex] != f).values
    dtrain = df.loc[traindex,:][trnidx].reset_index()
    dtrain['target'] = y[trnidx].values
    dnfimgtrn = dnimgtrn[trnidx]
    # 5 is the test fold
    if f == 5:
        dtest  = df.loc[testdex,:].reset_index()
        dnfimgtst = dnimgtst
    else:
        dtest  = df.loc[traindex,:][~trnidx].reset_index()
        dnfimgtst = dnimgtrn[~trnidx]
        dtest['target'] = y[~trnidx].values
    
    train_sorted_ix = np.array(map_sort(dtrain["title"].tolist(), dtrain["description"].tolist()))
    tst_sorted_ix = np.array(map_sort(dtest ["title"].tolist(), dtest ["description"].tolist()))
    
    y_pred_epochs = []
    epochs = 28
    batchSize = 512
    steps = (dtrain.shape[0]/batchSize+1)*epochs
    lr_init, lr_fin = 0.0014, 0.00001
    lr_decay  = (lr_init - lr_fin)/steps


    bags      = 3
    y_pred_ls = []
    y_sub_ls  = []
    for b in range(bags):
        model = get_model(128, .1,.00001)
        K.set_value(model.optimizer.lr, lr_init)
        K.set_value(model.optimizer.decay, lr_decay)
        #model.summary()
        for i in range(epochs):
            batchSize = min(512*(2**i),512)
            batchSizeTst = 256
            history = model.fit_generator(
                                Seq_generator(dtrain, dnfimgtrn, batchSize, train_sorted_ix)
                                , epochs=i+1
                                , max_queue_size=15
                                , verbose=2
                                , initial_epoch=i
                                , use_multiprocessing=True
                                , workers=3
                                )
            if i>3:
                y_sub_ls.append(model.predict_generator(
                                Seq_generator(dtest, dnfimgtst, batchSizeTst, tst_sorted_ix, target_out=False)
                                , max_queue_size=10
                                , verbose=2)[tst_sorted_ix.argsort()])
                if f == 5:
                    if len(y_sub_ls)>1:
                        y_pred_tst[:] = sum(y_sub_ls)/len(y_sub_ls)
                else:
                    print('RMSE:', np.sqrt(metrics.mean_squared_error(dtest['target'], y_sub_ls[-1].flatten())))
                    if len(y_sub_ls)>1:
                        y_pred = sum(y_sub_ls)/len(y_sub_ls)
                        print('RMSE bags:', np.sqrt(metrics.mean_squared_error(dtest['target'], y_pred.flatten()))) 
                        y_pred_trn[~trnidx] = y_pred
            gc.collect()
    #y_pred_trn.to_csv("rnndhCV_0506_trn.csv",index=True)
    y_pred_tst.to_csv("rnndhCV_0506_tst.csv",index=True) 
    del dtrain, dtest, dnfimgtrn, dnfimgtst
    gc.collect()

rnnsub = pd.concat([y_pred_trn, y_pred_tst]).reset_index()
rnnsub.rename(columns = {0 : 'deal_probability'}, inplace=True)
rnnsub['deal_probability'].clip(0.0, 1.0, inplace=True)
rnnsub.set_index('item_id', inplace = True)
print('RMSE for all :', np.sqrt(metrics.mean_squared_error(y, rnnsub.loc[traindex])))
# RMSE for all : 0.2168
rnnsub.to_csv("../sub/rnndhCV_0506.csv.gz",index=True,header=True, compression = 'gzip')
















'''

Epoch 1/1
2489/2492 [============================>.] - ETA: 0s - loss: 0.2349 - mean_absolute_error: 0.1496     Epoch 1/1
2492/2492 [==============================] - 71s 28ms/step - loss: 0.2348 - mean_absolute_error: 0.1496
RMSE: 0.21985621116345772
Epoch 2/2
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2272 - mean_absolute_error: 0.1456
RMSE: 0.21793590569493146
RMSE bags: 0.21792574992504865
Epoch 3/3
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2255 - mean_absolute_error: 0.1439
RMSE: 0.21712131928225736
RMSE bags: 0.21699600061434435
Epoch 4/4
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2246 - mean_absolute_error: 0.1428
RMSE: 0.21610996979925132
RMSE bags: 0.21627170564910336
Epoch 5/5
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2238 - mean_absolute_error: 0.1419
RMSE: 0.2159609722998397
RMSE bags: 0.2156799962490314
Epoch 6/6
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2233 - mean_absolute_error: 0.1411
RMSE: 0.2155344516861518
RMSE bags: 0.2153211645407326
Epoch 7/7
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2230 - mean_absolute_error: 0.1405
RMSE: 0.21551617273936377
RMSE bags: 0.21503730970522453
Epoch 8/8
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2226 - mean_absolute_error: 0.1397
RMSE: 0.21548786100245093
RMSE bags: 0.21482960198354567
Epoch 9/9
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2225 - mean_absolute_error: 0.1392
RMSE: 0.2154573516600981
RMSE bags: 0.21466066180563864
Epoch 10/10
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2222 - mean_absolute_error: 0.1384
RMSE: 0.21580469184790685
RMSE bags: 0.21453566109187186
Epoch 11/11
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2218 - mean_absolute_error: 0.1377
RMSE: 0.21550720798190948
RMSE bags: 0.21441288168130332
Epoch 12/12
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2214 - mean_absolute_error: 0.1369
Epoch 12/12
RMSE: 0.2166922086308751
RMSE bags: 0.2143186507634938
Epoch 13/13
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2212 - mean_absolute_error: 0.1364
RMSE: 0.2156060527421784
RMSE bags: 0.21419914263851797
Epoch 14/14
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2213 - mean_absolute_error: 0.1360
RMSE: 0.21599870241914323
RMSE bags: 0.2141337458282634
Epoch 15/15
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2221 - mean_absolute_error: 0.1360
RMSE: 0.215773220383164
RMSE bags: 0.21404439949219406
Epoch 16/16
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2216 - mean_absolute_error: 0.1354
RMSE: 0.21594672840701643
RMSE bags: 0.21398780559232394
Epoch 17/17
2489/2492 [============================>.] - ETA: 0s - loss: 0.2213 - mean_absolute_error: 0.1351   Epoch 17/17
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2214 - mean_absolute_error: 0.1351
RMSE: 0.21602352062186472
RMSE bags: 0.21393535525836174
Epoch 18/18
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2209 - mean_absolute_error: 0.1346
RMSE: 0.21604935445261483
RMSE bags: 0.21386952594126943
Epoch 19/19
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2206 - mean_absolute_error: 0.1342
RMSE: 0.21652610746586287
RMSE bags: 0.2138229456738561
Epoch 20/20
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2223 - mean_absolute_error: 0.1344
RMSE: 0.21799218524166106
RMSE bags: 0.21380270474005655
Epoch 21/21
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2228 - mean_absolute_error: 0.1345
RMSE: 0.21727741969091502
RMSE bags: 0.21375875966523503
Epoch 22/22
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2228 - mean_absolute_error: 0.1344
RMSE: 0.21812397882559076
RMSE bags: 0.21374454681808844
Epoch 23/23
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2223 - mean_absolute_error: 0.1340
RMSE: 0.21625086526756268
RMSE bags: 0.21371160638806588
Epoch 24/24
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2215 - mean_absolute_error: 0.1335
RMSE: 0.21659540782156436
RMSE bags: 0.21367538285761095
Epoch 25/25
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2209 - mean_absolute_error: 0.1332
RMSE: 0.2165070771204546
RMSE bags: 0.2136673261870648
Epoch 26/26
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2208 - mean_absolute_error: 0.1330
RMSE: 0.21785276916725976
RMSE bags: 0.21364576583685346
Epoch 27/27
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2207 - mean_absolute_error: 0.1325
RMSE: 0.21782723884272207
RMSE bags: 0.21363646607650713
Epoch 28/28
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2206 - mean_absolute_error: 0.1323
RMSE: 0.216986778296493
RMSE bags: 0.21361035188919517
Epoch 1/1
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2346 - mean_absolute_error: 0.1496
Epoch 1/1
RMSE: 0.219599362952194
RMSE bags: 0.21364201586612308
Epoch 2/2
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2271 - mean_absolute_error: 0.1456
RMSE: 0.21756606342132842
RMSE bags: 0.21366056990047644
Epoch 3/3
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2256 - mean_absolute_error: 0.1439
RMSE: 0.2169791553664575
RMSE bags: 0.21367767471701324
Epoch 4/4
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2245 - mean_absolute_error: 0.1429
RMSE: 0.2166907522681674
RMSE bags: 0.21369360909533588
Epoch 5/5
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2238 - mean_absolute_error: 0.1420
RMSE: 0.21694041896913788
RMSE bags: 0.21370630347910652
Epoch 6/6
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2236 - mean_absolute_error: 0.1414
RMSE: 0.21600917557970065
RMSE bags: 0.21369602543032307
Epoch 7/7
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2230 - mean_absolute_error: 0.1406
RMSE: 0.21619002303759985
RMSE bags: 0.21369506866687876
Epoch 8/8
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2228 - mean_absolute_error: 0.1399
RMSE: 0.2157734755973261
RMSE bags: 0.21368477010856654
Epoch 9/9
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2226 - mean_absolute_error: 0.1393
RMSE: 0.2152520275420221
RMSE bags: 0.2136727054236177
Epoch 10/10
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2224 - mean_absolute_error: 0.1387
RMSE: 0.21617819187928577
RMSE bags: 0.21367612401804503
Epoch 11/11
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2220 - mean_absolute_error: 0.1378
RMSE: 0.21549791822932785
RMSE bags: 0.2136584945268385
Epoch 12/12
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2216 - mean_absolute_error: 0.1372
RMSE: 0.2159367027570734
RMSE bags: 0.21365874030687146
Epoch 13/13
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2213 - mean_absolute_error: 0.1367
RMSE: 0.21598605011680228
RMSE bags: 0.21364745128035204
Epoch 14/14
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2211 - mean_absolute_error: 0.1361
RMSE: 0.21568760329589987
RMSE bags: 0.21363519260675304
Epoch 15/15
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2210 - mean_absolute_error: 0.1357
RMSE: 0.2165167321782653
RMSE bags: 0.21362075313132728
Epoch 16/16
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2207 - mean_absolute_error: 0.1352
RMSE: 0.2167496387073038
RMSE bags: 0.21361015029081792
Epoch 17/17
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2207 - mean_absolute_error: 0.1348
RMSE: 0.21604293139907196
RMSE bags: 0.21359742908345547
Epoch 18/18
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2207 - mean_absolute_error: 0.1344
RMSE: 0.21581631542950394
RMSE bags: 0.2135813763304528
Epoch 19/19
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2205 - mean_absolute_error: 0.1341
RMSE: 0.21632431571467522
RMSE bags: 0.2135717950173114
Epoch 20/20
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2207 - mean_absolute_error: 0.1337
RMSE: 0.21669397956182543
RMSE bags: 0.21355511134765356
Epoch 21/21
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2205 - mean_absolute_error: 0.1332
RMSE: 0.21732181079991877
RMSE bags: 0.21353927135462167
Epoch 22/22
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2206 - mean_absolute_error: 0.1329
Epoch 22/22
RMSE: 0.21774381787419053
RMSE bags: 0.21353159177971667
Epoch 23/23
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2206 - mean_absolute_error: 0.1325
RMSE: 0.21789482861877404
RMSE bags: 0.21352132486725683
Epoch 24/24
2492/2492 [==============================] - 62s 25ms/step - loss: 0.2214 - mean_absolute_error: 0.1323
RMSE: 0.21687340418135395
RMSE bags: 0.21350344810272634
Epoch 25/25
2492/2492 [==============================] - 62s 25ms/step - loss: 0.3081 - mean_absolute_error: 0.1367
RMSE: 0.2196382307565968
RMSE bags: 0.21350355710300192
Epoch 26/26
2492/2492 [==============================] - 62s 25ms/step - loss: 0.4601 - mean_absolute_error: 0.1407
RMSE: 0.21877706186693852
RMSE bags: 0.21349967081491064
Epoch 27/27
2492/2492 [==============================] - 62s 25ms/step - loss: 0.5131 - mean_absolute_error: 0.1403
RMSE: 0.21920902694359526
RMSE bags: 0.21349788912015047
Epoch 28/28
2492/2492 [==============================] - 62s 25ms/step - loss: 0.5696 - mean_absolute_error: 0.1404
RMSE: 0.21994297398257676
RMSE bags: 0.2135039377181423
Epoch 1/1
2490/2492 [============================>.] - ETA: 0s - loss: 0.2347 - mean_absolute_error: 0.1495     Epoch 1/1
2492/2492 [==============================] - 67s 27ms/step - loss: 0.2347 - mean_absolute_error: 0.1495
RMSE: 0.21934124365964477
RMSE bags: 0.21352975125550974
Epoch 2/2
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2272 - mean_absolute_error: 0.1455
RMSE: 0.21818887262178432
RMSE bags: 0.21354846708432135
Epoch 3/3
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2256 - mean_absolute_error: 0.1440
RMSE: 0.21703244411660336
RMSE bags: 0.21356043780007428
Epoch 4/4
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2245 - mean_absolute_error: 0.1427
RMSE: 0.21657861774704706
RMSE bags: 0.2135711913363945
Epoch 5/5
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2239 - mean_absolute_error: 0.1418
RMSE: 0.21648916606219762
RMSE bags: 0.21357374723064987
Epoch 6/6
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2233 - mean_absolute_error: 0.1410
RMSE: 0.21590358756435774
RMSE bags: 0.21357626706302693
Epoch 7/7
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2231 - mean_absolute_error: 0.1406
RMSE: 0.2156174197095797
RMSE bags: 0.2135786804981791
Epoch 8/8
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2228 - mean_absolute_error: 0.1398
RMSE: 0.21584325006974076
RMSE bags: 0.2135744875120727
Epoch 9/9
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2226 - mean_absolute_error: 0.1392
RMSE: 0.21671553202565857
RMSE bags: 0.21356584030579492
Epoch 10/10
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2223 - mean_absolute_error: 0.1385
RMSE: 0.21621227266412435
RMSE bags: 0.21356740551201747
Epoch 11/11
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2218 - mean_absolute_error: 0.1376
RMSE: 0.21614469070973258
RMSE bags: 0.21356577412383101
Epoch 12/12
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2216 - mean_absolute_error: 0.1371
RMSE: 0.2155990307714033
RMSE bags: 0.2135611538707667
Epoch 13/13
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2213 - mean_absolute_error: 0.1365
RMSE: 0.21555934148389067
RMSE bags: 0.2135564259397461
Epoch 14/14
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2210 - mean_absolute_error: 0.1359
RMSE: 0.21602174609618738
RMSE bags: 0.2135569274975923
Epoch 15/15
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2210 - mean_absolute_error: 0.1356
RMSE: 0.2160856744029581
RMSE bags: 0.21354998000057956
Epoch 16/16
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2209 - mean_absolute_error: 0.1351
RMSE: 0.2163169158255477
RMSE bags: 0.21354058941107287
Epoch 17/17
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2208 - mean_absolute_error: 0.1346
RMSE: 0.2169580279244466
RMSE bags: 0.21353983934381904
Epoch 18/18
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2207 - mean_absolute_error: 0.1341
RMSE: 0.2166800257619827
RMSE bags: 0.2135358314158332
Epoch 19/19
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2208 - mean_absolute_error: 0.1338
RMSE: 0.216418926961976
RMSE bags: 0.21352667522859922
Epoch 20/20
2489/2492 [============================>.] - ETA: 0s - loss: 0.2207 - mean_absolute_error: 0.1332   Epoch 20/20
2492/2492 [==============================] - 66s 26ms/step - loss: 0.2207 - mean_absolute_error: 0.1332
RMSE: 0.21725509461447742
RMSE bags: 0.21351971455214558
Epoch 21/21
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2207 - mean_absolute_error: 0.1328
RMSE: 0.2179361936244204
RMSE bags: 0.21351159277042248
Epoch 22/22
2489/2492 [============================>.] - ETA: 0s - loss: 0.2206 - mean_absolute_error: 0.1323   Epoch 22/22
2492/2492 [==============================] - 66s 26ms/step - loss: 0.2206 - mean_absolute_error: 0.1323
RMSE: 0.21841015888416448
RMSE bags: 0.2135030189301283
Epoch 23/23
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2207 - mean_absolute_error: 0.1319
RMSE: 0.21757174048797345
RMSE bags: 0.21349414354463694
Epoch 24/24
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2206 - mean_absolute_error: 0.1313
RMSE: 0.21774696540533245
RMSE bags: 0.21348438924048332
Epoch 25/25
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2206 - mean_absolute_error: 0.1308
RMSE: 0.21804694399929847
RMSE bags: 0.21347996621017698
Epoch 26/26
2489/2492 [============================>.] - ETA: 0s - loss: 0.2204 - mean_absolute_error: 0.1302   Epoch 26/26
2492/2492 [==============================] - 66s 26ms/step - loss: 0.2205 - mean_absolute_error: 0.1302
RMSE: 0.21878566936451774
RMSE bags: 0.2134726089587604
Epoch 27/27
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2206 - mean_absolute_error: 0.1298
RMSE: 0.21914651379348704
RMSE bags: 0.21346210087089582
Epoch 28/28
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2207 - mean_absolute_error: 0.1293
RMSE: 0.22120705324953258
RMSE bags: 0.21345108553731665


'''
