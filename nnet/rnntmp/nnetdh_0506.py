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
# path = '/home/ubuntu/avito/data/'

start_time = time.time()

validation = False
full       = False

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
dnimgtrn = np.load(path+'../imgfeatures/densenet_pool_array_train.npy')
dnimgval = dnimgtrn[validx]
dnimgtrn = dnimgtrn[trnidx]
scaler = preprocessing.StandardScaler()
dnimgtrn = scaler.fit_transform(dnimgtrn)
dnimgval = scaler.transform(dnimgval)
gc.collect()
dnimgtst = np.load(path+'../imgfeatures/densenet_pool_array_test.npy')
dnimgtst = scaler.transform(dnimgtst)
gc.collect()


print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
df['idx'] = range(df.shape[0])
del traindf,testdf
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

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
    df["bin_no_" + col] = (df[col].isna()).astype(np.int32)
    df[col].fillna(0, inplace = True)
    df[col] = np.log1p(df[col]+0.111).values
    df.rename(columns = { col : 'cont_'+col}, inplace = True)
    
print('[{}] Load bayes mean features'.format(time.time() - start_time))
featenc = pd.read_csv(path + '../features/alldf_bayes_mean.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
for col in featenc.columns:
    featenc.rename(columns = { col : 'cont_'+col}, inplace = True)
df = pd.concat([df.reset_index(),featenc],axis=1)
df = df.set_index('item_id')
df.head()

print('[{}] Load engineered price ratio features'.format(time.time() - start_time))

featrdgprc = pd.read_csv(path + '../features/price_category_ratios.gz', compression = 'gzip') # created with features/make/user_actagg_1705.py
for col in featrdgprc.columns:
    featrdgprc[col].fillna(featrdgprc[col].median(), inplace = True)
featrdgprc.isnull().sum()
for col in [c for c in featrdgprc.columns if '_log' not in c]:
    featrdgprc[col] = np.log1p(featrdgprc[col]+.001111)
featrdgprc.columns = ['cont_'+c for c in featrdgprc.columns]
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
    df["bin_no_" + col] = (df[col].isna()).astype(np.int32)
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


dtrain = df.loc[traindex,:][trnidx].reset_index()
dvalid = df.loc[traindex,:][validx].reset_index()
dtest  = df.loc[testdex,:].reset_index()
dtrain['target'] = y[trnidx].values
dvalid['target'] = y[validx].values


train_sorted_ix = np.array(map_sort(dtrain["title"].tolist(), dtrain["description"].tolist()))
val_sorted_ix = np.array(map_sort(dvalid["title"].tolist(), dvalid["description"].tolist()))
tst_sorted_ix = np.array(map_sort(dtest ["title"].tolist(), dtest ["description"].tolist()))

y_pred_epochs = []




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
    main_l = Dense(256, kernel_regularizer=l2(l2_val)) (main_l)
    main_l = PReLU()(main_l)
    #main_l = BatchNormalization()(main_l)
    main_l = Dropout(dr)(main_l)
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
# https://github.com/keras-team/keras/issues/1370
#norm = math.sqrt(sum(np.sum(K.get_value(w)) for w in model.optimizer.weights))

epochs = 28
batchSize = 512
steps = (dtrain.shape[0]/batchSize+1)*epochs
lr_init, lr_fin = 0.0014, 0.00001
lr_decay  = (lr_init - lr_fin)/steps


bags      = 3
y_pred_ls = []
y_sub_ls  = []
for b in range(bags):
    gc.collect()
    model = get_model(128, .1,.00001)
    #model = get_model(64, .1,.00001)
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)
    #model.summary()
    for i in range(epochs):
        batchSize = min(512*(2**i),512)
        batchSizeTst = 256
        model.fit_generator(
                            Seq_generator(dtrain, dnimgtrn, batchSize, train_sorted_ix)
                            , epochs=i+1
                            , max_queue_size=15
                            , verbose=1
                            , initial_epoch=i
                            , use_multiprocessing=True
                            , workers=3
                            )
        y_pred_ls.append(model.predict_generator(
                         Seq_generator(dvalid, dnimgval, batchSizeTst, val_sorted_ix, target_out=False)
                        , max_queue_size=10
                        , verbose=2)[val_sorted_ix.argsort()])
        y_sub_ls.append(model.predict_generator(
                        Seq_generator(dtest, dnimgtst, batchSizeTst, tst_sorted_ix, target_out=False)
                        , max_queue_size=10
                        , verbose=2)[tst_sorted_ix.argsort()])
        print('RMSE:', np.sqrt(metrics.mean_squared_error(dvalid['target'], y_pred_ls[-1].flatten())))
        if len(y_pred_ls)>1:
            y_pred = sum(y_pred_ls)/len(y_pred_ls)
            print('RMSE bags:', np.sqrt(metrics.mean_squared_error(dvalid['target'], y_pred.flatten())))    
    del model
    gc.collect()
        
def to_logit(ls):
    ls=np.array(ls)
    ls=np.clip(ls,.0001,.9999)
    return np.log(ls/(1-ls))

def to_proba(ls):
    return 1/(1+np.exp(-ls))
'''
res = np.full((epochs,epochs+1),1.)
for i in range(epochs):
    for j in range(i+1,epochs+1):
        preds = sum([sum(to_logit(y_pred_ls[i+epochs*bag:j+epochs*bag]))/len(y_pred_ls[i+epochs*bag:j+epochs*bag]) for bag in range(bags)])/bags
        res[i,j] = np.sqrt(metrics.mean_squared_error(dvalid['target'], to_proba(preds.flatten())))

        if res[i, j]<0.2136:
            print(i,' to ',j, 'RMSE bags:', res[i,j])
# 4  to  28 RMSE bags: 0.21356346841391569

for i in range(epochs):
    print(i,' ',np.argsort(res)[i,0], ':', res[i,np.argsort(res)[i,0]])
'''
i=4
j=28
y_sub = sum([sum(to_logit(y_sub_ls[i+epochs*bag:j+epochs*bag]))/len(y_sub_ls[i+epochs*bag:j+epochs*bag]) for bag in range(bags)])/bags
rnnsub = pd.DataFrame(to_proba(y_sub),columns=["deal_probability"],index=testdex)
rnnsub['deal_probability'] = rnnsub['deal_probability'] # Between 0 and 1
rnnsub.to_csv(path+"../sub/rnndhsub_0506_val.csv.gz",index=True,header=True, compression = 'gzip')
# print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
# print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

'''
Epoch 1/1
2492/2492 [==============================] - 83s 33ms/step - loss: 0.2375 - mean_absolute_error: 0.1507
RMSE: 0.21971732788312276
Epoch 2/2
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2289 - mean_absolute_error: 0.1451
RMSE: 0.21823631135367927
RMSE bags: 0.21789262291364989
Epoch 3/3
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2266 - mean_absolute_error: 0.1432
RMSE: 0.21746045993864224
RMSE bags: 0.21705316162179344
Epoch 4/4
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2254 - mean_absolute_error: 0.1419
RMSE: 0.21711198596636136
RMSE bags: 0.21644452168803377
Epoch 5/5
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2244 - mean_absolute_error: 0.1406
RMSE: 0.2166388121863551
RMSE bags: 0.2159107927366218
Epoch 6/6
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2234 - mean_absolute_error: 0.1397
RMSE: 0.21683525406247284
RMSE bags: 0.21557801365061383
Epoch 7/7
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2228 - mean_absolute_error: 0.1389
RMSE: 0.2171879806432999
RMSE bags: 0.2153122001146096
Epoch 8/8
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2221 - mean_absolute_error: 0.1382
RMSE: 0.21610617115541575
RMSE bags: 0.21500698381691224
Epoch 9/9
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2218 - mean_absolute_error: 0.1376
RMSE: 0.21548936830815804
RMSE bags: 0.2148076507483173
Epoch 10/10
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2215 - mean_absolute_error: 0.1372
RMSE: 0.21552966409573604
RMSE bags: 0.21462216648972973
Epoch 11/11
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2214 - mean_absolute_error: 0.1368
RMSE: 0.2167450203710096
RMSE bags: 0.21445232013750687
Epoch 12/12
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2213 - mean_absolute_error: 0.1364
RMSE: 0.21664675772755018
RMSE bags: 0.2143619048687673
Epoch 13/13
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2212 - mean_absolute_error: 0.1361
RMSE: 0.21613640147176458
RMSE bags: 0.21428116180845477
Epoch 14/14
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2213 - mean_absolute_error: 0.1359
RMSE: 0.2160485097170228
RMSE bags: 0.2142181336302171
Epoch 15/15
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1355
RMSE: 0.21647921517972907
RMSE bags: 0.2141342464793336
Epoch 16/16
2492/2492 [==============================] - 66s 26ms/step - loss: 0.2210 - mean_absolute_error: 0.1350
RMSE: 0.21725798625522827
RMSE bags: 0.21407695226779838
Epoch 17/17
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1347
RMSE: 0.21620652330316026
RMSE bags: 0.21401855210990442
Epoch 18/18
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2210 - mean_absolute_error: 0.1342
RMSE: 0.21616940289815825
RMSE bags: 0.2139824843452162
Epoch 19/19
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2208 - mean_absolute_error: 0.1338
RMSE: 0.21650131826307462
RMSE bags: 0.21394390500512767
Epoch 20/20
2489/2492 [============================>.] - ETA: 0s - loss: 0.2209 - mean_absolute_error: 0.1335   Epoch 20/20
2492/2492 [==============================] - 66s 26ms/step - loss: 0.2209 - mean_absolute_error: 0.1335
RMSE: 0.21696419836643896
RMSE bags: 0.21390168913721133
Epoch 21/21
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2209 - mean_absolute_error: 0.1331
RMSE: 0.2173355907175484
RMSE bags: 0.21388166253382326
Epoch 22/22
2492/2492 [==============================] - 65s 26ms/step - loss: 0.3155 - mean_absolute_error: 0.1381
RMSE: 0.21878691146320958
RMSE bags: 0.21385486767039083
Epoch 23/23
2492/2492 [==============================] - 65s 26ms/step - loss: 0.4101 - mean_absolute_error: 0.1394
RMSE: 0.22014303182745085
RMSE bags: 0.21386207487624936
Epoch 24/24
2492/2492 [==============================] - 65s 26ms/step - loss: 0.5594 - mean_absolute_error: 0.1401
RMSE: 0.22002086566994236
RMSE bags: 0.21388341081617485
Epoch 25/25
2492/2492 [==============================] - 65s 26ms/step - loss: 0.5433 - mean_absolute_error: 0.1391
RMSE: 0.2206547201185324
RMSE bags: 0.21389499429201747
Epoch 26/26
2492/2492 [==============================] - 66s 26ms/step - loss: 0.5149 - mean_absolute_error: 0.1390

RMSE: 0.21905090647803602
RMSE bags: 0.21391407955039501
Epoch 27/27
2492/2492 [==============================] - 66s 26ms/step - loss: 0.4630 - mean_absolute_error: 0.1388
RMSE: 0.22081256086588658
RMSE bags: 0.21394294706982614
Epoch 28/28
2492/2492 [==============================] - 65s 26ms/step - loss: 0.3802 - mean_absolute_error: 0.1388
RMSE: 0.21867609668814975
RMSE bags: 0.21394782585214445
Epoch 29/29
2492/2492 [==============================] - 65s 26ms/step - loss: 0.3135 - mean_absolute_error: 0.1386
RMSE: 0.21857820054761626
RMSE bags: 0.21396395835178592
Epoch 30/30
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2762 - mean_absolute_error: 0.1386
RMSE: 0.21873351888983877
RMSE bags: 0.2139643349242654
Epoch 1/1
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2372 - mean_absolute_error: 0.1509
RMSE: 0.22060107273934215
RMSE bags: 0.21400235801734277
Epoch 2/2
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2291 - mean_absolute_error: 0.1454
RMSE: 0.21879798363485228
RMSE bags: 0.21402856111979615
Epoch 3/3
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2268 - mean_absolute_error: 0.1433
RMSE: 0.21783729786581102
RMSE bags: 0.21403607940512215
Epoch 4/4
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2251 - mean_absolute_error: 0.1416
RMSE: 0.21669579616396203
RMSE bags: 0.21403593620673744
Epoch 5/5
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2241 - mean_absolute_error: 0.1405
RMSE: 0.21672678605475504
RMSE bags: 0.2140188211088136
Epoch 6/6
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2234 - mean_absolute_error: 0.1397
RMSE: 0.21616835965908696
RMSE bags: 0.2140102416210428
Epoch 7/7
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2229 - mean_absolute_error: 0.1389
RMSE: 0.21616842472732653
RMSE bags: 0.21399944865352286
Epoch 8/8
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2225 - mean_absolute_error: 0.1384
RMSE: 0.21665889639455005
RMSE bags: 0.2139963807847883
Epoch 9/9
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2224 - mean_absolute_error: 0.1379
RMSE: 0.21627614283063823
RMSE bags: 0.21398511273548282
Epoch 10/10
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2221 - mean_absolute_error: 0.1374
RMSE: 0.21554384956970418
RMSE bags: 0.2139568042124387
Epoch 11/11
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2219 - mean_absolute_error: 0.1370
RMSE: 0.21567181717334336
RMSE bags: 0.21392901593484198
Epoch 12/12
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2215 - mean_absolute_error: 0.1364
RMSE: 0.2162558116504721
RMSE bags: 0.21391079101160596
Epoch 13/13
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2213 - mean_absolute_error: 0.1361

RMSE: 0.21592802539989198
RMSE bags: 0.21388613333658935
Epoch 14/14
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2211 - mean_absolute_error: 0.1357
RMSE: 0.21721898223101307
RMSE bags: 0.21386854117279125
Epoch 15/15
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2212 - mean_absolute_error: 0.1355
RMSE: 0.21743041563124998
RMSE bags: 0.21384168331307765
Epoch 16/16
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2211 - mean_absolute_error: 0.1350
Epoch 16/16
RMSE: 0.21844872456861855
RMSE bags: 0.2138189976767084
Epoch 17/17
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2210 - mean_absolute_error: 0.1347
RMSE: 0.2171292200388435
RMSE bags: 0.21380202646919502
Epoch 18/18
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2209 - mean_absolute_error: 0.1343
RMSE: 0.21673305047436078
RMSE bags: 0.2137848858929853
Epoch 19/19
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2211 - mean_absolute_error: 0.1340
RMSE: 0.21666184138927744
RMSE bags: 0.21376296473457232
Epoch 20/20
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2208 - mean_absolute_error: 0.1337
Epoch 20/20
RMSE: 0.21680729268205112
RMSE bags: 0.21374734408764126
Epoch 21/21
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2209 - mean_absolute_error: 0.1334
RMSE: 0.217349864240077
RMSE bags: 0.2137295264179942
Epoch 22/22
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2209 - mean_absolute_error: 0.1330
RMSE: 0.21745324797711013
RMSE bags: 0.21371382433692274
Epoch 23/23
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2208 - mean_absolute_error: 0.1326
RMSE: 0.21703842363710785
RMSE bags: 0.2137021791197466
Epoch 24/24
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2208 - mean_absolute_error: 0.1323
RMSE: 0.21768452227616308
RMSE bags: 0.2136868832458267
Epoch 25/25
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2207 - mean_absolute_error: 0.1320
RMSE: 0.2193322776520206
RMSE bags: 0.21367568210953033
Epoch 26/26
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2207 - mean_absolute_error: 0.1316
RMSE: 0.21789269627102106
RMSE bags: 0.21366069613306385
Epoch 27/27
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2205 - mean_absolute_error: 0.1311
Epoch 27/27
RMSE: 0.21810016840529795
RMSE bags: 0.21364487740463226
Epoch 28/28
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2207 - mean_absolute_error: 0.1309
RMSE: 0.21882182272823794
RMSE bags: 0.21363549793372388
Epoch 29/29
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2206 - mean_absolute_error: 0.1305
RMSE: 0.21932876955281824
RMSE bags: 0.21362086546937373
Epoch 30/30
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2205 - mean_absolute_error: 0.1300
RMSE: 0.21950565936612942
RMSE bags: 0.21361120781127516
Epoch 1/1
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2372 - mean_absolute_error: 0.1504
RMSE: 0.2200374159176743
RMSE bags: 0.2136298201635869
Epoch 2/2
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2288 - mean_absolute_error: 0.1453
RMSE: 0.21933832344151974
RMSE bags: 0.2136490272762485
Epoch 3/3
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2265 - mean_absolute_error: 0.1432
RMSE: 0.21752650820190617
RMSE bags: 0.2136531468028395
Epoch 4/4
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2250 - mean_absolute_error: 0.1416
RMSE: 0.21690656333267858
RMSE bags: 0.21365808739847067
Epoch 5/5
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2240 - mean_absolute_error: 0.1405
RMSE: 0.21655868880395543
RMSE bags: 0.21366250722871769
Epoch 6/6
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2235 - mean_absolute_error: 0.1397
RMSE: 0.2175383454887492
RMSE bags: 0.21366580327180332
Epoch 7/7
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2229 - mean_absolute_error: 0.1390
RMSE: 0.21688887943898677
RMSE bags: 0.21366991092035448
Epoch 8/8
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2224 - mean_absolute_error: 0.1385
RMSE: 0.2168638235671314
RMSE bags: 0.21367405314417234
Epoch 9/9
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2222 - mean_absolute_error: 0.1378
RMSE: 0.21619496707133362
RMSE bags: 0.21367518165637886
Epoch 10/10
2491/2492 [============================>.] - ETA: 0s - loss: 0.2220 - mean_absolute_error: 0.1374   Epoch 10/10
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2219 - mean_absolute_error: 0.1374
RMSE: 0.2166895706764423
RMSE bags: 0.21367249042800956
Epoch 11/11
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2219 - mean_absolute_error: 0.1369
RMSE: 0.21649891262824103
RMSE bags: 0.21366914777339907
Epoch 12/12
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2218 - mean_absolute_error: 0.1366
RMSE: 0.21718013888289364
RMSE bags: 0.21367281512216424
Epoch 13/13
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2218 - mean_absolute_error: 0.1363
RMSE: 0.21769080325597776
RMSE bags: 0.21367164548688494
Epoch 14/14
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2216 - mean_absolute_error: 0.1359
RMSE: 0.2166890585264033
RMSE bags: 0.21366988602109235
Epoch 15/15
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2216 - mean_absolute_error: 0.1355
RMSE: 0.21674567918486992
RMSE bags: 0.21366930152590521
Epoch 16/16
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2217 - mean_absolute_error: 0.1352
RMSE: 0.21712141149782482
RMSE bags: 0.21366547491896423
Epoch 17/17
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2215 - mean_absolute_error: 0.1349
RMSE: 0.2176836504523436
RMSE bags: 0.21366391402673862
Epoch 18/18
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2214 - mean_absolute_error: 0.1345
RMSE: 0.21724125845103923
RMSE bags: 0.21365944703737338
Epoch 19/19
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2215 - mean_absolute_error: 0.1342
RMSE: 0.21848654354271985
RMSE bags: 0.21365800948241556
Epoch 20/20
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2214 - mean_absolute_error: 0.1337
RMSE: 0.21766279617597784
RMSE bags: 0.21365404659982898
Epoch 21/21
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2214 - mean_absolute_error: 0.1334
RMSE: 0.21862028058808422
RMSE bags: 0.213652361249084
Epoch 22/22
2492/2492 [==============================] - 64s 25ms/step - loss: 0.2213 - mean_absolute_error: 0.1331
RMSE: 0.21799154116013061
RMSE bags: 0.21364901182078197
Epoch 23/23
2492/2492 [==============================] - 64s 25ms/step - loss: 0.2213 - mean_absolute_error: 0.1327
RMSE: 0.21877220968021413
RMSE bags: 0.2136505439739601
Epoch 24/24
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2213 - mean_absolute_error: 0.1323
RMSE: 0.21978302325220433
RMSE bags: 0.2136516172023452
Epoch 25/25
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1318
RMSE: 0.22088693647906224
RMSE bags: 0.21364967787888478
Epoch 26/26
2492/2492 [==============================] - 64s 25ms/step - loss: 0.2212 - mean_absolute_error: 0.1315
RMSE: 0.21894879049740007
RMSE bags: 0.21364799016747882
Epoch 27/27
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1311
RMSE: 0.21973349307611592
RMSE bags: 0.2136453398292589
Epoch 28/28
2492/2492 [==============================] - 63s 25ms/step - loss: 0.2211 - mean_absolute_error: 0.1307
RMSE: 0.21997911826701794
RMSE bags: 0.21364231288344412
Epoch 29/29
2492/2492 [==============================] - 64s 26ms/step - loss: 0.4016 - mean_absolute_error: 0.1389
RMSE: 0.2221130553832405
RMSE bags: 0.21365965857143832
Epoch 30/30
2492/2492 [==============================] - 64s 26ms/step - loss: 0.7797 - mean_absolute_error: 0.1421
RMSE: 0.22085652459119454
RMSE bags: 0.21367209801646164

'''