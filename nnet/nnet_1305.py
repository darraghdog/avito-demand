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
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numba import jit
import nltk, re
from nltk.tokenize import ToktokTokenizer
from nltk.stem import PorterStemmer
import multiprocessing as mp
from collections import Counter
from keras.preprocessing.sequence import pad_sequences


import random
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, CuDNNGRU, Embedding, Flatten, Bidirectional, \
    MaxPooling1D, Conv1D, Add, Reshape, Lambda, PReLU, GaussianDropout, SpatialDropout1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras.layers import GlobalMaxPooling1D
from keras import initializers
from keras.utils import plot_model

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
df['idx'] = range(df.shape[0])
del traindf,testdf
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))    


print('[{}] Feature Engineering Price'.format(time.time() - start_time))
col = "price"
bins_ = 100
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
    var[~idx]    = 'locount'
    var.fillna('missing', inplace = True)
    return var.astype(str).values
for col_, cut_ in [("user_id", 5), ("image_top_1", 5), ("item_seq_number", 100)]: 
    df[col_] = lowCtCat(col_, cutoff = cut_)
for col_, cut_ in [('param_'+str(i+1), 20) for i in range(3)]: 
    df['cat_' + col_] = lowCtCat(col_, cutoff = cut_)

print('[{}] Encode Variables'.format(time.time() - start_time))
embed_me = ["item_seq_number", "user_id","image_top_1", "region", 'city', "emb_week_of_year", 'emb_price', \
            "parent_category_name", "category_name", "user_type", "emb_weekday"] + ['cat_param_'+str(i+1) for i in range(3)]
for col in embed_me:
    print('Label encode %s'%(col))
    lbl = preprocessing.LabelEncoder()
    df[col] = lbl.fit_transform(df[col].astype(str))

print('[{}] Embedding dimensions'.format(time.time() - start_time))
col_szs = dict((col, df[col].unique().shape[0]) for col in embed_me)
embed_szs = dict((col, int(np.ceil(np.log(col_szs[col])))+1) for col in embed_me)
embed_szs

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

print('[{}] Finished FITTING TEXT DATA...'.format(time.time() - start_time))  
tok_raw_dsc = myTokenizerFit(df['description'].loc[traindex].values, max_words = 60000)
tok_raw_ttl = myTokenizerFit(df['title'].loc[traindex].values, max_words = 15000)
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))

df["title"]       = fit_sequence(df.title, tok_raw_ttl)
df["description"] = fit_sequence(df.description, tok_raw_dsc)    
df["title"]       = [l if len(l)>0 else [0] for l in df["title"]]
gc.collect()
df["ttllen"] = [len(l) for l in df["title"]]
df["dsclen"] = [len(l) for l in df["description"]]

MAX_DSC = max(tok_raw_dsc.values())+1
MAX_TTL = max(tok_raw_ttl.values())+1



def get_keras_data(dataset):
    X = {
        'title': pad_sequences(dataset.title, 
                              maxlen=max([len(l) for l in dataset.title]))
        ,'description': pad_sequences(dataset.description, 
                              maxlen=max([len(l) for l in dataset.description]))
    }
    for col in embed_szs.keys():
        X[col] = dataset[col].values
    return X   

def map_sort(seq1, seq2):
	return sorted(range(len(seq1)), key=lambda x: len(seq1[x])*10+len(seq2[x]))
    
def reset_data(dt, bsize):
    max_step = dt.shape[0]
    n_batches = int(np.ceil(max_step*1. / float(bsize)))
    batch_steps = np.array(random.sample(range(n_batches), n_batches))
    sorted_ix = np.array(map_sort(dt["title"].tolist(), dt["description"].tolist()))
    dt.reset_index(drop=True, inplace = True)  
    return max_step, batch_steps, sorted_ix, dt

def trn_generator(dt, y, bsize):
    while True:
        max_step, batch_steps, sorted_ix, dt = reset_data(dt, bsize)
        for batch in batch_steps:
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, max_step)
            ix_   = sorted_ix[from_:to_]
            Xbatch = dt.iloc[ix_]
            Xbatch = get_keras_data(Xbatch)
            ybatch = dt.target.iloc[ix_]
            yield Xbatch, ybatch

def val_generator(dt, y, bsize):
    while 1:
        max_step, batch_steps, sorted_ix, dt = reset_data(dt, bsize)
        for batch in batch_steps:
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, max_step)
            ix_   = sorted_ix[from_:to_]
            Xbatch = dt.iloc[ix_]
            Xbatch = get_keras_data(Xbatch)
            ybatch = dt.target.iloc[ix_]
            yield Xbatch, ybatch
            
def tst_generator(dt, bsize):
    while 1:
        for batch in range(int(np.ceil(dt.shape[0]*1./bsize))):
            from_ = batch*bsize
            to_   = min((batch+1)*bsize, dt.shape[0])
            Xbatch = dt.iloc[from_:to_]
            Xbatch = get_keras_data(Xbatch)
            yield Xbatch

get_keras_data(df[:2])
def get_model():
    dr = 0.1
    ##Inputs
    title = Input(shape=[None], name="title")
    description = Input(shape=[None], name="description")
    
    # Categorical embeddings
    emb_inputs = dict((col, Input(shape=[1], name = col))  for col in embed_szs.keys())
    emb_model  = dict((col, Embedding(col_szs[col]+1, emb_n)(emb_inputs[col])) for (col, emb_n) in embed_szs.items())
    fe = concatenate([(emb_) for emb_ in emb_model.values()])
    #fe = SpatialDropout1D(dr)(fe)
    
    #Embeddings layers
    emb_size = 32
    emb_dsc = Embedding(MAX_DSC, emb_size)(description) 
    emb_ttl = Embedding(MAX_TTL, emb_size)(title) 
    
    # GRU Layer
    rnn_dsc = GRU(emb_size, recurrent_dropout=0.0) (emb_dsc)
    rnn_ttl = GRU(emb_size, recurrent_dropout=0.0) (emb_ttl)
    #rnn_dsc = CuDNNGRU(16) (emb_dsc)
    #rnn_ttl = CuDNNGRU(16) (emb_ttl)
    
    #main layer
    main_l = concatenate([
        rnn_dsc
        , rnn_ttl
        , Flatten()(fe)
    ])
    
    #main_l = BatchNormalization()(main_l)
    #main_l = Dropout(dr)(main_l)
    main_l = Dense(32) (main_l)
    main_l = PReLU()(main_l)
    #main_l = BatchNormalization()(main_l)
    main_l = Dropout(dr)(main_l)
    main_l = Dense(16) (main_l)
    main_l = PReLU()(main_l)
    #main_l = BatchNormalization()(main_l)
    main_l = Dropout(0.05)(main_l)
    
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([title, description] + [inp for inp in emb_inputs.values()], output)
    optimizer = optimizers.Adam()
    model.compile(loss='mse', 
                  optimizer=optimizer)
    
    return model


dtrain = df.loc[traindex,:][trnidx].reset_index()
dvalid = df.loc[traindex,:][validx].reset_index()
dtrain['target'] = y[trnidx].values
dvalid['target'] = y[validx].values


val_sorted_ix = np.array(map_sort(dvalid["title"].tolist(), dvalid["description"].tolist()))
y_pred_epochs = []

epochs = 3
batchSize = 512
#steps = (dtrain.shape[0]/batchSize+1)*epochs
#lr_init, lr_fin = 0.0014, 0.00001
#lr_decay  = (lr_init - lr_fin)/steps
model = get_model()
#K.set_value(model.optimizer.lr, lr_init)
#K.set_value(model.optimizer.decay, lr_decay)
model.summary()

y_pred_ls = []
for i in range(epochs):
    batchSize = 512*(2**i)
    model.fit_generator(
                        trn_generator(dtrain, dtrain.target, batchSize)
                        , epochs=1
                        , max_queue_size=1
                        , steps_per_epoch = int(np.ceil(dtrain.shape[0]/batchSize))
                        , validation_data = val_generator(dvalid, dvalid.target, batchSize)
                        , validation_steps = int(np.ceil(dvalid.shape[0]/batchSize))
                        , verbose=1
                        )
    batchSizeTst = 256
    y_pred_ls.append(model.predict_generator(
                    tst_generator(dvalid.iloc[val_sorted_ix], batchSizeTst)
                    , steps = int(np.ceil(dvalid.shape[0]*1./batchSizeTst))
                    , max_queue_size=1 
                    , verbose=2)[val_sorted_ix.argsort()])
    print('RMSE:', np.sqrt(metrics.mean_squared_error(dvalid['target'], y_pred_ls[-1].flatten())))
    if len(y_pred_ls)>1:
        y_pred = sum(y_pred_ls)/len(y_pred_ls)
        print('RMSE:', np.sqrt(metrics.mean_squared_error(dvalid['target'], y_pred.flatten())))

# Epoch1 - RMSE: #0.2196 #0.2269