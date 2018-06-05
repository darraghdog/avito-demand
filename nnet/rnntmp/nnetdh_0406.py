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

print('[{}] Load translated features'.format(time.time() - start_time))
usecols = [c + '_translated' for c in ['title', 'description', 'param_1', 'param_2', 'param_3']]+['item_id']
dftrl = pd.concat([pd.read_csv(path + '../features/translate_trn_en.csv.gz', usecols = usecols, compression = 'gzip').set_index('item_id'),
            pd.read_csv(path + '../features/translate_tst_en.csv.gz', usecols = usecols, compression = 'gzip').set_index('item_id')])
dftrl.columns = [c.replace('_translated', '') for c in dftrl.columns]
gc.collect()

print('[{}] Missing values'.format(time.time() - start_time))
for col in ['param_1', 'param_2', 'param_3', 'description', 'price', 'image']:
    df["bin_no_" + col] = (df[col] != df[col]).astype(np.int32)
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
dftrl =  expand_description(dftrl, category = False)

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
   dftrl[col] = parallelize(dftrl[col], tokCol)
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
                         df['title'].loc[traindex].values.tolist()+ \
                         dftrl['title'].loc[traindex].values.tolist()+ \
                         dftrl['description'].loc[traindex].values.tolist(), max_words = 80000)
print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))
df.head()
dftrl.head()

df["title"]       = fit_sequence(df.title, tok_raw)
df["description"] = fit_sequence(df.description, tok_raw)
df["title_translated"]       = fit_sequence(dftrl.title, tok_raw)
df["description_translated"] = fit_sequence(dftrl.description, tok_raw)
df["title"]       = [l if len(l)>0 else [0] for l in df["title"]]
gc.collect()
del dftrl
gc.collect()


MAX_DSC = max(tok_raw.values())+1
MAX_TTL = max(tok_raw.values())+1

bin_cols = [c for c in df.columns if 'bin_no' in c]
cont_cols = [c for c in df.columns if 'cont_' in c]


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
            ,'title_translated': pad_sequences(dataset.title_translated,
                                  maxlen=max([len(l) for l in dataset.title_translated]))
            ,'description_translated': pad_sequences(dataset.description_translated,
                                  maxlen=max([len(l) for l in dataset.description_translated]))
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
    title_translated = Input(shape=[None], name="title_translated")
    description_translated = Input(shape=[None], name="description_translated")
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
    emb_dsct = embs_text(description_translated)
    emb_ttlt = embs_text(title_translated)

    # GRU Layer
    rnn_dsc = (CuDNNGRU(emb_size))(emb_dsc)
    rnn_ttl = (CuDNNGRU(emb_size)) (emb_ttl)
    rnn_dsct = (CuDNNGRU(emb_size))(emb_dsct)
    rnn_ttlt = (CuDNNGRU(emb_size)) (emb_ttlt)
    #rnn_dsc = (GRU(emb_size))(emb_dsc)
    #rnn_ttl = (GRU(emb_size)) (emb_ttl)

    #main layer
    main_l = concatenate([
        rnn_dsc
        , rnn_ttl
        , Flatten()(fe)
        , bin_vars
        , cont_vars
        , dense_l
    ])  
    #main_l = BatchNormalization()(main_l)
    main_l = Dense(32, kernel_regularizer=l2(l2_val)) (main_l)
    main_l = PReLU()(main_l)
    #main_l = BatchNormalization()(main_l)
    main_l = Dropout(dr)(main_l)
    main_l = Dense(16, kernel_regularizer=l2(l2_val)) (main_l)
    main_l = PReLU()(main_l)
    #main_l = BatchNormalization()(main_l)
    main_l = Dropout(dr/2)(main_l)

    #output
    output = Dense(1,activation="linear", kernel_regularizer=l2(l2_val)) (main_l)

    #model
    model = Model([title, description, title_translated, description_translated] + \
                  [inp for inp in emb_inputs.values()] + [bin_vars] + [cont_vars] + [img_layer], output)
    #optimizer = optimizers.Adam(clipnorm=10)
    optimizer = optimizers.Adam(clipvalue=0.5)
    model.compile(loss=root_mean_squared_error,
                  optimizer=optimizer, metrics=['mae'])

    return model


# https://github.com/keras-team/keras/issues/1370
#norm = math.sqrt(sum(np.sum(K.get_value(w)) for w in model.optimizer.weights))

epochs = 27
batchSize = 512
steps = (dtrain.shape[0]/batchSize+1)*epochs
lr_init, lr_fin = 0.0014, 0.00001
lr_decay  = (lr_init - lr_fin)/steps


bags      = 3
y_pred_ls = []
y_sub_ls  = []
for b in range(bags):
    model = get_model(128, .1,.00001)
    #model = get_model(64, .1,.00001)
    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)
    #model.summary()
    for i in range(epochs):
        batchSize = min(512*(2**i),512)
        batchSizeTst = 256
        history = model.fit_generator(
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
        gc.collect()
        
def to_logit(ls):
    ls=np.array(ls)
    ls=np.clip(ls,.0001,.9999)
    return np.log(ls/(1-ls))

def to_proba(ls):
    return 1/(1+np.exp(-ls))

res = np.full((epochs,epochs+1),1.)
for i in range(epochs):
    for j in range(i+1,epochs+1):
        preds = sum([sum(to_logit(y_pred_ls[i+epochs*bag:j+epochs*bag]))/len(y_pred_ls[i+epochs*bag:j+epochs*bag]) for bag in range(bags)])/bags
        res[i,j] = np.sqrt(metrics.mean_squared_error(dvalid['target'], to_proba(preds.flatten())))

        if res[i, j]<0.2146:
            print(i,' to ',j, 'RMSE bags:', res[i,j])
# 5  to  15 RMSE bags: 0.2149315159490983
# 8  to  27 RMSE bags: 0.21418619733068925 with images concantenated and 128 emb size
# 5  to  27 RMSE bags: 0.21439159616847414 with images concantenated and 64 emb size

for i in range(epochs):
    print(i,' ',np.argsort(res)[i,0], ':', res[i,np.argsort(res)[i,0]])

i=6
j=26
y_sub = sum([sum(to_logit(y_sub_ls[i+epochs*bag:j+epochs*bag]))/len(y_sub_ls[i+epochs*bag:j+epochs*bag]) for bag in range(bags)])/bags
rnnsub = pd.DataFrame(to_proba(y_sub),columns=["deal_probability"],index=testdex)
rnnsub['deal_probability'] = rnnsub['deal_probability'] # Between 0 and 1
rnnsub.to_csv(path+"../sub/rnndhsub_2705A.csv.gz",index=True,header=True, compression = 'gzip')
# print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
# print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

'''
Epoch 1/1
2492/2492 [==============================] - 73s 29ms/step - loss: 0.2383 - mean_absolute_error: 0.1532
RMSE: 0.22058666111927083
Epoch 2/2
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2297 - mean_absolute_error: 0.1467
RMSE: 0.21961921280024124
RMSE bags: 0.21902097399410206
Epoch 3/3
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2270 - mean_absolute_error: 0.1441
RMSE: 0.21799077993327434
RMSE bags: 0.21780675777794087
Epoch 4/4
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2249 - mean_absolute_error: 0.1421
RMSE: 0.21763977030753698
RMSE bags: 0.2171990181996385
Epoch 5/5
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2235 - mean_absolute_error: 0.1408
RMSE: 0.21718274019545092
RMSE bags: 0.21674223910507906
Epoch 6/6
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2228 - mean_absolute_error: 0.1399
RMSE: 0.21736431014775606
RMSE bags: 0.21645412382339302
Epoch 7/7
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2225 - mean_absolute_error: 0.1395
RMSE: 0.21665605437888882
RMSE bags: 0.2161474717229319
Epoch 8/8
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2221 - mean_absolute_error: 0.1390
RMSE: 0.21691321889317688
RMSE bags: 0.21595448362818276
Epoch 9/9
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2218 - mean_absolute_error: 0.1386
RMSE: 0.21671132642413768
RMSE bags: 0.2157719694564262
Epoch 10/10
2492/2492 [==============================] - 66s 27ms/step - loss: 0.2218 - mean_absolute_error: 0.1383
RMSE: 0.21668072910192518
RMSE bags: 0.21565886503989437
Epoch 11/11
2489/2492 [============================>.] - ETA: 0s - loss: 0.2215 - mean_absolute_error: 0.1380   Epoch 11/11
2492/2492 [==============================] - 67s 27ms/step - loss: 0.2215 - mean_absolute_error: 0.1380
RMSE: 0.2169762960966161
RMSE bags: 0.21555719146779942
Epoch 12/12
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2215 - mean_absolute_error: 0.1376
RMSE: 0.2165060276235683
RMSE bags: 0.21540770177659843
Epoch 13/13
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2214 - mean_absolute_error: 0.1373
RMSE: 0.21720929671526368
RMSE bags: 0.21526960668524406
Epoch 14/14
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2213 - mean_absolute_error: 0.1371
RMSE: 0.216403902880614
RMSE bags: 0.21517585830924518
Epoch 15/15
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2212 - mean_absolute_error: 0.1368
RMSE: 0.21671033708223825
RMSE bags: 0.2150874775103638
Epoch 16/16
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2213 - mean_absolute_error: 0.1366
RMSE: 0.21630424104384546
RMSE bags: 0.21500675267185282
Epoch 17/17
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1363
RMSE: 0.2169372370611181
RMSE bags: 0.21494757748484303
Epoch 18/18
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1360
RMSE: 0.2168295402778363
RMSE bags: 0.21487896698349784
Epoch 19/19
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1357
RMSE: 0.21640937130816668
RMSE bags: 0.21482560578583038
Epoch 20/20
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1354
RMSE: 0.21721706717413697
RMSE bags: 0.21477062597025257
Epoch 21/21
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2211 - mean_absolute_error: 0.1351
RMSE: 0.21812247848585384
RMSE bags: 0.21472600389838079
Epoch 22/22
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2209 - mean_absolute_error: 0.1346

RMSE: 0.21732165603312067
RMSE bags: 0.2146889537727656
Epoch 23/23
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2210 - mean_absolute_error: 0.1344
RMSE: 0.2175217764161999
RMSE bags: 0.21464451194111783
Epoch 24/24
2492/2492 [==============================] - 65s 26ms/step - loss: 0.2209 - mean_absolute_error: 0.1340
RMSE: 0.2173059009025967
RMSE bags: 0.2146206042010009
Epoch 25/25
2492/2492 [==============================] - 66s 26ms/step - loss: 0.2208 - mean_absolute_error: 0.1336
RMSE: 0.21847795746254076
RMSE bags: 0.21461406758588464
Epoch 26/26
2492/2492 [==============================] - 64s 26ms/step - loss: 0.2207 - mean_absolute_error: 0.1332
RMSE: 0.2180920843767536
RMSE bags: 0.21459330716182254

'''