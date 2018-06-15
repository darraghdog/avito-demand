# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
import time, gc
import pandas as pd
import numpy as np
from sklearn import preprocessing
from nltk.corpus import stopwords 
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K


#path = '../input/'
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'

# path = '/home/ubuntu/avito/data/'
start_time = time.time()
full = False
validation = False

print('[{}] Load Train/Test'.format(time.time() - start_time))
usecols = ['activation_date', 'item_id', 'deal_probability']
traindf = pd.read_csv(path + 'train.csv.zip', index_col = "item_id", usecols = usecols, compression = 'zip') # , parse_dates = ["activation_date"]
traindex = traindf.index
testdf = pd.read_csv(path + 'test.csv.zip', index_col = "item_id", usecols = usecols[:-1]) #, parse_dates = ["activation_date"]
testdex = testdf.index
y = traindf.deal_probability.copy()
traindf.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*traindf.shape))
print('Test shape: {} Rows, {} Columns'.format(*testdf.shape))
traindf['activation_date'].value_counts()

print('[{}] Load Densenet image features'.format(time.time() - start_time))
dnimgtrn = csr_matrix(np.load(path+'../features/vgg19_pool_array_train.npy'))
save_npz(path+'../features/vgg19_pool_sparse_train.npz', dnimgtrn)
#dnimgtst = csr_matrix(np.load(path+'../features/vgg19_pool_array_test.npy'))
#save_npz(path+'../features/vgg19_pool_sparse_test.npz', dnimgtst)
dnimgtst = load_npz(path+'../features/vgg19_pool_sparse_test.npz')

print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
del traindf,testdf
gc.collect()
df['idx'] = range(df.shape[0])
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

'''
print('[{}] Combine Train and Test'.format(time.time() - start_time))
hotcols = ['user_type', 'parent_category_name', 'category_name']
dfhot = pd.get_dummies(df[hotcols])
dnimgtrn = np.hstack([dfhot.loc[traindex].values, dnimgtrn])
dnimgtst = np.hstack([dfhot.loc[testdex].values, dnimgtst])
'''

print('[{}] Set up folds'.format(time.time() - start_time))
foldls = [["2017-03-15", "2017-03-16", "2017-03-17"], \
           ["2017-03-18", "2017-03-19", "2017-03-20"], \
           ["2017-03-21", "2017-03-22", "2017-03-23"], \
           ["2017-03-24", "2017-03-25", "2017-03-26"], \
            ["2017-03-27", "2017-03-28", "2017-03-29", \
                "2017-03-30", "2017-03-31", "2017-04-01", \
                "2017-04-02", "2017-04-03","2017-04-07"]]
folds = [df.loc[traindex,:]['activation_date'].isin(f) for f in foldls]
[sum(f) for f in folds]

print('[{}] Modified sklearn wrapper'.format(time.time() - start_time))
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        

print('[{}] Start nnet'.format(time.time() - start_time))
ntrain  = df.loc[traindex,:].shape[0]
ntest   = df.loc[testdex,:].shape[0]

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def get_model(shape):
    model = Sequential()
    model.add(Dense(1000, input_dim=shape, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=root_mean_squared_error, optimizer=optim)
    return model


NFOLDS = 5  
bsize = 32
from sklearn import preprocessing
'''
scaler = preprocessing.StandardScaler()
dnimgtrnsc = scaler.fit_transform(dnimgtrn)
dnimgtstsc = scaler.transform(dnimgtst)
#del dnimgtrn, dnimgtst
'''
gc.collect()


oof_train = np.zeros((ntrain,))
oof_test = np.zeros((ntest,))
oof_test_skf = np.empty((NFOLDS, ntest))
for i, f in enumerate(folds):
    train_index, test_index = np.where(f==False), np.where(f)
    x_tr = dnimgtrn[train_index]
    y_tr = y.values[train_index]
    x_te = dnimgtrn[test_index]
    y_te = y.values[test_index]
    gc.collect()
    clf = get_model(dnimgtrn.shape[1])
    clf.fit(x_tr, y_tr, 
      epochs=2,
      batch_size=bsize,
      validation_data = (x_te, y_te), 
      verbose=1)
    del x_tr, y_tr
    gc.collect()
    oof_train[test_index] = clf.predict(x_te, batch_size=bsize*16).flatten()
    oof_test_skf[i, :] = clf.predict(dnimgtst, batch_size=bsize*16).flatten()
    del x_te, train_index, test_index
    gc.collect()
oof_test[:] = oof_test_skf.mean(axis=0)
oof_train_out, oof_test_out = oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
        
rms = sqrt(mean_squared_error(y, oof_train_out))
print('Ridge OOF RMSE: {}'.format(rms))
   
nnet_preds = np.concatenate([oof_train, oof_test_out])         
df['nnet_imgstr_preds'] = nnet_preds
df[['nnet_imgstr_preds']].to_csv(path + '../features/nnetImgStr5CV.csv.gz', compression = 'gzip', index = False)    


'''
Train on 1189868 samples, validate on 313556 samples
Epoch 1/2
  33472/1189868 [..............................] - ETA: 2:06 - loss: 0.2527   
1189868/1189868 [==============================] - 126s 106us/step - loss: 0.2379 - val_loss: 0.2378
Epoch 2/2
1189868/1189868 [==============================] - 127s 106us/step - loss: 0.2349 - val_loss: 0.2351
Train on 1176264 samples, validate on 327160 samples
Epoch 1/2
1176264/1176264 [==============================] - 125s 106us/step - loss: 0.2376 - val_loss: 0.2359
Epoch 2/2
1176264/1176264 [==============================] - 125s 106us/step - loss: 0.2353 - val_loss: 0.2358
Train on 1176532 samples, validate on 326892 samples
Epoch 1/2
1176532/1176532 [==============================] - 126s 107us/step - loss: 0.2562 - val_loss: 0.2538
Epoch 2/2
1176532/1176532 [==============================] - 129s 109us/step - loss: 0.2560 - val_loss: 0.2538
Train on 1195456 samples, validate on 307968 samples
Epoch 1/2
1195456/1195456 [==============================] - 128s 107us/step - loss: 0.2377 - val_loss: 0.2442
Epoch 2/2
1195456/1195456 [==============================] - 129s 108us/step - loss: 0.2347 - val_loss: 0.2361
Train on 1275576 samples, validate on 227848 samples
Epoch 1/2
1275576/1275576 [==============================] - 136s 107us/step - loss: 0.2383 - val_loss: 0.2339
Epoch 2/2
1275576/1275576 [==============================] - 135s 106us/step - loss: 0.2358 - val_loss: 0.2322
Ridge OOF RMSE: 0.24368481316948354
'''    
