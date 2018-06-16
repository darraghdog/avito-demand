
# https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lgbm-clone?scriptVersionId=3540839

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
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


#path = '../input/'
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'

path = '/home/ubuntu/avito/data/'
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
dnimgtrn = csr_matrix(np.load(path+'../features/vgg19_pool_array_train_float16.npy'))
dnimgtst = csr_matrix(np.load(path+'../features/vgg19_pool_array_test_float16.npy'))

print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
del traindf,testdf
gc.collect()
df['idx'] = range(df.shape[0])
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

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
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, f in enumerate(folds):
        train_index, test_index = np.where(f==False), np.where(f)
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y.values[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        gc.collect()

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

print('[{}] Start ridge'.format(time.time() - start_time))
SEED    = 200
NFOLDS  = 5
ntrain  = df.loc[traindex,:].shape[0]
ntest   = df.loc[testdex,:].shape[0]

ridge_params = { 'alpha':0.000001, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED }

ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
#ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])
ridge_oof_train, ridge_oof_test = get_oof(ridge, dnimgtrn , y, dnimgtst)

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))
# Ridge OOF RMSE: 0.25157
# With normalising 0.2584

gc.collect()


ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
df['ridge_img_preds'] = ridge_preds
df[['ridge_img_preds']].to_csv(path + '../features/ridgeImgVGG19_5CV.csv.gz', compression = 'gzip', index = False)

df['ridge_img_preds'].hist()
