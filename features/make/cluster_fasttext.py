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
from tqdm import tqdm
import hdbscan

#path = '../input/'
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'
#path = '/home/ubuntu/avito/data/'
start_time = time.time()
full = False

print('[{}] Load Train/Test'.format(time.time() - start_time))
usecols = ["item_id", 'title', 'description', "activation_date"]
traindf = pd.read_csv(path + 'train.csv.zip', index_col = "item_id", usecols = usecols,  parse_dates = ["activation_date"], compression = 'zip')
traindex = traindf.index
testdf = pd.read_csv(path + 'test.csv.zip', index_col = "item_id", usecols = usecols,  parse_dates = ["activation_date"])
testdex = testdf.index
print('Train shape: {} Rows, {} Columns'.format(*traindf.shape))
print('Test shape: {} Rows, {} Columns'.format(*testdf.shape))
traindf['activation_date'].value_counts()


print('[{}] Combine Train and Test'.format(time.time() - start_time))
df = pd.concat([traindf,testdf],axis=0)
del traindf,testdf
gc.collect()
df['idx'] = range(df.shape[0])
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

'''
df[['title']].to_csv(path + '../fasttext/titles.txt', index = False)
df['text'] = (df['description'].fillna('') + ' ' + df['title'])
df[['text']].to_csv(path + '../fasttext/descriptions.txt', index = False)
'''
# download the bin file from fasttext:
# ./fastText-0.1.0/fasttext print-sentence-vectors cc.ru.300.bin < descriptions.txt > descriptions.vec
# ./fastText-0.1.0/fasttext print-sentence-vectors cc.ru.300.bin < titles.txt > titles.vec


gc.collect()
EMBEDDING_FILE = path+'../fasttext/titles.vec'
def get_coefs(t, *arr): return t-1, np.asarray(arr, dtype='float16')
embeddings_index = dict(get_coefs(t, *o.split()[-300:]) for (t, o) in tqdm(enumerate(open(EMBEDDING_FILE))) if t>0)
ttl_embs = np.stack(embeddings_index.values())
del embeddings_index
print('[{}] Start clustering'.format(time.time() - start_time))
hdbscan_ = hdbscan.HDBSCAN(min_cluster_size=30)
ttl_clust = hdbscan_.fit_predict(ttl_embs)
print('[{}] Finish clustering'.format(time.time() - start_time))        
pd.Series(ttl_clust).to_csv(path + '../features/title_clust.csv')

