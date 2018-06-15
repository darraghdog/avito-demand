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
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#path = '../input/'
path = "/home/darragh/avito/data/"
#path = '/Users/dhanley2/Documents/avito/data/'
#path = '/home/ubuntu/avito/data/'
start_time = time.time()
full = False

print('[{}] Load Train/Test'.format(time.time() - start_time))
usecols = ["item_id", 'title', 'description', "activation_date", 'category_name']
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
batch_size = 200
for t, sz in enumerate([400, 200, 100]):
    clust_col = 'cluster'+str(sz)
    df[clust_col] = 0
    for cat in tqdm(df['category_name'].unique()[::-1]):
        idx = df['category_name'] == cat
        n_clusters = int(np.ceil(sum(idx)/sz))
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0, random_state=100)
        fit_size = min(sum(idx), 20000)
        mbk.fit(ttl_embs[idx][:fit_size])
        df[clust_col][idx] = mbk.predict(ttl_embs[idx])
print('[{}] Finish clustering'.format(time.time() - start_time))        

print('[{}] Write clusters'.format(time.time() - start_time))        
#pd.DataFrame(ttl_clust).to_csv(path + '../features/title_clusts.csv')
df[['cluster%s'%(i) for i in [100,200,400]]].head()
df[['cluster%s'%(i) for i in [100,200,400]]].to_csv(path + '../features/title_clusts.csv', index = False)





