import pandas as pd
import numpy as np
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split

lgb = pd.read_csv('../lgCV_2505.csv.gz', compression='gzip')
rnn = pd.read_csv('../rnnCV_2805.csv.gz', compression='gzip')
mlp = pd.read_csv('../mlpCV_2505.csv.gz', compression='gzip')
truth = pd.read_csv('../../train.csv.zip', compression='zip')
y = truth['deal_probability'].values
truth.drop('deal_probability', 1)
test =  pd.read_csv('../../test.csv.zip', compression='zip')
test['deal_probability']=float('NAN') 
truth = pd.concat([truth,test[truth.columns]],axis=0)


lgb.rename(columns={'deal_probability': 'lgb_preds' }, inplace=True)
mlp.rename(columns={'deal_probability': 'mlp_preds' }, inplace=True)
preds_df = lgb.merge(rnn, on='item_id').merge(mlp, on='item_id').merge(truth, on='item_id',how='left')
train_df = preds_df[~preds_df['deal_probability'].isnull()]
test_df = preds_df[preds_df['deal_probability'].isnull()]

cols=['lgb_preds','rnn_preds','mlp_preds']
train_X, valid_X, train_y, valid_y = train_test_split(train_df[cols], y, train_size=.8, random_state=12345)


clf = linear_model.RidgeCV (alphas = [.5,1,2,3,5,10], cv=None, store_cv_values=True)
clf.fit(train_X,train_y)
print('RMSE:', np.sqrt(metrics.mean_squared_error( valid_y, clf.predict(valid_X))))
clf = linear_model.Ridge(clf.alpha_)
clf.fit(train_X,train_y)


test_df['deal_probability'] = clf.predict(test_df[['lgb_preds','rnn_preds','mlp_preds']])
test_df['deal_probability'] = np.clip(test_df['deal_probability'], .0001, .9999)
test_df[['item_id', 'deal_probability']].to_csv('../lrsub_3005L2.csv.gz', compression='gzip', index=False, header=True)