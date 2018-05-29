import pandas as pd
import numpy as np
from sklearn import metrics

lgb = pd.read_csv('../lgCV_2505.csv.gz', compression='gzip')
rnn = pd.read_csv('../rnnCV_2805.csv.gz', compression='gzip')
mlp = pd.read_csv('../mlpCV_2505.csv.gz', compression='gzip')
truth = pd.read_csv('../../train.csv.zip', compression='zip', usecols=['item_id','deal_probability'])


lgb.rename(columns={'deal_probability': 'lgb_preds' }, inplace=True)
mlp.rename(columns={'deal_probability': 'mlp_preds' }, inplace=True)
preds_df = lgb.merge(rnn, on='item_id').merge(mlp, on='item_id').merge(truth, on='item_id',how='left')
train_df = preds_df[~preds_df['deal_probability'].isnull()]
test_df = preds_df[preds_df['deal_probability'].isnull()]

print ('Train RMSE (lgb,rnn, mlp)')
for col in ['lgb', 'rnn', 'mlp']:
    print('RMSE:', np.sqrt(metrics.mean_squared_error( train_df['deal_probability'], train_df[col+'_preds'])))
print ('Train Correlations (truth, lgb, rnn , mlp)\n', np.corrcoef([train_df['deal_probability'], train_df['lgb_preds'], train_df['rnn_preds'], train_df['mlp_preds']]))
print ('Test Correlations (truth, lgb, rnn , mlp)\n', np.corrcoef([test_df['deal_probability'],test_df['lgb_preds'], test_df['rnn_preds'], test_df['mlp_preds']]))