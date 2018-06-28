### Kaggle - Avito Demand Prediction
  
**5th Place solution - team `Optumize`**  
Predict demand for an online classified ads  
https://www.kaggle.com/c/avito-demand-prediction  
  

## Models Progrssion
### Single Models
```
Model         Small Val          5CV Val            Leaderboard     Comment
================================================================================================
lgb_1406A     0.2113             0.2136             ??????          tuning - from 250 to 1000 leaves
lgb_1406      0.2113             0.2136             ??????          tuning - from 250 to 1000 leaves
lgb_1106A     0.2117             0.2140             ??????          Add more encoding features
lgb_1006      0.2120             0.2153             ??????          Add param features                    
lgb_0906      0.2123             0.2158             ??????          Add price ranking reatures
lgb_0206      0.2132             0.2162             ??????          Add title translation as col and tfidf it
lgb_3105      0.2134             0.2168             0.2190          Meta image features
rnndh_0406a   0.2136                                0.2194          Add price ratios
lgb_2705B     0.2137             0.2168             0.2194          imgtop1 ratios;longer early stopping;remove categorical
lgb_2705A     0.2139             ??????             0.2197          Remove categoricals, add image_top_1 price ratio
lgb_2505      0.2143             0.2167             0.2202          More FE - Price & Item Seq ratios over category/title
lgb_2405D     0.2145             ??????             0.2204          Price ratios over category/title   
lgb_2405      0.2152             ??????             0.2211          pymorph on text
rnn_2605      0.2146             ??????             0.2213          Logit averaging and stopword removal
lgb_2205      0.2153             ??????             0.2213          Add oof ridge feature on text data and image data.
rnn_2205      0.2149             ??????             0.2215          Russian text processing
lgb_2205      0.2157             ??????             0.2215          Add oof ridge feature on text data
mlp_1905      0.2159             ??????             0.2217          Add in different kinds for grouping continuous
mlp_1705      0.2162             0.21875            0.2217          Add in aggregate features from active files
rnn_2105      0.2153             ??????             0.2221          RNN only submission, more regularization .2153 on validation
mlp_1605B     0.2166             ??????             0.2224          Add all item titles from avctive files per user
lgb_2105C     0.2162             ??????             0.2225          Add count and encoding
mlp_1605A     0.2170             ??????             0.2228
nnet_1505     0.2177             ??????             
lgb_2105      0.2174             ??????             0.2133             
lgb_1404      0.2182             ??????             0.2241
```

### Blend (Weighted Average)
```
Model         Small Val          5CV Val            Leaderboard     Comment
================================================================================================
blend3x_2605  ????               ?????              0.2188          Blend of 0.25 * mlp_1705, 0.5 * lgb_2505 and 0.25 * rnn_2205
all_2405      ????               ?????              0.2193          Equal Blend of mlp_1905, lgb_2205 and rnn_2205
mlp_1905      ????               ?????              0.2204          MLP 1705A and 1905 50/50 and mix 50/50 with best LB            
mlp_1705A     ????               ??????             0.2204          Weighted avg mlp and best lb kernel https://www.kaggle.com/lscoelho/blending-
mlp_1605B     ????               ??????             0.2208          Weighted avg mlp and best lb kernel https://www.kaggle.com/lscoelho/blending-models-lb-0-2216   
```

### Stack
```
Model         Small CV Val       5CV Val         Leaderboard     Comment
              in blend script
================================================================================================
L1GBM_2306    0.2113             ?????              0.2147          More ridge and add user entropy features
L1GBM_2006A   0.2118             ?????              0.2151          different tfidf
L1GBM_1606B   0.2122             ?????              0.2153          More features at L2
L1GBM_1506B   0.2126             ?????              0.2154          Bag the L2 lgb; Bag the 1406 sub
L1GBM_1506    0.2127             ?????              0.2155          Add lgb 1406 leaves tuning
L1GBM_1006A   0.2133             ?????              0.2161          Add param features
L1GBM_1006    ??????             ?????              0.2163          Add price ranking
L1GBM_0406A   ??????             ?????              0.2166          fixed lgb bug at L1, hash 'text' instead of description
L1GBM_0306A   ??????             ?????              0.2167          Included title translation LGB 

```

### Word embeddings
```
=================================================================================================
features/wiki.ru.vec                  - https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
cc.ru.300.vec.gz                      - https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
all.norm-sz100-w10-cb0-it1-min100.w2v - http://panchenko.me/data/dsl-backup/w2v-ru/
```
