

### Single Models
```
Model         Small Val          5CV Val            Leaderboard     Comment
================================================================================================
rnn_2605      0.2146             ??????             0.2213          Logit averaging and stopword removal
lgb_2505      0.2143             ??????             0.2202          More FE - Price & Item Seq ratios over category/title
lgb_2405D     0.2145             ??????             0.2204          Price ratios over category/title   
lgb_2405      0.2152             ??????             0.2211          pymorph on text
lgb_2205      0.2153             ??????             0.2213          Add oof ridge feature on text data and image data.
rnn_2205      0.2149             ??????             0.2215          Russian text processing
lgb_2205      0.2157             ??????             0.2215          Add oof ridge feature on text data
mlp_1905      0.2159             ??????             0.2217          Add in different kinds for grouping continuous
mlp_1704A     0.2162             ??????             0.2217          Add in aggregate features from active files
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
Model         Small Val          5CV Val            Leaderboard     Comment
================================================================================================
```
