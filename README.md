

### Single Models 
```
Model         Small Val          5CV Val            Leaderboard     Comment 
================================================================================================
mlp_1905      0.2159             ??????             0.2217          Add in different kinds for grouping continuous
mlp_1704A     0.2162             ??????             0.2217          Add in aggregate features from active files
mlp_1605B     0.2166             ??????             0.2224          Add all item titles from avctive files per user
mlp_1605A     0.2170             ??????             0.2228
nnet_1505     0.2177             ??????             
lgb_1404      0.2182             ??????             0.2241
```

### Blend (Weighted Average)
```
Model         Small Val          5CV Val            Leaderboard     Comment 
================================================================================================
mlp_1905      ????               ?????              0.2204          MLP 1705A and 1905 50/50 and mix 50/50 with best LB            
mlp_1705A     ????               ??????             0.2204          Weighted avg mlp and best lb kernel https://www.kaggle.com/lscoelho/blending-
mlp_1605B     ????               ??????             0.2208          Weighted avg mlp and best lb kernel https://www.kaggle.com/lscoelho/blending-models-lb-0-2216   
```

### Stack
```
Model         Small Val          5CV Val            Leaderboard     Comment
================================================================================================
```

