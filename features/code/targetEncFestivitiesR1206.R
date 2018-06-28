#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(readr)

path = '~/avito/data/'
path = '/Users/dhanley2/Documents/avito/data/'

# Write out the <ip, device, os> level
keepcols = c("parent_category_name", "category_name", "price",  'region', 'city', 'user_type', 'activation_date',
             'param_1', 'param_2', 'param_3', 'image_top_1', 'title', 'description', 'deal_probability')
trndf = data.table(read_csv(paste0(path, 'train.csv'), col_types = list('user_id' = col_skip())))
trndf = trndf[,keepcols,with=F]
trndf[,is_train:=1]
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv'), col_types = list( 'user_id' = col_skip())))
tstdf = tstdf[,keepcols[1:(length(keepcols)-1)],with=F]
tstdf[, is_train:=0]
tstdf[, deal_probability := NA]
gc(); gc()
alldf = rbind(trndf, tstdf)
train_length = nrow(trndf)
alldf[, index := 1:nrow(alldf)]
rm(trndf, tstdf)
gc(); gc()


##### These are our five folds
foldls = list(c("2017-03-15", "2017-03-16", "2017-03-17"), 
              c("2017-03-18", "2017-03-19", "2017-03-20"), 
              c("2017-03-21", "2017-03-22", "2017-03-23"), 
              c("2017-03-24", "2017-03-25", "2017-03-26"), 
              c("2017-03-27", "2017-03-28", "2017-03-29", 
                "2017-03-30", "2017-03-31", "2017-04-01",
                "2017-04-02", "2017-04-03","2017-04-07"))
for (f in foldls) print(nrow(alldf[(is_train == 1)][(activation_date %in% as.Date(f))]))
for (col in keepcols) print(paste(col, length(unique(alldf[[col]]))))

table(alldf[is_train==1]$activation_date)
alldf[, fold_:=6]
alldf[activation_date %in% as.Date(foldls[[1]]), fold_:= 1]
alldf[activation_date %in% as.Date(foldls[[2]]), fold_:= 2]
alldf[activation_date %in% as.Date(foldls[[3]]), fold_:= 3]
alldf[activation_date %in% as.Date(foldls[[4]]), fold_:= 4]
alldf[activation_date %in% as.Date(foldls[[5]]), fold_:= 5]
table(alldf$activation_date, alldf$fold_)

# Function for out of fold Bayes mean
oofBMean = function(df, fold, cols_, prior = 200, allrows_ = FALSE){
  glob_mean = mean(df[(fold_ != fold) & (is_train == 1)]$deal_probability)
  glob_ct   = nrow(df[(fold_ != fold) & (is_train == 1)])
  # Aggregate the infold data
  indf = df[(fold_ != fold) & (is_train == 1),.(.N, mean(deal_probability)), by = c(cols_)]
  if (allrows_){
    outdf = df[(is_train == 0), c(cols_), with=F]
  }else{
    outdf = df[(fold_ == fold) & (is_train == 1), c(cols_, "fold_"), with=F]
  }
  setnames(indf, c(cols_, "count", "mean"))
  indf[, bmean := ((mean*count)+(glob_mean*prior))/(count+prior)]
  indf[,mean := NULL]
  indf[,count := NULL]
  outdf[, idx := 1:nrow(outdf)]
  outdf = merge(outdf, indf, by = cols_, all.x = T)[order(idx)]
  outdf[ is.na(bmean), bmean:= glob_mean]
  return(outdf$bmean)
}


folds_ = sort(unique(alldf[(is_train == 1)]$fold_))
for (col in keepcols) print(paste(col, length(unique(alldf[[col]]))))

keepcols
# Get bayes mean user_id using folds
encodeDF = function(alldf, col, prior = 1000){
  alldf[, tmpcol := -1]
  for (fold in folds_) alldf[(fold_ == fold) & (is_train == 1), tmpcol := oofBMean(alldf, fold, col, prior )]
  alldf[(is_train == 0), tmpcol :=  oofBMean(alldf, 6, col, allrows_ = TRUE, prior)]
  setnames(alldf, "tmpcol", paste0(col, "_enc4", collapse = '_'))
  return(alldf)
}

# Write out the <ip, device, os> level
keepcols = c("parent_category_name", "category_name", "price",  'region', 'city', 'user_type', 'activation_date',
             'param_1', 'param_2',  'image_top_1', 'title', 'deal_probability')

keepcols = c("parent_category_name", "category_name", "price",  'region', 'city', 'user_type',
             'param_1', 'param_2', 'deal_probability')

ls = list(list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city'), 20), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'title'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'image_top_1'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'image_top_1', 'price'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'image_top_1', 'title'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'image_top_1', 'title', 'price'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'image_top_1', 'title', 'description'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'title'), 20),
          list(c("parent_category_name",  'region', 'title'), 20),
          list(c("parent_category_name",  'region', 'title', 'description'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'title', 'description'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'image_top_1'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'user_type'), 20), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'title', 'user_type'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'image_top_1', 'user_type'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'image_top_1', 'title', 'user_type'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'title', 'user_type'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'image_top_1', 'user_type'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'user_type'), 20))
for (l in ls){
  cols = l[[1]]
  prior = l[[2]]
  print(cols)
  alldf = encodeDF(alldf, cols, prior = 50)
  alldf[, tmpct := .N, by= cols]
  setnames(alldf, "tmpct", paste0(cols, "_count4", collapse = '_'))
}

#Create a second title column
alldf_enc = alldf[, grep('_enc4|_count4', colnames(alldf), value = T), with = F]

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'alldf_bayes_fest_1206')
rm(list=ls())
gc();gc();gc()

