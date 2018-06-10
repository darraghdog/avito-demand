#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(readr)

path = '~/avito/data/'
#path = '/Users/dhanley2/Documents/avito/data/'

# Load up train/test and active files and join them all together, keeping the position of train/test
keepcols = c("title","parent_category_name", "category_name", "price", 'region', 'city','activation_date',
             'param_1', 'param_2', 'param_3')
trndf = data.table(read_csv(paste0(path, 'train.csv')))
trndf = trndf[,keepcols,with=F]
gc(); gc()
gc(); gc()
trnadf = data.table(read_csv(paste0(path, 'train_active.csv')))
trnadf = trnadf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv')))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstadf = data.table(read_csv(paste0(path, 'test_active.csv')))
tstadf = tstadf[,keepcols,with=F]
gc(); gc()
gc(); gc()
alldfsm = rbind(trndf, tstdf)
train_length = nrow(trndf)
rm(trndf, tstdf)
gc(); gc()
alladf = rbind(trnadf, tstadf)
rm(trnadf, tstadf)
gc(); gc()
alldfsm[, index := 1:nrow(alldfsm)]
alladf[, index := -1]
alldf = rbind(alladf, alldfsm)
rm(alladf, alldfsm)
gc(); gc(); gc(); gc(); gc(); gc()

# Build rank features by title
cols = c("title","parent_category_name", "category_name", 'region', 'city',
             'param_1', 'param_2', 'param_3', "price")
cols_np = c("title","parent_category_name", "category_name", 'region', 'city',
         'param_1', 'param_2', 'param_3')
#setkeyv(alldf, cols)
alldf = alldf[!(is.na(title)&index==-1)]
alldf[, ct:= .N, by = cols_np]
alldf[,rank_ttl_asc := -1]
alldf[,rank_ttl_dsc := -1]
alldf[ct>1,rank_ttl_asc := rank(price, ties.method = "min"), by = cols_np]
alldf[ct>1,rank_ttl_dsc := rank(1-price, ties.method = "min"), by = cols_np]
alldf[is.na(price), rank_ttl_asc := -1]
alldf[is.na(price), rank_ttl_dsc := -1]


# Build rank features by title
cols = c("parent_category_name", "category_name", 'region', 'city',
         'param_1', 'param_2', 'param_3', "price")
cols_np = c("parent_category_name", "category_name", 'region', 'city',
            'param_1', 'param_2', 'param_3')
alldf[, ct:= .N, by = cols_np]
alldf[ct>2,rank_prm_asc := rank(price, ties.method = "min"), by = cols_np]
alldf[ct>2,rank_prm_dsc := rank(1-price, ties.method = "min"), by = cols_np]
alldf[ct<3,rank_prm_asc := -1]
alldf[ct<3,rank_prm_dsc := -1]
alldf[is.na(price), rank_prm_asc := -1]
alldf[is.na(price), rank_prm_dsc := -1]

# Build rank features by title
cols = c("parent_category_name", "category_name", 'region', 'city', "price")
cols_np = c("parent_category_name", "category_name", 'region', 'city')
alldf[, ct:= .N, by = cols_np]
alldf[ct>2,rank_cat_asc := rank(price, ties.method = "min"), by = cols_np]
alldf[ct>2,rank_cat_dsc := rank(1-price, ties.method = "min"), by = cols_np]
alldf[ct<3,rank_cat_asc := -1]
alldf[ct<3,rank_cat_dsc := -1]
alldf[is.na(price), rank_cat_asc := -1]
alldf[is.na(price), rank_cat_dsc := -1]

# Check the features
trndf = data.table(read_csv(paste0(path, 'train.csv')))
y = trndf$deal_probability
rm(trndf)
gc()

# Take out the relevant rows
alldfout  = alldf[index!=-1][order(index)]
alldfout  = alldfout[,grep('rank_', colnames(alldfout), value = T), with = F]

idx = 1:length(y)
table(cut2(alldfout[idx]$rank_cat_asc, g = 20), cut2(y, g = 10))


# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfout, 'price_seq_category_ranks')
#rm(list=ls())
gc();gc();gc()