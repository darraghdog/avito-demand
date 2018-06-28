#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(readr)

path = '~/avito/data/'
# path = '/Users/dhanley2/Documents/avito/data/'

# Load up train/test and active files and join them all together, keeping the position of train/test
keepcols = c("title","parent_category_name", "category_name", "price", 'item_seq_number', 'user_id', 'user_type')
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



# Build variance features
alldf[, pcat_price_varr := var(price, na.rm = T), by = parent_category_name]
alldf[, cat_price_varr  := var(price, na.rm = T), by = .(parent_category_name, category_name)]
alldf[, typc_price_varr  := var(price, na.rm = T), by = .(parent_category_name, category_name, user_type)]
alldf[, typp_price_varr  := var(price, na.rm = T), by = .(parent_category_name, user_type)]

# Take out the relevant rows
alldfout  = alldf[index!=-1][order(index)]
alldfout  = alldfout[,grep('_varr', colnames(alldf), value = T), with = F]/(10^14)
alldfout

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfout, 'price_variance_categories')
rm(list=ls())
gc();gc();gc()