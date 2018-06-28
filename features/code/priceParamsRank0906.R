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
keepcols = c('title' ,"parent_category_name", "category_name", "price", 'region', 'city','param_1', 'param_2', 'param_3')
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
gc(); gc(); gc(); gc(); gc(); gc()
alldf = rbind(alladf, alldfsm)
rm(alladf, alldfsm)
gc(); gc(); gc(); gc(); gc(); gc()




# Build stem ratio features
alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, title_stem)]
alldf[, ttlst_price_rratio := (((price/meanpr)*ct)+(10))/(ct+10)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, title_stem, region, city)]
alldf[, ttlst_city_price_rratio := (((price/meanpr)*ct)+(10))/(ct+10)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, title_stem, region, city, param_1, param_2, param_3)]
alldf[, ttlst_prm_price_rratio := (((price/meanpr)*ct)+(5))/(ct+5)]

# Build count features
alldf[, ttlst_count := .N, by = title_stem]
alldf[, ttlst_cat_count := .N, by = .(title_stem, category_name, region)]
alldf[, ttlst_catpr_count := .N, by = .(title_stem, category_name, region, price)]
alldf[, ttlst_catprm_count := .N, by = .(title_stem, category_name, region, price, param_1, param_2, param_3)]

# Build rank features by title
cols = c("title","parent_category_name", "category_name", 'region', 'city',
         'param_1', 'param_2', 'param_3', "price")
cols_np = c("title","parent_category_name", "category_name", 'region', 'city',
            'param_1', 'param_2', 'param_3')
alldf = alldf[!(is.na(title)&index==-1)]
alldf[, ct:= .N, by = cols_np]
alldf[,rank_ttl_asc := NULL]
alldf[,rank_ttl_dsc := NULL]
alldf[ct>1&(!is.na(price)),rank_ttl_asc := rank(price, ties.method = "min"), by = cols_np]
alldf[ct>1&(!is.na(price)),rank_ttl_dsc := rank(1-price, ties.method = "min"), by = cols_np]
alldf[ct<2, rank_ttl_asc := -1]
alldf[ct<2, rank_ttl_dsc := -1]
alldf[is.na(price), rank_ttl_asc := -1]
alldf[is.na(price), rank_ttl_dsc := -1]
gc();gc()

# Build rank features by title stem
cols = c("title_stem","parent_category_name", "category_name", 'region', 'city',
         'param_1', 'param_2', 'param_3', "price")
cols_np = c("title_stem","parent_category_name", "category_name", 'region', 'city',
            'param_1', 'param_2', 'param_3')
alldf[, ct:= .N, by = cols_np]
alldf[,rank_ttlst_asc := NULL]
alldf[,rank_ttlst_dsc := NULL]
alldf[ct>2&(!is.na(price)), rank_ttlst_asc := rank(price, ties.method = "min"), by = cols_np]
alldf[ct>2&(!is.na(price)), rank_ttlst_dsc := rank(1-price, ties.method = "min"), by = cols_np]
alldf[ct<3,rank_ttlst_asc := -1]
alldf[ct<3,rank_ttlst_dsc := -1]
alldf[is.na(price), rank_ttlst_asc := -1]
alldf[is.na(price), rank_ttlst_dsc := -1]

# Build rank features by title
cols = c("parent_category_name", "category_name", 'region', 'city',
         'param_1', 'param_2', 'param_3', "price")
cols_np = c("parent_category_name", "category_name", 'region', 'city',
            'param_1', 'param_2', 'param_3')
alldf[, ct:= .N, by = cols_np]
alldf[ct>2&(!is.na(price)),rank_prm_asc := rank(price, ties.method = "min"), by = cols_np]
alldf[ct>2&(!is.na(price)),rank_prm_dsc := rank(1-price, ties.method = "min"), by = cols_np]
alldf[ct<3,rank_prm_asc := -1]
alldf[ct<3,rank_prm_dsc := -1]
alldf[is.na(price), rank_prm_asc := -1]
alldf[is.na(price), rank_prm_dsc := -1]

# Take out the relevant rows
alldfout  = alldf[index!=-1][order(index)]
keepcols = c(grep('ratio', colnames(alldfout), value = T)  ,  grep('rank_', colnames(alldfout), value = T) ,  grep('_count', colnames(alldfout), value = T))
alldfout  = alldfout[, keepcols, with = F]
alldfout[is.na(alldfout )] = -1

# Check 
y = data.table(read_csv(paste0(path, 'train.csv')))$deal_probability
idx = 1:length(y)
plot(table(cut2(alldfout[idx]$ttlst_price_rratio, g = 20), cut2(y, g = 10)))
plot(table(cut2(alldfout[idx]$ttlst_city_price_rratio, g = 20), cut2(y, g = 10)))
plot(table(cut2(alldfout[idx]$ttlst_prm_price_rratio, g = 20), cut2(y, g = 10)))
plot(table(cut2(alldfout[idx]$rank_ttl_asc, g = 20), cut2(y, g = 10)))
plot(table(cut2(alldfout[idx]$rank_ttl_dsc, g = 20), cut2(y, g = 10)))
plot(table(cut2(alldfout[idx]$ttlst_count, g = 20), cut2(y, g = 10)))

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfout, 'price_rank_ratios0906')
#rm(list=ls())
gc();gc();gc()