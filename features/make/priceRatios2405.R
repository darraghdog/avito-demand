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

# Write out the <ip, device, os> level
keepcols = c("title","parent_category_name", "category_name", "price")
trndf = data.table(read_csv(paste0(path, 'train.csv')))
trndf = trndf[,keepcols,with=F]
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv')))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
alldf = rbind(trndf, tstdf)
train_length = nrow(trndf)
alldf[, index := 1:nrow(alldf)]
rm(trndf, tstdf)
gc(); gc()

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), by = parent_category_name]
alldf[, pcat_price_rratio := (((price/meanpr)*.N)+(20))/(.N+20)]
alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), by = .(parent_category_name, category_name)]
alldf[, cat_price_rratio := (((price/meanpr)*.N)+(20))/(.N+20)]
hist(alldf$cat_price_rratio, breaks = 1000000)
alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), by = .(parent_category_name, category_name, title)]
alldf[, ttl_price_rratio := (((price/meanpr)*.N)+(5))/(.N+5)]
hist(alldf$ttl_price_rratio, xlim = c(0, 3), breaks = 10000)
alldf[, `:=`(ct = .N, meanpr = mean(log(price+1), na.rm = T)), by = parent_category_name]
alldf[, pcat_log_price_rratio := (((log(price+1)/meanpr)*.N)+(20))/(.N+20)]
alldf[, `:=`(ct = .N, meanpr = mean(log(1+price), na.rm = T)), by = .(parent_category_name, category_name)]
alldf[, cat_log_price_rratio := (((log(1+price)/meanpr)*.N)+(20))/(.N+20)]
hist(alldf$cat_log_price_rratio, breaks = 1000)

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf[,grep('ratio', colnames(alldf), value = T), with = F], 'price_category_ratios')
rm(list=ls())
gc();gc();gc()

