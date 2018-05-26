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

# Load up train/test and active files and join them all together, keeping the position of train/test
keepcols = c("title","parent_category_name", "category_name", "price", 
             'user_id', 'user_type', 'region', 'city', 'item_seq_number')
trndf = read_csv(paste0(path, 'train.csv'))
trndf = trndf[,keepcols]
gc(); gc()
trndf = data.table(trndf)
gc(); gc()

trnadf = read_csv(paste0(path, 'train_active.csv'))
trnadf = trnadf[,keepcols]
gc(); gc()
trnadf = data.table(trnadf)
gc(); gc()

tstdf = read_csv(paste0(path, 'test.csv'))
tstdf = tstdf[,keepcols]
gc(); gc()
tstdf = data.table(tstdf)
gc(); gc()


tstadf = read_csv(paste0(path, 'test_active.csv'))
tstadf = tstadf[,keepcols]
gc(); gc()
tstadf = data.table(tstadf)
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

# Build ratio features
alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)),
      by = .(parent_category_name, user_type)]
alldf[, pcat_price_rratio := (((price/meanpr)*ct)+(200))/(ct+200)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, user_type)]
alldf[, cat_price_rratio := (((price/meanpr)*ct)+(50))/(ct+50)]
alldf[, cat_itseq_rratio := (((item_seq_number/meansq)*ct)+(50))/(ct+50)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, title)]
alldf[, ttl_price_rratio := (((price/meanpr)*ct)+(5))/(ct+5)]

alldf[, `:=`(ct = .N, meanpr = mean(log(price+1), na.rm = T)), 
      by = .(parent_category_name, user_type)]
alldf[, pcat_log_price_rratio := (((log(price+1)/meanpr)*ct)+(200))/(ct+200)]

alldf[, `:=`(ct = .N, meanpr = mean(log(price+1), na.rm = T)), 
      by = .(parent_category_name, user_id)]
alldf[, user_log_price_rratio := (((log(price+1)/meanpr)*ct)+(10))/(ct+10)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, user_type, region)]
alldf[, reg_price_rratio := (((price/meanpr)*ct)+(50))/(ct+50)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, user_type, region, city)]
alldf[, cty_price_rratio := (((price/meanpr)*ct)+(50))/(ct+50)]

# Take out the relevant rows
alldfout  = alldf[index!=-1][order(index)]
alldfout[,grep('_rratio', colnames(alldfout), value = T), with = F]

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfout[,grep('_rratio', colnames(alldfout), value = T), with = F], 'price_category_ratios_2505A')
rm(list=ls())
gc();gc();gc()