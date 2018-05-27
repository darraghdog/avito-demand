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
keepcols = c("title","parent_category_name", "category_name", "price", 'item_seq_number', 'user_id', 'region', 'city',
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

# Build ratio features
alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)),
      by = parent_category_name]
alldf[, pcat_price_rratio := (((price/meanpr)*ct)+(200))/(ct+200)]
alldf[, pcat_itseq_rratio := (((item_seq_number/meansq)*ct)+(200))/(ct+200)]
hist(alldf$pcat_price_rratio, breaks = 10000)
hist(alldf$pcat_itseq_rratio, breaks = 10000)

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name)]
alldf[, cat_price_rratio := (((price/meanpr)*ct)+(50))/(ct+50)]
alldf[, cat_itseq_rratio := (((item_seq_number/meansq)*ct)+(50))/(ct+50)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, region)]
alldf[, reg_price_rratio := (((price/meanpr)*ct)+(50))/(ct+50)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, region, city)]
alldf[, cty_price_rratio := (((price/meanpr)*ct)+(10))/(ct+10)]


alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, param_1)]
alldf[, par1_price_rratio := (((price/meanpr)*ct)+(50))/(ct+50)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, param_1, param_2)]
alldf[, par2_price_rratio := (((price/meanpr)*ct)+(50))/(ct+50)]


alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, param_1, param_2, param_3)]
alldf[, par3_price_rratio := (((price/meanpr)*ct)+(50))/(ct+50)]

# alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
#       by = .(parent_category_name, category_name, image_top_1)]
# alldf[, imgt1_price_rratio := (((price/meanpr)*ct)+(10))/(ct+10)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, title)]
alldf[, ttl_price_rratio := (((price/meanpr)*ct)+(5))/(ct+5)]
alldf[, ttl_itseq_rratio := (((item_seq_number/meansq)*ct)+(5))/(ct+5)]


alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, title, region)]
alldf[, ttlr_price_rratio := (((price/meanpr)*ct)+(5))/(ct+5)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = .(parent_category_name, category_name, title, region, city)]
alldf[, ttlc_price_rratio := (((price/meanpr)*ct)+(5))/(ct+5)]


# alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
#       by = .(parent_category_name, category_name, title, image_top_1)]
# alldf[, ttli_price_rratio := (((price/meanpr)*ct)+(5))/(ct+5)]

alldf[, `:=`(ct = .N, meanpr = mean(log(price+1), na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = parent_category_name]
alldf[, pcat_log_price_rratio := (((log(price+1)/meanpr)*ct)+(200))/(ct+200)]
alldf[, pcat_log_itseq_rratio := (((log(item_seq_number+1)/meansq)*ct)+(200))/(ct+200)]

alldf[, `:=`(ct = .N, meanpr = mean(log(price+1), na.rm = T), meansq = mean(item_seq_number, na.rm = T)), 
      by = parent_category_name]
alldf[, user_log_price_rratio := (((log(price+1)/meanpr)*ct)+(10))/(ct+10)]

# Take out the relevant rows
alldfout  = alldf[index!=-1][order(index)]
alldfout[,grep('_rratio', colnames(alldfout), value = T), with = F]

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfout[,grep('_rratio', colnames(alldfout), value = T), with = F], 'price_seq_category_ratios2705')
#rm(list=ls())
gc();gc();gc()