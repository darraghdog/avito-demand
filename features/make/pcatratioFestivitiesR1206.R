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
             'param_1', 'param_2', 'param_3', 'title', 'user_id')
# Load up train/test and active files and join them all together, keeping the position of train/test
trndf = data.table(read_csv(paste0(path, 'train.csv'),  col_types = list('item_id' = col_skip(), 'description' = col_skip())))
trndf = trndf[,keepcols,with=F]
y = data.table(read_csv(paste0(path, 'train.csv'),  col_types = list('item_id' = col_skip(), 'description' = col_skip())))$deal_probability

gc(); gc()
gc(); gc()
trnadf = data.table(read_csv(paste0(path, 'train_active.csv'),col_types = list('item_id' = col_skip(), 'description' = col_skip())))
trnadf = trnadf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv'), col_types = list('item_id' = col_skip(), 'description' = col_skip())))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstadf = data.table(read_csv(paste0(path, 'test_active.csv'), col_types = list('item_id' = col_skip(), 'description' = col_skip())))
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

priceRatios = function(df, cols1, cols2, prior){
  df[, `:=`(ct1 = .N, meanpr1 = mean(price, na.rm = T)), by = cols1]
  df[, `:=`(ct2 = .N, meanpr2 = mean(price, na.rm = T)), by = c(cols1, cols2)]
  df[, tmpcol1:= (((meanpr2/meanpr1)*ct2)+(prior))/(ct1+prior)]
  df[, tmpcol2:= ct2/ct1]
  df[is.na(tmpcol1), tmpcol1:= 1]
  df[is.na(tmpcol2), tmpcol2:= 1]
  setnames(alldf, "tmpcol1", paste0(paste0(cols1, collapse = '_'), '___',paste0(cols2, collapse = '_'), "_pcatratio"))
  setnames(alldf, "tmpcol2", paste0(paste0(cols1, collapse = '_'), '___',paste0(cols2, collapse = '_'), "_ctratio"))
  return(alldf)
}
cols = c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city')

# Write out the <ip, device, os> level
ls = list(list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region'), c('city'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region'), c('user_id'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3'), c('city'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3'), c('title'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region'), c('title'), 10),
          list(c("parent_category_name", "category_name"), c('title'), 10),
          list(c("parent_category_name"), c("category_name"), 10),
          list(c("parent_category_name", "category_name"), c('user_id'), 10),
          list(c("parent_category_name", "category_name", 'activation_date'), c('user_id'), 10),
          list(c("user_id" ), c('activation_date'), 10),
          list(c("region" ), c('city'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region'), c('activation_date'), 20),
          list(c("parent_category_name", "category_name"), c('activation_date'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'title'), c('activation_date'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2'), c('param_3'), 20),
          list(c("parent_category_name", "category_name", 'param_1'), c('param_2', 'param_3'), 20),
          list(c("parent_category_name", "category_name"), c('param_1', 'param_2', 'param_3'), 20))
  
for (l in ls){
  cols1 = l[[1]]
  cols2 = l[[2]]
  prior = l[[3]]
  print(paste0(paste0(cols1, collapse = '_'), '___',paste0(cols2, collapse = '_')))
  alldf = priceRatios(alldf, cols1, cols2, prior = prior)
}

#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('_pcatratio|_ctratio', colnames(alldf), value = T), with = F]
View(alldf_enc[1:1000])
sum(is.na(alldf_enc))


# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'pcat_ratios_fest_1806')
rm(list=ls())
gc();gc();gc()

tdf = rbind(trndf, trnadf)
tdf[,ct:=.N, by=user_id]
View(tdf[ct>3,.(user_id, item_seq_number, price)][order(user_id, item_seq_number)][1:1000])
tdf


tdf[,ctis:=.N, by=.(user_id, item_seq_number)]
table(tdf$ctis)

table(tdf[1:length(y)]$ctis)

# hist(log1p(trndf$price))
# View(trndf[price>100000000])
# trndf = data.table(read_csv(paste0(path, 'train.csv'),  col_types = list('item_id' = col_skip())))
# 
# trndf[,l :=  nchar(description)]
# trndf[parent_category_name=='Услуги', .N, by = l][order(l)]
# 
# 
# 
# trndf$deal_probability
# 
# trndf[, mean(deal_probability==0), by = category_name]
# trndf[, mean(deal_probability==0.5), by = parent_category_name]
# trndf[, mean(deal_probability==1), by = parent_category_name]
# 
# trndf[, .N, by = activation_date][order(activation_date)]
# trndf[, mean(log1p(price), na.rm = T), by = activation_date][order(activation_date)]
# 
# 
# trndf[ parent_category_name=='Услуги', .N, by = deal_probability][order(deal_probability)]
# par(mfrow=c(2,1))
# hist(trndf[ (parent_category_name=='Услуги')  & is.na(price)]$deal_probability)
# hist(trndf[ (parent_category_name=='Услуги')  & !is.na(price)]$deal_probability)
# 
# View(trndf[ (parent_category_name=='Услуги')])
# View(trndf[ (parent_category_name=='Личные вещи')])
# trndf[, .N, by= parent_category_name]
# 
# idx = trndf$parent_category_name=='Услуги'
# plot(table(cut2(trndf[idx]$price, g = 5), cut2(trndf[idx]$deal_probability, g = 5)))
# 
# 99%%100
# plot(table(cut2(trndf[idx]$deal_probability, g = 5), trndf[idx]$price%%100 == 0))
# plot(table(cut2(trndf[idx]$price, g = 5), trndf[idx]$price%%100 == 0))
# 
# plot(table(cut2(trndf[idx]$deal_probability, g = 5), trndf[idx]$price%%1000 == 0))
# plot(table(cut2(trndf[idx]$price, g = 5), trndf[idx]$price%%1000 == 0))
# 
# plot(table(cut2(trndf[idx]$deal_probability, g = 5), trndf[idx]$price%%100 == 0))
# plot(table(cut2(trndf[idx]$deal_probability, g = 5), trndf[idx]$price%%1000 == 0))
# 
# plot(table(cut2(trndf[idx]$l, g = 20), cut2(trndf[idx]$deal_probability, g = 5)))
# plot(table(cut2(trndf[idx]$l, g = 20), cut2(trndf[idx]$deal_probability, g = 5)))
# 
# 
# table(trndf[1:100]$price)
# 
# 
