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

base_secs = (47*24*3600*365)
date2num = function(df, col){
  setkeyv(df, col)
  tmpdf = data.table(from = unique(df[[col]]))
  tmpdf[,to:= as.numeric(fasttime::fastPOSIXct(from)) - base_secs]
  setkeyv(tmpdf, 'from')
  df = df[tmpdf]
  df[[col]] = NULL
  setnames(df, 'to', col)
  return(df)
}

# Load train periods
tstpdf <- data.table(read_csv(paste0(path, "periods_test.csv")))
tstpdf = date2num(tstpdf, 'date_to')
tstpdf = date2num(tstpdf, 'date_from')
tstdf[, activation_date:= NULL]
setkeyv(tstpdf, c('item_id'))
setnames(tstpdf, 'activation_date', 'activation_date_period')
gc();gc()

# Load active file
keepcols = c("item_id", 'activation_date', "category_name", "price",  'region', 'city', 'param_1', 'param_2', 'param_3')
tstadf = data.table(read_csv(paste0(path, 'test_active.csv')))
tstadf= date2num(tstadf, 'activation_date')
tstadf = tstadf[,keepcols,with=F]
setkeyv(tstadf, c('item_id'))
gc(); gc()
gc(); gc()

# Join periods and active file
tstdf = merge(tstadf, tstpdf, by = 'item_id', all.x = T, all.y = F)


# Load up train/test and active files and join them all together, keeping the position of train/test
keepcols = c("item_id", 'activation_date', "category_name", "price",  'region', 'city', 'param_1', 'param_2', 'param_3')
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


# Join periods and the rest
alladf = date2num(alladf, 'activation_date')
alladf = alladf[(!is.na(category_name))]

setkeyv(alladf, c('item_id'))
setkeyv(allpdf, c('item_id'))
alladf = alladf[allpdf]

allpdf[item_id == '29be8c9abe10']

nrow(alldf[index>1])






usrpdf = alldf[, .(sum(activation_len), min(activation_len), max(activation_len), mean(activation_len), 
                   length(activation_len), var(activation_len)), by = user_id]
usrpdf[is.na(V5)]$V5 = 1
setnames(usrpdf, c('user_id', "user_activ_sum", "user_activ_min", "user_activ_max", "user_activ_mean", 
                   "user_activ_len", "user_activ_var" ))

writeme = function(df, name) write.csv(df, gzfile(paste0(path, '../features/', name, '.gz')), row.names = F, quote = F)
writeme(usrpdf, "user_activ_period_stats")


