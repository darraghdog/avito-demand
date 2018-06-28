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
keepcols = c('item_id', "parent_category_name", "category_name", "price",  'region', 'city', 'user_type', 'activation_date',
             'param_1', 'param_2', 'param_3', 'title', 'user_id')
# Load up train/test and active files and join them all together, keeping the position of train/test
trndf = data.table(read_csv(paste0(path, 'train.csv'),  col_types = list('description' = col_skip())))
trndf = trndf[,keepcols,with=F]
gc(); gc()
gc(); gc()
trnadf = data.table(read_csv(paste0(path, 'train_active.csv'),col_types = list('description' = col_skip())))
trnadf = trnadf[,keepcols,with=F]
gc(); gc()
gc(); gc()

alldf = rbind(trndf, trnadf)
alldf[, index := c((1:nrow(trndf)), rep(-1, nrow(alldf)-nrow(trndf)))]
gc(); gc(); gc(); gc(); gc(); gc()
rm(trnadf, trndf)
gc(); gc(); gc(); gc(); gc(); gc()


# tstdf = data.table(read_csv(paste0(path, 'test.csv'), col_types = list('description' = col_skip())))
# tstdf = tstdf[,keepcols,with=F]
# gc(); gc()
# gc(); gc()
# tstadf = data.table(read_csv(paste0(path, 'test_active.csv'), col_types = list('description' = col_skip())))
# tstadf = tstadf[,keepcols,with=F]
# gc(); gc()
# gc(); gc()
# alldfsm = rbind(trndf, tstdf)
# train_length = nrow(trndf)
# rm(trndf, tstdf)
# gc(); gc()
# alladf = rbind(trnadf, tstadf)
# rm(trnadf, tstadf)
# gc(); gc()
# alldfsm[, index := 1:nrow(alldfsm)]
# alladf[, index := -1]
# gc(); gc(); gc(); gc(); gc(); gc()
# alldf = rbind(alladf, alldfsm)
# rm(alladf, alldfsm)
# gc(); gc(); gc(); gc(); gc(); gc()

#ptrndf = fread(paste0(path, 'periods_train.csv'))
#ptstdf = fread(paste0(path, 'periods_test.csv'))
#pdf    = rbind(ptstdf, ptrndf)
#rm(ptstdf, ptrndf)
pdf = fread(paste0(path, 'periods_train.csv'))
gc(); gc(); gc(); gc(); gc(); gc()

pdf[,ct:= .N, by = item_id]
pdf = pdf[order(item_id, date_from)]
hist(pdf$ct)
View(pdf[1:1000])

# Check the overlap
table(pdf$item_id %in% alldf[index==-1]$item_id)
table(pdf$item_id %in% alldf[index!=-1]$item_id)

# Lets join it up
alldftmp = alldf[,.(title, user_id, price, city, activation_date, item_id)]
setnames(pdf, "activation_date", "activation_date_p")
?merge
alldftmp = merge(alldftmp, pdf, 'item_id', all = T)

alldftmp[, activation_date_x := ifelse(is.na(activation_date_p), as.character(activation_date), activation_date_p)]
alldftmp = alldftmp[order(user_id, title, activation_date_x, date_to)]
View(alldftmp[1:10000])
gc(); gc()
# fasttime::fastPOSIXct(date_to)

# extending the values forward 
rollfwd = function(x){
  values <- c(NA, x[!is.na(x)])
  ind <- cumsum(!is.na(x)) + 1
  y <- values[ind]
  return(y)
}
alldftmp[, date_from_roll := rollfwd(date_from)]
alldftmp[, date_to_roll := rollfwd(date_to)]
alldftmp[((user_id!=shift(user_id, 1, type = 'lag')) |(title!=shift(title, 1, type = 'lag')) )&(is.na(date_to)),  `:=`(date_from_roll = NA, date_to_roll =NA )]

alldfsm = alldftmp[is.na(date_to)]
alldfsm[, `:=`(date_to = NULL, date_from=NULL, activation_date_p=NULL, activation_date_x=NULL, ct=NULL)]
alldfsm[, `:=`(date_to_roll = fasttime::fastPOSIXct(date_to_roll), date_from_roll = fasttime::fastPOSIXct(date_from_roll))]
alldfsm[, `:=`(activation_date = fasttime::fastPOSIXct(activation_date))]
alldfsm[, since_last_end := (as.numeric(activation_date)-as.numeric(date_to_roll))/(3600*24)]
alldfsm[, since_last_start := (as.numeric(activation_date)-as.numeric(date_to_roll))/(3600*24)]
alldfsm[, since_last_len   := (as.numeric(date_to_roll)-as.numeric(date_from_roll))/(3600*24)]


writeme(alldfsm[!is.na(since_last_len), .(item_id, since_last_end, since_last_start, since_last_len) ], 'since_last_2406')



#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('user_id|category_name|title', colnames(alldf), value = T), with = F]
alldf_enc[,index := 1:nrow(alldf_enc)]
alldf_enc = merge(alldf_enc, entdt, by = "user_id" )[order(index)]
alldf_enc = merge(alldf_enc, catdt, by = "category_name" )[order(index)]
alldf_enc = merge(alldf_enc, ttldt, by = "title" )[order(index)]
View(alldf_enc[1:1000])
alldf_enc[, `:=`(user_id=NULL, category_name=NULL, title=NULL, parent_category_name=NULL, index = NULL)]

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'since_last_2406')
rm(list=ls())
gc();gc();gc()

