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

priceRatios = function(df, cols, prior){
  df[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), by = cols]
  df[, tmpcol:= (((price/meanpr)*ct)+(prior))/(ct+prior)]
  df[is.na(tmpcol), tmpcol:= 1]
  setnames(alldf, "tmpcol", paste0(paste0(cols, collapse = '_'), "_fratio5"))
  return(alldf)
}
cols = c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city')

# Write out the <ip, device, os> level
ls = list(list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city'), 20), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'title'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'title'), 20),
          list(c("parent_category_name",  'region', 'title'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2',  'user_id'), 20),
          list(c("parent_category_name", "category_name", 'user_id'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'user_id'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'user_id', 'city'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'user_id', 'city'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'user_id'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'user_type'), 20), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'title', 'user_type'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'title', 'user_type'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'user_type'), 20))
for (l in ls){
  cols = l[[1]]
  prior = l[[2]]
  print(cols)
  alldf = priceRatios(alldf, cols, prior = prior)
  alldf[, tmpct := .N, by= setdiff(cols, 'price')]
  setnames(alldf, "tmpct", paste0(paste0(cols, collapse = '_'), "_count5"))
}

#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('_count5|_fratio5', colnames(alldf), value = T), with = F]
View(alldf_enc[1:1000])
sum(is.na(alldf_enc))


# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'pratios_fest_1206')
rm(list=ls())
gc();gc();gc()

