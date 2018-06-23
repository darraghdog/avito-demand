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

#compute entropy by group, over subgrp
calc_entropy <- function(df, group, subgrp, tgt_vn_prefix) {
  sum1 <- df[, .N, by=list(df[[group]], df[[subgrp]])]
  setnames(sum1, c(group, subgrp, 'subgrpcnt'))
  sum2 <- df[, .N, by=list(df[[group]])]
  setnames(sum2, c(group, 'cnt'))
  sum3 <- merge(sum2, sum1, by=c(group))
  sum3[, entropy := - log(subgrpcnt * 1.0 / cnt) * subgrpcnt * 1.0 / cnt]
  sum3[is.na(entropy), entropy := 0]
  sum4 <- sum3[, sum(entropy), by=list(sum3[[group]])]
  setnames(sum4, c(group, paste(tgt_vn_prefix, 'entropy', sep='_')))
  return(sum4)
}


# get the entropy features 
ls = list(list(c("user_id"), c("category_name")), 
          list(c("user_id"), c("parent_category_name")),
          list(c("user_id"), c("title")),
          list(c("user_id"), c("param_1")),
          list(c("user_id"), c("city")),
          list(c("user_id"), c("param_2")),
          list(c("user_id"), c("param_3")),
          list(c("user_id"), c("activation_date")))


entdt = calc_entropy(alldf, ls[[1]] [[1]], ls[[1]] [[2]], colnm)[order(user_id)]
entdt = entdt[, "user_id", with=F]
for(l in ls){
  colnm = paste(l[[1]], l[[2]], 'entropy', sep = '__')
  print(colnm)
  entdttmp = calc_entropy(alldf, l[[1]], l[[2]], colnm)[order(user_id)]
  entdt[, tmp := entdttmp[[2]]  ]
  setnames(entdt, "tmp", colnm)
}

View(entdt[1:1000])

#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('user_id', colnames(alldf), value = T), with = F]
alldf_enc[,index := 1:nrow(alldf_enc)]
alldf_enc = merge(alldf_enc, entdt, by = "user_id" )[order(index)]
View(alldf_enc[1:1000])

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'user_entropy_2306')
rm(list=ls())
gc();gc();gc()

