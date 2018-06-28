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
             'param_1', 'param_2', 'param_3', 'title', 'user_id', 'item_seq_number')
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

# Write out the <ip, device, os> level
ls = list(list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city'), 20), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'title'), 5),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'title'), 5),
          list(c("parent_category_name",  'region', 'title'), 5),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'region'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'region'), 20),
          list(c("parent_category_name", "category_name"), 20),
          list(c("parent_category_name", "category_name", 'param_1'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3'), 20))

# Create User types
table(alldf$user_type)
alldf[, user_type_1 := as.integer(user_type=='Private')]
alldf[, user_type_2 := as.integer(user_type=='Shop')]
alldf[, user_type_3 := as.integer(user_type=='Company')]
table(alldf$user_type_2)

cols = c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3')
globutyp1 = mean(alldf$user_type_1, na.rm = T)
globutyp2 = mean(alldf$user_type_2, na.rm = T)
globutyp3 = mean(alldf$user_type_3, na.rm = T)
globseqmn = mean(log1p(alldf$item_seq_number), na.rm = T)

for (l in ls){
  cols = l[[1]]
  prior = l[[2]]
  print(cols)
  # itemseq number
  alldf[, `:=`(tmpmn = mean(log1p(item_seq_number), na.rm=T), tmpct=.N), by= cols]
  alldf[, tmpbmn := ((tmpmn*tmpct)+(globseqmn*prior))/(prior+tmpct)]
  alldf[is.na(tmpmn), tmpbmn := globseqmn]
  setnames(alldf, "tmpbmn", paste0(paste0(cols, collapse = '_'), "_itemseqmean"))
  # user types
  alldf[, `:=`(tmpmn = mean(user_type_1, na.rm=T), tmpct=.N), by= cols]
  alldf[, tmpbmn := ((tmpmn*tmpct)+(globutyp1*prior))/(prior+tmpct)]
  alldf[is.na(tmpmn), tmpbmn := globutyp1]
  setnames(alldf, "tmpbmn", paste0(paste0(cols, collapse = '_'), "_itemtyp1"))
  
  alldf[, `:=`(tmpmn = mean(user_type_2, na.rm=T), tmpct=.N), by= cols]
  alldf[, tmpbmn := ((tmpmn*tmpct)+(globutyp2*prior))/(prior+tmpct)]
  alldf[is.na(tmpmn), tmpbmn := globutyp2]
  setnames(alldf, "tmpbmn", paste0(paste0(cols, collapse = '_'), "_itemtyp2"))
  
  alldf[, `:=`(tmpmn = mean(user_type_3, na.rm=T), tmpct=.N), by= cols]
  alldf[, tmpbmn := ((tmpmn*tmpct)+(globutyp3*prior))/(prior+tmpct)]
  alldf[is.na(tmpmn), tmpbmn := globutyp3]
  setnames(alldf, "tmpbmn", paste0(paste0(cols, collapse = '_'), "_itemtyp3"))
}

#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('_itemseqmean|_itemtyp', colnames(alldf), value = T), with = F]
View(alldf_enc[1:1000])
sum(is.na(alldf_enc))


# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'agg_seqtyp_1306')
#rm(list=ls())
gc();gc();gc()


# Write out the <ip, device, os> level
keepcols = c("parent_category_name", "category_name", "price",  'region', 'city', 'user_type', 'activation_date',
             'param_1', 'param_2', 'param_3', 'title', 'user_id', 'item_seq_number', 'image_top_1')
# Load up train/test and active files and join them all together, keeping the position of train/test
trndf = data.table(read_csv(paste0(path, 'train.csv'),  col_types = list('item_id' = col_skip(), 'description' = col_skip())))
trndf = trndf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv'), col_types = list('item_id' = col_skip(), 'description' = col_skip())))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
gc(); gc()
alldfsm = rbind(trndf, tstdf)
train_length = nrow(trndf)
rm(trndf, tstdf)
alldfsm[, index := 1:nrow(alldfsm)]
gc(); gc()

alldfsm[, image_top_1_there := as.integer(!is.na(image_top_1))]
table(alldfsm$image_top_1_there)
globutyp1 = mean(alldfsm$image_top_1_there, na.rm = T)

for (l in ls){
  cols = l[[1]]
  prior = l[[2]]
  print(cols)
  # user types
  alldfsm[, `:=`(tmpmn = mean(image_top_1_there, na.rm=T), tmpct=.N), by= cols]
  alldfsm[, tmpbmn := ((tmpmn*tmpct)+(globutyp1*prior))/(prior+tmpct)]
  alldfsm[is.na(tmpmn), tmpbmn := globutyp1]
  setnames(alldfsm, "tmpbmn", paste0(paste0(cols, collapse = '_'), "_aggimgtop1"))
}

#Create a second title column
alldf_enc = alldfsm[index!=-1][order(index)][, grep('_aggimgtop1', colnames(alldfsm), value = T), with = F]
View(alldf_enc[1:1000])
sum(is.na(alldf_enc))
writeme(alldf_enc, 'agg_imgpresence_1306')
#rm(list=ls())
gc();gc();gc()