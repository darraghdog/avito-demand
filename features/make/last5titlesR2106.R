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
keepcols = c("title", 'activation_date', 'user_id', 'item_seq_number')
# Load up train/test and active files and join them all together, keeping the position of train/test
trndf = data.table(read_csv(paste0(path, 'train.csv'),  col_types = list('item_id' = col_skip(), 'image' = col_skip(),  'description' = col_skip())))
trndf = trndf[,keepcols,with=F]
trndf = trndf[,test:=0]
gc(); gc()
gc(); gc()
trnadf = data.table(read_csv(paste0(path, 'train_active.csv'),col_types = list('item_id' = col_skip(), 'image' = col_skip(), 'description' = col_skip())))
trnadf = trnadf[,keepcols,with=F]
trnadf = trnadf[,test:=0]
gc(); gc()
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv'), col_types = list('item_id' = col_skip(), 'image' = col_skip(), 'description' = col_skip())))
tstdf = tstdf[,keepcols,with=F]
tstdf = tstdf[,test:=1]
gc(); gc()
gc(); gc()
tstadf = data.table(read_csv(paste0(path, 'test_active.csv'), col_types = list('item_id' = col_skip(), 'image' = col_skip(), 'description' = col_skip())))
tstadf = tstadf[,keepcols,with=F]
tstadf = tstadf[,test:=1]
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

# 
alldf = alldf[order(user_id, test, activation_date, item_seq_number)]
alldf[, titles_last_five := ' ' ]
for(i in 1:5){
  print(paste0('Shift ', i))
  alldf[, `:=` (title_shift = shift(title, i), user_shift = shift(user_id, i), test_shift = shift(test, i))]
  alldf[user_id!=user_shift, title_shift := ' ']
  alldf[test   !=test_shift, title_shift := ' ']
  alldf[is.na(test_shift), title_shift := ' ']
  alldf[, titles_last_five := paste(titles_last_five, title_shift)]
}
alldf[, `:=` (title_shift = NULL, user_shift = NULL, test_shift = NULL)]
View(alldf[1:1000])


#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('titles_last_five', colnames(alldf), value = T), with = F]
View(alldf_enc[1:1000])
sum(is.na(alldf_enc))
alldf_enc[is.na(alldf_enc)] = -99999

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'last_five_title_2106')
rm(list=ls())
gc();gc();gc()