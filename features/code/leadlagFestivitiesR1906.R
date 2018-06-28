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
keepcols = c("parent_category_name", "category_name", "price", 'user_type', 'activation_date', 'user_id', 'item_seq_number')
# Load up train/test and active files and join them all together, keeping the position of train/test
trndf = data.table(read_csv(paste0(path, 'train.csv'),  col_types = list('item_id' = col_skip(), 'title' = col_skip(), 'description' = col_skip())))
trndf = trndf[,keepcols,with=F]
trndf = trndf[,test:=0]
gc(); gc()
gc(); gc()
trnadf = data.table(read_csv(paste0(path, 'train_active.csv'),col_types = list('item_id' = col_skip(), 'title' = col_skip(), 'description' = col_skip())))
trnadf = trnadf[,keepcols,with=F]
trnadf = trnadf[,test:=0]
gc(); gc()
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv'), col_types = list('item_id' = col_skip(), 'title' = col_skip(), 'description' = col_skip())))
tstdf = tstdf[,keepcols,with=F]
tstdf = tstdf[,test:=1]
gc(); gc()
gc(); gc()
tstadf = data.table(read_csv(paste0(path, 'test_active.csv'), col_types = list('item_id' = col_skip(), 'title' = col_skip(), 'description' = col_skip())))
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

alldf = alldf[, c(keepcols, 'test', 'index'), with = F]
# Make the features
# Lead lag features based on activation date
alldf = alldf[order(user_id, test, activation_date)]
alldf[, `:=`(price_datesort_lag  = shift(price, 1, type = 'lag') , seq_datesort_lag  = shift(item_seq_number, 1, type = 'lag' ))]
alldf[, `:=`(price_datesort_lead = shift(price, 1, type = 'lead'), seq_datesort_lead = shift(item_seq_number, 1, type = 'lead'))]
alldf[(user_id != shift(user_id, 1, type = 'lag')) | (test != shift(test, 1, type = 'lag')), `:=`(price_datesort_lag  = -1, seq_datesort_lag  = -1)]
alldf[(user_id != shift(user_id, 1, type = 'lead')) | (test != shift(test, 1, type = 'lead')), `:=`(price_datesort_lead  = -1, seq_datesort_lead  = -1)]
alldf[, `:=`(seq_datesort_lag_diff = item_seq_number - seq_datesort_lag, seq_datesort_lead_diff = item_seq_number - seq_datesort_lead)]
alldf[seq_datesort_lag  == -1, `:=`(seq_datesort_lag_diff = -999)]
alldf[seq_datesort_lead == -1, `:=`(seq_datesort_lead_diff = -999)]

# Lead lag features based on item sequence
alldf = alldf[order(user_id, test, item_seq_number)]
alldf[, `:=`(price_seqsort_lag  = shift(price, 1, type = 'lag') , seq_seqsort_lag  = shift(item_seq_number, 1, type = 'lag' ))]
alldf[, `:=`(price_seqsort_lead = shift(price, 1, type = 'lead'), seq_seqsort_lead = shift(item_seq_number, 1, type = 'lead'))]
alldf[(user_id != shift(user_id, 1, type = 'lag')) | (test != shift(test, 1, type = 'lag')), `:=`(price_seqsort_lag  = -1, seq_seqsort_lag  = -1)]
alldf[(user_id != shift(user_id, 1, type = 'lead')) | (test != shift(test, 1, type = 'lead')), `:=`(price_seqsort_lead  = -1, seq_seqsort_lead  = -1)]
alldf[, `:=`(seq_seqsort_lag_diff = item_seq_number - seq_seqsort_lag, seq_seqsort_lead_diff = item_seq_number - seq_seqsort_lead)]
alldf[seq_seqsort_lag  == -1, `:=`(seq_seqsort_lag_diff = -999)]
alldf[seq_seqsort_lead == -1, `:=`(seq_seqsort_lead_diff = -999)]

# number of repeat postings of that item
alldf[, item_seq_repeat_postings := .N, by = .(user_id, item_seq_number)]
alldf[, category_repeat_postings := .N, by = .(user_id, category_name)]

# Compare to min and max of sequences
alldf = alldf[order(user_id, item_seq_number)]
alldf[, price_min_sequence := price[1], by = .(user_id)]
alldf[, price_min_sequence_diff := price - price_min_sequence]
alldf[, seq_min_sequence := min(item_seq_number), by = .(user_id)]
alldf[, seq_min_sequence_diff := item_seq_number - seq_min_sequence]

# NA and 0 ratio
alldf[, na_ratio_user := sum(is.na(price))/.N, by = .(user_id)]
alldf[, na_ratio_user := sum(is.na(price))/.N, by = .(category_name)]
alldf[, zero_ratio_user := sum(is.na(price))/.N, by = .(user_id)]
alldf[, zero_ratio_user := sum(is.na(price))/.N, by = .(category_name)]

# Rolling mean by sequence
alldf = alldf[order(user_id, item_seq_number, test)]
alldf[, rmean_price_byseq3_1 := roll_mean(price, 3, fill = -1, na.rm = T, align = 'left'), by = .(user_id, test)]
alldf[, rmean_price_byseq3_2 := roll_mean(price, 3, fill = -1, na.rm = T, align = 'right'), by = .(user_id, test)]
alldf[, rmean_price_byseq3_3 := roll_mean(price, 3, fill = -1, na.rm = T, align = 'center'), by = .(user_id, test)]
alldf[, rmean_price_byseq8_1 := roll_mean(price, 8, fill = -1, na.rm = T, align = 'left'), by = .(user_id, test)]
alldf[, rmean_price_byseq8_2 := roll_mean(price, 8, fill = -1, na.rm = T, align = 'right'), by = .(user_id, test)]
alldf[, rmean_price_byseq8_3 := roll_mean(price, 8, fill = -1, na.rm = T, align = 'center'), by = .(user_id, test)]

#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('_datesort_|_seqsort_|_repeat_postings|_min_sequence|_ratio_|rmean_price_byseq', colnames(alldf), value = T), with = F]
View(alldf_enc[1:1000])
sum(is.na(alldf_enc))
alldf_enc[is.na(alldf_enc)] = -99999

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'pseq_leadlag_festivities_1906')
rm(list=ls())
gc();gc();gc()