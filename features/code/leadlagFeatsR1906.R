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

getRollMinLead = function(df, cols_, fname, path, df_out = FALSE, shift_n = 1){
  df$click_time = fasttime::fastPOSIXct(df$click_time)
  df$click_sec = as.numeric(df$click_time)
  df$click_day  = wday(df$click_time)
  df = df[,c(cols_, "click_sec", "click_time", "click_day"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec"))
  df[,click_sec_shift_lead := shift(click_sec, shift_n, type = "lead")]
  df[,seq_lead := .N:1, by = cols_ ]
  df[,click_sec_lead := click_sec_shift_lead - click_sec]
  df[seq_lead %in% 1:shift_n, click_sec_lead := -1]
  df[,lead_app_cum_min := click_sec_lead]
  df[lead_app_cum_min == -1 ,lead_app_cum_min:= 99999]
  df[,lead_app_cum_min := cummin(lead_app_cum_min) , by = c(cols_, "click_day")]
  df[,lead_app_cum_click_day := 1:length(lead_app_cum_min) , by = c(cols_, "click_day")]
  df = df[,.(lead_app_cum_min, lead_app_cum_click_day)]
  if (df_out){
    return(df)
  }else{
    write.csv(df, 
              gzfile(paste0(path, fname)), 
              row.names = F, quote = F)
  }
  gc();gc();gc()
}

getSplitLead = function(df, cols_, fname, path, shift_n = 1){
  df$click_sec = as.numeric(fasttime::fastPOSIXct(df$click_time))
  df[, split_sec := round((0:(.N-1))/.N, 4), by = click_time]
  df = df[,c(cols_, "click_sec", "split_sec"), with = F]
  df[, index := 1:nrow(df)]
  setorderv(df, c(cols_, "click_sec", "split_sec"))
  df[,click_sec_shift_lead := shift(click_sec+split_sec, shift_n, type = "lead")]
  df[,seq_lead := .N:1, by = cols_ ]
  df[,click_sec_lead := click_sec_shift_lead - (click_sec + split_sec)]
  df[,click_sec_lead := round(click_sec_lead, 4)]
  df[seq_lead %in% 1:shift_n, click_sec_lead := 999999]
  setorderv(df, "index")
  new_name = "click_sec_lead_split_sec"
  setnames(df, "click_sec_lead", new_name)
  df = df[,new_name,with=F]
  return(df)
}

# Make the features
View(alldf[order(user_id, test, activation_date)][1:1000])

alldf = alldf[order(user_id, test, activation_date)]
alldf[, `:=`(price_lag  = shift(price, 1, type = 'lag') , seq_lag  = shift(item_seq_number, 1, type = 'lag' ))]
alldf[, `:=`(price_lead = shift(price, 1, type = 'lead'), seq_lead = shift(item_seq_number, 1, type = 'lead'))]
alldf[(user_id != shift(user_id, 1, type = 'lag')) | (test != shift(test, 1, type = 'lag')), `:=`(price_lag  = -1, seq_lag  = -1)]
alldf[(user_id != shift(user_id, 1, type = 'lead')) | (test != shift(test, 1, type = 'lead')), `:=`(price_lead  = -1, seq_lead  = -1)]
alldf[, `:=`(seq_lag_diff = item_seq_number - seq_lag, seq_lead_diff = item_seq_number - seq_lead)]
alldf[seq_lag  == -1, `:=`(seq_lag_diff = -999)]
alldf[seq_lead == -1, `:=`(seq_lead_diff = -999)]


#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('_lag|_lead', colnames(alldf), value = T), with = F]
View(alldf_enc[1:1000])
sum(is.na(alldf_enc))
alldf_enc[is.na(alldf_enc)] = -999

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'pseq_lead_lag_1806')
rm(list=ls())
gc();gc();gc()