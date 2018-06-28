#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(Metrics)
library(readr)

path = '~/avito/data/'
path = '/Users/dhanley2/Documents/avito/data/'

trnpdf <- data.table(read_csv(paste0(path, "periods_train.csv")))
tstpdf <- data.table(read_csv(paste0(path, "periods_test.csv")))
allpdf = rbind(trnpdf, tstpdf)
rm(trnpdf, tstpdf); gc()
base_secs = (47*24*3600*365)
allpdf[, date_to         := as.numeric(fasttime::fastPOSIXct(date_to)) - base_secs] # Remove about 47 years from base seconds
allpdf[, date_from       := as.numeric(fasttime::fastPOSIXct(date_from)) - base_secs]
allpdf[, activation_date_sec := as.numeric(activation_date) - base_secs]
allpdf[, activation_len  := (date_to - date_from)/86400]
allpdf[, active_since  := abs((activation_date_sec - date_from)/86400)]
allpdf[, active_until  := abs((activation_date_sec - date_to)/86400)]
allpdf[, active_diff  := active_until - active_since]
gc();gc()

## No overlap of item_id with period and these files 
# trndf <- data.table(read_csv(paste0(path, "train.csv")))[,.(item_id, user_id, activation_date)]
# tstdf <- data.table(read_csv(paste0(path, "test.csv")))[,.(item_id, user_id, activation_date)]
# alldf = rbind(trndf, tstdf)
# rm(trndf, tstdf); gc()

trnadf <- data.table(read_csv(paste0(path, "train_active.csv")))[,.(item_id, user_id)]
gc();gc()
tstadf <- data.table(read_csv(paste0(path, "test_active.csv")))[,.(item_id, user_id)]
gc();gc()
alladf = rbind(trnadf, tstadf)
alladf = unique(alladf)
rm(trnadf, tstadf); gc()

setkeyv(alladf, "item_id")
setkeyv(allpdf, "item_id")

alldf = merge(allpdf, alladf, by = "item_id", all.x = F, all.y = F)
allaggdf = alldf[,(.N), by = .(item_id, user_id)]
allaggdf = allaggdf[, mean(N), by = user_id]
setnames(allaggdf, "", "user_mean_activeperiod_ct")

alldf = alldf[,.(user_id, activation_len, active_since, active_until, active_diff)]
gc();gc()
alldf

usrpdf = alldf[, .(sum(activation_len), min(activation_len), max(activation_len), mean(activation_len), 
                   length(activation_len), var(activation_len)), by = user_id]
usrpdf[is.na(V5)]$V5 = 1
setnames(usrpdf, c('user_id', "user_activ_sum", "user_activ_min", "user_activ_max", "user_activ_mean", 
                   "user_activ_len", "user_activ_var" ))

writeme = function(df, name) write.csv(df, gzfile(paste0(path, '../features/', name, '.gz')), row.names = F, quote = F)
writeme(usrpdf, "user_activ_period_stats")


