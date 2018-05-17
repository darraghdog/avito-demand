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

trnpdf <- fread(paste0(path, "periods_train.csv"))
tstpdf <- fread(paste0(path, "periods_test.csv"))
allpdf = rbind(trnpdf, tstpdf)
rm(trnpdf, tstpdf); gc()
allpdf[, date_to         := as.numeric(fasttime::fastPOSIXct(date_to))]
allpdf[, date_from       := as.numeric(fasttime::fastPOSIXct(date_from))]
allpdf[, activation_date := as.numeric(fasttime::fastPOSIXct(activation_date))]
allpdf[, activation_len  := date_to - date_from]
gc();gc()

table(allpdf$activation_len%)
102%%10
