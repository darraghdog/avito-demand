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

trndf <- data.table(read_csv(paste0(path, "train.csv")))
tstdf <- data.table(read_csv(paste0(path, "test.csv")))
trnadf <- data.table(read_csv(paste0(path, "train_active.csv")))
tstadf <- data.table(read_csv(paste0(path, "test_active.csv")))
gc();gc()
