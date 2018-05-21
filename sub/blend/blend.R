#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(Metrics)

path = '~/avito/data/'
path = '/Users/dhanley2/Documents/avito/data/'

subbl = fread(paste0(path, '../sub/mlpblend_1905.csv'))
subrn = fread(paste0(path, '../sub/rnnsub_2005.csv'))
subbl = subbl[order(item_id)]
subrn = subrn[order(item_id)]

cor(subbl$deal_probability, subrn$deal_probability)
subbld = subbl
subbld$deal_probability = subbl$deal_probability*0.9 + subrn$deal_probability*0.1
fwrite(subbld, paste0(path, '../sub/mlpblend_2105.csv'))
