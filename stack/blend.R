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

subml = fread(paste0(path, '../sub/mlpsub_1705.csv'))[order(item_id)]
subrn = fread(paste0(path, '../sub/rnnsub_2205.csv'))[order(item_id)]
sublg = fread(paste0(path, '../sub/lgsub_2505.csv'))[order(item_id)]

cor(subrn$deal_probability, subml$deal_probability)
cor(sublg$deal_probability, subml$deal_probability)
cor(sublg$deal_probability, subrn$deal_probability)
mean(subrn$deal_probability)
mean(sublg$deal_probability)
mean(sublg$deal_probability)

subbld  = copy(sublg)
subbld$deal_probability = 0.5*(sublg$deal_probability) + 0.25*(subrn$deal_probability) + 0.25*(subml$deal_probability)

fwrite(subbld, paste0(path, '../sub/blend3x_2605.csv'))
