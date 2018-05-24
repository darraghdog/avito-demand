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

subbl = fread(paste0(path, '../sub/submission.csv'))[order(item_id)] # Public kernel -- https://www.kaggle.com/moussaid/moyen-just-simple-moyen-of-my-predictions/output
sublg = fread(paste0(path, '../sub/lgsub_2405.csv'))[order(item_id)]
subm1 = fread(paste0(path, '../sub/mlpsub_1905.csv'))[order(item_id)]
subm2 = fread(paste0(path, '../sub/mlpsub_1705.csv'))[order(item_id)]
subrn = fread(paste0(path, '../sub/rnnsub_2205.csv'))[order(item_id)]

cor(subbl$deal_probability, subm1$deal_probability)
cor(subrn$deal_probability, subm1$deal_probability)
cor(subm1$deal_probability, subm2$deal_probability)
cor(sublg$deal_probability, subbl$deal_probability)

mean()

subbld  = copy(subbl)
subbld$deal_probability = 0.2*(subbl$deal_probability + sublg$deal_probability) + 
                            0.15*(subm1$deal_probability + subm2$deal_probability) + 
                            0.3*(subrn$deal_probability)

fwrite(subbld, paste0(path, '../sub/blendall_2405.csv'))
