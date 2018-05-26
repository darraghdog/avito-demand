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

# Load up train/test and active files and join them all together, keeping the position of train/test
trndf = data.table(read_csv(paste0(path, 'train.csv')))
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv')))
tstdf[, deal_probability := -1]
gc(); gc()
nrow_train = nrow(trndf)
alldf = rbind(trndf, tstdf)
rm(trndf, tstdf)
gc(); gc(); gc(); gc(); gc(); gc()
table(alldf$activation_date, alldf$deal_probability==-1)

alldf[1:nrow_train,.(mean(deal_probability), .N), by=city][order(N)]



# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfout[,grep('_rratio', colnames(alldfout), value = T), with = F], 'lag_')
rm(list=ls())
gc();gc();gc()