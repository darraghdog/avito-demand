#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(readr)

path = '~/avito/data/'
#path = '/Users/dhanley2/Documents/avito/data/'

# Load up train/test and active files and join them all together, keeping the position of train/test
keepcols = c("title","parent_category_name", "category_name", "price", 'image_top_1', 'region', 'city')
trndf = data.table(read_csv(paste0(path, 'train.csv')))
trndf = trndf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv')))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
gc(); gc()
alldf = rbind(trndf, tstdf)
alldf[, index := 1:nrow(alldf)]
train_length = nrow(trndf)
rm(trndf, tstdf)
gc(); gc()
gc(); gc(); gc(); gc(); gc(); gc()

# Build ratio features
alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, image_top_1)]
alldf[, cat_price_iratio := (((price/meanpr)*ct)+(20))/(ct+20)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, image_top_1, region)]
alldf[, reg_price_iratio := (((price/meanpr)*ct)+(20))/(ct+20)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, region)]
alldf[, reg_price_gratio := (((price/meanpr)*ct)+(200))/(ct+200)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, region, city)]
alldf[, cty_price_gratio := (((price/meanpr)*ct)+(20))/(ct+20)]


# Take out the relevant rows
alldfout  = alldf[order(index)]
alldfout  = alldfout[,grep('ratio', colnames(alldfout), value = T), with = F]
alldfout[is.na(alldfout)] = -1

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfout[,grep('ratio', colnames(alldfout), value = T), with = F], 'price_imagetop1_ratios')
rm(list=ls())
gc();gc();gc()
