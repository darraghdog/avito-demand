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
keepcols = c("parent_category_name", "category_name", "price",  'region', 'city', 'user_type',
             'param_1', 'param_2')
trndf = data.table(read_csv(paste0(path, 'train.csv')))
trndf = trndf[,keepcols,with=F]
gc(); gc()
gc(); gc()
trnadf = data.table(read_csv(paste0(path, 'train_active.csv')))
trnadf = trnadf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv')))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstadf = data.table(read_csv(paste0(path, 'test_active.csv')))
tstadf = tstadf[,keepcols,with=F]
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
alldf = rbind(alladf, alldfsm)
rm(alladf, alldfsm)
gc(); gc(); gc(); gc(); gc(); gc()

# Build ratio features
alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, region, city, param_1)]
alldf[, par1cty_price_prratio := (((price/meanpr)*ct)+(20))/(ct+20)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, region, city, param_1, param_2)]
alldf[, par2cty_price_prratio := (((price/meanpr)*ct)+(20))/(ct+20)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, region, user_type, param_1)]
alldf[, par1utyp_price_prratio := (((price/meanpr)*ct)+(20))/(ct+20)]

alldf[, `:=`(ct = .N, meanpr = mean(price, na.rm = T)), 
      by = .(parent_category_name, category_name, region, user_type, param_1, param_2)]
alldf[, par2utyp_price_prratio := (((price/meanpr)*ct)+(20))/(ct+20)]

for(col in grep('_prratio', colnames(alldfout), value = T)){
  alldf[[col]][is.na(alldf[[col]])] = median(alldf[[col]], na.rm = T)
}


# Take out the relevant rows
alldfout  = alldf[index!=-1][order(index)]
alldfout  = alldfout[,grep('_prratio', colnames(alldfout), value = T), with = F]

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfout, 'price_param_ratios1006')
#rm(list=ls())
gc();gc();gc()