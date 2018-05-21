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

# Write out the <ip, device, os> level
keepcols = c("user_id","region","city", "category_name", "param_1","param_2","param_3",
             "image_top_1", "item_seq_number", "activation_date", "deal_probability")
trndf = data.table(read_csv(paste0(path, 'train.csv')))
trndf = trndf[,keepcols,with=F]
trndf[,is_train:=1]
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv')))
tstdf = tstdf[,keepcols[1:(length(keepcols)-1)],with=F]
tstdf[, is_train:=0]
tstdf[, deal_probability := NA]
gc(); gc()
alldf = rbind(trndf, tstdf)
train_length = nrow(trndf)
alldf[, index := 1:nrow(alldf)]
rm(trndf, tstdf)
gc(); gc()

alldf[, user_id_ct := .N, by = user_id]
alldf[, region_ct  := .N, by = region]
alldf[, city_ct    := .N, by = city]
alldf[, category_name_ct := .N, by = category_name]
alldf[, param_1_ct := .N, by = param_1]
alldf[, param_2_ct := .N, by = param_2]
alldf[, param_3_ct := .N, by = param_3]
alldf[, image_top_1_ct := .N, by = image_top_1]
alldf[, item_seq_number_ct  := .N, by = item_seq_number]

alldf_ct = alldf[, colnames(alldf) %in% paste0(keepcols, "_ct"), with = F]

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_ct, 'alldf_count')
rm(list=ls())
gc();gc();gc()

