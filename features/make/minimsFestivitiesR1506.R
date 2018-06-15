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
keepcols = c("parent_category_name", "category_name", "price",  'region', 'city', 'user_type', 'activation_date',
             'param_1', 'param_2', 'param_3', 'title', 'user_id')
# Load up train/test and active files and join them all together, keeping the position of train/test
trndf = data.table(read_csv(paste0(path, 'train.csv'),  col_types = list('item_id' = col_skip(), 'description' = col_skip())))
y = trndf$deal_probability
trndf = trndf[,keepcols,with=F]
gc(); gc()
gc(); gc()
trnadf = data.table(read_csv(paste0(path, 'train_active.csv'),col_types = list('item_id' = col_skip(), 'description' = col_skip())))
trnadf = trnadf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv'), col_types = list('item_id' = col_skip(), 'description' = col_skip())))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
gc(); gc()
tstadf = data.table(read_csv(paste0(path, 'test_active.csv'), col_types = list('item_id' = col_skip(), 'description' = col_skip())))
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
gc(); gc(); gc(); gc(); gc(); gc()
alldf = rbind(alladf, alldfsm)
rm(alladf, alldfsm)
gc(); gc(); gc(); gc(); gc(); gc()

# Write out the <ip, device, os> level
ls = list(list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city'), 20), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'title'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'title'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'region', 'city'), 20), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'region', 'city', 'title'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'region', 'title'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'region', 'city'), 20), 
          list(c("parent_category_name", "category_name", 'param_1', 'region', 'city', 'title'), 10),
          list(c("parent_category_name", "category_name", 'param_1', 'region', 'title'), 20),
          list(c("parent_category_name", "category_name", 'param_1', 'region'), 20))

#cols = c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3', 'region', 'city', 'title')

rankmin = function(df, cols){
  df[, `:=`(ct = .N, minval = min(price, na.rm=T)), by = cols]
  df[, minprice := as.integer(price == minval)]
  df[, summins := sum(minprice, na.rm = T), by = cols]
  df[summins>1, minprice := 0]
  df[price==0, minprice := 0]
  df[ , min_price := 2]
  df[ minprice == 1, min_price := 1/ct]
  df[ , `:=`(summins=NULL, ct=NULL, minval = NULL, minprice = NULL)]
  return(df)
}
rankmax = function(df, cols){
  df[, `:=`(ct = .N, maxval = max(price, na.rm=T)), by = cols]
  df[, maxprice := as.integer(price == maxval)]
  df[, summaxs := sum(maxprice, na.rm = T), by = cols]
  df[summaxs>1, maxprice := 0]
  df[price==0, maxprice := 0]
  df[ , max_price := 2]
  df[ maxprice == 1, max_price := 1/ct]
  df[ , `:=`(summaxs=NULL, ct=NULL, maxval = NULL, maxprice = NULL)]
  return(df)
}

for (l in ls){
  cols = l[[1]]
  print(paste0(paste0(cols, collapse = '_')))
  # Get the min price from a group and divide by the count of the group
  alldf = rankmin(alldf, cols)
  alldf = rankmax(alldf, cols)
  #hist(alldf[min_price < 1]$min_price)
  #hist(alldf[max_price < 1]$max_price)
  #plot(table(cut2(alldf[index!=-1][1:length(y)]$min_price, g = 1000 ), cut2(y, g = 10)))
  #plot(table(cut2(alldf[index!=-1][1:length(y)]$max_price, g = 1000 ), cut2(y, g = 10)))
  setnames(alldf, "min_price", paste0(paste0(cols, collapse = '_'), "___min_price"))
  setnames(alldf, "max_price", paste0(paste0(cols, collapse = '_'), "___max_price"))
}

#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('___max_price|___min_price', colnames(alldf), value = T), with = F]
View(alldf_enc[1:10000])
sum(is.na(alldf_enc))


# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'minmax_fest_1206')
rm(list=ls())
gc();gc();gc()

