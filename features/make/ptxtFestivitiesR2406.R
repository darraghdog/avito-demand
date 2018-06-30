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

# Write out the <ip, device, os> level
drop_cols = c("price",  'region', 'city', 'user_type', 'activation_date',
              'user_id', 'image', 'image_top_1', 'deal_probability', 'item_seq_number')
drop_ls = list()
for (c in drop_cols) drop_ls[[c]] = col_skip()

parseNChars = function(f){
  df  = data.table(read_csv(paste0(path, f ), col_types = drop_ls))
  df[is.na(title), title := '']
  df[is.na(description), description := '']
  df[,`:=`(title_chars = nchar(title), description_chars = nchar(description))]
  df[,`:=`(title = NULL, description = NULL)]
  gc();gc()
  return(df)  
}

# Load up train/test and active files and join them all together, keeping the position of train/test
trndf  = parseNChars('train.csv')
trnadf = parseNChars('train_active.csv')
tstdf  = parseNChars('test.csv')
tstadf = parseNChars('test_active.csv')
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

ttlRatios = function(df, cols, prior){
  df[, `:=`(ct = .N, meanpr = mean(title_chars, na.rm = T)), by = cols]
  df[, tmpcol:= (((title_chars/meanpr)*ct)+(prior))/(ct+prior)]
  df[is.na(tmpcol), tmpcol:= 1]
  setnames(alldf, "tmpcol", paste0(paste0(cols, collapse = '_'), "_ttlratio5"))
  return(alldf)
}
dscRatios = function(df, cols, prior){
  df[, `:=`(ct = .N, meanpr = mean(description_chars, na.rm = T)), by = cols]
  df[, tmpcol:= (((description_chars/meanpr)*ct)+(prior))/(ct+prior)]
  df[is.na(tmpcol), tmpcol:= 1]
  setnames(alldf, "tmpcol", paste0(paste0(cols, collapse = '_'), "_dscratio5"))
  return(alldf)
}

# Write out the <ip, device, os> level
ls = list(list(c("parent_category_name", "category_name"), 100), 
          list(c("parent_category_name", "category_name", 'param_1'), 50), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2'), 30), 
          list(c("parent_category_name", "category_name", 'param_1', 'param_2', 'param_3'), 20), 
          list(c("parent_category_name"), 500))

for (l in ls){
  cols = l[[1]]
  prior = l[[2]]
  print(cols)
  alldf = ttlRatios(alldf, cols, prior = prior)
  alldf = dscRatios(alldf, cols, prior = prior)
}

#Create a second title column
alldf_enc = alldf[index!=-1][order(index)][, grep('ratio5', colnames(alldf), value = T), with = F]
View(alldf_enc[1:1000])
alldf_enc[is.na(alldf_enc)] = -1
sum(is.na(alldf_enc))


# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldf_enc, 'ptext_fest_2506')
rm(list=ls())
gc();gc();gc()

