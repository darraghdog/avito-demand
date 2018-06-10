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
keepcols = c('title' ,"parent_category_name", "category_name", "price", 'region', 'city','param_1', 'param_2', 'param_3')
trndf = data.table(read_csv(paste0(path, 'train.csv')))
trndf = trndf[,keepcols,with=F]
gc(); gc()
gc(); gc()
#trnadf = data.table(read_csv(paste0(path, 'train_active.csv')))
#trnadf = trnadf[,keepcols,with=F]
#gc(); gc()
#gc(); gc()
tstdf = data.table(read_csv(paste0(path, 'test.csv')))
tstdf = tstdf[,keepcols,with=F]
gc(); gc()
gc(); gc()
#tstadf = data.table(read_csv(paste0(path, 'test_active.csv')))
#tstadf = tstadf[,keepcols,with=F]
#gc(); gc()
#gc(); gc()
alldfsm = rbind(trndf, tstdf)
rain_length = nrow(trndf)
rm(trndf, tstdf)
gc(); gc()
#alladf = rbind(trnadf, tstadf)
#rm(trnadf, tstadf)
#gc(); gc()
alldfsm[, index := 1:nrow(alldfsm)]
#alladf[, index := -1]
gc(); gc(); gc(); gc(); gc(); gc()
#alldf = rbind(alladf, alldfsm)
#rm(alladf, alldfsm)
#gc(); gc(); gc(); gc(); gc(); gc()


alldfsm[ , year := NULL]
alldfsm[ , year := '']
alldfsm[category_name=="Автомобили"]$year = unlist(lapply(strsplit(alldfsm[category_name=="Автомобили"]$title, ', '), tail, n=1L))
alldfsm[year == 'до 1960', year := "1950"]
alldfsm[, year := as.numeric(year)]
alldfsm[is.na(year), year := -1]

# Check 
y = data.table(read_csv(paste0(path, 'train.csv')))$deal_probability
idx = 1:length(y)
plot(table(cut2(alldfsm[idx]$year, g = 200), cut2(y, g = 10)))

alldfsmout = alldfsm[,.(year)]

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfsmout, 'carYearBrand')
#rm(list=ls())
gc();gc();gc()