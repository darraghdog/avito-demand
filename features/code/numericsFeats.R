#install.packages("fasttime")
rm(list=ls())
gc();gc();gc()
library(data.table)
library(RcppRoll)
library(fasttime)
library(Hmisc)
library(readr)
library(plyr)
library(stringi)

path = '~/avito/data/'
#path = '/Users/dhanley2/Documents/avito/data/'

# Load up train/test and active files and join them all together, keeping the position of train/test
keepcols = c('title' ,"parent_category_name", "category_name", "price", 'region', 'city','param_1', 'param_2', 'param_3')
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

# Get target
y = data.table(read_csv(paste0(path, 'train.csv')))$deal_probability
idx = 1:length(y)


sort(table(alldf$category_name))
# Make feature on clother size - Childrens Clothes - 282753 - Children's clothing and footwear
table(alldf[category_name=='Детская одежда и обувь']$param_3)
s1 = '134-140 см (8-10 лет)'
s2 = '50-56 cм (0-2 мес)'
getMonthBin = function(x) as.numeric(grepl('лет', x))
getYearBin = function(x) as.numeric(grepl('мес', x))
getYearSmBin = function(x) as.numeric(grepl('года', x))
getNoSize = function(x) as.numeric(grepl('Без размера', x))

alldf[ , `:=`(cont_kidsclothes_mthb=NULL, cont_kidsclothes_yrb=NULL, cont_kidsclothes_yrsb=NULL, cont_kidsclothes_nosz=NULL)]
alldf[ , `:=`(cont_kidsclothes_mthb=0, cont_kidsclothes_yrb=0, cont_kidsclothes_yrsb=0, cont_kidsclothes_nosz=0)]
alldf[category_name=='Детская одежда и обувь', cont_kidsclothes_mthb:=  getMonthBin(alldf[category_name=='Детская одежда и обувь']$param_3)]
alldf[category_name=='Детская одежда и обувь', cont_kidsclothes_yrb:=  getYearBin(alldf[category_name=='Детская одежда и обувь']$param_3)]
alldf[category_name=='Детская одежда и обувь', cont_kidsclothes_yrsb:=  getYearSmBin(alldf[category_name=='Детская одежда и обувь']$param_3)]
alldf[category_name=='Детская одежда и обувь', cont_kidsclothes_nosz:=  getNoSize(alldf[category_name=='Детская одежда и обувь']$param_3)]

plot(table(alldf[idx]$cont_kidsclothes_mthb, cut2(y, g = 10)))
plot(table(alldf[idx]$cont_kidsclothes_yrb, cut2(y, g = 10)))
plot(table(alldf[idx]$cont_kidsclothes_yrsb, cut2(y, g = 10)))
plot(table(alldf[idx]$cont_kidsclothes_nosz, cut2(y, g = 10)))



# Make feature on clother size - Adults Clothes - 282753 - Clothes, shoes, accessories
alldf[category_name=='Одежда, обувь, аксессуары']
table(alldf[category_name=='Одежда, обувь, аксессуары']$param_3)
substrRight <- function(x, n) substr(x, nchar(x)-n+1, nchar(x))
getBrackets <- function(x){ ifelse(substrRight(x, 1) == ')' , 
                                   mapvalues(sub(".*\\((.*)\\).*", "\\1", x, perl=TRUE), from=c('XS', "S", "M", "L", 'XL', 'XXL', 'XXXL'), 
                                             to=as.character(1:7), warn_missing=F),0)} 
getFirstSize <- function(x) as.numeric(stri_extract_first_regex(x, "[0-9]+"))

alldf[ , `:=`(cont_adultclothes_sizel=NULL, cont_adultclothes_sizev=NULL, cont_adultclothes_sizeb=NULL)]
alldf[ , `:=`(cont_adultclothes_sizel='0', cont_adultclothes_sizev='0', cont_adultclothes_sizeb='0')]
alldf[category_name=='Одежда, обувь, аксессуары', cont_adultclothes_sizel:=  unlist(lapply(alldf[category_name=='Одежда, обувь, аксессуары']$param_3, getBrackets))]
alldf[category_name=='Одежда, обувь, аксессуары', cont_adultclothes_sizev:=  unlist(lapply(alldf[category_name=='Одежда, обувь, аксессуары']$param_3, getFirstSize))]
alldf[ , `:=`(cont_adultclothes_sizel=as.numeric(cont_adultclothes_sizel), 
              cont_adultclothes_sizev=as.numeric(cont_adultclothes_sizev))]
alldf[category_name=='Одежда, обувь, аксессуары', cont_adultclothes_sizeb:=  cont_adultclothes_sizev%%2]

plot(table(alldf[idx]$cont_adultclothes_sizel, cut2(y, g = 10)))
plot(table(alldf[idx]$cont_adultclothes_sizev, cut2(y, g = 10)))
plot(table(alldf[idx]$cont_adultclothes_sizeb, cut2(y, g = 10)))

# Apartments - Make features
#View(alldf[category_name=='Квартиры'])
# Cut up this
s= '1-к квартира, 44 м², 13/16 эт.'
totalFloors  = function(x) as.numeric(sub(".*/", "", gsub(' ', '', gsub('эт.', '', gsub(".*м²,", "", x)))))
apptFloors   = function(x) as.numeric(sub("/.*", "", gsub(' ', '', gsub('эт.', '', gsub(".*м²,", "", x)))))
apptsqMetres = function(x) as.numeric(gsub(' ', '', gsub(".*квартира,|.*Студия, ", "", gsub("м².*", "", x))))
apptRooms    = function(x) as.numeric(gsub(' ', '', gsub('Сниму|Куплю', '', gsub("-к .*", "", x))))
alldf[ , `:=`(cont_appt_totFl=NULL, cont_appt_apFl=NULL, cont_appt_sqM = NULL, cont_appt_rooms=NULL)]
alldf[ , `:=`(cont_appt_totFl='0', cont_appt_apFl='0', cont_appt_sqM = '0', cont_appt_rooms='0')]
alldf[category_name=="Квартиры", cont_appt_totFl:=  unlist(lapply(alldf[category_name=="Квартиры"]$title, totalFloors))]
alldf[category_name=="Квартиры", cont_appt_apFl :=  unlist(lapply(alldf[category_name=="Квартиры"]$title, apptFloors))]
alldf[category_name=="Квартиры", cont_appt_sqM  :=  unlist(lapply(alldf[category_name=="Квартиры"]$title, apptsqMetres))]
alldf[category_name=="Квартиры", cont_appt_rooms:=  unlist(lapply(alldf[category_name=="Квартиры"]$title, apptRooms))]
changeCols = grep('cont_', colnames(alldf), value = T)
alldf[,(changeCols):= lapply(.SD, as.numeric), .SDcols = changeCols]



plot(table(cut2(alldf[idx]$cont_appt_totFl, g=200), cut2(y, g = 10)))
plot(table(cut2(alldf[idx]$cont_appt_apFl, g=200), cut2(y, g = 10)))
plot(table(cut2(alldf[idx]$cont_appt_sqM , g=200), cut2(y, g = 10)))
plot(table(cut2(alldf[idx]$cont_appt_rooms , g=200), cut2(y, g = 10)))


# Automobiles
# View(alldf[category_name=='Автомобили'])
s = 'Mercedes-Benz S-класс, 2016'
alldf[ , cont_car_year := NULL]
alldf[ , cont_car_year := '0']
alldf[category_name=="Автомобили"]$cont_car_year = unlist(lapply(strsplit(alldf[category_name=="Автомобили"]$title, ', '), tail, n=1L))
alldf[cont_car_year == 'до 1960', cont_car_year := "1950"]
alldf[, cont_car_year := as.numeric(cont_car_year)]

plot(table(cut2(alldf[idx]$cont_car_year, g = 100), cut2(y, g = 10)))



# Check 
changeCols = grep('cont_', colnames(alldf), value = T)
alldfsmout = alldf[,changeCols, with = F]

# Write out the files
writeme = function(df, name){
  write.csv(df, 
            gzfile(paste0(path, '../features/', name,'.gz')), 
            row.names = F, quote = F)
}
writeme(alldfsmout, 'numericFeats')
#rm(list=ls())
gc();gc();gc()