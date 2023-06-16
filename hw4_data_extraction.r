airquality
without_na_airquality = airquality[complete.cases(airquality),]
length(without_na_airquality[,1])

# write.csv(without_na_airquality, file="./data/airquality_data")
pairs(airquality[,1:4], panel = panel.smooth, main = "airquality data")

hist(airquality[,2])
