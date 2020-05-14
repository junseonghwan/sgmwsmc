library(ggplot2)
d0<-read.csv("Google Drive/Research/repo/sgmwsmc/output/simulated_knot_matching0/simulatedDataPerformance0.csv", header=T)
d1<-read.csv("Google Drive/Research/repo/sgmwsmc/output/simulated_knot_matching0/simulatedDataPerformance1.csv", header=T)
d<-rbind(d0[1:50,], d1[1:50,])
sum(d$PredictionAccuracy)/sum(d$NumMatchings)

d0<-read.csv("Google Drive/Research/repo/sgmwsmc/output/simulated_knot_matching1/simulatedDataPerformance0.csv", header=T)
d1<-read.csv("Google Drive/Research/repo/sgmwsmc/output/simulated_knot_matching1/simulatedDataPerformance1.csv", header=T)
d<-rbind(d0[1:50,], d1[1:50,])
sum(d$PredictionAccuracy)/sum(d$NumMatchings)

d0<-read.csv("Google Drive/Research/repo/sgmwsmc/output/simulated_knot_matching2/simulatedDataPerformance0.csv", header=T)
d1<-read.csv("Google Drive/Research/repo/sgmwsmc/output/simulated_knot_matching2/simulatedDataPerformance1.csv", header=T)
d2<-rbind(d0[1:50,], d1[1:50,])
sum(d2$PredictionAccuracy)/sum(d2$NumMatchings)

plot(d$NumNodes, d$PredictionTimes)
points(d2$NumNodes, d2$PredictionTimes, col='red', pch=19)
plot(d$NumMatchings, d$PredictionTimes)
hist(d$NumNodes)
hist(d$NumMatchings)

j<-1
dir<-paste("Google Drive/Research/repo/sgmwsmc/output/knot_matching", j, sep="")
option_list<-read.csv(paste(dir, "/executionInfo/options.map", sep=""), header=F, sep="\t", row.names = 1)
lambda<-as.numeric(option_list["lambda",])

x<-read.csv(paste(dir, "/realDataPerformance.csv", sep=""), header=T)
x$idx<-1:30

plot(d0$NumNodes, d0$PredictionTimes, pch=19, col='red')
points(d1$NumNodes, d1$PredictionTimes, pch=19, col='blue')
points(x$NumNodes, x$PredictionTimes, pch=10, col='black')


