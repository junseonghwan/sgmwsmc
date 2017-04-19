setwd("Dropbox/Research/papers/probabilistic-matching/sgmwsmc/data/16Oct2015")
library(ggplot2)

boards<-c(4, 8, 17, 18, 20, 24)

f<-function(i)
{
  file<-paste("Board ", boards[i], "/labelledMatching.csv", sep="")
  d<-read.csv(file, header=F)
  names(d)
  dd<-subset(d, V9 > 0)
  labels<-unique(dd$V9)
  stat<-matrix(0, ncol = 12, nrow = length(labels))
  for (label in labels)
  {
    knots<-subset(dd, V9 == label)
    dim1_diff<-knots[1,]$V7 - knots[2,]$V7
    dim2_diff<-knots[1,]$V8 - knots[2,]$V8
    a1 <- knots[1,]$V7 * knots[1,]$V8
    a2 <- knots[2,]$V7 * knots[2,]$V8
    wide <- sum(knots$V1 %%  2) == 0
    diff <- abs(a1 - a2)
    ratio <- min(a1, a2)/max(a1, a2)
    #dist <- sqrt(sum((knots[1, c(3, 5, 6)] - knots[2, c(3, 5, 6)])^2))
    dist <- sqrt(sum((knots[1, c(4)] - knots[2, c(4)])^2))
    pidx1 <- knots[1,]$V1
    pidx2 <- knots[2,]$V1
    stat[label,]<-c(min(a1, a2), max(a1, a2), wide, diff, ratio, dist, boards[i], pidx1, pidx2, label, dim1_diff, dim2_diff)
  }
  
  stat<-as.data.frame(stat)
  names(stat)<-c("min_area", "max_area", "wide_surface", "diff", "ratio", "dist", "board", "pidx1", "pidx2", "label", "dim1_diff", "dim2_diff")
  return(stat)
}

stat<-data.frame()
for (i in 1:length(boards))
{
  stat<-rbind(stat, f(i))
}

ggplot(stat, aes(factor(wide_surface), ratio)) + geom_boxplot()
ggplot(stat, aes(factor(wide_surface), diff/max_area)) + geom_boxplot()

mean(subset(stat, wide_surface == 0)[,4])
mean(subset(stat, wide_surface == 1)[,4])

median(subset(stat, wide_surface == 0)[,4])
median(subset(stat, wide_surface == 1)[,4])

mean(subset(stat, wide_surface == 0)[,5])
mean(subset(stat, wide_surface == 1)[,5])

median(subset(stat, wide_surface == 0)[,5])
median(subset(stat, wide_surface == 1)[,5])

plot(stat$dist, stat$ratio)
cor(stat$dist, stat$ratio)

plot(stat$dist, stat$diff/stat$max_area)
cor(stat$dist, stat$diff/stat$max_area)

hist(stat$diff/stat$max_area, breaks=50)

ggplot(stat, aes(dim1_diff)) + geom_density()
ggplot(stat, aes(dim2_diff)) + geom_density()
hist(stat$dim1_diff, breaks=10)
hist(stat$dim2_diff, breaks=10)
cor(stat$dim1_diff, stat$dim2_diff)
plot(stat$dim1_diff, stat$dim2_diff)
ggplot(stat, aes(dim1_diff, dim2_diff)) + geom_density2d()

# fit covariance matrix
cov<-cov(stat[,c("dim1_diff", "dim2_diff")])
means<-colMeans(stat[,c("dim1_diff", "dim2_diff")])
corr<-colMeans(stat[,c("dim1_diff", "dim2_diff")])

# evaluate the likelihood of each of the observations
library(mvtnorm)
stat$board
stat<-subset(stat, board == 17)
dat<-stat[,c("dim1_diff", "dim2_diff")]
cov(stat[,c("dim1_diff", "dim2_diff")])
colMeans(stat[,c("dim1_diff", "dim2_diff")])
pp<-dmvnorm(dat, mean = means, sigma=cov)
sum(log(pp))

pp<-dmvnorm(dat, sigma=cov)
sum(log(pp))

dmvnorm(dat[1,], mean = means, sigma=cov)
dmvnorm(dat[3,], mean = means, sigma=cov)

