setwd("Dropbox/Research/papers/probabilistic-matching/sgmwsmc/output/")
rm(list=ls())
library(RColorBrewer)
library(rgl)
library(plot3D)

filename = 'log-density-10-4-10-2-20160912-1.0'
datafile = paste(filename, ".txt", sep='')
figurename = paste('figures/', filename, '.pdf', sep='')
d<-read.table(datafile, sep=",")
head(d)
d[which.min(d[,3]),]
truth<-as.numeric(d[dim(d)[1]-1,])
estimated<-as.numeric(d[dim(d)[1],])
truth
estimated
txt = paste("loglik@MLE: ", -round(estimated[3],3), '\nloglik@truth: ', -round(truth[3], 3), sep='')

rbPal<-colorRampPalette(c('red', 'yellow', 'blue'))
dd<-d[order(d[,3]),]
col<-rbPal(length(dd[,3]))
# NOTE: the plot shows that the objective is convex, and hence, we are findnig a global minimum.
plot3d(dd[,1], dd[,2], -dd[,3], col = col, xlab='', ylab='', zlab='loglik', zlim = c(min(-dd[,3]), 0)) 
text3d(estimated[1], estimated[2], 100, texts = txt, col='black', size=10)
#points3d(estimated[1], estimated[2], estimated[3], col='black', size=10)

pdf(file = figurename)
scatter3D(dd[,1], dd[,2], -dd[,3], zlab='loglik', main=txt)
#scatter3D(estimated[1], estimated[2], -estimated[3], add = TRUE, colkey = FALSE, col='black', pch=19, cex=1.5)
dev.off()


# restric the domain of the objective -- if needed
sub1<-subset(d, V1 > -1 & V2 > -1)
dd<-sub1[order(sub1[,3]),]
col<-rbPal(length(sub1[,3]))
plot3d(dd[,1], dd[,2], dd[,3], col = col) 
