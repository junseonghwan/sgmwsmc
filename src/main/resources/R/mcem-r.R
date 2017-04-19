setwd("Dropbox/Research/papers/probabilistic-matching/sgmwsmc/")
library(RColorBrewer)
library(rgl)
library(plot3D)

d<-read.csv("mcem_surface_3.csv", header=F)
rbPal<-colorRampPalette(c('red', 'yellow', 'blue'))
dd<-d[order(d[,3]),]
col<-rbPal(length(dd[,3]))
# NOTE: the plot shows that the objective is convex, and hence, we are findnig a global minimum.
#plot3d(dd[,1], dd[,2], -dd[,3], col = col, xlab='', ylab='', zlab='loglik', zlim = c(min(-dd[,3]), 0)) 
plot3d(dd[,1], dd[,2], dd[,3], col = col, xlab='', ylab='', zlab='loglik') 

d[1,]
min(d[,3])
