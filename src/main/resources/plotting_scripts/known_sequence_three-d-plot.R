rm(list=ls())
library(RColorBrewer)
library(rgl)
library(plot3D)

filename = 'known_sequence_param_estimation_surface_2'
datafile = paste('~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/simulation/', filename, ".csv", sep='')
figurename = paste('~/Google Drive/Research/papers/probabilistic-matching/paper/figures/', filename, '.pdf', sep='')
d<-read.table(datafile, sep=",", header = T)
truth<-as.numeric(d$nllk_truth[1])
estimated<-as.numeric(d$nllk_map[1])
truth
estimated
txt = paste("loglik@MAP: ", -round(estimated,3), '\n loglik@truth: ', -round(truth, 3), sep='')

rbPal<-colorRampPalette(c('red', 'yellow', 'blue'))
dd<-d[order(d$nllk),]
col<-rbPal(length(dd[,3]))
# NOTE: the plot shows that the objective is convex, and hence, we are findnig a global minimum.
plot3d(dd$f0, dd$f1, -dd$nllk, col = col, xlab='', ylab='', zlab='loglik', zlim = c(min(-dd$nllk), 0)) 
text3d(0, 0, 100, texts = txt, col='black', size=10)
#points3d(estimated[1], estimated[2], estimated[3], col='black', size=10)

pdf(file = figurename)
scatter3D(dd$f0, dd$f1, -dd$nllk, zlab='loglik', main=txt)
#scatter3D(estimated[1], estimated[2], -estimated[3], add = TRUE, colkey = FALSE, col='black', pch=19, cex=1.5)
dev.off()


# restric the domain of the objective -- if needed
#sub1<-subset(d, V1 > -1 & V2 > -1)
#dd<-sub1[order(sub1[,3]),]
#col<-rbPal(length(sub1[,3]))
#plot3d(dd[,1], dd[,2], dd[,3], col = col) 
