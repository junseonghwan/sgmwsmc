setwd("Dropbox/Research/papers/probabilistic-matching/sgmwsmc/output/mcem/_1476174320822")
d<-read.csv("results/results.csv")
boards<-unique(d$board)
ret<-matrix(0, ncol=2, nrow=length(boards))
for (i in 1:length(boards))
{
  b<-boards[i]
  dd<-subset(d, board == b)
  ret[i,1]<-mean(dd$consensus)/dd$total[1]
  ret[i,2]<-mean(dd$MAP)/dd$total[1]
}
cbind(boards, ret)
