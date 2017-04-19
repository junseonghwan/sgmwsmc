setwd("Dropbox/Research/papers/probabilistic-matching/sgmwsmc/data/16Oct2015")
rm(list=ls())
library(ggplot2)

# plot the knots
f<-function(kn)
{
  names(kn)<-c("pidx", "idx", "x", "x0", "y", "z", "w", "h", "label")
  knots<-subset(kn, label > 0)

  # split the knots by the label, compute the Manhattan distance between the width and the height
  md<-rep(0, length(unique(knots$label)))
  diff<-matrix(0, ncol = 2, nrow = length(unique(knots$label)))
  for (l in unique(knots$label))
  {
    s <- subset(knots, label == l)
    a1 <- s[1,'w'] * s[1,'h']
    a2 <- s[2,'w'] * s[2,'h']
    md[l] <- (abs(s[1,'w']/a1 - s[2,'w']/a2) + abs(s[1,'h']/a1 - s[2,'h']/a2))
    #md[l] <- (abs(s[1,'w'] - s[2,'w']) + abs(s[1,'h'] - s[2,'h']))
    diff[l,1] <- s[1,'w'] - s[2,'w']
    diff[l,2] <- s[1,'h'] - s[2,'h']
  }
  a <- list()
  a$normalize_size_diff<-md
  a$dimension_diff<-diff
  return(a)
}

kn4 <- subset(read.table("Board 4/tracheids/labelledMatching.csv", sep=","), V9 > 0)
kn8 <- subset(read.table("Board 8/tracheids/labelledMatching.csv", sep=","), V9 > 0)
kn17 <- subset(read.table("Board 17/tracheids/labelledMatching.csv", sep=","), V9 > 0)
kn18 <- subset(read.table("Board 18/tracheids/labelledMatching.csv", sep=","), V9 > 0)
kn20 <- subset(read.table("Board 20/tracheids/labelledMatching.csv", sep=","), V9 > 0)
kn24 <- subset(read.table("Board 24/tracheids/labelledMatching.csv", sep=","), V9 > 0)
md<-f(kn4)
md<-c(md, f(kn8))
md<-c(md, f(kn17))
md<-c(md, f(kn18))
md<-c(md, f(kn20))
md<-c(md, f(kn24))

param = c(-0.6492831196829282,
          -0.7406358298282127,
          -0.11889469700258537,
          -0.21355491014508277)

