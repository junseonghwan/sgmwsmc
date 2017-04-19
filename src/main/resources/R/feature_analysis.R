# compute the features and the likelihood for a given board (or segmented part)
library(ggplot2)
library(gridExtra)
library(cowplot)
library(ellipse)
library(rgl)
library(lbfgs)
library("quadprog")
library(rootSolve)
library(dplyr)
#rm(list=ls())

p<-c(-5.255771678342197, -7.210103975308426, -3.704217657986097, -6.3501096081657655, -0.008068219778903703, 0.282944954011892)

LEVEL<-0.975
SQRT_CRITICAL_VALUE<-sqrt(qchisq(0.975, 2))
c<-300
AREA_NORM_CONST<-1000

board<-7
board<-"54003"
#enhanced_ellipses<-read.csv(paste("~/Desktop/scans/21Oct2015/Board ", board, "/knotdetection/matching/enhanced_matching_segmented.csv", sep=""), header=T)
enhanced_ellipses<-read.csv(paste("~/Desktop/scans/16Mar2016/", board, "/knotdetection/matching/enhanced_matching_segmented.csv", sep=""), header=T)

# now get a segment
temp<-group_by(enhanced_ellipses, segments) %>% summarise(count=n())
segs<-which(temp$count > 3)

for (seg in segs)
{
  knots<-subset(enhanced_ellipses, segments == seg)
  
  # compute the features for the true matching
  mm<-unique(knots$matching)
  truth<-rep(0, length(mm))
  for (i in 1:length(mm))
  {
    m<-mm[i]
    ff<-compute_features(subset(knots, matching == m))
    truth[i]<-sum(ff*p)
  }
  
  # compute the features for other combinations
  decisions<-matrix(-Inf, dim(knots)[1], dim(knots)[1])
  for (i in 1:dim(knots)[1])
  {
    for (j in 1:dim(knots)[1])
    {
      if (knots[i,"surface"] != knots[j,"surface"]) {
        ff<-compute_features(rbind(knots[i,], knots[j,]))
        decisions[i,j]<-sum(ff*p)
      }
    }
  }
  round(exp(decisions)/rowSums(exp(decisions)), 3)
}


compute_features<-function(knots)
{
  dist_f<-distance_features(knots)
  area_f<-area_features(knots)
  return(c(dist_f, area_f))
}

area_features<-function(knots)
{
  f<-rep(0, 2)
  if (dim(knots)[1] == 2) {
    if (knots[1,"surface"] %% 2 == 0 & knots[2,"surface"] %% 2 == 0) {
      a1<-compute_area(knots[1,])
      a2<-compute_area(knots[2,])
      f[1]<-abs(a2-a1)/AREA_NORM_CONST
    }
  } else {
    a<-rep(0, 3)
    a[1]<-compute_area(knots[1,])
    a[2]<-compute_area(knots[2,])
    a[3]<-compute_area(knots[3,])
    sort(a)
    a1<-a[1] + a[2]
    a2<-a[3]
    f[2]<-abs(a2-a1)/AREA_NORM_CONST
  }
  return(f)
}

distance_features<-function(knots)
{
  f<-rep(0, 4)
  if (dim(knots)[1] == 2) {
    k1<-knots[1,]
    k2<-knots[2,]
    if (k1$surface %% 2 == 0 & k2$surface %% 2 == 0) {
      f[1]<-compute_distance(k1[c("x", "y", "z")], k2[c("x", "y", "z")])/DIST_NORM_CONST
    } else {
      f[2]<-compute_distance(k1[c("x", "y", "z")], k2[c("x", "y", "z")])/DIST_NORM_CONST
    }
  } else {
    k1<-knots[1,]
    k2<-knots[2,]
    k3<-knots[3,]
    mm<-rbind(k1[1:3], k2[1:3], k3[1:3])
    dd<-dist(mm)
    #which.min(dd)
    #which.max(dd)
    f[3]<-min(dd)/DIST_NORM_CONST
    f[4]<-max(dd)/DIST_NORM_CONST
  }
  return(f)
}

compute_distance<-function(x, y)
{
  return(sqrt(sum((x - y)^2)))
}

compute_area<-function(knot)
{
  S<-matrix(c(knot$var_x, knot$cov, knot$cov, knot$var_y), 2, 2)
  lambdas<-eigen(S)$values
  return(pi * sqrt(lambdas[1]) * SQRT_CRITICAL_VALUE * sqrt(lambdas[2]) * SQRT_CRITICAL_VALUE)
}
