library(ggplot2)
library(gridExtra)
library(cowplot)
library(ellipse)
library(rgl)
library(lbfgs)
library("quadprog")
library(rootSolve)
#rm(list=ls())

LEVEL<-0.975

board<-7
board<-"54003"
#enhanced_ellipses<-read.csv(paste("~/Desktop/scans/21Oct2015/Board ", board, "/knotdetection/matching/enhanced_matching_segmented.csv", sep=""), header=T)
enhanced_ellipses<-read.csv(paste("~/Desktop/scans/16Mar2016/", board, "/knotdetection/matching/enhanced_matching_segmented.csv", sep=""), header=T)
#enhanced_ellipses<-read.csv(paste("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/data/simmatch/enhanced_matching", board, ".csv", sep=""), header=T)
cbind(enhanced_ellipses$boundary_axis_idx1, enhanced_ellipses$boundary_axis_idx2, enhanced_ellipses$yaxis, enhanced_ellipses$zaxis)
names(enhanced_ellipses)
dim(enhanced_ellipses)

# plot this board to see where it can be improved
maxy<-600
maxz<-600
maxx<-5000
ref_pts<-c(1700, 2500, 3500, 4400)
dimensions<-matrix(0, ncol=8, nrow = 4)
num_matching<-length(unique(enhanced_ellipses$matching))
hash<-data.frame(key=unique(enhanced_ellipses$matching), val=order(unique(enhanced_ellipses$matching)))
cols<-palette(rainbow(num_matching))
plots<-list()
for (s in 0:3)
{
  temp <- NULL
  if (s==0 || s==2) { ylimmax <- maxy} else { ylimmax <- maxz}
  p<-ggplot() + geom_point() + xlim(0, maxx) + ylim(0, ylimmax)
  p <- p + theme(legend.position="none", axis.title.y=element_blank(), axis.title.x=element_blank())
  p <- p + geom_vline(xintercept = ref_pts[1]) + geom_vline(xintercept = ref_pts[2]) + geom_vline(xintercept = ref_pts[3]) + geom_vline(xintercept = ref_pts[4])
  
  #matching_s<-subset(matching, surface == s & matching > 0)
  matching_s<-subset(enhanced_ellipses, surface == s)
  for (i in 1:dim(matching_s)[1])
  {
    #if (s==0 && i==1){show(e)}
    if (s == 0 || s == 2) {
      mu<-c(matching_s[i,"x"], matching_s[i,"y"])
      ypos <- matching_s[i,"y"]
      if (matching_s[i,"y"] < 0) { ypos <- 0 }
      if (matching_s[i,"y"] > ylimmax) { ypos <- ylimmax}
    } else {
      mu<-c(matching_s[i,"x"], matching_s[i,"z"])
      ypos <- matching_s[i,"z"]
      if (matching_s[i,"z"] < 0) { ypos <- 0 }
      if (matching_s[i,"z"] > ylimmax) { ypos <- ylimmax}
      
    }
    S<-matrix(c(matching_s[i,"var_x"], matching_s[i,"cov"], matching_s[i,"cov"], matching_s[i,"var_y"]), ncol=2, byrow = T)
    e<-data.frame(ellipse(S, level = LEVEL, centre = t(mu)))    
    e$matching<-matching_s[i,"matching"]
    p<-p + geom_path(data = e, aes(x, y), col=cols[hash[hash$key == unique(e$matching),"val"]]) + annotate("text", x=matching_s[i,"x"], y=ypos, label = paste(matching_s[i,"matching"],",",matching_s[i,"idx"],sep=""), col='black', size=3)
    #p<-p + geom_path(data = e, aes(x, y))
    #p<-p + geom_path(data = e, aes(x, y), col=cols[hash[hash$key == unique(e$matching),"val"]]) + annotate("text", x=matching_s[i,"x"], y=ypos, label = paste(matching_s[i,"surface"],",",matching_s[i,"idx"],sep=""), col='black', size=3)
  }
  plots[[s+1]]<-p
}
pp<-plot_grid(plots[[1]], plots[[2]], plots[[3]], plots[[4]], nrow = 4)
pp
#save_plot(filename = "~/Desktop/plot_board7.pdf", pp, base_width = 25, base_height=11)
names(enhanced_ellipses)
cbind(enhanced_ellipses[,c("x", "boundary_axis_idx1", "boundary_axis_idx2", "surface", "idx", "matching", "segments", "var_x", "var_y", "cov")])
