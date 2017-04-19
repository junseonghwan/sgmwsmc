setwd("Dropbox/Research/papers/probabilistic-matching/sgmwsmc/output/image/")
library(ggplot2)
library(plyr)

f<-function(nrep, dir_name, file_name, k, splits, large, method)
{
  ret<-data.frame()
  for (i in 1:length(k))
  {
    kk<-k[i]
    avgs<-rep(0, nrep+1)
    for (j in 0:nrep) 
    {
      if (j == nrep) {
        if (!large)
          file<-paste(dir_name, file_name, "_", kk, ".csv", sep="")
        else
          file<-paste(dir_name, file_name, "_", kk, "_large_training", ".csv", sep="")
      } else {
        if (!large)
          file<-paste(dir_name, file_name, "_", kk, "_", j, ".csv", sep="")
        else
          file<-paste(dir_name, file_name, "_", kk, "_large_training_", j, ".csv", sep="")
      }
      d<-read.csv(file, header=F)
      avgs[j+1]<-mean(d[,3])/30
    }
    df<-data.frame("avg"=mean(avgs), "sd"=sd(avgs), "split"=splits[i], "method"=method)
    ret<-rbind(ret, df)
  }
  return(ret)
}

nrep = 5;
file_name<-"edge_feature_performance"
dir_name<-"edge-feature-performance/"
k<-c(2, 3, 5, 10, 15, 25)
splits<-rev(c("56/55", "37/74", "23/88", "12/99", "8/103", "5/106"))
method<-"SGM w/ Edge Features"
ret_edge<-f(nrep, dir_name, file_name, k, splits, FALSE, method)

splits<-c("106/5", "103/8", "99/12", "88/23", "74/37", "55/56")
ret_edge_large<-f(nrep, dir_name, file_name, k, splits, TRUE, method)

ret<-rbind(ret_edge, ret_edge_large)

### read in the linear feature experiments results

file_name<-"image_matching_performance"
dir_name<-"linear-feature-performance/"
splits<-rev(c("56/55", "37/74", "23/88", "12/99", "8/103", "5/106"))
method<-"SGM w/ Node Features"
ret_linear<-f(nrep, dir_name, file_name, k, splits, FALSE, method)

splits<-c("106/5", "103/8", "99/12", "88/23", "74/37", "55/56")
ret_linear_large<-f(nrep, dir_name, file_name, k, splits, TRUE, method)

ret<-rbind(ret, ret_linear, ret_linear_large)

# hard code the nubmers from Caetano et. al. (2009)
caetano<-as.data.frame(matrix(c(
  17.7, 17.1,
  15.6, 9.1,
  13.1, 10.2,
  16.6, 14.3,
  15.0, 9.9,
  12.7, 7.3,
  13.1, 9.0,
  12.1, 10.1,
  11.9, 8.4,
  14.2, 10.8,
  13.5, 8.1,
  10.7, 5.7), ncol=2, byrow = TRUE))
splits<-c("5/106", "8/103", "12/99", "23/88", "37/74", "56/55", "55/56", "74/37", "88/23", "99/12", "103/8", "106/5")
c1<-data.frame("avg"=1-as.numeric(caetano[,1])/100, "sd"=0, "split"=splits, "method"="LA Learning")
c2<-data.frame("avg"=1-as.numeric(caetano[,2])/100, "sd"=0, "split"=splits, "method"="GA Learning")
cc<-rbind(c1, c2)

dd<-rbind(ret, cc)
dd$method <- factor(dd$method, levels = c("LA Learning", "GA Learning", "SGM w/ Node Features", "SGM w/ Edge Features"))
limits <- aes(ymax = (1-avg) + sd, ymin = (1-avg) - sd)
p <- ggplot(dd, aes(x=split, y=1-avg, fill=factor(method))) + geom_bar(stat = "identity", position=position_dodge(width=.9)) + geom_errorbar(limits, width=.2, position=position_dodge(width=.9))
p <- p + scale_y_continuous(limits=c(0, 0.2)) + labs(x = "Training/Testing Split", y = "Average Error Rate", fill = "Methods")
p <- p + theme_bw()
p <- p + theme(axis.title.x	= element_text(size = rel(2)), axis.title.y	= element_text(size = rel(2)))
p <- p + theme(axis.text.x = element_text(size = rel(0.8)), axis.text.y = element_text(size = rel(2)))
p <- p + theme(legend.title = element_text(size = rel(2)), legend.text = element_text(size = rel(1)))
p
ggsave(filename = "../../../AISTATS/figures/image-matching-results.pdf", plot = p)

