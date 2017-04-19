setwd("Dropbox/Research/papers/probabilistic-matching/sgmwsmc/output/image/")
library(ggplot2)

files<-c("edge_feature_performance_2_large_training.csv", "edge_feature_performance_3_large_training.csv",
         "edge_feature_performance_5_large_training.csv", "edge_feature_performance_10_large_training.csv",
         "edge_feature_performance_15_large_training.csv", "edge_feature_performance_25_large_training.csv",
         "edge_feature_performance_2.csv", "edge_feature_performance_3.csv",
         "edge_feature_performance_5.csv", "edge_feature_performance_10.csv",
         "edge_feature_performance_15.csv", "edge_feature_performance_25.csv")
k<-c("106/5", "103/8", "99/12", "88/23", "74/37", "55/56", "56/55", "37/74", "23/88", "12/99", "8/103", "5/106")
edge_ret<-as.data.frame(matrix(0, nrow = length(files), ncol = 3))
for (i in 1:length(files))
{
  file<-paste("edge-feature-performance/", files[i], sep="")
  d<-read.csv(file, header=F)
  edge_ret[i,1]<-mean(d[,3])/30
  edge_ret[i,2]<-"edge_feature"
  edge_ret[i,3]<-k[i]
}

node_files<-c("image_matching_performance_2_large_training.csv", "image_matching_performance_3_large_training.csv",
         "image_matching_performance_5_large_training.csv", "image_matching_performance_10_large_training.csv",
         "image_matching_performance_15_large_training.csv", "image_matching_performance_25_large_training.csv",
         "image_matching_performance_2.csv", "image_matching_performance_3.csv",
         "image_matching_performance_5.csv", "image_matching_performance_10.csv",
         "image_matching_performance_15.csv", "image_matching_performance_25.csv")
node_ret<-as.data.frame(matrix(0, nrow = length(files), ncol = 3))
for (i in 1:length(node_files))
{
  file<-paste("linear-feature-performance/", node_files[i], sep="")
  d<-read.csv(file, header=F)
  node_ret[i,1]<-mean(d[,3])/30
  node_ret[i,2]<-"node_feature"
  node_ret[i,3]<-k[i]
}

edge_ret
node_ret

d<-rbind(edge_ret, node_ret)
p <- ggplot(d, aes(x=factor(V3), y=1-V1, fill=factor(V2))) + geom_bar(stat = "identity", position="dodge") + scale_y_continuous(limits=c(0, 0.3))
p
ggsave(filename = "../../../AISTATS/figures/image-matching-results.pdf", plot = p)
