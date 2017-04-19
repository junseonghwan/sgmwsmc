library(ggplot2)
library(dplyr)

d_real<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/segmented_real_boards_em_training.csv", header=T)
d_real$idx<-1:dim(d_real)[1]
d_real$accuracy<-d_real$prediction/d_real$total
sum(d_real$prediction)/sum(d_real$total)
d_real$jaccard_accuracy<-d_real$jaccard/d_real$num_nodes
names(d_real)
dd<-rbind(data.frame(Board=1:30, "value"=as.numeric(d_real[,c("accuracy")]), Type="Prediction"), cbind(Board=1:30, "value"=as.numeric(d_real[,"jaccard_accuracy"]), Type="Jaccard Index"))
dd$value<-as.numeric(dd$value)
dd$Board<-as.numeric(dd$Board)
p <- ggplot(dd, aes(x=Board, y=value, fill=Type)) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + xlab("Board") + ylab("Accuracy")
p <- p + theme_bw() + theme(legend.text=element_text(size=10), axis.title = element_text(size=15))
p
ggsave(filename = "Google Drive/Research/talks-presentations/aistats2017-poster/figures/knot-matching.pdf", p)