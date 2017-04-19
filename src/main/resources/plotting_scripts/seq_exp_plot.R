library(ggplot2)

d<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/simulation/unknown_sequence_param_simulation.csv", header=T)
#names(d)<-c("numData", "numMCSamples", "numPartitions", "maxNodesPerPartition", "numFeatures", "RMSE", "NLLK", "Exp_Type")
d10<-subset(d, numMCSamples == 10)
fs<-unique(d10$Exp_Type)
unknown<-summary(subset(d10, as.character(Exp_Type) == fs[1])$RMSE)
known<-summary(subset(d10, as.character(Exp_Type) == fs[2])$RMSE)

d1<-subset(d, numMCSamples == 1)
fs<-unique(d10$max_nllk)
unknown<-summary(subset(d1, as.character(Exp_Type) == fs[1])$RMSE)
known<-summary(subset(d1, as.character(Exp_Type) == fs[2])$RMSE)

p <- ggplot(d) + geom_boxplot(aes(x = Exp_Type, y=RMSE))
p <- p + theme_bw()
p

p <- ggplot(d10) + geom_boxplot(aes(x = Exp_Type, y=RMSE))
p <- p + theme_bw()
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/unknown_seq_exp_boxplot10.pdf")

p <- ggplot(d1) + geom_boxplot(aes(x = Exp_Type, y=RMSE))
p <- p + theme_bw()
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/unknown_seq_exp_boxplot1.pdf")
