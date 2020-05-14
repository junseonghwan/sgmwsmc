library(ggplot2)
library(dplyr)

d<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/simulation/known_sequence_param_estimation_exp.csv", header=T)
d2<-subset(d, I == 10)
#dd<-group_by(d2, N) %>% summarise(rmse_bar=mean(rmse), rmse_se=sd(rmse))
p<-ggplot(d2, aes(N, rmse)) + geom_smooth()
p<-p+theme_bw()+xlab("Num Nodes per Partition")+ylab("RMSE")
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/known_sequence_rmse.pdf", width = 3.5, height = 2.5)
