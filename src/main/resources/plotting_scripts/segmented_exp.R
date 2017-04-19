library(dplyr)
library(ggplot2)

d<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/segmented_simulated_data_exp.csv", header=T)
d$accuracy<-d$prediction/d$total
d$jaccard_accuracy<-d$jaccard/d$num_nodes
#p <- ggplot(d, aes(x=as.factor(board), y=accuracy, fill=I("black"))) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- ggplot(d, aes(x=idx, y=accuracy, fill=I("black"))) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + theme_bw()
p
ggsave("~/Google Drive/Research/papers/probabilistic-matching/paper/figures/sim_data_segmented_exp.pdf", p)
p <- ggplot(d, aes(x=as.factor(board), y=jaccard_accuracy, fill=I("black"))) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + theme_bw()
p
ggsave("~/Google Drive/Research/papers/probabilistic-matching/paper/figures/sim_data_jaccard.pdf", p)

### Below is the code used to generate the figure for the paper

d_simulated<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/segmented_sim_data_validation_exp.csv", header=T)
d_simulated$idx<-1:30
d_simulated$accuracy<-d_simulated$prediction/d_simulated$total
d_simulated$jaccard_accuracy<-d_simulated$jaccard/d_simulated$num_nodes
sum(d_simulated$prediction)/sum(d_simulated$total)
p <- ggplot(d_simulated, aes(x=as.factor(idx), y=accuracy, fill=I("black"))) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + theme_bw() + xlab("Board")
p
#ggsave("~/Google Drive/Research/papers/probabilistic-matching/paper/figures/sim_data_segmented_exp.pdf", p)


#d_real<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/segmented_real_data_exp.csv", header=T)
d_real<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/segmented_real_boards_em_training.csv", header=T)
d_real$idx<-1:dim(d_real)[1]
d_real$accuracy<-d_real$prediction/d_real$total
sum(d_real$prediction)/sum(d_real$total)
d_real$jaccard_accuracy<-d_real$jaccard/d_real$num_nodes
p <- ggplot(d_real, aes(x=as.factor(idx), y=accuracy, fill=I("black"))) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + theme_bw()
p
#ggsave("~/Google Drive/Research/papers/probabilistic-matching/paper/figures/real_data_segmented_exp.pdf", p)

# combine d_simulated and d_real
d_simulated$TrainingType<-"SIMULATED_DATA"
d_real$TrainingType<-"LOO_CV"

dim(d_real)
dim(d_simulated)
names(d_simulated)
names(d_real)

dd<-rbind(d_simulated[,-7], d_real)
dd$accuracy<-dd$prediction/dd$total
dd$jaccard_accuracy<-dd$jaccard/dd$num_nodes

p <- ggplot(dd, aes(x=as.factor(idx), y=accuracy, fill=TrainingType)) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + theme_bw() + xlab("Board") + ylab("Prediction Accuracy") + theme(legend.position="none")
p
ggsave("~/Google Drive/Research/papers/probabilistic-matching/paper/figures/real_data_prediction_accuracy.pdf", p)

p <- ggplot(dd, aes(x=as.factor(idx), y=jaccard_accuracy, fill=TrainingType)) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + theme_bw() + xlab("Board") + ylab("Jaccard Index")
p
ggsave("~/Google Drive/Research/papers/probabilistic-matching/paper/figures/real_data_jaccard.pdf", p)
