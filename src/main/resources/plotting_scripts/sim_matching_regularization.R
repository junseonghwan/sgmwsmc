library(ggplot2)
library(dplyr)

d<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/simmatching/simmatching_regularization_results.csv", header=T)
dim(d)
names(d)
jaccard<-group_by(d, lambda) %>% summarize(avg=mean(jaccard), se=sd(jaccard))
p<-ggplot(jaccard, aes(lambda, avg)) + geom_smooth() + geom_point()
p<-p + theme_bw()
#limits<-aes(ymin=avg - se, ymax= avg + se)
#p<-ggplot(jaccard, aes(lambda, avg)) + geom_line() + theme_bw()
#p <- p + geom_errorbar(limits, width=.2, position=position_dodge(width=.9))
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/sim_matching_l2_cv_mean_jaccard.pdf")

p<-ggplot(d, aes(lambda, jaccard)) + geom_smooth() + geom_point()
p<-p + theme_bw()
#limits<-aes(ymin=avg - se, ymax= avg + se)
#p<-ggplot(jaccard, aes(lambda, avg)) + geom_line() + theme_bw()
#p <- p + geom_errorbar(limits, width=.2, position=position_dodge(width=.9))
p
#ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/sim_matching_l2_cv_mean_jaccard.pdf")
