library(ggplot2)
library(dplyr)

d<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/regularization1/segmented_l2_regularization1.csv", header=T)
#d1<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/regularization20/segmented_l2_regularization20.csv", header=T)
#d<-rbind(d, d1)

jaccard<-group_by(d, lambda) %>% summarize(avg=mean(jaccard), se=sd(jaccard))
p<-ggplot(jaccard, aes(lambda, avg)) + geom_smooth() + theme_bw()
p<-p + geom_point()
#limits<-aes(ymin=avg - se, ymax= avg + se)
#p <- p + geom_errorbar(limits, width=.2, position=position_dodge(width=.9))
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/l2_cv_mean_jaccard.pdf")

p<-ggplot(d, aes(x=lambda, y=jaccard)) + geom_smooth(method = "loess") + geom_point()
p<-p+theme_bw()
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/l2_cv_jaccard.pdf")

p<-ggplot(d, aes(x=lambda, y=zeroOne)) + geom_smooth(method = "loess")
p<-p+theme_bw()
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/l2_cv_zero_one.pdf")
