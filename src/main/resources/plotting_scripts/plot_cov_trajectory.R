library(ggplot2)

params<-read.csv("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/realParamTrejectory.csv", header=F)
names(params)<-c("ITER", "COVARIATE", "VALUES")
param_names<-unique(params$V2)
iter<-unique(params$V1)

params<-subset(params, ITER < 20)

p<-ggplot(params, aes(ITER, VALUES)) + geom_line() + facet_grid(.~factor(PARAM_NAMES))
p<-p+theme_bw()
p

p<-ggplot(params, aes(ITER, VALUES, col=COVARIATE)) + geom_line()
p<-p+theme_bw()
p<-p+xlab("Training Iterations")
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/param_trajectory.pdf")
