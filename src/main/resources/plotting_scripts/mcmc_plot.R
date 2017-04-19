library(ggplot2)
d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/mcmc_output.csv", header=F)
head(d)
names(d)<-c("idx", "rep", "iter", "T", "numCorrect", "total", "jaccard", "numSamples", "time")
ggplot(d, aes(iter, 1-(jaccard/numSamples), col=as.factor(idx))) + geom_smooth()
ggplot(d, aes(time, 1-(jaccard/numSamples), col=as.factor(idx))) + geom_smooth()

ggplot(d, aes(iter, 1-(jaccard/numSamples), col=as.factor(idx))) + geom_smooth(method="glm", method.args=list(family="binomial") , se=FALSE)
ggplot(d, aes(time, 1-(jaccard/numSamples), col=as.factor(idx))) + geom_smooth(method="glm", method.args=list(family="binomial") , se=FALSE)
