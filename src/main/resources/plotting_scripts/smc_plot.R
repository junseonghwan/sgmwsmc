library(ggplot2)
d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/smc_output.csv", header=F)
head(d)
names(d)<-c("idx", "rep", "numParticles", "numCorrect", "total", "jaccard", "time")
ggplot(d, aes(numParticles, 1-(jaccard/(2*total)), col=as.factor(idx))) + geom_smooth(method="glm", method.args=list(family="binomial") , se=FALSE)
ggplot(d, aes(time, 1-(jaccard/(2*total)), col=as.factor(idx))) + geom_smooth(method="glm", method.args=list(family="binomial") , se=FALSE)

