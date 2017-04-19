library(ggplot2)
library(dplyr)
rm(list=ls())

mcmc<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/mcmc_output.csv", header=F)
mcmc$idx<-mcmc$idx + 1
names(mcmc)<-c("idx", "rep", "iter", "T", "numCorrect", "total", "jaccard", "numSamples", "time")
smc<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/smc_output.csv", header=F)
names(smc)<-c("idx", "rep", "numParticles", "numCorrect", "total", "jaccard", "time")

p<-ggplot(mcmc, aes(iter, 1 - numCorrect/total)) + geom_smooth()
p<-p+theme_bw()
p<-p+ylab("Average Error")
p
ggsave(filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/mcmc_error_vs_chain_length.pdf")

p<-ggplot(smc, aes(numParticles, 1- numCorrect/total)) + geom_smooth(se = F)
p<-p+theme_bw()
p<-p+ylab("Average Error")
p
ggsave(filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/smc_error_vs_num_particles.pdf")

# combine the two data frames so that each board can be compared in terms of runtime vs accuracy
mcmc$Type<-"MCMC"
smc$Type<-"SMC"
dd<-rbind(mcmc[,c("idx", "jaccard", "numCorrect", "total", "time", "Type", "rep")], smc[,c("idx", "jaccard", "numCorrect", "total", "time", "Type", "rep")])

p<-ggplot(dd, aes(time, 1-numCorrect/total, col=Type)) + geom_smooth(se=FALSE)
p<-p+xlim(0, 3)
p<-p+xlab("Time (sec)")+ylab("Error")
p
ggsave(filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/mcmc_smc_time_all_boards.pdf", plot = p)


## compute the average time for 750 particles vs 10000 chain lengths
temp_mcmc<-subset(mcmc, iter == 10000)
mean(temp_mcmc$time)

temp_smc<-subset(smc, numParticles == 640)
mean(temp_smc$time)

# SMC execution time is slower -- but this can be improved by limiting the number of decisions and using random proposal
# not necessary to compare the time of execution -- just compare the accuracies.

mcmc2<-subset(mcmc, iter == 10000)
smc2<-subset(smc, numParticles == 640)

mcmc_accuracy<-mcmc2 %>% group_by(idx) %>% summarise(accuracy=mean(numCorrect/total))
smc_accuracy<-smc2 %>% group_by(idx) %>% summarise(accuracy=mean(numCorrect/total))
mcmc_accuracy$Type<-"MCMC"
smc_accuracy$Type<-"SMC"
dd_accuracy<-rbind(mcmc_accuracy, smc_accuracy)
p <- ggplot(dd_accuracy, aes(x=factor(idx), y=accuracy, fill=Type)) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + theme_bw()
p <- p + xlab("Board") + ylab("Accuracy")
p <- p + theme(legend.position="none")
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/MCMC-SMC-accuracy.pdf")

mcmc_jaccard<-mcmc2 %>% group_by(idx) %>% summarise(jaccard=mean(jaccard/numSamples))
smc_jaccard<-smc2 %>% group_by(idx) %>% summarise(jaccard=mean(jaccard/(2*total)))
mcmc_jaccard$Type<-"MCMC"
smc_jaccard$Type<-"SMC"
dd_jaccard<-rbind(mcmc_jaccard, smc_jaccard)
p <- ggplot(dd_jaccard, aes(x=factor(idx), y=jaccard, fill=Type)) + geom_bar(stat = "identity", position=position_dodge(width=.9))
p <- p + theme_bw()
p <- p + xlab("Board") + ylab("Jaccard")
p
ggsave(plot = p, filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/MCMC-SMC-jaccard.pdf")

