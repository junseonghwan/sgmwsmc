library(ggplot2)
d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_10.csv", header=F)
sum(d$V5 < 0.5)
p<-ggplot(d, aes(V1, V5)) + geom_line() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_sample10.pdf")

d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_100.csv", header=F)
p<-ggplot(d, aes(V1, V5)) + geom_line() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_sample100.pdf")
sum(d$V5 < 0.5)

d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_10_32.csv", header=F)
dim(d)
p<-ggplot(d, aes(V1, V5)) + geom_line() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_10_32.pdf")
sum(d$V5 < 0.5)

d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_10_16.csv", header=F)
p<-ggplot(d, aes(V1, V5)) + geom_line() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_10_16.pdf")
sum(d$V5 < 0.5)

d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_10_8.csv", header=F)
p<-ggplot(d, aes(V1, V5)) + geom_line() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_10_8.pdf")
sum(d$V5 < 0.5)

d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_10_4.csv", header=F)
p<-ggplot(d, aes(V1, V5)) + geom_line() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_10_4.pdf")
sum(d$V5 < 0.5)

d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_10_2.csv", header=F)
p<-ggplot(d, aes(V1, V5)) + geom_line() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_10_2.pdf")
sum(d$V5 < 0.5)
