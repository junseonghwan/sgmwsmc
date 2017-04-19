library(ggplot2)
library(dplyr)

dd<-data.frame()
reps<-5

for (rep in 0:(reps-1))
{
  d<-read.csv(paste("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_10_5_", rep, ".csv", sep=""), header=F)
  d$d<-10
  d$rep<-(rep+1)
  dd<-rbind(dd, d)
}
p<-ggplot(d, aes(V1, V5)) + geom_line() + geom_point() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_sample10.pdf")

for (rep in 0:(reps-1))
{
  d<-read.csv(paste("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_100_5_", rep, ".csv", sep=""), header=F)
  d$d<-100
  d$rep<-(rep+1)
  dd<-rbind(dd, d)
}
p<-ggplot(d, aes(V1, V5)) + geom_line() + geom_point() + xlab("Iteration") + ylab("Relative ESS") + geom_hline(yintercept = 0.5, col="red")
p<-p + theme_bw()
p
ggsave(plot = p, filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_sample100.pdf")


ff<-c(10, 20, 40, 60, 80, 100)
dd<-data.frame()
reps<-5
for (f in ff)
{
  for (rep in 0:(reps-1))
  {
    d<-read.csv(paste("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/smc/ess_", f, "_5_", rep, ".csv", sep=""), header=F)
    d$d<-f
    d$rep<-(rep+1)
    dd<-rbind(dd, d)
  }
}

# plot f vs num_resamplings
ret<-group_by(dd, d, rep) %>% summarise(count=sum(V5 < 0.5))
p<-ggplot(ret, aes(log(d), count)) + geom_smooth() + theme_bw()
p # this plot shows that the number of resampling operations is not affected by the number of features
