library(ggplot2)

dir<-"~/Google Drive/Research/repo/sgmwsmc/output/knot_matching_timing_realdata/"
targetESSs <- c(100, 200, 400, 600, 800, 1000, 1200)
dd<-data.frame()
for (target in targetESSs)
{
  d<-read.csv(paste(dir, "Target", target, "/realDataPerformance.csv", sep=""), header=T)
  d$Idx<-1:dim(d)[1]
  d$TargetESS <- target
  dd<-rbind(dd, d)
}
names(d)
p <- ggplot(dd, aes(TargetESS, PredictionTimes, col=factor(Idx))) + geom_line() + theme_bw() + theme(legend.position = "none")
p <- p + xlab("Target ESS") + ylab("Prediction Time (seconds)")
p
ggsave(filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/ess_vs_times.pdf", p)

dd2<-subset(dd, TargetESS == 100)
# read in the simulated data with TargetESS=100
p<-ggplot(dd2, aes(NumNodes, PredictionTimes)) + geom_point() + theme_bw()
p<-p+xlab("No. Knots")+ylab("Prediction Time (seconds)")
ggsave(filename = "Google Drive/Research/papers/probabilistic-matching/paper/figures/num_knots_vs_times.pdf", p)

# generate a figure with num knots on the x-axis and time on the y-axis
sim1<-read.csv("~/Google Drive/Research/repo/sgmwsmc/output/simulated_knot_matching2/simulatedDataPerformance0.csv", header=T)
sim2<-read.csv("~/Google Drive/Research/repo/sgmwsmc/output/simulated_knot_matching2/simulatedDataPerformance1.csv", header=T)
sim<-rbind(sim1, sim2)
sim$Type<-"Simulated Boards"
real<-read.csv("~/Google Drive/Research/repo/sgmwsmc/output/knot_matching1/realDataPerformance.csv", header=T)
real$Type<-"Real Boards"
dat<-rbind(sim, real)
names(dat)
p <- ggplot(dat, aes(NumNodes, PredictionTimes, col=factor(Type))) + geom_point() + theme_bw()
p <- p + ylab("Time (Seconds)") + xlab("Number of Knot Faces")# + labs(color="Data Type")
p <- p + theme(legend.title = element_blank(), legend.text = element_text(size=12))
p <- p + theme(axis.title=element_text(size=18,face="bold"))
p
ggsave(filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/timing-results.pdf", p)
