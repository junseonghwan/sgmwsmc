library(dplyr)
library(ggplot2)
library(cowplot)
library(stringr)

param_real<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/real_param_estimate.csv", header=F)
param_simul<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/knot-matching/simulated_param_estimate.csv", header=F)
names(param_real)<-c("idx", "Covariate", "Value")
names(param_simul)<-c("Covariate", "Value")
param_real$Covariate<-trimws(as.character(param_real$Covariate))
param_simul$Covariate<-trimws(as.character(param_simul$Covariate))
param_real$idx<-param_real$idx + 1

ff<-trimws(as.character(unique(param_real$Covariate)))
for (i in 1:length(ff))
{
  temp1<-subset(param_real, Covariate == ff[i])
  temp2<-subset(param_simul, Covariate == ff[i])
  temp1$Type<-"LOO"
  temp2$Type<-"SIMULATED"
  temp<-rbind(temp1[,-1], temp2)
  #p<-ggplot(temp, aes(Covariate, Value)) + geom_boxplot() + xlab("") + theme_bw()
  #p<-p+geom_point(temp2, aes(Covariate, Value), col='red')
  p<-ggplot(temp, aes(Value)) + geom_line(stat="density") + xlab("") + theme_bw()
  p<-p+geom_point(aes(x=temp$Value, y=0, col=Type), size=4)
  p<-p+ggtitle(ff[i]) + ylab("")
  if (i %% 2 == 1)
    p <- p + theme(legend.position="none")
  ggsave(filename = paste("~/Google Drive/Research/papers/probabilistic-matching/paper/figures/", ff[i], ".pdf", sep=""), plot = p)
}

