library(ggplot2)
library(reshape2)

# plot mc-em convergence param plot, likelihood plot, performance, and timing plots
param_names<-c("TWO_MATCHING_DISTANCE_1",
               "THREE_MATCHING_DISTANCE_2",
               "THREE_MATCHING_DISTANCE_1",
               "TWO_MATCHING_DISTANCE_2",
               "TWO_MATCHING_AREA_DIFF",
               "THREE_MATCHING_AREA_DIFF")

for (j in 1:3)
{
  dir<-paste("Google Drive/Research/repo/sgmwsmc/output/knot_matching", j, sep="")
  option_list<-read.csv(paste(dir, "/executionInfo/options.map", sep=""), header=F, sep="\t", row.names = 1)
  lambda<-as.numeric(option_list["lambda",])
  
  x<-read.csv(paste(dir, "/realDataPerformance.csv", sep=""), header=T)
  x$idx<-1:30
  
  if (dir.exists(paste(dir, "/plots/", sep="")) == FALSE)
    dir.create(paste(dir, "/plots/", sep=""))
  for (i in 0:29)
  {
    params<-read.csv(paste(dir, "/rep", i, "/params.csv", sep=""), header=F)
    params<-params[-1,]
    names(params)<-param_names
    params$Iter <- 1:dim(params)[1]
    melted_params<-melt(params, id.vars = c("Iter"))
    names(melted_params)<-c("Iter", "Covariates", "Values")

    p<-ggplot(melted_params, aes(Iter, Values, col=Covariates)) + geom_line()# + geom_point()
    p<-p+theme_bw()+xlab("Iterations")+ylab("Values")
    p <- p + theme(axis.text=element_text(size=12), axis.title=element_text(size=14,face="bold"))
    ggsave(filename = paste(dir,"/plots/param_trajectory", i, ".pdf", sep=""), p)

    means<-read.csv(paste(dir, "/rep", i, "/sumOfMeans.csv", sep=""), header=F)
    vars<-read.csv(paste(dir, "/rep", i, "/sumOfVars.csv", sep=""), header=F)
    df<-data.frame(iter=1:dim(means)[1], logLik=means[,1], sd=sqrt(vars[,1]))
    p<-ggplot(df, aes(x=iter, y=logLik)) + geom_line() + theme_bw()
    p<-p + geom_errorbar(aes(ymin = logLik - 2*sd, ymax = logLik + 2*sd), width=0.5)
    p<-p + ylab("Log Likelihood") + xlab("Iterations")
    p<-p + theme(axis.text=element_text(size=12), axis.title=element_text(size=14,face="bold"))
    ggsave(filename = paste(dir, "/plots/mcem_lik", i, ".pdf", sep=""), p)
  }

  p <- ggplot(x, aes(NumNodes, PredictionTimes)) + geom_point() + theme_bw()
  p <- p + xlab("Number of Knots") + ylab("Time (seconds)")
  ggsave(filename = paste(dir, "/plots/timing.pdf", sep=""), p)
  
  ### board-by-board prediction accuracy and the Jaccard index plot
  p<-ggplot(x, aes(as.factor(idx), PredictionAccuracy/NumMatchings)) + geom_bar(stat = "identity", position=position_dodge(width=.9))
  p <- p + theme_bw() + xlab("Boards") + ylab("Prediction Accuracy")
  ggsave(paste(dir, "/plots/prediction_accuracy.pdf", sep=""), p)
  
  p<-ggplot(x, aes(as.factor(idx), AvgJaccardIndex/NumNodes)) + geom_bar(stat = "identity", position=position_dodge(width=.9))
  p <- p + theme_bw() + xlab("Boards") + ylab("Jaccard Index")
  ggsave(paste(dir, "/plots/jaccard_accuracy.pdf", sep=""), p)

  # create a bar plot with prediction accuracy and Jaccard together
  x$SingleSample <- x$PredictionAccuracy/x$NumMatchings
  x$JaccardIndex <- x$AvgJaccardIndex/x$NumNodes
  x2 <- melt(x, id.vars = c("idx"))
  x2 <- subset(x2, variable == "SingleSample" | variable == "JaccardIndex")
  p <- ggplot(x2, aes(as.factor(idx), value, fill=variable)) + geom_bar(stat = "identity", position=position_dodge(width=.9))
  p <- p + theme_bw() + xlab("Boards") + ylab("Accuracy")
  p <- p + theme(legend.title = element_blank(), legend.text = element_text(size=12))
  p <- p + theme(axis.title=element_text(size=14,face="bold"))
  ggsave(filename = "~/Google Drive/Research/papers/probabilistic-matching/paper/figures/combined_accuracy_plot.pdf", p)

  print(paste("Total accuracy:", sum(x$PredictionAccuracy), "/", sum(x$NumMatchings), "=", sum(x$PredictionAccuracy)/sum(x$NumMatchings)))
}

