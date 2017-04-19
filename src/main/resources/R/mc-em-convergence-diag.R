setwd("Dropbox/Research/papers/probabilistic-matching/sgmwsmc/output/mcem/")
library(ggplot2)

files<-c("H_DIFF", "W_DIFF", "TWO_DISTANCE_1", "TWO_DISTANCE_2", "objective")

f<-function(dir_name, file_name, board_num, nrep)
{
  dd<-data.frame()
  for (i in 0:(nrep-1))
  {
    file <- paste(folder, file_name, "_", board_num, "_", i, sep="")
    val = read.table(file, header=F)
    d <- data.frame(val, "iter"=1:dim(val)[1], "rep"=(i+1))
    dd<-rbind(dd, d)
  }
  names(dd)<-c("value", "iter", "rep")
  return(dd)
}

file_idx<-5
folders<-c("_1476326188871/", "_1476303647129/", "_1476326557416/", "_1476326693784/", "_1476326848151/")
boards<-c(8, 17, 18, 20, 24)
board_idx<-5
board_num <- boards[board_idx]
folder<-folders[board_idx]
dd<-f(folder, files[idx], board_num, 5)
output_file <- paste("../../../AISTATS-Appendix/figures/mcem", "_", files[file_idx], "_", board_num, ".pdf", sep="")
p <- ggplot(dd, aes(iter, value, col=factor(rep))) + geom_line()
p <- p + labs(x = "MC-EM Iterations", y = "Negative log likelihood", color = "Replication", title =paste("Board", board_num))
p <- p + theme_bw() + theme(plot.title = element_text(size = rel(2))) + theme(legend.position="none")
#p <- p + geom_hline(yintercept = mean(dd[,1]))
p
ggsave(output_file, p)
