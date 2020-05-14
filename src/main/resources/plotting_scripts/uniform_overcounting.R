library(ggplot2)

d<-read.csv("Google Drive/Research/papers/probabilistic-matching/sgmwsmc/output/overcounting/uniform_4_1.csv", header=T)
limits <- aes(ymax = RMSE + 1.96*sd_RMSE, ymin = RMSE - 1.96*sd_RMSE)

p <- ggplot(d, aes(N, RMSE, col=OvercountingCorrected)) + geom_line()
p <- p + theme_bw()
p <- p + theme(legend.position="none")
p <- p + geom_errorbar(limits, width=.2, position=position_dodge(width=.9))
p
ggsave("Google Drive/Research/papers/probabilistic-matching/paper/figures/overcounting_uniform_4_1.pdf", p, width=3.5, height=2.5)
