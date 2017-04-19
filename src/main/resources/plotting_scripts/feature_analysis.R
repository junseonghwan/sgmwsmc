library(dplyr)
# read the matching data -- explore the features
data_dirs<-c("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/data/16Mar2016/", "~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/data/21Oct2015/")
boards<-c()
for (data_dir in data_dirs)
{
  boards<-c(boards, paste(data_dir, list.files(data_dir), "/enhanced_matching_segmented.csv", sep=""))
}

# now read the data -- get matchings
mm<-list()
k<-1
for (board in boards)
{
  d<-read.csv(board, header=T)
  matching<-unique(d$matching)
  for (m in matching)
  {
    mm[[k]]<-subset(d, matching == m)
    k <- k + 1
  }
}

mm[[1]]
