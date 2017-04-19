H_SPAN<-150
# for simulated data:
num_boards<-100
boards<-1:num_boards
DATA_DIR<-"Google Drive/Research/papers/probabilistic-matching/sgmwsmc/data/simmatching/"

for (board in boards)
{
  filename<-paste(DATA_DIR, "enhanced_matching", board, ".csv", sep="")
  enhanced_ellipses<-read.csv(filename, header=T)
  
  segments<-rep(0, dim(enhanced_ellipses)[1])
  segments[1]<-1
  for (i in 1:dim(enhanced_ellipses)[1])
  {
    ss<-segments[i]
    if (ss == 0)
      ss<-segments[i-1] + 1
    segments[which(abs(enhanced_ellipses[,"x"] - enhanced_ellipses[i,"x"]) < H_SPAN)]<-ss
  }
  
  # save each of the segments to a separate file
  enhanced_ellipses$segments<-segments
  
  output_file<-paste(DATA_DIR, "enhanced_matching_segmented", board, ".csv", sep="")
  write.csv(enhanced_ellipses, output_file, row.names = FALSE)
}

print("Done!")
