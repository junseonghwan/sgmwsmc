H_SPAN<-100
# for simulated data:
#num_boards<-20
#boards<-1:num_boards

# for real data:
#boards<-c(2,3,4,5,7,8,9,13,14,15,16)

# make it as a script
#BOARD_NO <- commandArgs(TRUE)[1]
#BOARD_NO<-"D4021"
#IMAGE_DIR <- commandArgs(TRUE)[2]
#IMAGE_DIR<-"/Users/sjun/Desktop/scans/16Mar2016/"
#IMAGE_DIR<-"~/Dropbox/Research/papers/probabilistic-matching/knotclassifier/imageprocessor/data/21Oct2015/"
#BOARD_DIR<-paste(IMAGE_DIR, BOARD_NO, sep="")

BOARD_DIR<-commandArgs(TRUE)[1]
TR_DIR<-paste(BOARD_DIR, "/knotdetection/tracheids/", sep="")
MATCHING_DIR<-paste(BOARD_DIR, "/knotdetection/matching/", sep="")

enhanced_matching_file <- paste(MATCHING_DIR, "enhanced_matching.csv", sep="")
if (file.exists(enhanced_matching_file)) {
  enhanced_ellipses<-read.csv(enhanced_matching_file, header=T)

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
  
  output_file<-paste(MATCHING_DIR, "enhanced_matching_segmented.csv", sep="")
  write.csv(enhanced_ellipses, output_file, row.names = FALSE)
  #write.csv(enhanced_ellipses, paste("~/Google Drive/Research/papers/probabilistic-matching/sgmwsmc/data/simmatch/enhanced_matching_segmented", board, ".csv", sep=""), row.names = FALSE)
  print("Done!")
} else {
  print(paste(BOARD_DIR, "not processed."))
}
