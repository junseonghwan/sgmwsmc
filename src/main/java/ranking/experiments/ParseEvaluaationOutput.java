package ranking.experiments;

import java.io.File;
import java.io.PrintWriter;

import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class ParseEvaluaationOutput implements Runnable
{
	//@Option public static String path = "/Users/seonghwanjun/Dropbox/Research/ranking/OHSUMED/Scores/";
	@Option public static String path = "/Users/seonghwanjun/Dropbox/Research/ranking/TD2003/Scores/";
	@Option public static String outputPath = "/Users/seonghwanjun/Dropbox/Research/ranking/letor3baselines/";
	//@Option public static String outputFile = "OHSUMED_RESULTS.txt";
	@Option public static String outputFile = "TD2003_RESULTS.txt";
	@Override
	public void run()
	{
		PrintWriter out = BriefIO.output(new File(outputPath + outputFile));
		for (int fold = 1; fold <= 5; fold++)
		{
			String filePath = path + "Fold" + fold + "/" + "evaluation.txt";
			// read the file
			for (String line : BriefIO.readLines(new File(filePath)))
			{
				if (line.indexOf(":") > 0)
				{
					String [] row = line.split(":");
					if (row[0].equalsIgnoreCase("NDCG"))
					{
						String [] metric = row[1].split("\\s+");
						for (int j = 1; j < metric.length; j++)
						{
							double ndcg = Double.parseDouble(metric[j].trim());
							out.print(ndcg + " ");
							System.out.print(ndcg + " ");
						}
						out.println();
						System.out.println();
						break;
					}
				}
			}				
		}
		out.close();
	}
	
	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new ParseEvaluaationOutput());
	}

}
