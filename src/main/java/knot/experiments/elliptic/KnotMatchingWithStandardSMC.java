package knot.experiments.elliptic;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.EllipticalKnot;
import knot.data.KnotDataReader.Segment;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

import org.apache.commons.math3.util.Pair;

import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;

import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class KnotMatchingWithStandardSMC implements Runnable 
{
	@Option public static double lambda = 1.0;
	@Option public static double tol = 1e-6;
	@Option Random rand = new Random(1);
	@Option public static boolean exactSampling = false;
	@Option public static boolean sequentialMatching = true;
	//public static int [] lumbers = {7,2,3,4,5,8,9,13,14,15,16};
	public static String [] BOARDS = {};

	//public static String dataDirectory = "/Users/sjun/Desktop/scans/21Oct2015/";
	public static String [] dataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	@Option public static String fileName = "enhanced_matching_segmented";
	public static String outputPath = "output/knot-matching/";

	static {
		if (BOARDS.length == 0) {
			List<String> boards = new ArrayList<>();
			for (String dataDirectory : dataDirectories)
			{
				List<File> dirs = BriefFiles.ls(new File(dataDirectory));
				for (int i = 0; i < dirs.size(); i++)
				{
					String board = dirs.get(i).getName();
					if (board.charAt(0) == '.') continue;
					boards.add(dataDirectory + "" + board);
				}
			}
			BOARDS = boards.toArray(new String[boards.size()]);
		}
	}

	
	@Override
	public void run()
	{
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<List<Segment>> instances = KnotExpUtils.readSegmentedBoard(fileName, BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = SegmentedExperimentsRealData.unpack(instances);

		//DecisionModel<String, EllipticalKnot> decisionModel = new KnotPairwiseMatchingDecisionModel();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();

		// 1. train the model parameters
		// 2. make a deterministic pass on the segments to match any obvious cases
		// 3. run SMC on any remaining segments

		// leave-one-out cross validation
		List<String> lines = new ArrayList<>();
		long seed = rand.nextLong();
		//System.out.println("seed: " + seed);
		Random random = new Random(seed);
		KnotExpUtils.evaluateSMCPerformance(random, decisionModel, fe, data, BOARDS, lambda, tol, sequentialMatching, exactSampling, lines);

		PrintWriter writer = BriefIO.output(new File(outputPath + "smc_performance_exp.csv"));
		//writer.println("board, consensus, MAP, total, time");
		writer.println("board, prediction, best, total, segments");
		for (String line : lines) {
			writer.println(line);
		}
		writer.close();
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new KnotMatchingWithStandardSMC());
	}
}
