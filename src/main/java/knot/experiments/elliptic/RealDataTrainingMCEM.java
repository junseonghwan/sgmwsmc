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

import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class RealDataTrainingMCEM implements Runnable 
{
	@Option public static double lambda = 0.05;
	@Option public static double tol = 1e-6;
	@Option Random rand = new Random(1);
	@Option public static boolean exactSampling = true;
	@Option public static boolean sequentialMatching = true;
	public static String [] BOARDS = {};

	public static String [] dataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	@Option public static String fileName = "enhanced_matching_segmented.csv";
	public static String outputPath = "output/knot-matching/";
	public static String paramOutputPath = outputPath + "real_param_estimate.csv";

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
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = unpack(instances);

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
		//Random random = new Random(seed);
		
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe);
		SupervisedLearning<String, EllipticalKnot> sl = new SupervisedLearning<>();
		double [] initial = new double[fe.dim()];
		Pair<Double, double[]> ret = sl.MAPviaMCEM(command, KnotExpUtils.pack(data), 20, 1_000, 10_000, 0.05, initial, 1e-4, false);
		System.out.println(ret.getFirst());
		for (double w : ret.getSecond())
			System.out.println(w);
		//KnotExpUtils.evaluateSegmentedMatching(random, decisionModel, fe, data, BOARDS, lambda, tol, sequentialMatching, exactSampling, lines, paramOutputPath);

		/*
		PrintWriter writer = BriefIO.output(new File(outputPath + "segmented_real_data_exp.csv"));
		//writer.println("board, consensus, MAP, total, time");
		writer.println("board, prediction, best, total, jaccard, num_nodes, segments");
		for (String line : lines) {
			writer.println(line);
		}
		writer.close();
		*/
	}
	
	public static List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> unpack(List<List<Segment>> instances)
	{
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = new ArrayList<>();
		for (List<Segment> instance : instances)
		{
			List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> datum = new ArrayList<>(); 
			for (Segment segment : instance)
			{
				datum.add(Pair.create(new ArrayList<>(segment.label2Edge.values()), segment.knots));
			}
			data.add(datum);
		}
		return data;
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new RealDataTrainingMCEM());
	}
}
