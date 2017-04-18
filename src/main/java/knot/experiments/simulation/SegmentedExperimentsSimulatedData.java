package knot.experiments.simulation;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import knot.data.EllipticalKnot;
import knot.data.KnotDataReader.Segment;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.KnotPairwiseMatchingDecisionModel;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

public class SegmentedExperimentsSimulatedData implements Runnable 
{
	@Option public static double lambda = 1.0;
	@Option public static double tol = 1e-6;
	@Option Random rand = new Random(1);
	@Option public static boolean exactSampling = true;
	@Option public static boolean sequentialMatching = false;
	@Option public static int numData = 20;
	public static String [] BOARDS = {};
	static {
		if (BOARDS.length == 0 && numData > 0) {
			BOARDS = new String[numData];
			for (int i = 0; i < numData; i++) {
				BOARDS[i] = i + 1 + "";
			}
		}
	}

	@Option public static String dataDirectory = "data/simmatch/";
	@Option public static String fileName = "enhanced_matching_segmented";
	public static String outputPath = "output/knot-matching/";

	@Override
	public void run()
	{
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<List<Segment>> instances = KnotExpUtils.readSegmentedSimulatedBoard(dataDirectory, fileName, BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = KnotExpUtils.unpack(instances);

		DecisionModel<String, EllipticalKnot> decisionModel = new KnotPairwiseMatchingDecisionModel();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();

		// 1. train the model parameters
		// 2. make a deterministic pass on the segments to match any obvious cases
		// 3. run SMC on any remaining segments

		// leave-one-out cross validation
		List<String> lines = new ArrayList<>();
		long seed = rand.nextLong();
		//System.out.println("seed: " + seed);
		Random random = new Random(seed);
		KnotExpUtils.evaluateSegmentedMatching(random, decisionModel, fe, data, BOARDS, lambda, tol, sequentialMatching, exactSampling, lines);

		PrintWriter writer = BriefIO.output(new File(outputPath + "segmented_simulated_data_exp.csv"));
		//writer.println("board, consensus, MAP, total, time");
		writer.println("board, prediction, best, total, jaccard_index, num_nodes, segments");
		for (String line : lines) {
			writer.println(line);
		}
		writer.close();
	}
	
	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new SegmentedExperimentsSimulatedData());
	}
	
}
