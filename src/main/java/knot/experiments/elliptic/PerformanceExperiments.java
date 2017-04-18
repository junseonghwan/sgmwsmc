package knot.experiments.elliptic;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;
import knot.data.EllipticalKnot;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.SingletonExplicitKnotMatchingDecisionModel;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class PerformanceExperiments implements Runnable
{
	@Option public static double lambda = 1.0;
	@Option public static double tol = 1e-6;
	@Option Random rand = new Random(1);
	@Option public static boolean exactSampling = false;
	@Option public static boolean sequentialMatching = true;
	public static int [] lumbers = {2,3,4,5,7,8,9,13,14,15,16};

	public static String dataDirectory = "/Users/sjun/Desktop/scans/21Oct2015/";
	public static String outputPath = "output/knot-matching/";

	@Override
	public void run()
	{
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances = KnotExpUtils.readEllipticalData(dataDirectory, lumbers, false);

		//DecisionModel<String, EllipticalKnot> decisionModel = new SingletonExplicitKnotMatchingDecisionModel();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();

		// leave-one-out cross validation
		List<String> lines = new ArrayList<>();
		long seed = rand.nextLong();
		//System.out.println("seed: " + seed);
		Random random = new Random(seed);
		KnotExpUtils.evaluate(random, decisionModel, fe, instances, lumbers, lambda, tol, sequentialMatching, exactSampling, lines);

		PrintWriter writer = BriefIO.output(new File(outputPath + "performance_exp.csv"));
		writer.println("board, consensus, MAP, total, time");
		for (String line : lines) {
			writer.println(line);
		}
		writer.close();
	}


	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new PerformanceExperiments());
	}

}
