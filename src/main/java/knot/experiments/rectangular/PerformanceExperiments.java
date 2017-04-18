package knot.experiments.rectangular;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import knot.data.RectangularKnot;
import knot.model.KnotDoubletonDecisionModel;
import knot.model.features.rectangular.DistanceSizeFeatureExtractor;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class PerformanceExperiments implements Runnable
{
	@Option public static double lambda = 1.0;
	@Option public static double tol = 1e-6;
	@Option Random rand = new Random(1);
	@Option public static boolean sequentialMatching = true;
	@Option public static boolean exactSampling = true;
	public static int [] lumbers = {4, 8, 17, 18, 20, 24};

	public static String dataDirectory = "data/16Oct2015/";
	public static String outputPath = "output/knot-matching/";

	@Override
	public void run()
	{
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances = KnotExpUtils.prepareData(dataDirectory, lumbers, false);

		DecisionModel<String, RectangularKnot> decisionModel = new KnotDoubletonDecisionModel<>();
		GraphFeatureExtractor<String, RectangularKnot> fe = new DistanceSizeFeatureExtractor();

		// leave-one-out cross validation
		List<String> lines = new ArrayList<>();
		long seed = rand.nextLong();
		System.out.println("seed: " + seed);
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
