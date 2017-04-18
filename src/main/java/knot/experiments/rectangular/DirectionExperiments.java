package knot.experiments.rectangular;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.RectangularKnot;
import knot.model.KnotDoubletonDecisionModel;
import knot.model.features.rectangular.DistanceSizeFeatureExtractor;

import org.apache.commons.math3.util.Pair;

import common.evaluation.LearningUtils;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import briefj.BriefIO;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;

public class DirectionExperiments implements Runnable
{
	
	@Option public double lambda = 1.0;
	@Option public double tol = 1e-6;
	@Option public boolean sequentialMatching = true;
	@Option public boolean exactSampling = true;
	@Option public int numConcreteParticles = 1000;
	@Option public int maxNumVirtualParticles = 1000;
	@Option Random rand = new Random(1);
	
	public String outputPath = "output/knot-matching/"; 
	public String dataDirectory = "data/16Oct2015/";
	public int [] lumbers = {4, 8, 17, 18, 20, 24};

	@Override
	public void run()
	{
		Random random = new Random(rand.nextLong());
		DecisionModel<String, RectangularKnot> decisionModel = new KnotDoubletonDecisionModel<>();
		GraphFeatureExtractor<String, RectangularKnot> fe = new DistanceSizeFeatureExtractor();

		// estimate the parameters using different directions
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances1 = KnotExpUtils.prepareData(dataDirectory, lumbers, false);
		Pair<Double, Counter<String>> ret1 = LearningUtils.learnParameters(random, decisionModel, fe, instances1, lambda, tol);

		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances2 = KnotExpUtils.prepareData(dataDirectory, lumbers, true);
		Pair<Double, Counter<String>> ret2 = LearningUtils.learnParameters(random, decisionModel, fe, instances2, lambda, tol);

		// prediction accuracy
		// leave-one-out cross validation
		List<String> dir1 = new ArrayList<>();
		List<String> dir2 = new ArrayList<>();
		KnotExpUtils.evaluate(random, decisionModel, fe, instances1, lumbers, lambda, tol, sequentialMatching, exactSampling, dir1);
		KnotExpUtils.evaluate(random, decisionModel, fe, instances2, lumbers, lambda, tol, sequentialMatching, exactSampling, dir2);
		
		PrintWriter writer = BriefIO.output(new File(outputPath + "direction_exp.csv"));
		writer.println("board, consensus, MAP, total, time, dir");
		for (String line : dir1)
		{
			writer.println(line + ", " + 1);
		}
		for (String line : dir2)
		{
			writer.println(line + ", " + 2);
		}
		writer.close();

		System.out.println("=== Parameter Estimation Experiments Summary ===");
		System.out.println("Likelihood: " + ret1.getFirst() + " vs " + ret2.getFirst());
		double diff = 0.0;
		for (String f : ret1.getSecond())
		{
			System.out.println(f + ": " + ret1.getSecond().getCount(f) + " vs " + ret2.getSecond().getCount(f));
			diff += Math.pow(ret1.getSecond().getCount(f) - ret2.getSecond().getCount(f), 2.0);
		}
		System.out.println("MSD: " + diff/ret1.getSecond().size());
	}

	public void MAPEstimation()
	{
		// pass in data, return the MAP estimates
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new DirectionExperiments());
	}

}
