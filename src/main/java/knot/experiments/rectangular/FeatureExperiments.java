package knot.experiments.rectangular;

import java.io.File;
import java.io.PrintWriter;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import knot.data.RectangularKnot;
import knot.model.DistanceFeatureExtractor;
import knot.model.KnotDoubletonDecisionModel;
import knot.model.features.rectangular.DistanceSizeFeatureExtractor;
import knot.model.features.rectangular.SizeFeatureExtractor;
import common.evaluation.MatchingSampleEvaluation;
import common.model.DecisionModel;
import common.model.NoFeatureExtractor;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class FeatureExperiments implements Runnable
{
	@Option public Random random = new Random(1);
	@Option public int [] lumbers = {4, 8, 17, 18, 20, 24};
	@Option public String output = "output/feature-analysis/output.csv";

	@Override
	public void run()
	{
		List<MatchingSampleEvaluation<String, RectangularKnot>> evals0 = noFeatureAnalysis();
		List<MatchingSampleEvaluation<String, RectangularKnot>> evals1 = distanceFeatureAnalysis();
		List<MatchingSampleEvaluation<String, RectangularKnot>> evals2 = sizeFeatureAnalysis();
		List<MatchingSampleEvaluation<String, RectangularKnot>> evals3 = distanceSizeFeatureAnalysis();
		
		// compare the evaluations, output CSV file
		PrintWriter writer = BriefIO.output(new File(output));
		writer.println("Experiment, Consensus, MAP, Average, Total");
		outputResults(evals0, "NoFeature", writer);
		outputResults(evals1, "Distance", writer);
		outputResults(evals2, "Size", writer);
		outputResults(evals3, "DistanceSize", writer);
		writer.close();
	}
	
	public void outputResults(List<MatchingSampleEvaluation<String, RectangularKnot>> evals, String experimentType, PrintWriter writer)
	{
		for (MatchingSampleEvaluation<String, RectangularKnot> eval : evals)
		{
			String str = experimentType + ", " + eval.consensusMatching.getSecond() + ", " + eval.bestLogLikMatching.getSecond().getSecond() + ", " + eval.avgAccuracy + ", " + eval.cardinality;
			writer.println(str);
		}

	}
	
	public List<MatchingSampleEvaluation<String, RectangularKnot>> noFeatureAnalysis()
	{
		NoFeatureExtractor<String, RectangularKnot> noFeatures = new NoFeatureExtractor<>();
		DecisionModel<String, RectangularKnot> decisionModel = new KnotDoubletonDecisionModel();
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances = KnotExpUtils.prepareData(KnotExpUtils.DEFAULT_DATA_DIR, lumbers, false);
		return KnotExpUtils.evaluate(random, decisionModel, noFeatures, instances, lumbers);
	}

	public List<MatchingSampleEvaluation<String, RectangularKnot>> distanceFeatureAnalysis()
	{
		DistanceFeatureExtractor distanceFeatures = new DistanceFeatureExtractor();
		DecisionModel<String, RectangularKnot> decisionModel = new KnotDoubletonDecisionModel();
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances = KnotExpUtils.prepareData(KnotExpUtils.DEFAULT_DATA_DIR, lumbers, false);
		return KnotExpUtils.evaluate(random, decisionModel, distanceFeatures, instances, lumbers);
	}
	
	public List<MatchingSampleEvaluation<String, RectangularKnot>> sizeFeatureAnalysis()
	{
		SizeFeatureExtractor sizeFeatures = new SizeFeatureExtractor();
		DecisionModel<String, RectangularKnot> decisionModel = new KnotDoubletonDecisionModel();
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances = KnotExpUtils.prepareData(KnotExpUtils.DEFAULT_DATA_DIR, lumbers, false);
		return KnotExpUtils.evaluate(random, decisionModel, sizeFeatures, instances, lumbers);
	}

	public List<MatchingSampleEvaluation<String, RectangularKnot>> distanceSizeFeatureAnalysis()
	{
		DistanceSizeFeatureExtractor fe = new DistanceSizeFeatureExtractor();
		DecisionModel<String, RectangularKnot> decisionModel = new KnotDoubletonDecisionModel();
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances = KnotExpUtils.prepareData(KnotExpUtils.DEFAULT_DATA_DIR, lumbers, false);
		return KnotExpUtils.evaluate(random, decisionModel, fe, instances, lumbers);
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new FeatureExperiments());
	}
}
