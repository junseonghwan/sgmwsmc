package knot.experiments.rectangular;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.RectangularKnot;
import knot.model.KnotDoubletonDecisionModel;

import org.apache.commons.math3.util.Pair;

import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.NoFeatureExtractor;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import briefj.BriefFiles;
import briefj.opt.Option;
import briefj.run.Mains;

public class KnotMatchingNoLearning implements Runnable
{
	@Option public static double lambda = 1.0;
	@Option public static double tol = 1e-6;
	@Option public static int numConcreteParticles = 1000;
	@Option public static int maxNumVirtualParticles = numConcreteParticles;
	@Option Random rand = new Random(1);
	@Option public static int numRep = 1;
	@Option public static boolean exactSampling = true;
	@Option public static String outputPath = "output/knot-matching/";
	
	public String dataDirectory = "data/16Oct2015/";
	public int [] lumbers = {4, 8, 17, 18, 20, 24};

	@Override
	public void run()
	{
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances = KnotExpUtils.prepareData(dataDirectory, lumbers, false);

		DecisionModel<String, RectangularKnot> decisionModel = new KnotDoubletonDecisionModel<>();
		GraphFeatureExtractor<String, RectangularKnot> fe = new NoFeatureExtractor<>();

		// leave-one-out cross validation
		for (int n = 0; n < numRep; n++)
		{
			long seed = rand.nextLong();
			//Long seed = Long.parseUnsignedLong("7564655870752979346");
			Random random = new Random(seed);
			evaluate(random, decisionModel, fe, instances);
			System.out.println("seed: " + seed);
		}
	}
	
	public static List<MatchingSampleEvaluation<String, RectangularKnot>> evaluate(Random random, 
			DecisionModel<String, RectangularKnot> decisionModel, 
			GraphFeatureExtractor<String, RectangularKnot> fe, 
			List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances)
	{
		List<MatchingSampleEvaluation<String, RectangularKnot>> evals = new ArrayList<>();

		for (int i = 0; i < instances.size(); i++)
		{
			Pair<List<Set<RectangularKnot>>, List<RectangularKnot>> heldOut = instances.remove(i);
			Command<String, RectangularKnot> command = new Command<>(decisionModel, fe); 
			ExactProposalObservationDensity<String, RectangularKnot> observationDensity = new ExactProposalObservationDensity<>(command);

			GraphMatchingState<String, RectangularKnot> initial = GraphMatchingState.getInitialState(heldOut.getSecond());
			LatentSimulator<GenericGraphMatchingState<String, RectangularKnot>> transitionDensity = new GenericMatchingLatentSimulator<String, RectangularKnot>(command, initial, true, exactSampling);

			List<Object> emissions = new ArrayList<>();
			for (int j = 0; j < heldOut.getSecond().size(); j++) emissions.add(null);

			// draw samples using SMC
			long start = System.currentTimeMillis();
			SequentialGraphMatchingSampler<String, RectangularKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
			smc.sample(numConcreteParticles, maxNumVirtualParticles);
			List<GenericGraphMatchingState<String, RectangularKnot>> samples = smc.getSamples();
			MatchingSampleEvaluation<String, RectangularKnot> eval = MatchingSampleEvaluation.evaluate(samples, heldOut.getFirst());
			long end = System.currentTimeMillis();
			evals.add(eval);

			System.out.println("===== Evaluation Summary =====");
			System.out.println("Consensus matching accuracy: " + eval.consensusMatching.getSecond() + ", size: " + eval.consensusMatching.getFirst().size());
			//System.out.println(eval.consensusMatching.getFirst().toString());
			System.out.println("Maximum likelihood matching: " + eval.bestLogLikMatching.getSecond().getFirst() + ", accuracy: " + eval.bestLogLikMatching.getSecond().getSecond());
			//System.out.println(eval.bestLogLikMatching.getFirst().toString());
			System.out.println("Best accuracy: " + eval.bestAccuracyMatching.getSecond());
			System.out.println("Average correctness: " + eval.avgAccuracy);
			System.out.println("Total # of matching: " + heldOut.getFirst().size());
			System.out.println("Time (s): " + (end-start)/1000.0);
			System.out.println("===== End Evaluation Summary =====");			

			instances.add(i, heldOut);
		}

		return evals;
	}
	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new KnotMatchingNoLearning());
	}

}
