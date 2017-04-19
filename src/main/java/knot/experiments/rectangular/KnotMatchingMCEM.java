package knot.experiments.rectangular;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import knot.data.RectangularKnot;
import knot.model.KnotDoubletonDecisionModel;
import knot.model.NegativeParameterValueSupportSet;
import knot.model.features.rectangular.DistanceSizeFeatureExtractor;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GraphMatchingState;
import common.learning.MonteCarloExpectationMaximization;
import common.model.Command;
import common.processor.RealParametersProcessor;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.RandomProposalObservationDensity;
import common.smc.components.SequentialGraphMatchingSampler;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class KnotMatchingMCEM implements Runnable
{
	@Option public static double lambda = 5.0;
	@Option public static double tol = 1e-6;
	@Option public static int numConcreteParticles = 100;
	@Option public static int maxNumVirtualParticles = 10000;
	@Option Random random = new Random(1);
	@Option public static boolean exactSampling = true;
	@Option public static int numRep = 5;
	
	public static final String outputDir = "output/mcem/";
	public String dataDirectory = "data/16Oct2015/";
	//public int [] lumbers = {4, 8, 17, 18, 20, 24};
	public int [] lumbers = {24};
	
	@Override
	public void run()
	{
		long time = System.currentTimeMillis();

		// read in the knots, perform MC-EM to jointly infer the matching + parameters
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> data = KnotExpUtils.prepareData(dataDirectory, lumbers, false);

		MonteCarloExpectationMaximization<String, RectangularKnot> mcem = new MonteCarloExpectationMaximization<>();
		KnotDoubletonDecisionModel<RectangularKnot> decisionModel = new KnotDoubletonDecisionModel<>();
		DistanceSizeFeatureExtractor fe = new DistanceSizeFeatureExtractor();
		Command<String, RectangularKnot> command = new Command<>(decisionModel, fe);
		NegativeParameterValueSupportSet supportSet = new NegativeParameterValueSupportSet(command);

		List<String> results = new ArrayList<>();
		for (int idx = 0; idx < data.size(); idx++)
		{
			List<RectangularKnot> nodes = data.get(idx).getSecond();
			GraphMatchingState<String, RectangularKnot> initial = GraphMatchingState.getInitialState(nodes);
			GenericMatchingLatentSimulator<String, RectangularKnot> transitionDensity = new GenericMatchingLatentSimulator<>(command, initial, true, true);
			//ExactProposalObservationDensity<String, RectangularKnot> observationDensity = new ExactProposalObservationDensity<>();
			RandomProposalObservationDensity<String, RectangularKnot> observationDensity = new RandomProposalObservationDensity<>(command);

			// carry out MC-EM with multiple starting points
			System.out.println("==========");
			System.out.println("Board " + lumbers[idx]);
			for (int i = 0; i < numRep; i++) {
				// randomly initialize the starting point
				double [] w = new double[fe.dim()];
				supportSet.initParam(random, w);
				command.updateModelParameters(w);

				RealParametersProcessor<String> processor = new RealParametersProcessor<>(outputDir + "_" + time, lumbers[idx] + "_" + i);
				double nllk = mcem.execute(random, command, nodes, data.get(idx).getFirst(), initial, transitionDensity, observationDensity, supportSet, lambda, processor);
				System.out.println("final nllk: " + nllk);
				for (int k = 0; k < fe.dim(); k++)
				{
					String f = command.getIndexer().i2o(k);
					System.out.println(command.getIndexer().i2o(k) + ": " + command.getModelParameters().getCount(f));
				}
				processor.output();

  			// now do prediction
  			List<Object> emissions = new ArrayList<>();
  			for (int j = 0; j < nodes.size(); j++) emissions.add(null);
  			
  			SequentialGraphMatchingSampler<String, RectangularKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
  			double logZ = smc.sample(numConcreteParticles, maxNumVirtualParticles);
  			System.out.println("logZ=" + logZ);
  			MatchingSampleEvaluation<String, RectangularKnot> mse = MatchingSampleEvaluation.evaluate(smc.getSamples(), data.get(idx).getFirst());
  			String outputString = lumbers[idx] + ", " + i + ", " + mse.avgAccuracy + ", " + mse.bestLogLikMatching.getSecond().getSecond() + ", " + mse.consensusMatching.getSecond() + ", " + mse.bestAccuracyMatching.getSecond() + ", " + data.get(idx).getFirst().size();
  			System.out.println(outputString);
  			results.add(outputString);
			}
			System.out.println("==========");
		}

		PrintWriter writer = BriefIO.output(new File(outputDir + "_" + time + "/results/results.csv"));
		writer.println("board,rep,avg,MAP,consensus,best,total");
		for (String ret : results)
		{
			writer.println(ret);
		}
		writer.close();
	}
	
	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new KnotMatchingMCEM());
	}

}
