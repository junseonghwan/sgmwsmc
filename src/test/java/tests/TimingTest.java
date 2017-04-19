package tests;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import common.experiments.simulation.SimpleNode;
import common.experiments.simulation.SimulationUtils;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.learning.SupervisedLearning.ObjectiveFunction;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;
import briefj.opt.Option;
import briefj.run.Mains;

public class TimingTest implements Runnable
{
	@Option public int numData = 4000;
	@Option public int numPartitions = 2;
	@Option public int numNodesPerPartition = 30;
	@Option public int numFeatures = 60;
	@Option public int randomSeed = 20160912;
	@Option public boolean writeToFile = false;
	@Option public int gridSize = 100;
	@Option public double lambda = 1.0;

	@Override
	public void run()
	{
		// generate data, compare timing of executing Objective.valueAt
		Random random = new Random(randomSeed);

		// generate parameters
		double [] w = new double[numFeatures];
		for (int n = 0; n < numFeatures; n++) w[n] = random.nextGaussian();
		
		GraphFeatureExtractor<String, SimpleNode> fe = null; 
		DecisionModel<String, SimpleNode> decisionModel = null;
		Command<String, SimpleNode> command = null;

		// generate the data
		List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances = new ArrayList<>();		
		for (int n = 0; n < numData; n++)
		{
			List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, 1.0);

			if (command == null)
			{
				// initialize the model settings
				fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
				decisionModel = new DoubletonDecisionModel<>();
				command = new Command<>(decisionModel, fe);
				command.updateModelParameters(w);
			}
			
			// generate a matching
			GraphMatchingState<String, SimpleNode> state = GraphMatchingState.getInitialState(nodes);
			while (state.hasNextStep())
			{
				state.sampleNextState(random, command, true, true);
			}
			//System.out.println("instance1=" + state);			
			instances.add(Pair.create(state.getMatchings(), state.getVisitedNodes()));
		}

		ObjectiveFunction<String, SimpleNode> obj = new ObjectiveFunction<>(command, instances);
		long start = System.currentTimeMillis();
		obj.valueAt(w);
		long end = System.currentTimeMillis();
		System.out.println("Parallel execution: " + (end-start)/1000.0 + " seconds.");
		SupervisedLearning.parallelize = false;
		start = System.currentTimeMillis();
		obj.valueAt(w);
		end = System.currentTimeMillis();
		System.out.println("Serial execution: " + (end-start)/1000.0 + " seconds.");
		
		// if the data size is super large, there is benefit to parallelization
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new TimingTest());
	}

}
