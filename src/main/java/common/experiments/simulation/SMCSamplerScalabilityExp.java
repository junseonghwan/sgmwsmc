package common.experiments.simulation;

import java.util.List;
import java.util.Random;

import knot.experiments.rectangular.KnotExpUtils;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;
import briefj.opt.Option;
import briefj.run.Mains;

// Work in progress: tuning SMC sampler's speed.
public class SMCSamplerScalabilityExp implements Runnable 
{
	@Option public static final int numParticles = 1000;
	@Option public static final int numNodesPerPartition = 8;
	@Option public static final int numPartitions = 4;
	@Option public static final int numFeatures = 10;
	@Option public static final int numReps = 5;

	@Option public static final double sigma_var = 1.4;
	@Option public static final double nu_var = 1.1;
	
	@Option public static final boolean exactSampling = true;
	@Option public static final boolean sequentialMatching = false;
	@Option public static final boolean useStreaming = false;
	@Option public static final boolean plotESS = false;
	@Option public static String outputFilePath = "output/smc/ess" + "_" + numFeatures + "_" + numNodesPerPartition;

	@Option public static final Random random = new Random(999);

	@Override
	public void run() 
	{
		// generate the parameters
		// generate a graph with numNodes, sample features and assign to partitions
		// run SMC with varying number of particles -- plot ESS per iteration
		// can we show some sort of coverage probability?
		double [] w = SimulationUtils.sampleParameters(random, numFeatures, 0.0, sigma_var);

		PairwiseMatchingModel<String, SimpleNode> decisionModel = new PairwiseMatchingModel<>();
		Command<String, SimpleNode> command = null;
		GraphFeatureExtractor<String, SimpleNode> fe = null;
		
		for (int rep = 0; rep < numReps; rep++)
		{
			
			// generate nodes: sample features, assign the nodes to partitions
			List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, nu_var);
			
			if (command == null)
			{
				// initialize the model settings
				fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
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
			System.out.println(nodes);
			System.out.println(state.getVisitedNodes());

	
			// sample matching using SMC -- output ESS per iteration to a file
			List<GenericGraphMatchingState<String, SimpleNode>> samples = KnotExpUtils.runSMC(random, nodes, decisionModel, fe, w, sequentialMatching, exactSampling, useStreaming, outputFilePath + "_" + rep + ".csv");

			// do stuff with samples, some way to quantify that the number of particles is sufficient? Estimate something?
			MatchingSampleEvaluation<String, SimpleNode> mes = MatchingSampleEvaluation.evaluate(samples, state.getMatchings());
			System.out.println(mes.bestLogLikMatching.getSecond().getSecond() + "/" + state.getMatchings().size());
		}

	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new SMCSamplerScalabilityExp());
	}

}
