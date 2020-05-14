package common.experiments.simulation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import bayonet.distributions.Normal;
import briefj.opt.Option;
import briefj.run.Mains;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;

public class UnknownSequenceExperiments  implements Runnable
{
	@Option public static int numRep = 10;
	@Option public static int numRandomStarts = 5;
	@Option public static int numData = 10;
	@Option public static int numPartitions = 4;
	@Option public static int numNodesPerPartition = 10;
	@Option public static int numMCSamples = 100;
	@Option public static int numFeatures = 2;
	@Option public static boolean useSPF = true;

	@Option public static final double sigma_var = 1.4;
	public static final double lambda = 1/(2*sigma_var);
	@Option public Random rand = new Random(12);
	
	public static DecisionModel<String, SimpleNode> decisionModel = new PairwiseMatchingModel<>();
	public static Command<String, SimpleNode> command = null;
	public static GraphFeatureExtractor<String, SimpleNode> fe = null;
	
	@Override
	public void run()
	{
		for (int i = 0; i < numRep; i++)
		{
			Random random = new Random(rand.nextLong());
			
			// generate parameters
			double [] w = new double[numFeatures];
			for (int n = 0; n < numFeatures; n++) {
				w[n] = Normal.generate(random, 0.0, sigma_var);
				System.out.println("w[" + n + "]: " + w[n]);
			}

			// generate the data
			Pair<List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>>, List<List<Set<SimpleNode>>>> truth = generateData(random, w);
			command.updateModelParameters(w);
			double logLik = 0.0;
			List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances = truth.getFirst();
			List<List<Set<SimpleNode>>> decisions = truth.getSecond();
			for (int j = 0; j < instances.size(); j++)
			{
				logLik += SupervisedLearning.evaluate(command, command.getModelParameters(), Pair.create(decisions.get(j), instances.get(j).getSecond())).getFirst();
			}

			SupervisedLearning<String, SimpleNode> sl = new SupervisedLearning<>();
			Pair<Double, double[]> ret = sl.MAPviaMCEM(random, i, command, instances, 100, 100, 1_000, 0, w, 1e-4, false, useSPF);
			command.updateModelParameters(ret.getSecond());
			double logLik2 = 0.0;
			for (int j = 0; j < instances.size(); j++)
			{
				logLik2 += SupervisedLearning.evaluate(command, command.getModelParameters(), Pair.create(decisions.get(j), instances.get(j).getSecond())).getFirst();
			}

			System.out.println(logLik + ", " + logLik2);
			System.out.println("w: ");
			for (int j = 0; j < ret.getSecond().length; j++)
				System.out.println(ret.getSecond()[j] + ", " + w[j]);
		}
	}	

	private Pair<List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>>, List<List<Set<SimpleNode>>>> generateData(Random random, double [] w)
	{
		List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances = new ArrayList<>();
		List<List<Set<SimpleNode>>> decisions = new ArrayList<>();

		for (int n = 0; n < numData; n++)
		{
			List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, sigma_var);
			Collections.shuffle(nodes, random); // randomize the sequence 
			
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
			
			instances.add(Pair.create(state.getMatchings(), nodes)); 
			decisions.add(state.getDecisions());
		}
		return Pair.create(instances, decisions);
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new UnknownSequenceExperiments());
	}

}
