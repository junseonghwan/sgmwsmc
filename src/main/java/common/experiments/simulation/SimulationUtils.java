package common.experiments.simulation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import bayonet.distributions.Normal;
import bayonet.math.NumericalUtils;
import briefj.collections.Counter;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;

public class SimulationUtils 
{
	/**
	 * Generate TestNodes
	 * 
	 * @param random
	 * @param numPartitions
	 * @param numNodesPerPartition
	 * @param numFeatures
	 * @return
	 */
	public static List<SimpleNode> generateSimpleNodes(Random random, int numPartitions, int numNodesPerPartition, int numFeatures, double mu, double var)
	{
		double [] features = new double[numFeatures];
		List<SimpleNode> nodes = new ArrayList<>();
		for (int pidx = 0; pidx < numPartitions; pidx++)
		{
			for (int idx = 0; idx < numNodesPerPartition; idx++)
			{
				for (int j = 0; j < numFeatures; j++)
					features[j] = Normal.generate(random, mu, var);
				SimpleNode node = new SimpleNode(pidx, idx, features);
				nodes.add(node);
			}
		}

		return nodes;
	}

	public static List<SimpleNode> generateSimpleNodes(Random random, int numPartitions, int [] numNodesPerPartition, int numFeatures, double mu, double var)
	{
		double [] features = new double[numFeatures];
		List<SimpleNode> nodes = new ArrayList<>();
		for (int pidx = 0; pidx < numPartitions; pidx++)
		{
			for (int idx = 0; idx < numNodesPerPartition[pidx]; idx++)
			{
				for (int j = 0; j < numFeatures; j++)
					features[j] = Normal.generate(random, mu, var);
				SimpleNode node = new SimpleNode(pidx, idx, features);
				nodes.add(node);
			}
		}

		return nodes;
	}

	public static Pair<List<Set<SimpleNode>>, List<SimpleNode>> sampleMatching(Random random, DecisionModel<String, SimpleNode> decisionModel, double [] parameters, List<SimpleNode> nodes)
	{
		GraphFeatureExtractor<String, SimpleNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
		Command<String, SimpleNode> command = new Command<>(decisionModel, fe);
		command.updateModelParameters(parameters);
		
		GraphMatchingState<String, SimpleNode> state = GraphMatchingState.getInitialState(nodes);
		while (state.hasNextStep())
		{
			state.sampleNextState(random, command, true, true);
		}
		return Pair.create(state.getMatchings(), state.getVisitedNodes());
	}
	
	public static Pair<List<Set<SimpleNode>>, List<SimpleNode>> sampleMatchingSMC(Random random, DecisionModel<String, SimpleNode> decisionModel, double [] parameters, List<SimpleNode> nodes)
	{
		GraphFeatureExtractor<String, SimpleNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
		Command<String, SimpleNode> command = new Command<>(decisionModel, fe);
		command.updateModelParameters(parameters);
		
		GraphMatchingState<String, SimpleNode> state = GraphMatchingState.getInitialState(nodes);
		
		return Pair.create(state.getMatchings(), state.getVisitedNodes());
	}
	
	/**
	 * Generate a bipartite matching, compute log likelihood and log gradient 
	 * @param random
	 * @param numPartitions
	 * @param numNodesPerPartition
	 * @param numFeatures
	 * @param param
	 * @param nodes
	 * @param instance
	 * @return
	 */
	public static Pair<Double, Counter<String>> generateData(Random random, int numPartitions, int numNodesPerPartition, int numFeatures, Counter<String> param, List<SimpleNode> nodes, List<Set<SimpleNode>> instance)
	{
		// randomly set the features
		double [] features = null;
		for (int pidx = 0; pidx < numPartitions; pidx++)
		{
			for (int idx = 0; idx < numNodesPerPartition; idx++)
			{
				System.out.print("[" + pidx + ", " + idx + "]: (");
				features = new double[numFeatures];
				for (int j = 0; j < numFeatures; j++)
				{
					features[j] = random.nextDouble();
					System.out.print(features[j] + " ");
				}
				System.out.println(")");
				SimpleNode node = new SimpleNode(pidx, idx, features);
				nodes.add(node);
			}
		}

		List<SimpleNode> p0 = new ArrayList<>();
		List<SimpleNode> p1 = new ArrayList<>();
		for (SimpleNode node : nodes)
		{
			if (node.getPartitionIdx() == 0) 
				p0.add(node);
			else if (node.getPartitionIdx() == 1)
				p1.add(node);
		}

		GraphFeatureExtractor<String, SimpleNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(p0.get(0));

		double expectedLogDensity = 0.0;
		Counter<String> expectedDeriv = new Counter<>();
		for (int idx = 0; idx < numNodesPerPartition; idx++)
		{
			Counter<String> num = new Counter<>();			

			Set<SimpleNode> e = new HashSet<>();
			e.add(p0.get(idx));
			e.add(p1.get(idx));
			instance.add(e);

			double logNorm = Double.NEGATIVE_INFINITY;
			for (int idx2 = idx; idx2 < numNodesPerPartition; idx2++)
			{
				Set<SimpleNode> d = new HashSet<>();
				d.add(p1.get(idx2));
				Counter<String> ff = fe.extractFeatures(p0.get(idx), d, null);

				double dotProduct = 0.0;
				for (String f : ff)
				{
					dotProduct += param.getCount(f) * ff.getCount(f);
					if (idx == idx2)
						expectedDeriv.incrementCount(f, ff.getCount(f));
				}
				if (idx == idx2) {
					expectedLogDensity += dotProduct;
				}
				logNorm = NumericalUtils.logAdd(logNorm, dotProduct);
				for (String f : ff)
				{
					num.incrementCount(f, Math.exp(dotProduct) * ff.getCount(f));
				}
			}
			expectedLogDensity -= logNorm;
			double den = Math.exp(logNorm);
			for (String f : expectedDeriv)
			{
				expectedDeriv.incrementCount(f, -num.getCount(f)/den);
			}
		}
		
		return Pair.create(expectedLogDensity, expectedDeriv);
	}

	public static double [] sampleParameters(Random random, int d, double mu, double var)
	{
		double [] w = new double[d];
		for (int n = 0; n < d; n++) {
			w[n] = Normal.generate(random, mu, var);
			System.out.println("w[" + n + "]: " + w[n]);
		}
		return w;
	}
	
	public static List<String> evaluateSurfaceOverRegularGrid(SupervisedLearning<String, SimpleNode> sl, Command<String, SimpleNode> command,  List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances, GraphFeatureExtractor<String, SimpleNode> fe, int gridSize)
	{
		double min = -3.0, max = 3.0;
		double stepSize = (max - min)/gridSize;
		double [] x = new double[2];
		List<String> lines = new ArrayList<>();
		for (int i = 0; i <= gridSize; i++)
		{
			//System.out.println("i: " + i);
			x[0] = min + stepSize * i;
			for (int j = 0; j <= gridSize; j++)
			{
				x[1] = min + stepSize * j;
				command.updateModelParameters(x);
				// evaluate
				double val = 0.0;
				for (Pair<List<Set<SimpleNode>>, List<SimpleNode>> instance : instances)
				{
					val += SupervisedLearning.value(command, command.getModelParameters(), instance).getFirst();
				}
				
				lines.add(x[0] + ", " + x[1] + ", " + -val);
			}
		}
		return lines;
	}
}
