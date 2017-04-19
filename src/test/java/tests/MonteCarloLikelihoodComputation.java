package tests;

import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.junit.Assert;
import org.apache.commons.math3.util.Pair;
import org.junit.Test;

import com.google.common.collect.Collections2;

import bayonet.math.NumericalUtils;
import briefj.collections.Counter;
import common.experiments.simulation.SimpleNode;
import common.experiments.simulation.SimulationUtils;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.DoubletonDecisionModel;

public class MonteCarloLikelihoodComputation 
{

	public static Random random = new Random(44290123);
	public static int numPartitions = 2;
	public static int numNodesPerPartition = 3;
	public static int numFeatures = 5;

	@Test
	public void randomizeSequenceTest()
	{
		// compute the likelihood when the true sequence is unknown -- for small problem, you can enumerate all possible node visitation sequence
		// compare to Monte Carlo averaging over the possible sequences (in this case, the number of samples in Monte Carlo > total number of sequences so should be able to get a decent estimate) 

		// generate nodes
		List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, 1.0);

		// set parameters
		CanonicalFeatureExtractor<String, SimpleNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
		Counter<String> params = fe.getDefaultParameters();
		for (String f : params)
		{
			double w = random.nextGaussian();
			System.out.println(w);
			params.setCount(f, w);
		}
		DecisionModel<String, SimpleNode> decisionModel = new DoubletonDecisionModel<>();
		Command<String, SimpleNode> command = new Command<>(decisionModel, fe, params);

		// generate matching
		GraphMatchingState<String, SimpleNode> initial = GraphMatchingState.getInitialState(nodes);
		while (initial.hasNextStep())
		{
			initial.sampleNextState(random, command, true, true);
		}
		double logLik1 = initial.getLogDensity();
		System.out.println("logLik1: " + logLik1);

		// now compute the likelihood 
		Pair<List<Set<SimpleNode>>, List<SimpleNode>> instance = Pair.create(initial.getMatchings(), nodes);
		double logLik2 = SupervisedLearning.value(command, params, instance).getFirst();
		System.out.println("logLik2: " + logLik2);
		//Assert.assertTrue(NumericalUtils.isClose(logLik1, logLik2, 1e-6));

		// enumerate over all nodes permutation
		double logLik3 = 0.0;
		Collection<List<SimpleNode>> permutations = Collections2.permutations(nodes);
		int n = permutations.size();
		for (List<SimpleNode> permutation : permutations)
		{
			instance = Pair.create(initial.getMatchings(), permutation);
			double logLik = SupervisedLearning.value(command, params, instance).getFirst();
			logLik3 += logLik;
		}
		System.out.println("logLik3: " + logLik3/n);

		int numSamples = 10000;
		long start = System.currentTimeMillis();
		Pair<Double, Counter<String>> ret1 = SupervisedLearning.valueRandomizedSequence(new Random(5), numSamples, command, instance, fe, params, false);
		double logLik4 = ret1.getFirst();
		long end = System.currentTimeMillis();
		System.out.println("Parallel computation: " + (end - start)/1000.0 + " seconds.");
		System.out.println("logLik4: " + logLik4/numSamples);
		double diff = Math.abs(logLik3/n -logLik4/numSamples);
		//System.out.println(diff);
		Assert.assertTrue(NumericalUtils.isClose(diff, 0.0, 1e-2));

		start = System.currentTimeMillis();
		Pair<Double, Counter<String>> ret2 = SupervisedLearning.valueRandomizedSequence(new Random(5), numSamples, command, instance, fe, params, true);
		logLik4 = ret2.getFirst();
		end = System.currentTimeMillis();
		System.out.println("Serial computation: " + (end - start)/1000.0 + " seconds.");
		System.out.println("logLik4: " + logLik4/numSamples);
		diff = Math.abs(logLik3/n -logLik4/numSamples);
		Assert.assertTrue(NumericalUtils.isClose(diff, 0.0, 1e-2));

		for (String f : ret1.getSecond())
		{
			System.out.println(ret1.getSecond().getCount(f) + ", " + ret2.getSecond().getCount(f));
			Assert.assertTrue(NumericalUtils.isClose(ret1.getSecond().getCount(f), ret2.getSecond().getCount(f), 1e-6));
		}
		
	}

}
