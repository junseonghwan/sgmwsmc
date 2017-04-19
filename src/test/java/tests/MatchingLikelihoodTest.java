package tests;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;
import org.junit.Assert;
import org.junit.Test;

import bayonet.math.NumericalUtils;
import briefj.Indexer;
import briefj.collections.Counter;
import common.experiments.simulation.SimpleNode;
import common.experiments.simulation.SimulationUtils;
import common.learning.SupervisedLearning;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;

public class MatchingLikelihoodTest 
{
	
	@Test
	public void test()
	{
		// test that the code for evaluating the likelihood and the derivative works
		Random random = new Random(5132);
		int numFeatures = 3;
		int numPartitions = 2;
		int numNodesPerPartition = 3;
		List<SimpleNode> nodes = new ArrayList<>();
		List<Set<SimpleNode>> instance = new ArrayList<>();
		
		Counter<String> w = new Counter<>();
		for (int i = 0; i < numFeatures; i++)
		{
			w.setCount(i + "", random.nextGaussian() * 10);
		}

		Pair<Double, Counter<String>> expectedDeriv = SimulationUtils.generateData(random, numPartitions, numNodesPerPartition, numFeatures, w, nodes, instance);
		GraphFeatureExtractor<String, SimpleNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));

		List<SimpleNode> permutation = new ArrayList<>();
		for (int i = 0; i < nodes.size(); i++)
		{
			permutation.add(nodes.get(i));
		}

		DecisionModel<String, SimpleNode> decisionModel = new DoubletonDecisionModel<>();
		Command<String, SimpleNode> command = new Command<>(decisionModel, fe, w);

		Pair<Double, Counter<String>> actualRet = SupervisedLearning.value(command, w, Pair.create(instance, permutation));

		double actualLogDensity = actualRet.getFirst();
		System.out.println(" Expected: " + expectedDeriv.getFirst() + "\n Actual: " + actualLogDensity + "\n =====");
		Assert.assertTrue(NumericalUtils.isClose(actualLogDensity, expectedDeriv.getFirst(), 1e-3));

		Indexer<String> indexer = new Indexer<>();
		Counter<String> actualDeriv = actualRet.getSecond();
		indexer.addAllToIndex(actualDeriv.keySet());
		for (String f : actualDeriv)
		{
			System.out.println(" Expected: " + expectedDeriv.getSecond().getCount(f) + "\n Actual:" + actualDeriv.getCount(f));
			Assert.assertTrue(NumericalUtils.isClose(actualDeriv.getCount(f), expectedDeriv.getSecond().getCount(f), 1e-3));
		}
	}
		
}
