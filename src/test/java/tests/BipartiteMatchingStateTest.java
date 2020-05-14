package tests;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.Pair;
import org.junit.Assert;
import org.junit.Test;

import bayonet.math.NumericalUtils;
import briefj.collections.Counter;
import common.experiments.simulation.SimpleNode;
import common.experiments.simulation.SimulationUtils;
import common.graph.BipartiteMatchingState;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.BipartiteDecisionModel;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;

public class BipartiteMatchingStateTest
{
	
	@Test
	public void sampleBipartiteMatching()
	{
		int seed = 51235;
		Random random = new Random(seed);
		int numPartitions = 2;
		int numNodesPerPartition = 3;
		int numFeatures = 2;
		List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, 1.0);
		GraphFeatureExtractor<String, SimpleNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
		Command<String, SimpleNode> command = new Command<>(new BipartiteDecisionModel<>(), fe);
		BipartiteMatchingState<String, SimpleNode> bpState = BipartiteMatchingState.getInitial(nodes);
		double logLik1 = 0.0;
		while (bpState.hasNextStep())
		{
			logLik1 += bpState.sampleNextState(random, command, true, true);
		}
		System.out.println(bpState);
		
		random = new Random(seed);
		command = new Command<>(new DoubletonDecisionModel<>(), fe);
		GraphMatchingState<String, SimpleNode> gmState = GraphMatchingState.getInitialState(nodes);
		double logLik2 = 0.0;
		while (gmState.hasNextStep())
		{
			logLik2 += gmState.sampleNextState(random, command, true, true);
		}
		System.out.println(gmState);
		
		System.out.println(logLik1 + ", " + logLik2);
		Assert.assertTrue(NumericalUtils.isClose(logLik1, logLik2, 1e-6));
	}

	@Test
	public void testSMCSamplerUniformWeight()
	{
		int seed = 123;
		Random random = new Random(seed);
		int numPartitions = 2;
		int numNodesPerPartition = 3;
		int numFeatures = 2;
		List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, 1.0);
		GraphFeatureExtractor<String, SimpleNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
		Command<String, SimpleNode> command = new Command<>(new BipartiteDecisionModel<>(), fe);
		BipartiteMatchingState<String, SimpleNode> bpState = BipartiteMatchingState.getInitial(nodes);
		
		GenericMatchingLatentSimulator<String, SimpleNode> transitionDensity = new GenericMatchingLatentSimulator<>(command, bpState, true, true);
		ObservationDensity<GenericGraphMatchingState<String, SimpleNode>, Object> observationDensity = new ExactProposalObservationDensity<>(command);
		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < nodes.size(); i++) emissions.add(null);
		SequentialGraphMatchingSampler<String, SimpleNode> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
		int numSamples = 10000;
		double logZ = smc.sample(random, numSamples, numSamples);
		System.out.println(logZ);
		Assert.assertTrue(NumericalUtils.isClose(1.0, Math.exp(logZ), 1e-6));

		// check that the weights are uniform
		List<GenericGraphMatchingState<String, SimpleNode>> samples = smc.getSamples();
		Counter<Set<SimpleNode>> distribution = new Counter<>();
		for (GenericGraphMatchingState<String, SimpleNode> sample : samples)
		{
			List<Set<SimpleNode>> matchings = sample.getMatchings();
			for (Set<SimpleNode> edge : matchings)
			{
				distribution.incrementCount(edge, 1.0);
			}
		}
		
		for (Set<SimpleNode> edge : distribution)
		{
			System.out.println("{");
			for (SimpleNode knot : edge)
			{
				System.out.print(knot.toString());
			}
			double fraction = distribution.getCount(edge)/numSamples;
			System.out.println(", " + fraction);
			System.out.println("}");
			Assert.assertTrue(NumericalUtils.isClose(1/3.0, fraction, 0.01));
		}
	}
	
	@Test
	public void estimateParametersTest()
	{
		int seed = 123;
		Random random = new Random(seed);
		int numPartitions = 2;
		int numNodesPerPartition = 3;
		int numFeatures = 2;
		int numData = 100;
		int numReplicates = 100;
		
		double [] w = new double[numFeatures];
		for (int i = 0; i < numFeatures; i++)
			w[i] = random.nextGaussian();

		List<SummaryStatistics> summ = new ArrayList<>(numFeatures);
		for (int i = 0; i < numFeatures; i++) summ.add(new SummaryStatistics());
		for (int n = 0; n < numReplicates; n++)
		{
			Pair<Double, double[]> ret = estimateParameters(random, w, numPartitions, numNodesPerPartition, numFeatures, numData);
			for (int i = 0; i < numFeatures; i++)
				summ.get(i).addValue(ret.getSecond()[i]);
		}

		// build 95% CI baed on asymptotic properties of MLE
		for (int i = 0; i < numFeatures; i++) {
			double lower = summ.get(i).getMean() - 1.96 * summ.get(i).getStandardDeviation()/Math.sqrt(numReplicates);
			double upper = summ.get(i).getMean() + 1.96 * summ.get(i).getStandardDeviation()/Math.sqrt(numReplicates);
			System.out.println("mean: " + summ.get(i).getMean() + ", sd: " + summ.get(i).getStandardDeviation());
			System.out.println(w[i] + " in (" + lower + ", " + upper + ")? " + (w[i] > lower && w[i] < upper));
			Assert.assertTrue((w[i] >= lower && w[i] <= upper));
		}

	}

	private Pair<Double, double[]> estimateParameters(Random random, double [] w, int numPartitions, int numNodesPerPartition, int numFeatures, int numData)
	{
		List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances = new ArrayList<>();
		GraphFeatureExtractor<String, SimpleNode> fe = null;
		Command<String, SimpleNode> command = null;
		for (int n = 0; n < numData; n++)
		{
			List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, 1.0);
			if (command == null)
			{
				fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
				command = new Command<>(new BipartiteDecisionModel<>(), fe);
				command.updateModelParameters(w);
			}

			BipartiteMatchingState<String, SimpleNode> bpState = BipartiteMatchingState.getInitial(nodes);
			// sample the final matching
			while (bpState.hasNextStep())
			{
				bpState.sampleNextState(random, command, true, true);
			}
			
			instances.add(Pair.create(bpState.getMatchings(), nodes));
		}
		
		// run parameter estimation
		SupervisedLearning<String, SimpleNode> sl = new SupervisedLearning<>();
		Pair<Double, double[]> ret = sl.MAP(command, instances, 0.0, new double[fe.dim()], 1e-6, true);
		System.out.println("logLik= " + ret.getFirst());
		System.out.println("Truth, Estimate");
		for (int i = 0; i < ret.getSecond().length; i++)
		{
			System.out.println(w[i] + ", " + ret.getSecond()[i]);
		}
		return ret;
	}
	
}
