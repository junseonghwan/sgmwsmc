package tests;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

import bayonet.math.NumericalUtils;
import briefj.collections.Counter;
import briefj.opt.Option;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.SingletonExplicitDecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import knot.data.KnotDataReader;
import knot.data.RectangularKnot;

public class UniformSingletonExplicityMatchingTest
{
	@Option
	public static int numConcreteParticles = 15000;
	@Option
	public static int numVirtualParticles = 100000;
	@Option
	public static boolean sequential = false;
	@Option
	public static int numPartitions = 4;
	@Option
	public static int numNodesPerPartitions = 1;

	@Test
	public void test()
	{
		Random random = new Random(1);

		List<RectangularKnot> knots = KnotDataReader.simulateKnots(random, numPartitions, numNodesPerPartitions);
		GraphFeatureExtractor<String, RectangularKnot> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(knots.get(0));
		Counter<String> params = fe.getDefaultParameters();
		for (String f : params)
		{
			params.setCount(f, 0.0); // uniform so all parameters are set to 0
		}
		DecisionModel<String, RectangularKnot> decisionModel = new SingletonExplicitDecisionModel<>();
		Command<String, RectangularKnot> command = new Command<>(decisionModel, fe, params);

		GraphMatchingState<String, RectangularKnot> initial = GraphMatchingState.getInitialState(knots);
		GenericMatchingLatentSimulator<String, RectangularKnot> transitionDensity = new GenericMatchingLatentSimulator<String, RectangularKnot>(command, initial, sequential, true);
		ObservationDensity<GenericGraphMatchingState<String, RectangularKnot>, Object> observationDensity = new ExactProposalObservationDensity<>(command);
		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < knots.size(); i++) emissions.add(null);

		SequentialGraphMatchingSampler<String, RectangularKnot> sgm = new SequentialGraphMatchingSampler<String, RectangularKnot>(transitionDensity, observationDensity, emissions);
		double logZ = sgm.sample(random, numConcreteParticles, numVirtualParticles);
		double Z = Math.exp(logZ);
		System.out.println(Z);
		//Assert.assertTrue(NumericalUtils.isClose(Z, 1, 10-6));

		// check number of distinct states as well as that uniform probability

		// check to see if SMC samples contains state and see if the estimate by the SMC is accurate
		Counter<GenericGraphMatchingState<String, RectangularKnot>> population = new Counter<>();
		for (GenericGraphMatchingState<String, RectangularKnot> sample : sgm.getSamples())
		{
			if (!population.containsKey(sample))
				System.out.println(sample);
			population.incrementCount(sample, 1.0);
		}

		System.out.println("num states=" + population.size());
		int expectedNumStates = 14;
		Assert.assertTrue(population.size() == expectedNumStates);
		
		System.out.println("Under uniform distribution, all state must have prob estimate: " + 1.0/expectedNumStates);
		for (GenericGraphMatchingState<String, RectangularKnot> state : population)
		{
			double p = (double)population.getCount(state)/numConcreteParticles;
			System.out.println("prob= " + p);
			System.out.println(state);
			Assert.assertTrue(NumericalUtils.isClose(p, 1.0/expectedNumStates, 0.015));
		}

	}
}
