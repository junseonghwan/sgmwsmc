package tests;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import knot.data.KnotDataReader;
import knot.data.RectangularKnot;

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
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;

public class NonUniformBipartiteGraphTest 
{
	@Option
	public static int numConcreteParticles = 10000;
	@Option
	public static int numVirtualParticles = 10000;
	@Option
	public static boolean sequential = true;
	@Option
	public static int numPartitions = 2;
	@Option
	public static int numNodesPerPartitions = 3;
	
	@Test
	public void test()
	{
		Random random = new Random(1245141);
		
		List<RectangularKnot> knots = KnotDataReader.simulateKnots(random, numPartitions, numNodesPerPartitions);
		GraphFeatureExtractor<String, RectangularKnot> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(knots.get(0));
		Counter<String> params = fe.getDefaultParameters();
		for (String f : params)
		{
			params.setCount(f, 0.1); // generate the parameters
		}
		DecisionModel<String, RectangularKnot> decisionModel = new DoubletonDecisionModel<>();
		Command<String, RectangularKnot> command = new Command<>(decisionModel, fe, params);

		GraphMatchingState<String, RectangularKnot> initial = GraphMatchingState.getInitialState(knots);
		GenericMatchingLatentSimulator<String, RectangularKnot> transitionDensity = new GenericMatchingLatentSimulator<String, RectangularKnot>(command, initial, true, true);
		ObservationDensity<GenericGraphMatchingState<String, RectangularKnot>, Object> observationDensity = new ExactProposalObservationDensity<>(command);
		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < knots.size(); i++) emissions.add(null);

		SequentialGraphMatchingSampler<String, RectangularKnot> sgm = new SequentialGraphMatchingSampler<String, RectangularKnot>(transitionDensity, observationDensity, emissions);
		double logZ = sgm.sample(random, numConcreteParticles, numVirtualParticles);
		double Z = Math.exp(logZ);
		System.out.println(Z);
		Assert.assertTrue(NumericalUtils.isClose(Z, 1, 10-6));

		// for one specific matching, compute its probability
		// check to see if SMC approximates this probability accurately
		GraphMatchingState<String, RectangularKnot> state = GraphMatchingState.getInitialState(knots);
		double sumLogProb = 0.0; 
		for (int i = 0; i < knots.size(); i++)
		{
			sumLogProb += state.sampleNextState(random, command, sequential, true);
		}

		System.out.println("target state: " + sumLogProb);
		System.out.println(state.toString());
		System.out.println("process samples...");
		
		// check to see if SMC samples contains state and see if the estimate by the SMC is accurate
		Counter<GenericGraphMatchingState<String, RectangularKnot>> population = new Counter<>();
		for (GenericGraphMatchingState<String, RectangularKnot> sample : sgm.getSamples())
		{
			if (!population.containsKey(sample))
				System.out.println(sample);
			population.incrementCount(sample, 1.0);
		}

		double diff = population.getCount(state)/numConcreteParticles - Math.exp(sumLogProb);
		System.out.println(diff);
		Assert.assertTrue(NumericalUtils.isClose(diff, 0, 0.01));
	}
}
