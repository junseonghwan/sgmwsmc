package tests;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.KnotDataReader;
import knot.data.RectangularKnot;

import org.junit.Assert;
import org.junit.Test;

import bayonet.math.NumericalUtils;
import briefj.collections.Counter;
import briefj.opt.Option;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.Command;
import common.model.DecisionModel;
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.NoFeatureExtractor;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;

// edges have same weight
public class DoubletonMatchingUniformWeightTest 
{
	@Option
	public static int numConcreteParticles = 10000;
	@Option
	public static int numVirtualParticles = 10000;
	@Option
	public static Random random = new Random(1);
	@Option
	public static int LUMBER_FOLDER = 8;

	@Test
	public void test()
	{
		List<RectangularKnot> knots = KnotDataReader.simulateKnots(random, 4, 1);

		GraphFeatureExtractor<String, RectangularKnot> fe = new NoFeatureExtractor<>();
		DecisionModel<String, RectangularKnot> decisionModel = new DoubletonDecisionModel<>();
		GenericGraphMatchingState<String, RectangularKnot> initial = GraphMatchingState.getInitialState(knots);
		Command<String, RectangularKnot> command = new Command<>(decisionModel, fe);
		LatentSimulator<GenericGraphMatchingState<String, RectangularKnot>> transitionDensity = new GenericMatchingLatentSimulator<String, RectangularKnot>(command, initial, true, true);
		ObservationDensity<GenericGraphMatchingState<String, RectangularKnot>, Object> observationDensity = new ObservationDensity<GenericGraphMatchingState<String, RectangularKnot>, Object>() {

			@Override
      public double logDensity(GenericGraphMatchingState<String, RectangularKnot> latent, Object emission) {
	      return 0.0;
      }

			@Override
      public double logWeightCorrection(GenericGraphMatchingState<String, RectangularKnot> curLatent, GenericGraphMatchingState<String, RectangularKnot> oldLatent) {
	      return 0;
      }

			@Override
      public boolean cancellationApplied() {
	      return true;
      }
		};

		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < knots.size(); i++) emissions.add(null);
		
		SequentialGraphMatchingSampler<String, RectangularKnot> sgm = new SequentialGraphMatchingSampler<String, RectangularKnot>(transitionDensity, observationDensity, emissions);
		double logZ = sgm.sample(numConcreteParticles, numVirtualParticles);
		System.out.println(logZ);
		double Z = Math.exp(logZ);
		System.out.println(Z);
		Assert.assertTrue(NumericalUtils.isClose(Z, 1, 10-6));

		List<GenericGraphMatchingState<String, RectangularKnot>> samples = sgm.getSamples();
		Counter<Set<RectangularKnot>> distribution = new Counter<>();
		for (GenericGraphMatchingState<String, RectangularKnot> sample : samples)
		{
			List<Set<RectangularKnot>> matchings = sample.getMatchings();
			for (Set<RectangularKnot> edge : matchings)
			{
				distribution.incrementCount(edge, 1.0);
			}
		}
		
		for (Set<RectangularKnot> edge : distribution)
		{
			System.out.println("{");
			for (RectangularKnot knot : edge)
			{
				System.out.print(knot.toString());
			}
			double fraction = distribution.getCount(edge)/numConcreteParticles;
			System.out.println(", " + fraction);
			System.out.println("}");
			Assert.assertTrue(NumericalUtils.isClose(1/3.0, fraction, 0.01));
		}
	}
}
