package tests;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.KnotDataReader;
import knot.data.RectangularKnot;
import knot.model.features.rectangular.DistanceSizeFeatureExtractor;

import org.apache.commons.math3.util.CombinatoricsUtils;
import org.apache.commons.math3.util.Pair;
import org.junit.Assert;
import org.junit.Test;

import bayonet.math.NumericalUtils;
import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingStateFactory;
import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.DoubletonDecisionModel;
import common.smc.DiscreteParticleFilter;
import common.smc.DiscreteParticleFilter.DiscreteLatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.GenericDiscreteLatentSimulator;

public class DiscreteParticleFilterTest 
{
	
	@Test
	public void uniformWeightTripartiteTest()
	{
		Random random = new Random(19850110); 
		int numPartition = 3;
		int numNodesPerPartition = 3;
		List<RectangularKnot> knots = KnotDataReader.simulateKnots(random, numPartition, numNodesPerPartition);

		DistanceSizeFeatureExtractor fe = new DistanceSizeFeatureExtractor();
		Counter<String> params = fe.getDefaultParameters();
		for (String f : params)
		{
			double w = 0.0;
			params.setCount(f, w);
		}

		DecisionModel<String, RectangularKnot> decisionModel = new DoubletonDecisionModel<>();
		Command<String, RectangularKnot> command = new Command<>(decisionModel, fe, params);
		
		GenericGraphMatchingState<String, RectangularKnot> initial = GraphMatchingStateFactory.createInitialGraphMatchingState(command, knots);
		DiscreteLatentSimulator<GenericGraphMatchingState<String, RectangularKnot>> transitionDensity = new GenericDiscreteLatentSimulator<>(command, initial, true);		
		ObservationDensity<GenericGraphMatchingState<String, RectangularKnot>, Object> observationDensity = new ObservationDensity<GenericGraphMatchingState<String,RectangularKnot>, Object>() {

			@Override
			public double logDensity(
					GenericGraphMatchingState<String, RectangularKnot> latent,
					Object emission) {
				return latent.getLogDensity();
			}

			@Override
			public double logWeightCorrection(
					GenericGraphMatchingState<String, RectangularKnot> curLatent,
					GenericGraphMatchingState<String, RectangularKnot> oldLatent) {
				return 0;
			}

			@Override
			public boolean cancellationApplied() {
				return true;
			}
		};

		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < knots.size(); i++) emissions.add(null);

 		DiscreteParticleFilter<GenericGraphMatchingState<String, RectangularKnot>, Object> dpf = new DiscreteParticleFilter<>(transitionDensity, observationDensity, emissions);
 		dpf.options.numberOfConcreteParticles = 1000;
 		double logZ = dpf.sample();
 		List<GenericGraphMatchingState<String, RectangularKnot>> samples = dpf.getPropagationResults().samples;
 		System.out.println("num particles: " + samples.size());

 		// check that all particles are unique
 		Set<GenericGraphMatchingState<String, RectangularKnot>> pmf = new HashSet<>(samples);
 		Assert.assertTrue(pmf.size() == samples.size());

 		double logSum = Double.NEGATIVE_INFINITY;
 		for (GenericGraphMatchingState<String, RectangularKnot> sample : samples)
 		{
 			double logWeight = observationDensity.logDensity(sample, null);
 			logSum = NumericalUtils.logAdd(logSum, logWeight);
 		}
 		// since num particles > num possible matchings, we should have the probability sum to 1.
 		System.out.println("logSum: " + logSum);
 		Assert.assertTrue(NumericalUtils.isClose(1.0, Math.exp(logSum), 1e-6));

 		for (GenericGraphMatchingState<String, RectangularKnot> sample : samples)
 		{
 			double normalizedWeight = Math.exp(observationDensity.logDensity(sample, null) - logSum);
 			Pair<List<Set<RectangularKnot>>, List<RectangularKnot>> instance = Pair.create(sample.getMatchings(), knots);
 	 		double truthNormalizedLikelihood = Math.exp(SupervisedLearning.value(command, params, instance).getFirst());
 	 		System.out.println(normalizedWeight + ", " + truthNormalizedLikelihood);
 	 		System.out.println(sample.toString());
 	 		Assert.assertTrue(NumericalUtils.isClose(normalizedWeight, truthNormalizedLikelihood, 1e-6));
 		}

 		System.out.println(logZ);

	}
	
	@Test
	public void uniformWeightBipartiteTest()
	{
		Random random = new Random(19850110); 
		int numPartition = 2;
		int numNodesPerPartition = 3;
		List<RectangularKnot> knots = KnotDataReader.simulateKnots(random, numPartition, numNodesPerPartition);

		DistanceSizeFeatureExtractor fe = new DistanceSizeFeatureExtractor();
		Counter<String> params = fe.getDefaultParameters();
		for (String f : params)
		{
			//double w = -random.nextDouble();
			double w = 0.0;
			params.setCount(f, w);
		}

		DecisionModel<String, RectangularKnot> decisionModel = new DoubletonDecisionModel<>();
		Command<String, RectangularKnot> command = new Command<>(decisionModel, fe, params);
		
		GenericGraphMatchingState<String, RectangularKnot> initial = GraphMatchingStateFactory.createInitialGraphMatchingState(command, knots);
		DiscreteLatentSimulator<GenericGraphMatchingState<String, RectangularKnot>> transitionDensity = new GenericDiscreteLatentSimulator<>(command, initial, true);		
		ObservationDensity<GenericGraphMatchingState<String, RectangularKnot>, Object> observationDensity = new ObservationDensity<GenericGraphMatchingState<String,RectangularKnot>, Object>() {

			@Override
			public double logDensity(
					GenericGraphMatchingState<String, RectangularKnot> latent,
					Object emission) {
				return latent.getLogDensity();
			}

			@Override
			public double logWeightCorrection(
					GenericGraphMatchingState<String, RectangularKnot> curLatent,
					GenericGraphMatchingState<String, RectangularKnot> oldLatent) {
				return 0;
			}

			@Override
			public boolean cancellationApplied() {
				return true;
			}
		};

		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < knots.size(); i++) emissions.add(null);

 		DiscreteParticleFilter<GenericGraphMatchingState<String, RectangularKnot>, Object> dpf = new DiscreteParticleFilter<>(transitionDensity, observationDensity, emissions);
 		dpf.options.numberOfConcreteParticles = 1000;
 		double logZ = dpf.sample();
 		List<GenericGraphMatchingState<String, RectangularKnot>> samples = dpf.getPropagationResults().samples;
 		System.out.println("num particles: " + samples.size());
 		// there are only 2 partitions, so we know exactly how many particles there are
 		if (dpf.options.numberOfConcreteParticles >= CombinatoricsUtils.factorial(numNodesPerPartition))
 			Assert.assertTrue(samples.size() == CombinatoricsUtils.factorial(numNodesPerPartition));

 		double logSum = Double.NEGATIVE_INFINITY;
 		for (GenericGraphMatchingState<String, RectangularKnot> sample : samples)
 		{
 			double logWeight = observationDensity.logDensity(sample, null);
 			logSum = NumericalUtils.logAdd(logSum, logWeight);
 		}
 		// since num particles > num possible matchings, we should have the probability sum to 1.
 		System.out.println("logSum: " + logSum);
 		Assert.assertTrue(NumericalUtils.isClose(1.0, Math.exp(logSum), 1e-6));

 		for (GenericGraphMatchingState<String, RectangularKnot> sample : samples)
 		{
 			double normalizedWeight = Math.exp(observationDensity.logDensity(sample, null) - logSum);
 			Pair<List<Set<RectangularKnot>>, List<RectangularKnot>> instance = Pair.create(sample.getMatchings(), knots);
 	 		double truthNormalizedLikelihood = Math.exp(SupervisedLearning.value(command, params, instance).getFirst());
 	 		System.out.println(normalizedWeight + ", " + truthNormalizedLikelihood);
 	 		System.out.println(sample.toString());
 	 		Assert.assertTrue(NumericalUtils.isClose(normalizedWeight, truthNormalizedLikelihood, 1e-6));
 		}

 		System.out.println(logZ);
	}

}
