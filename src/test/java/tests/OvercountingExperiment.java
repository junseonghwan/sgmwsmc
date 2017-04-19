package tests;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.junit.Test;

import common.experiments.simulation.SimpleNode;
import common.experiments.simulation.SimulationUtils;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.SingletonExplicitDecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;

public class OvercountingExperiment 
{
	
	@Test
	public void test()
	{
		int numNodes = 4;
		Random random = new Random(1); 
		List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, 4, 1, 0, 0.0, 1.0);
		
		GraphMatchingState<String, SimpleNode> initial = GraphMatchingState.getInitialState(nodes);
		SingletonExplicitDecisionModel<String, SimpleNode> decisionModel = new SingletonExplicitDecisionModel<>();
		GraphFeatureExtractor<String, SimpleNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
		Command<String, SimpleNode> command = new Command<>(decisionModel, fe);
		LatentSimulator<GenericGraphMatchingState<String, SimpleNode>> transitionDensity = new GenericMatchingLatentSimulator<>(command, initial, true, true);
		ExactProposalObservationDensity<String, SimpleNode> observationDensity = new ExactProposalObservationDensity<>(command);
		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < numNodes; i++) emissions.add(null);
		SequentialGraphMatchingSampler<String, SimpleNode> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
		int numParticles = 1000;
		double logZ = smc.sample(numParticles, numParticles * 10);
		System.out.println("logZ: " + logZ);
		Map<GenericGraphMatchingState<String, SimpleNode>, Integer> map = new HashMap<>();
		for (GenericGraphMatchingState<String, SimpleNode> sample : smc.getSamples())
		{
			if (map.containsKey(sample)) {
				map.put(sample, map.get(sample) + 1);
			} else {
				map.put(sample, 1);
			}
		}
		System.out.println(map.keySet().size());
		for (GenericGraphMatchingState<String, SimpleNode> key : map.keySet())
		{
			System.out.println(map.get(key)/(double)numParticles);
			System.out.println(key);
		}

	}

}
