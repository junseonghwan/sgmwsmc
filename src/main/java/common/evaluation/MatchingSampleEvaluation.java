package common.evaluation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.math3.util.Pair;

import com.google.common.collect.Sets;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;

public class MatchingSampleEvaluation<F, NodeType extends GraphNode<?>>
{
	public Pair<GenericGraphMatchingState<F, NodeType>, Pair<Double, Integer>> bestLogLikMatching;
	public Pair<GenericGraphMatchingState<F, NodeType>, Integer> bestAccuracyMatching;
	public Pair<List<Set<NodeType>>, Integer> consensusMatching;
	public double avgAccuracy;
	public double avgJaccardIndex;
	public int cardinality;
	public static boolean verbose = true;
	
	private MatchingSampleEvaluation() { }
	
	public static <NodeType extends GraphNode<?>> List<NodeType> getNodes(List<Set<NodeType>> truth)
	{
		List<NodeType> nodes = new ArrayList<>();
		for (Set<NodeType> e : truth)
		{
			nodes.addAll(e);
		}
		return nodes;
	}
	
	public static <F, NodeType extends GraphNode<?>> MatchingSampleEvaluation<F, NodeType> evaluate(List<GenericGraphMatchingState<F, NodeType>> samples, List<Set<NodeType>> truth)
	{
		if (!validateMatchingSamples(samples, getNodes(truth)))
			throw new RuntimeException("SMC has bug.");
		
		List<NodeType> nodes = getNodes(truth);

		MatchingSampleEvaluation<F, NodeType> eval = new MatchingSampleEvaluation<>();

		double sum = 0.0;
		int bestAccuracy = 0;
		double bestLogLik = Double.NEGATIVE_INFINITY;
		for (GenericGraphMatchingState<F, NodeType> sample : samples)
		{
			int accuracy = computeAccuracy(sample, truth);
			sum += accuracy;

			if (accuracy > bestAccuracy)
			{
				bestAccuracy = accuracy;
				eval.bestAccuracyMatching = Pair.create(sample, bestAccuracy);
				if (verbose) {
  				System.out.println("new best accuracy: " + accuracy + "\n logDensity@new best: " + sample.getLogDensity());
  				System.out.println(sample.toString());
				}
			}
			if (sample.getLogDensity() > bestLogLik) {
				bestLogLik = sample.getLogDensity();
				eval.bestLogLikMatching = Pair.create(sample, Pair.create(bestLogLik, accuracy));
				if (verbose) {
  				System.out.println("new best loglik: " + sample.getLogDensity() + "\n accuracy@new best loglik: " + accuracy);
  				System.out.println(sample.toString());
				}
			}
		}
		
		eval.avgAccuracy = sum / samples.size();

		List<Set<NodeType>> consensus = findConsensusMatching(samples);
		int accuracy = 0;
		for (Set<NodeType> eTrue : truth)
		{
			for (Set<NodeType> eSampled : consensus)
			{
				if (eSampled.containsAll(eTrue) && eTrue.containsAll(eSampled)) {
					accuracy += 1;
					break;
				}
			}
		}
		eval.consensusMatching = Pair.create(consensus, accuracy);
		eval.cardinality = truth.size();
		
		// compute the Jaccard index
		eval.avgJaccardIndex = jaccardIndex(nodes, truth, samples);
		
		return eval;
	}
	
	public static <F, NodeType extends GraphNode<?>> int computeAccuracy(GenericGraphMatchingState<F, NodeType> sample, List<Set<NodeType>> truth)
	{
		List<Set<NodeType>> sampledMatching = sample.getMatchings();
		int accuracy = 0;
		for (Set<NodeType> eTrue : truth)
		{
			for (Set<NodeType> eSampled : sampledMatching)
			{
				if (eSampled.containsAll(eTrue) && eTrue.containsAll(eSampled)) {
					accuracy += 1;
					break;
				}
			}
		}
		return accuracy;
	}
	
	public static <F, NodeType extends GraphNode<?>> List<Set<NodeType>> findConsensusMatching(List<GenericGraphMatchingState<F, NodeType>> listOfMatchings)
	{
		// get all nodes
		List<NodeType> nodes = listOfMatchings.get(0).getVisitedNodes();
		
		Map<Set<NodeType>, Integer> edgeCounts = new HashMap<>();
		
		for (GenericGraphMatchingState<F, NodeType> state : listOfMatchings)
		{
			for (Set<NodeType> edge : state.getMatchings())
			{
				if (!edgeCounts.containsKey(edge))
					edgeCounts.put(edge, 0);
				
				edgeCounts.put(edge, edgeCounts.get(edge) + 1);
			}
		}

		List<Map.Entry<Set<NodeType>, Integer>> list = new ArrayList<>(edgeCounts.entrySet());
		Collections.sort(list, new Comparator<Map.Entry<Set<NodeType>, Integer>>() {

			@Override
			public int compare(Entry<Set<NodeType>, Integer> o1, Entry<Set<NodeType>, Integer> o2) {
				if (o1.getValue() < o2.getValue())
					return 1;
				else if (o1.getValue() > o2.getValue())
					return -1;
				return 0;
			}
		});

		Set<Set<NodeType>> matching = new HashSet<>();
		Set<NodeType> coveredNodes = new HashSet<>();
		for (Map.Entry<Set<NodeType>, Integer> entry : list)
		{
			boolean exists = false;
			for (Set<NodeType> e : matching)
			{
				if (!Sets.intersection(e, entry.getKey()).isEmpty()) {
					exists = true;
				}
			}
			if (!exists) {
				matching.add(entry.getKey());
				coveredNodes.addAll(entry.getKey());
			}
		}
		
		for (NodeType node : nodes)
		{
			if (!coveredNodes.contains(node)) {
				// create a singleton
				Set<NodeType> e = new HashSet<>();
				e.add(node);
				matching.add(e);
			}
		}

		return new ArrayList<>(matching);
	}

	/**
	 * Check for any violation in the matching samples
	 * 
	 * @param samples
	 * @return
	 */
	public static <F, NodeType extends GraphNode<?>> boolean validateMatchingSamples(List<GenericGraphMatchingState<F, NodeType>> samples, List<NodeType> nodes)
	{
		for (GenericGraphMatchingState<F, NodeType> sample : samples)
		{
			try {
				sample.getNode2EdgeView(); // this will actually throw an exception if there is an invalid matching
			} catch (RuntimeException ex) {
				System.out.println(sample);
				return false;
			}

			// also, check that all nodes are covered
			for (NodeType node : nodes)
			{
				if (!sample.covers(node)) {
					System.out.println(sample);
					return false;
				}
			}
		}

		return true;
	}

	public static <F, NodeType extends GraphNode<?>> double jaccardIndex(List<NodeType> nodes, List<Set<NodeType>> truth, List<GenericGraphMatchingState<F, NodeType>> samples) {
		double sum = 0.0;
		Set<GenericGraphMatchingState<F, NodeType>> diversity = new HashSet<>();
		for (GenericGraphMatchingState<F, NodeType> sample : samples)
		{
			diversity.add(sample);
			double val = 0.0;
			for (NodeType node : nodes) 
			{
				// retrieve the edge that contains this node from the truth
				Set<NodeType> a = getEdgeContainingNode(node, truth);
				// retrieve the edge that contains this node from the sample
				Set<NodeType> b = getEdgeContainingNode(node, sample.getMatchings());
				// find the symmetric difference between a and b
				double intersectionSize = Sets.intersection(a, b).size();
				double unionSize = Sets.union(a, b).size();
				val += intersectionSize/unionSize;
			}
			//val /= nodes.size();
			sum += val;
		}
		
		//System.out.println("num unique particles=" + diversity.size());
		return sum/samples.size();
	}
	
	public static <F, NodeType extends GraphNode<?>> double jaccardIndex(List<NodeType> nodes, List<Set<NodeType>> truth, GenericGraphMatchingState<F, NodeType> sample) {
		double val = 0.0;
		for (NodeType node : nodes) 
		{
			// retrieve the edge that contains this node from the truth
			Set<NodeType> a = getEdgeContainingNode(node, truth);
			// retrieve the edge that contains this node from the sample
			Set<NodeType> b = getEdgeContainingNode(node, sample.getMatchings());
			// find the symmetric difference between a and b
			double intersectionSize = Sets.intersection(a, b).size();
			double unionSize = Sets.union(a, b).size();
			val += intersectionSize/unionSize;
		}
		return val;
	}

	public static <F, NodeType extends GraphNode<?>> Set<NodeType> getEdgeContainingNode(NodeType v, List<Set<NodeType>> matching) {
		for (Set<NodeType> e : matching) {
			if (e.contains(v)) {
				return e;
			}
		}
		return null; // an edge containing v does not exist in the provided matching
	}
}
