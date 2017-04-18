package common.model;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;

public class MultinomialLogisticModel<F, NodeType extends GraphNode<?>> 
{
	private GraphFeatureExtractor<F, NodeType> fe;
	private Counter<F> params;

	public MultinomialLogisticModel(GraphFeatureExtractor<F, NodeType> fe, Counter<F> params)
	{
		if (params.size() != fe.dim()) throw new RuntimeException("Feature and parameter dimension do not match.");

		this.fe = fe;
		this.params = params;
	}

	// compute the log probability and return the features
	public Pair<Double, Counter<F>> logProb(NodeType node, Set<NodeType> decision, GenericGraphMatchingState<F, NodeType> matchingState)
	{
		Counter<F> features = fe.extractFeatures(node, decision, matchingState);
		double logProb = 0.0;
		for (F f : features)
		{
			logProb += features.getCount(f) * params.getCount(f);
		}
		return Pair.create(logProb, features);
	}

	public Pair<Double, Counter<F>> logProb(NodeType node, Set<NodeType> decision, List<Set<NodeType>> matchingState)
	{
		Set<NodeType> e = new HashSet<>(decision);
		e.add(node);
		Counter<F> features = fe.extractFeatures(e, matchingState);
		double logProb = 0.0;
		for (F f : features)
		{
			logProb += features.getCount(f) * params.getCount(f);
		}
		return Pair.create(logProb, features);
	}

	public int numVariables()
	{
		return fe.dim();
	}
	
}
