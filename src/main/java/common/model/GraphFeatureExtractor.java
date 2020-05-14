package common.model;

import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;

public interface GraphFeatureExtractor<F, NodeType extends GraphNode<?>> 
{
	// extract features from the node, decision, and overall state of the matching
	public Counter<F> extractFeatures(NodeType node, Set<NodeType> decision, GenericGraphMatchingState<F, NodeType> matchingState);
	/*
	default Counter<F> extractFeatures(Set<NodeType> e, List<Set<NodeType>> matchingState) {
		throw new RuntimeException("Not implemented.");
	}
	*/
	public Counter<F> extractFeatures(Set<NodeType> e, List<Set<NodeType>> matchingState);
	public Counter<F> getDefaultParameters();
	public int dim();
	default void setStandardization(Counter<F> mean, Counter<F> sd)
	{
		
	}
}
