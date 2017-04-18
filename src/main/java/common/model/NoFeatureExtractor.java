package common.model;

import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;

public class NoFeatureExtractor<F, NodeType extends GraphNode<F>> implements GraphFeatureExtractor<F, NodeType> {

	@Override
  public Counter<F> extractFeatures(NodeType node, Set<NodeType> decision, GenericGraphMatchingState<F, NodeType> matchingState) {
	  return new Counter<>();
  }

	@Override
  public Counter<F> getDefaultParameters() {
	  return new Counter<>();
  }

	@Override
  public int dim() {
	  return 0;
  }

	@Override
	public Counter<F> extractFeatures(Set<NodeType> e, List<Set<NodeType>> matchingState) {
		return new Counter<>();
	}

}
