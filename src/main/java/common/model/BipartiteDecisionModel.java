package common.model;

import java.util.List;
import java.util.Set;

import common.graph.BipartiteMatchingState;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;

public class BipartiteDecisionModel<F, NodeType extends GraphNode<?>> implements DecisionModel<F, NodeType>
{

	@Override
  public List<Set<NodeType>> getDecisions(NodeType node,
  		GenericGraphMatchingState<F, NodeType> s) {
		
		if (s instanceof BipartiteMatchingState) {
			BipartiteMatchingState<F, NodeType> state = (BipartiteMatchingState<F, NodeType>)s;
			return getDecisions(node, state.getPartition2(), state.getCoveredNodes(), state.getMatchings(), state.getNode2EdgeView());
		}
	  
		throw new RuntimeException("BipartiteDecisionModel can only be used with BipartiteMatchingState.");
  }

}
