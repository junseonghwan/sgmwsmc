package common.model;

import java.util.List;
import java.util.Set;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;

/**
 * Generic pairwise decision model that only considers uncovered nodes in the current state as possible decisions
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 * @param <NodeType>
 * @param <F>
 */
public class DoubletonDecisionModel<F, NodeType extends GraphNode<?>> implements DecisionModel<F, NodeType> 
{
	
	@Override
  public List<Set<NodeType>> getDecisions(NodeType node, GenericGraphMatchingState<F, NodeType> state) {
		return getDecisions(node, state.getUnvisitedNodes(), state.getCoveredNodes(), state.getMatchings(), state.getNode2EdgeView());
	}

	@Override
	public boolean inSupport(GenericGraphMatchingState<F, NodeType> state) {
		return true;
	}
	
}
