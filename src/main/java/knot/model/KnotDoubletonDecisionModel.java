package knot.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import knot.data.Knot;
import common.graph.GenericGraphMatchingState;
import common.model.DoubletonDecisionModel;

public class KnotDoubletonDecisionModel<NodeType extends Knot> extends DoubletonDecisionModel<String, NodeType>
{
	public static final double H_SPAN = 100.0;

	@Override
  public List<Set<NodeType>> getDecisions(NodeType node,
  		GenericGraphMatchingState<String, NodeType> state) {
		return getDecisions(node, state.getUnvisitedNodes(), state.getCoveredNodes(), state.getMatchings(), state.getNode2EdgeView());
  }

	@Override
  public List<Set<NodeType>> getDecisions(NodeType node,
      List<NodeType> unvisitedNodes, Set<NodeType> coveredNodes,
      List<Set<NodeType>> matching, Map<NodeType, Set<NodeType>> node2Matching) {

		List<Set<NodeType>> decisions = new ArrayList<>();
		
		if (coveredNodes.contains(node))
		{
			// this node is already covered, under this decision model, this node does not get to make any decision
			decisions.add(new HashSet<>());
			return decisions;
		}

		for (NodeType otherNode : unvisitedNodes)
		{
			 // Caution: this if statement is needed in case the GraphMatchingState or its subclass does not remove candidate nodes from its state (see: BipartiteMatchingState and GraphMatchingState)
			// under this decision model, covered nodes are not considered as a candidate
			if (coveredNodes.contains(otherNode))
				continue;

			if (node.getPartitionIdx() == otherNode.getPartitionIdx())
				continue;

			if (Math.abs(node.getNodeFeatures().getCount("x") - otherNode.getNodeFeatures().getCount("x")) > H_SPAN)
				continue;

			Set<NodeType> instance = new HashSet<>();
			instance.add(otherNode);
			decisions.add(instance);
		}

		if (decisions.size() == 0)
		{
			decisions.add(new HashSet<>());
		}
		return decisions;
  }

}
