package common.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.graph.GraphNode;

public interface DecisionModel<F, NodeType extends GraphNode<?>> 
{
	public List<Set<NodeType>> getDecisions(NodeType node, GenericGraphMatchingState<F, NodeType> state);
	
	default List<Set<NodeType>> getDecisions(NodeType node, List<NodeType> candidateNodes, Set<NodeType> coveredNodes, List<Set<NodeType>> matching, Map<NodeType, Set<NodeType>> node2Edge)
	{
		List<Set<NodeType>> decisions = new ArrayList<>();
		
		if (coveredNodes.contains(node))
		{
			// this node is already covered, under this decision model, this node does not get to make any decision
			decisions.add(new HashSet<>());
			return decisions;
		}

		for (NodeType otherNode : candidateNodes)
		{
			 // Caution: this if statement is needed in case the GraphMatchingState or its subclass does not remove candidate nodes from its state (see: BipartiteMatchingState and GraphMatchingState)
			// under this decision model, covered nodes are not considered as a candidate
			if (coveredNodes.contains(otherNode))
				continue;

			if (node.getPartitionIdx() == otherNode.getPartitionIdx())
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
	
	default boolean inSupport(GenericGraphMatchingState<F, NodeType> state)
	{
		return true;
	}

	default boolean pathExists(GraphMatchingState<F, NodeType> currState, Map<NodeType, Set<NodeType>> finalState)
	{
		return true;
	}
	
	default int numParents(GenericGraphMatchingState<F, NodeType> curLatent)
	{
		return 1;
	}
}
