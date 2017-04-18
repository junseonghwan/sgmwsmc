package common.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.graph.GraphNode;

public class SingletonExplicitDecisionModel<F, NodeType extends GraphNode<?>> implements DecisionModel<F, NodeType>
{
	@Override
	public List<Set<NodeType>> getDecisions(NodeType node, GenericGraphMatchingState<F, NodeType> state) {
		return getDecisions(node, state.getUnvisitedNodes(), state.getCoveredNodes(), state.getMatchings(), state.getNode2EdgeView());
	}

	@Override
  public List<Set<NodeType>> getDecisions(NodeType node,
  		List<NodeType> unvisitedNodes, Set<NodeType> coveredNodes,
      List<Set<NodeType>> matching, Map<NodeType, Set<NodeType>> node2Edge) {

		List<Set<NodeType>> decisions = new ArrayList<>();
		decisions.add(new HashSet<>()); // empty decision == no decision/singleton 

		if (node2Edge.containsKey(node)) {
			if (node2Edge.get(node).size() >= 3) return decisions; // maximum number of nodes in a matching is 3 -- no decision to be made
		}

		if (!coveredNodes.contains(node))
		{
	  		// consider all existing matching if the node is not covered
	  		for (Set<NodeType> edge : matching)
	  		{
	  			if (edge.size() >= 3) continue;
	  			decisions.add(edge);
	  		}
		}

		for (NodeType otherNode : unvisitedNodes)
		{
			if (coveredNodes.contains(otherNode)) // covered nodes are already handled by the above code for adding the edges
				continue;
			if (node.getPartitionIdx() == otherNode.getPartitionIdx())
				continue;

			Set<NodeType> instance = new HashSet<>();
			instance.add(otherNode);
			decisions.add(instance);
		}

	  return decisions;
  }

	@Override
	public boolean pathExists(GraphMatchingState<F, NodeType> currState, Map<NodeType, Set<NodeType>> finalState) 
	{
		Set<NodeType> visitedNodes = new HashSet<>(currState.getVisitedNodes());
		for (Set<NodeType> e : finalState.values())
		{
			Set<Set<NodeType>> edgesFormed = new HashSet<>();
			for (NodeType v : e)
			{
				if (!visitedNodes.contains(v)) continue;

				Set<NodeType> eprime = currState.getNode2EdgeView().get(v);
				edgesFormed.add(eprime);
				if (edgesFormed.size() > 1)
					return false; // if two nodes are in separate edges, they cannot be merged under this decision model
				if (!e.containsAll(eprime))
					return false;
			}
		}
		return true;
	}
	
	@Override
	public int numParents(GenericGraphMatchingState<F, NodeType> state)
	{
		int numParents = 0;
		//System.out.println(curLatent.toString());
		for (Set<NodeType> e : state.getMatchings())
		{
			int numVisited = 0;
			for (NodeType node : e)
			{
				if (state.getVisitedNodesAsSet().contains(node)) {
					numVisited++;
				}
			}

			if (e.size() == 1) {
				numParents += 1;
			} else if (e.size() == 2) {
				if (numVisited == 1)
					numParents += 1;
				else if (numVisited == 2)
					numParents += 4;
			} else if (e.size() == 3) {
				if (numVisited == 1)
					throw new RuntimeException();
				else if (numVisited == 2)
					numParents += 4;
				else if (numVisited == 3)
					numParents += 6;
			}
		}
		return numParents;
	}
}
