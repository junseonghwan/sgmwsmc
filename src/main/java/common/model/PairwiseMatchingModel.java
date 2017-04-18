package common.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.graph.GraphNode;

public class PairwiseMatchingModel<F, NodeType extends GraphNode<?>> implements DecisionModel<F, NodeType> {

	public static int H_SPAN = 100;

	@Override
	public List<Set<NodeType>> getDecisions(NodeType node, GenericGraphMatchingState<F, NodeType> state) 
	{
		return getDecisions(node, state.getUnvisitedNodes(), state.getCoveredNodes(), state.getMatchings(), state.getNode2EdgeView());	
	}

	public boolean isCandidateEdge(NodeType node, Set<NodeType> edge)
	{
		for (NodeType v : edge)
		{
			if (v.getPartitionIdx() == node.getPartitionIdx())
				return false;
		}
		return true;
	}

	@Override
  public List<Set<NodeType>> getDecisions(NodeType node,
  		List<NodeType> unvisitedNodes, Set<NodeType> coveredNodes,
      List<Set<NodeType>> matching, Map<NodeType, Set<NodeType>> node2Edge) {

		List<Set<NodeType>> decisions = new ArrayList<>();

		if (coveredNodes.contains(node))
		{
			 // if the node is already covered, then the only decision that is available is an empty decision for now
			decisions.add(new HashSet<>()); 
			return decisions;
		} 
		else 
		{
  		// this node is not yet covered so consider all existing edges as long as the edge does contain a node from the same partition as the node being considered
  		for (Set<NodeType> edge : matching)
  		{
  			if (edge.size() >= 3) continue;
  			if (isCandidateEdge(node, edge))
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

		if (decisions.size() == 0)
			decisions.add(new HashSet<>()); // empty decision only if there is no other decisions available 

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
		int numSingletons = 0;
		for (Set<NodeType> e : state.getMatchings())
		{
			if (e.size() == 1) {
				numSingletons++;
			}
		}

		boolean singletonExists = numSingletons > 0; 
		numParents += numSingletons;
		for (Set<NodeType> e : state.getMatchings())
		{
			if (e.size() == 1) continue;

			int numVisited = 0;
			for (NodeType node : e)
			{
				if (state.getVisitedNodes().contains(node)) {
					numVisited++;
				}
			}

			if (e.size() == 2) {
				if (singletonExists) {
					if (numVisited == 1)
						return 1;
					else if (numVisited == 2)
						numParents += 2;
					else
						throw new RuntimeException();
				} else {
					if (numVisited == 1) {
						numParents += 1;
					} else if (numVisited == 2) {
						numParents += 2;
					} else
						throw new RuntimeException();
				}
			} else if (e.size() == 3) {
				if (singletonExists) {
					if (numVisited == 2)
						return numSingletons;
					else if (numVisited == 3)
						numParents += 3;
					else
						throw new RuntimeException();
				} else {
					if (numVisited == 2)
						numParents += 2;
					else if (numVisited == 3)
						numParents += 6;
					else
						throw new RuntimeException();
				}
			}
		}

		/*
		int numSingletons = 0;
		for (Set<NodeType> e : state.getMatchings())
		{
			if (e.size() == 1) {
				numSingletons++;
			}
		}

		boolean singletonExists = numSingletons > 0; 
		numParents += numSingletons;
		for (Set<NodeType> e : state.getMatchings())
		{
			if (e.size() == 1) continue;

			int numVisited = 0;
			for (NodeType node : e)
			{
				if (state.getVisitedNodes().contains(node)) {
					numVisited++;
				}
			}

			if (e.size() == 2) {
				if (singletonExists) {
					if (numVisited == 1)
						return 1;
					else if (numVisited == 2)
						numParents += 2;
					else
						throw new RuntimeException();
				} else {
					if (numVisited == 1) {
						numParents += 1;
					} else if (numVisited == 2) {
						numParents += 2;
					} else
						throw new RuntimeException();
				}
			} else if (e.size() == 3) {
				if (singletonExists) {
					if (numVisited == 2)
						return 1;
					else if (numVisited == 3)
						numParents += 3;
					else
						throw new RuntimeException();
				} else {
					if (numVisited == 2)
						numParents += 4;
					else if (numVisited == 3)
						numParents += 6;
					else
						throw new RuntimeException();
				}
			}
		}
		*/
		return numParents;
	}

}
