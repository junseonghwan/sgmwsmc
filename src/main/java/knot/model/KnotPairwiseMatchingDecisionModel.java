package knot.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import common.graph.GenericGraphMatchingState;
import common.model.PairwiseMatchingModel;
import knot.data.EllipticalKnot;

public class KnotPairwiseMatchingDecisionModel extends PairwiseMatchingModel<String, EllipticalKnot> {

	@Override
	public boolean isCandidateEdge(EllipticalKnot node, Set<EllipticalKnot> edge)
	{
		//boolean sharesAxis = false;
		for (EllipticalKnot v : edge)
		{
			if (v.getPartitionIdx() == node.getPartitionIdx())
				return false;
			/*
			if (!sharesAxis && ThreeMatchingFeatureExtractor.sharesAxis(node, v) == 1)
				sharesAxis = true;
				*/
		}
		//return sharesAxis;
		return true;
	}

	@Override
  public List<Set<EllipticalKnot>> getDecisions(EllipticalKnot node,
  		List<EllipticalKnot> unvisitedNodes, Set<EllipticalKnot> coveredNodes,
      List<Set<EllipticalKnot>> matching, Map<EllipticalKnot, Set<EllipticalKnot>> node2Edge) {

		List<Set<EllipticalKnot>> decisions = new ArrayList<>();

		if (coveredNodes.contains(node))
		{
			 // if the node is already covered, then the only decision that is available is an empty decision for now
			decisions.add(new HashSet<>()); 
			return decisions;
		} 
		else 
		{
  		// this node is not yet covered so consider all existing edges as long as the edge does contain a node from the same partition as the node being considered
			// three matching is only considered if and only if this node and one of the nodes in the candidate edge shares an axis
  		for (Set<EllipticalKnot> edge : matching)
  		{
  			if (edge.size() >= 3) continue;
  			if (isCandidateEdge(node, edge))
  				decisions.add(edge);
  		}
		}

		for (EllipticalKnot otherNode : unvisitedNodes)
		{
			if (coveredNodes.contains(otherNode)) // covered nodes are already handled by the above code for adding the edges
				continue;
			if (node.getPartitionIdx() == otherNode.getPartitionIdx())
				continue;

			Set<EllipticalKnot> instance = new HashSet<>();
			instance.add(otherNode);
			decisions.add(instance);
		}

		if (decisions.size() == 0)
			decisions.add(new HashSet<>()); // empty decision only if there is no other decisions available 

	  return decisions;
  }

	@Override
	public int numParents(GenericGraphMatchingState<String, EllipticalKnot> state)
	{
		int numParents = 0;
		int numSingletons = 0;
		for (Set<EllipticalKnot> e : state.getMatchings())
		{
			if (e.size() == 1) {
				numSingletons++;
			}
		}

		boolean singletonExists = numSingletons > 0; 
		numParents += numSingletons;
		for (Set<EllipticalKnot> e : state.getMatchings())
		{
			if (e.size() == 1) continue;

			int numVisited = 0;
			for (EllipticalKnot node : e)
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

		return numParents;
	}

}
