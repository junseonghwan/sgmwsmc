package knot.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.DecisionModel;
import knot.data.EllipticalKnot;

// (Experimental code) allow for bin packing point of view of knot matching.
// Note: Poor performance.
public class SingletonExplicitKnotMatchingDecisionModel implements DecisionModel<String, EllipticalKnot>
{
	public static int H_SPAN = 100;

	@Override
	public List<Set<EllipticalKnot>> getDecisions(EllipticalKnot node, GenericGraphMatchingState<String, EllipticalKnot> state) {
		return getDecisions(node, state.getUnvisitedNodes(), state.getCoveredNodes(), state.getMatchings(), state.getNode2EdgeView());
	}

	public boolean isCandidateEdge(EllipticalKnot node, Set<EllipticalKnot> edge)
	{
		for (EllipticalKnot v : edge)
		{
			if (Math.abs(v.getNodeFeatures().getCount("x") - node.getNodeFeatures().getCount("x")) > H_SPAN)
				return false;

			if (v.getPartitionIdx() == node.getPartitionIdx())
				return false;
		}
		return true;
	}

	@Override
  public List<Set<EllipticalKnot>> getDecisions(EllipticalKnot node,
      List<EllipticalKnot> unvisitedNodes, Set<EllipticalKnot> coveredNodes,
      List<Set<EllipticalKnot>> matching, Map<EllipticalKnot, Set<EllipticalKnot>> node2Edge) {

		List<Set<EllipticalKnot>> decisions = new ArrayList<>();
		decisions.add(new HashSet<>()); // empty decision == no decision/singleton 

		if (node2Edge.containsKey(node)) {
			if (node2Edge.get(node).size() >= 3) return decisions; // maximum number of nodes in a matching is 3 -- no decision to be made
		}

		if (!coveredNodes.contains(node))
		{
	  		// consider all existing matching if the node is not covered
	  		for (Set<EllipticalKnot> edge : matching)
	  		{
	  			if (edge.size() >= 3) continue;
	  			if (isCandidateEdge(node, edge)) {
	  				decisions.add(edge);
	  			}
	  		}
		}

		for (EllipticalKnot otherNode : unvisitedNodes)
		{
			if (coveredNodes.contains(otherNode))
				continue;
			if (node.getPartitionIdx() == otherNode.getPartitionIdx())
				continue;
			if (Math.abs(node.getNodeFeatures().getCount("x") - otherNode.getNodeFeatures().getCount("x")) > H_SPAN)
				continue;

			Set<EllipticalKnot> instance = new HashSet<>();
			instance.add(otherNode);
			decisions.add(instance);
		}
		
	  return decisions;
  }

	// executes move in backward and compute the log likelihood as well as log gradient
	@Override
	public boolean pathExists(GraphMatchingState<String, EllipticalKnot> currState, Map<EllipticalKnot, Set<EllipticalKnot>> finalState)
	{
		Set<EllipticalKnot> visitedNodes = new HashSet<>(currState.getVisitedNodes());
		for (Set<EllipticalKnot> e : finalState.values())
		{
			Set<Set<EllipticalKnot>> edgesFormed = new HashSet<>();
			for (EllipticalKnot v : e)
			{
				if (!visitedNodes.contains(v)) continue;

				Set<EllipticalKnot> eprime = currState.getNode2EdgeView().get(v);
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
	public int numParents(GenericGraphMatchingState<String, EllipticalKnot> state)
	{
		int numParents = 0;
		//System.out.println(curLatent.toString());
		for (Set<EllipticalKnot> e : state.getMatchings())
		{
			int numVisited = 0;
			for (EllipticalKnot node : e)
			{
				if (state.getNode2EdgeView().containsKey(node)) {
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
