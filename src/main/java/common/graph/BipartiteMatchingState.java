package common.graph;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.google.common.collect.ImmutableList;

public class BipartiteMatchingState<F, NodeType extends GraphNode<?>> extends GraphMatchingState<F, NodeType>
{
	private ImmutableList<NodeType> partition2;

	public static <F, NodeType extends GraphNode<?>> BipartiteMatchingState<F, NodeType> getInitial(List<NodeType> p1, List<NodeType> p2)
	{
		BipartiteMatchingState<F, NodeType> initial = new BipartiteMatchingState<>();
		initial.unvisitedNodes = new ArrayList<>(p1);
		initial.partition2 = ImmutableList.copyOf(p2);
		//initial.visitedNodes = new HashSet<>();
		initial.visitedNodes = new ArrayList<>();
		//initial.matching = new ArrayList<>();
		initial.matching = new HashSet<>();
		initial.coveredNodes = new HashSet<>();
		return initial;
	}

	public static <F, NodeType extends GraphNode<?>> BipartiteMatchingState<F, NodeType> getInitial(List<NodeType> nodes)
	{
		// separate nodes into unvisitedNodes and partition2
		List<NodeType> p1 = new ArrayList<>();
		List<NodeType> p2 = new ArrayList<>();
		for (NodeType node : nodes)
		{
			if (p1.size() == 0) {
				p1.add(node);
			} else {
				if (p1.get(0).getPartitionIdx() == node.getPartitionIdx())
					p1.add(node);
				else
					p2.add(node);
			}
		}
		
		return getInitial(p1, p2);
	}

	@Override
	public BipartiteMatchingState<F, NodeType> copyState()
	{
		BipartiteMatchingState<F, NodeType> copyState = new BipartiteMatchingState<>();
		//copyState.matching = new ArrayList<>(this.matching);
		copyState.matching = new HashSet<>(this.matching);
		copyState.coveredNodes = new HashSet<>(this.coveredNodes);
		//copyState.visitedNodes = new HashSet<>(this.visitedNodes);
		copyState.visitedNodes = new ArrayList<>(this.visitedNodes);
		copyState.unvisitedNodes = new ArrayList<>(this.unvisitedNodes);
		copyState.partition2 = ImmutableList.copyOf(this.partition2);

		copyState.logDensity = this.logDensity;
		if (this.logGradient != null) {
  		for (F f : this.logGradient)
  		{
  			copyState.logGradient.setCount(f, this.logGradient.getCount(f));
  		}
		}

		return copyState;
	}
	
	public static <F, NodeType extends GraphNode<F>> BipartiteMatchingState<F, NodeType> constructFinalState(List<NodeType> p1InSequence, List<NodeType> p2InSequence, List<Set<NodeType>> matching)
	{
		BipartiteMatchingState<F, NodeType> finalState = new BipartiteMatchingState<>();
		//finalState.matching = new ArrayList<>(matching);
		finalState.matching = new HashSet<>(matching);
		//finalState.visitedNodes = new HashSet<>(p1InSequence);
		finalState.visitedNodes = new ArrayList<>(p1InSequence);
		finalState.unvisitedNodes = new ArrayList<>();
		finalState.partition2 = ImmutableList.copyOf(p2InSequence);

		finalState.coveredNodes = new HashSet<>();
		finalState.coveredNodes.addAll(p1InSequence);
		finalState.coveredNodes.addAll(p2InSequence);

		return finalState;
	}
	
	public List<NodeType> getPartition2()
	{
		return this.partition2;
	}

	/**
	 * Make a move where idx1 from partition1 and idx2 from partition2 are matched. Use for testing only.
	 * 
	 * @param idx1
	 * @param idx2
	 */
	public void move(int idx1, int idx2)
	{
		NodeType node1 = this.unvisitedNodes.remove(idx1);
		NodeType node2 = this.partition2.get(idx2);
		Set<NodeType> e = new HashSet<>();
		e.add(node1);
		e.add(node2);
		this.coveredNodes.add(node1);
		this.coveredNodes.add(node2);
		this.visitedNodes.add(node1);
		this.matching.add(e);
	}

	public void move(NodeType node1, NodeType node2)
	{
		this.unvisitedNodes.remove(node1);
		Set<NodeType> e = new HashSet<>();
		e.add(node1);
		e.add(node2);
		this.coveredNodes.add(node1);
		this.coveredNodes.add(node2);
		this.visitedNodes.add(node1);
		this.matching.add(e);
	}


}
