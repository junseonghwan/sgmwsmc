package common.model;

import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import common.graph.GraphMatchingState;
import common.graph.GraphNode;

/**
 * Provide functionality to evaluate p(\sigma, d_{\sigma} | \theta)
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 */
public class SequentialDecisionModel 
{

	public static <F, NodeType extends GraphNode<?>> double logProb(Counter<F> params, List<NodeType> nodes, List<Set<NodeType>> decisions)
	{
		double logProb = 0.0;
		
		// construct an initial GraphMatchingState and build up the state
		GraphMatchingState<F, NodeType> state = GraphMatchingState.getInitialState(nodes);
		
		for (NodeType v : nodes)
		{
			
		}

		return logProb;
	}
}
