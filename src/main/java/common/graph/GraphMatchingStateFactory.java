package common.graph;

import java.util.List;

import common.model.BipartiteDecisionModel;
import common.model.Command;

public class GraphMatchingStateFactory 
{

	public static <F, NodeType extends GraphNode<?>> GenericGraphMatchingState<F, NodeType> createInitialGraphMatchingState(Command<F, NodeType> command, List<NodeType> nodes)
	{
		if (command.getDecisionModel() instanceof BipartiteDecisionModel)
		{
			return BipartiteMatchingState.getInitial(nodes);
		}
		else
		{
			return GraphMatchingState.getInitialState(nodes);
		}
	}

}
