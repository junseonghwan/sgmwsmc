package common.smc.components;

import java.util.List;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;
import common.model.Command;
import common.smc.DiscreteParticleFilter.DiscreteLatentSimulator;

public class GenericDiscreteLatentSimulator<F, NodeType extends GraphNode<?>> implements DiscreteLatentSimulator<GenericGraphMatchingState<F, NodeType>> 
{
	private GenericGraphMatchingState<F, NodeType> initial = null;
	private Command<F, NodeType> command = null;
	private boolean sequentialSampling = true;
	
	public GenericDiscreteLatentSimulator(Command<F, NodeType> command, GenericGraphMatchingState<F, NodeType> initial, boolean sequentialSampling) 
	{
		this.command = command;
		this.initial = initial;
		this.sequentialSampling = sequentialSampling;
	}

	@Override
	public List<GenericGraphMatchingState<F, NodeType>> generateInitialStates() {
		return generateDescendants(initial);
	}

	@Override
	public List<GenericGraphMatchingState<F, NodeType>> generateDescendants(GenericGraphMatchingState<F, NodeType> state) {
		return command.generateNextSamples(state, sequentialSampling);
	}

	@Override
	public int numIterations() {
		return initial.getUnvisitedNodes().size();
	}

}
