package common.smc.components;

import java.util.Random;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;
import common.model.Command;
import common.smc.StreamingParticleFilter.LatentSimulator;

public class GenericMatchingLatentSimulator<F, NodeType extends GraphNode<?>> implements LatentSimulator<GenericGraphMatchingState<F, NodeType>> 
{
	private GenericGraphMatchingState<F, NodeType> initial = null;
	private Command<F, NodeType> command = null;
	private boolean sequentialSampling = true;
	private boolean exactSampling = true;

	public GenericMatchingLatentSimulator(Command<F, NodeType> command, GenericGraphMatchingState<F, NodeType> initial, boolean sequentialSampling, boolean exactSampling) 
	{
		this.command = command;
		this.initial = initial;
		this.sequentialSampling = sequentialSampling;
		this.exactSampling = exactSampling;
  }

	@Override
	public GenericGraphMatchingState<F, NodeType> sampleInitial(Random random) {
		return sampleForwardTransition(random, initial);
	}

	@Override
  public GenericGraphMatchingState<F, NodeType> sampleForwardTransition(Random random, GenericGraphMatchingState<F, NodeType> state) {
		GenericGraphMatchingState<F, NodeType> next = state.copyState();
		command.sampleNext(random, next, sequentialSampling, exactSampling);
		return next;
  }

	@Override
  public int numIterations() {
	  return initial.getUnvisitedNodes().size();
  }
	
}
