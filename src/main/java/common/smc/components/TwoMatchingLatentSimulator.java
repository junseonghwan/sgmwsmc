package common.smc.components;

import java.util.Random;

import common.graph.GraphMatchingState;
import common.graph.GraphNode;
import common.model.Command;
import common.smc.StreamingParticleFilter.LatentSimulator;

public class TwoMatchingLatentSimulator<F, NodeType extends GraphNode<?>> implements LatentSimulator<GraphMatchingState<F, NodeType>> {

	private GraphMatchingState<F, NodeType> initial = null;
	private Command<F, NodeType> command = null;
	private boolean sequentialSampling = true;
	private boolean exactSampling = true;

	public TwoMatchingLatentSimulator(Command<F, NodeType> command, GraphMatchingState<F, NodeType> initial, boolean sequentialSampling, boolean exactSampling) 
	{
		this.command = command;
		this.initial = initial;
		this.sequentialSampling = sequentialSampling;
		this.exactSampling = exactSampling;
	}

	@Override
	public GraphMatchingState<F, NodeType> sampleInitial(Random random) {
		return sampleForwardTransition(random, initial);
	}

	@Override
	public GraphMatchingState<F, NodeType> sampleForwardTransition(Random random, GraphMatchingState<F, NodeType> state) {
		GraphMatchingState<F, NodeType> next = state.copyState();
		next.sampleNextUnCoveredNode(random, command, sequentialSampling, exactSampling);
		return next;
	}

	@Override
	public int numIterations() {
		return 0;
	}

}
