package common.smc.components;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;
import common.model.Command;
import common.smc.StreamingParticleFilter.ObservationDensity;

public class ExactProposalObservationDensity<F, NodeType extends GraphNode<?>> implements ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> 
{
	private Command<F, NodeType> command;
	public ExactProposalObservationDensity(Command<F, NodeType> command)
	{
		this.command = command;
	}

	@Override
	public double logDensity(GenericGraphMatchingState<F, NodeType> latent, Object emission) 
	{
		if (latent == null) return 0.0;
		return latent.getLogDensity();
	}

	@Override
	public double logWeightCorrection(
			GenericGraphMatchingState<F, NodeType> curLatent,
			GenericGraphMatchingState<F, NodeType> oldLatent) {
		double numParents = command.getDecisionModel().numParents(curLatent);
		return -Math.log(numParents) - curLatent.getLogForwardProposal();
	}

	@Override
	public boolean cancellationApplied() {
		return false;
	}

}
