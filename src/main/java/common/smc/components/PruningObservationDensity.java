package common.smc.components;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;
import common.smc.StreamingParticleFilter.ObservationDensity;

public class PruningObservationDensity<F, NodeType extends GraphNode<?>> implements ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object>
{
	private Map<NodeType, Set<NodeType>> targetState;
	public PruningObservationDensity(List<Set<NodeType>> matching)
	{
		this.targetState = new HashMap<>();
		
		// construct nodeToEdge view
		for (Set<NodeType> e : matching)
		{
			for (NodeType v : e)
			{
				targetState.put(v, e);
			}
		}
	}

	@Override
	public double logDensity(GenericGraphMatchingState<F, NodeType> latent, Object emission) 
	{
		if (latent == null) return 0.0;

		// check if the latent is contained in the target
		if (!inSupport(latent.getNode2EdgeView(), targetState))
			return Double.NEGATIVE_INFINITY;
		
		//return latent.getLogDensity();
		return 0.0;
	}
	
	public static <NodeType> boolean inSupport(Map<NodeType, Set<NodeType>> subset, Map<NodeType, Set<NodeType>> superset)
	{
		for (NodeType v : subset.keySet())
		{
			if (!superset.get(v).containsAll(subset.get(v)))
				return false;
		}
		return true;
	}

	@Override
	public double logWeightCorrection(
			GenericGraphMatchingState<F, NodeType> curLatent,
			GenericGraphMatchingState<F, NodeType> oldLatent) {
		/*
		double numParents = command.getDecisionModel().numParents(curLatent);
		return -Math.log(numParents) - curLatent.getLogForwardProposal();
		*/
		//return -curLatent.getLogForwardProposal();
		return 0.0;
	}

	@Override
	public boolean cancellationApplied() {
		return false;
	}

}
