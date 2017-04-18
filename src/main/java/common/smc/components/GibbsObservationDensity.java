package common.smc.components;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.graph.GraphNode;
import common.model.Command;
import common.smc.StreamingParticleFilter.ObservationDensity;

public class GibbsObservationDensity<F, NodeType extends GraphNode<?>> implements ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object>
{
	private Command<F, NodeType> command;

	public GibbsObservationDensity(Command<F, NodeType> command) 
	{
		this.command = command;
	}

	@Override
	public double logDensity(GenericGraphMatchingState<F, NodeType> curr, GenericGraphMatchingState<F, NodeType> prev, Object emission) {
		Set<Set<NodeType>> mcurr = new HashSet<>(curr.getMatchings());
		Set<Set<NodeType>> mprev = new HashSet<>(prev.getMatchings());
		mcurr.removeAll(mprev);
		if (mcurr.size() == 0)
			return 0.0;
		else if (mcurr.size() > 1) 
			throw new RuntimeException();
		else {
			List<Set<NodeType>> e = new ArrayList<>(mcurr);
			Counter<F> features = command.getFeatureExtractor().extractFeatures(e.get(0), curr.getMatchings());
			double val = 0.0;
			for (F f : features)
			{
				val += features.getCount(f) * command.getModelParameters().getCount(f);
			}
			return val;
		}
	}
	
	@Override
	public double logDensity(GenericGraphMatchingState<F, NodeType> latent, Object emission) {
		if (latent == null) {
			return 0.0;
		}
		/*
		NodeType u = latent.getVisitedNodes().get(latent.getVisitedNodes().size() - 1);
		Set<NodeType> e = latent.getNode2EdgeView().get(u);
		
		// compute the loglike of this edge
		Counter<F> p = command.getModelParameters();
		Counter<F> features = command.getFeatureExtractor().extractFeatures(e, latent.getMatchings());
		double val = 0.0;
		for (F f : features)
		{
			val += features.getCount(f) * p.getCount(f);
		}
		val += logWeightCorrection(latent, null);
		return val;
		*/
		return computeLogLikelihood(command, (GraphMatchingState<F, NodeType>)latent);
	}
	
	public static <F, NodeType extends GraphNode<?>> double computeLogLikelihood(Command<F, NodeType> command, GraphMatchingState<F, NodeType> m)
	{
		// use Gibbs measure to weight the matching
		Counter<F> p = command.getModelParameters();
		List<Set<NodeType>> matching = m.getMatchings();
		double loglik = 0.0;
		for (Set<NodeType> e : matching)
		{
			Counter<F> features = command.getFeatureExtractor().extractFeatures(e, matching);
			double val = 0.0;
			for (F f : features)
			{
				val += features.getCount(f) * p.getCount(f);
			}
			loglik += val;
		}
		return loglik;
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
