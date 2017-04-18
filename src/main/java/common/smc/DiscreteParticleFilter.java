package common.smc;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Stack;

import org.apache.commons.lang3.tuple.Pair;

import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.StreamingPropagator.PropagationResult;
import common.smc.StreamingPropagator.PropagatorOptions;

/**
 * Implementation of: Fearnhead and Clifford (2003). On-line inference for hidden Markov models via particle filters.
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 * @param <L>
 * @param <E>
 */
public class DiscreteParticleFilter<L, E> 
{
	private final List<E> emissions;
	private final DiscreteLatentSimulator<L> transitionDensity;
	private final ObservationDensity<L,E> observationDensity;
	public Random mainRandom = new Random(1);
  
	private PropagationResult<L> propResults;
  
	public DiscreteParticleFilter(DiscreteLatentSimulator<L> transitionDensity, ObservationDensity<L, E> observationDensity, List<E> emissions)
	{
		this.emissions = emissions;
		this.transitionDensity = transitionDensity;
		this.observationDensity = observationDensity;
	}
  
	public PropagatorOptions options = new PropagatorOptions();

	
	public static interface DiscreteLatentSimulator<L>
	{
	    public List<L> generateInitialStates();
	    public List<L> generateDescendants(L state);
	  	public int numIterations();
	}

	  /**
	   * @return The estimate for log(Z)
	   */
	  public double sample()
	  {
	    // initial distribution
  		DiscreteParticleFilterProposal proposal = getInitialDistributionProposal();
  		StreamingPropagator<L> propagator = new StreamingPropagator<>(proposal, options);
	    propResults = propagator.executeDPF();
	    double logZ = propResults.population.logZEstimate();
      if (options.verbose)
      	System.out.println("log Z_1: " + logZ);

	    // recursion
	    for (int i = 1; i < transitionDensity.numIterations(); i++)
	    {
	      proposal = new DiscreteParticleFilterProposal(mainRandom.nextLong(), emissions.get(i), emissions.get(i-1), propResults.samples);
	      propagator = new StreamingPropagator<>(proposal, options);
	      propResults = propagator.executeDPF();
	      double logZi = propResults.population.logZEstimate();
	      logZ += logZi;
	      if (options.verbose)
	      	System.out.println("logZ_" + (i+1) + ": " + logZi + ", log Z: " + logZ);
	    }
	    return logZ;
	  }
	  
	  public PropagationResult<L> getPropagationResults() {
	  	return propResults;
	  }
	  
	  public List<L> getSamples()
	  {
		  return propResults.samples;
	  }

	  private DiscreteParticleFilterProposal getInitialDistributionProposal()
	  {
	    return new DiscreteParticleFilterProposal(mainRandom.nextLong(), emissions.get(0), null, null);
	  }

	  private class DiscreteParticleFilterProposal implements ProposalWithRestart<L>
	  {
		    private final long seed;
		    private final PermutationStream permutationStream;
		    private final Random random;
		    
		    private final E curEmission;
		    private final E oldEmission;
		    private final List<L> oldLatents;
		    private int nCalls = 0;
		    
		    private List<L> oldLatentsCopy; // copy of the old latent states
		    private Stack<L> currLatentStack; // latent states to be served up
		    
		    private DiscreteParticleFilterProposal(long seed, E curEmission, E oldEmission, List<L> oldLatents)
		    {
		      this.seed = seed;
		      this.curEmission = curEmission;
		      this.oldEmission = oldEmission;
		      this.oldLatents = oldLatents;
		      this.random = new Random(seed * 171);
		      this.permutationStream = oldLatents == null ? null : new PermutationStream(oldLatents.size(), random);
	    	  currLatentStack = new Stack<>();
		      if (oldLatents != null) {
		    	  this.oldLatentsCopy = new ArrayList<>(oldLatents);
		      } else {
		    	  this.currLatentStack.addAll(transitionDensity.generateInitialStates());
		    	  this.oldLatentsCopy = null;
		      }
		    }

			public void populateStack(L oldLatent) 
			{
				currLatentStack.addAll(transitionDensity.generateDescendants(oldLatent));
			}
			
			private boolean isInitial()
			{
				return oldLatents == null;
			}
			
			L currOldLatent = null;

		    @Override
		    public Pair<Double, L> nextLogWeightSamplePair()
		    {
		    	if (currLatentStack.isEmpty()) {
	    			currOldLatent = oldLatents.get(permutationStream.popIndex());
	    			oldLatentsCopy.remove(currOldLatent);
	    			populateStack(currOldLatent);
		    	}
		    	// consume a newly proposed latent state
		    	L latent = currLatentStack.pop();
		    	double logWeight = observationDensity.logDensity(latent, curEmission);
	        if (!observationDensity.cancellationApplied() && !isInitial())
	        	logWeight -= observationDensity.logWeightCorrection(latent, currOldLatent); // adjust the backward kernel
			    nCalls++;
			    
		    	return Pair.of(logWeight, latent);
		    }

		    @Override
			public int numberOfCalls() {
				return nCalls;
			}
	
		    @Override
			public DiscreteParticleFilterProposal restart() {
			      return new DiscreteParticleFilterProposal(seed, curEmission, oldEmission, oldLatents);
			}
		    
		    @Override
		    public boolean hasNextLogWeightSamplePair()
		    {
		    	if (currLatentStack.size() > 0) {
		    		return true;
		    	}else {
		    		if (oldLatentsCopy == null || oldLatentsCopy.size() == 0) {
		    			return false;
		    		} else {
		    			return true;
		    		}
		    	}
		    }

	  }
}
