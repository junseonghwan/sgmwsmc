package common.smc.components;

import java.util.List;
import java.util.Random;

import org.apache.commons.math3.util.Pair;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;
import common.smc.ParticlePopulation;
import common.smc.SMCAlgorithm;
import common.smc.SMCOptions;
import common.smc.StreamingParticleFilter;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;

public class SequentialGraphMatchingSampler<F, NodeType extends GraphNode<?>> 
{
	private LatentSimulator<GenericGraphMatchingState<F, NodeType>> transitionDensity; 
	private ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity;
	private List<Object> emissions;
	private List<GenericGraphMatchingState<F, NodeType>> samples;
	private boolean useStreamingSMC = true;
	
	public SequentialGraphMatchingSampler(
			LatentSimulator<GenericGraphMatchingState<F, NodeType>> transitionDensity, 
			ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity,
			List<Object> emissions)
	{
		this(transitionDensity, observationDensity, emissions, true);
	}
	
	public SequentialGraphMatchingSampler(
			LatentSimulator<GenericGraphMatchingState<F, NodeType>> transitionDensity, 
			ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity,
			List<Object> emissions,
			boolean useStreaming)
	{
		this.transitionDensity = transitionDensity;
		this.observationDensity = observationDensity;
		this.emissions = emissions;
		this.useStreamingSMC = useStreaming;
	}

	public double sample(Random random, int numConcreteParticles, int maxNumVirtualParticles)
	{
		return sample(random, numConcreteParticles, maxNumVirtualParticles, null);
	}

	public double sample(Random random, int numConcreteParticles, int maxNumVirtualParticles, String outputFilePath)
	{
		double logZ = Double.NEGATIVE_INFINITY;
		// generate matching using sequential Monte Carlo
		if (useStreamingSMC) {
			StreamingParticleFilter<GenericGraphMatchingState<F, NodeType>, Object> sbf = new StreamingParticleFilter<>(transitionDensity, observationDensity, emissions);
			sbf.options.numberOfConcreteParticles = numConcreteParticles;
			sbf.options.maxNumberOfVirtualParticles = maxNumVirtualParticles;
			sbf.options.verbose = false;
			sbf.mainRandom = new Random(random.nextLong());
			sbf.options.targetedRelativeESS = 1.0;
			logZ = sbf.sample();
			samples = sbf.getSamples().samples;
		} else {
			SMCOptions options = new SMCOptions();
			options.nParticles = numConcreteParticles;
			options.random = new Random(random.nextLong());
			if (outputFilePath != null)
				options.loggerFilePath = outputFilePath;
			SMCAlgorithm<GenericGraphMatchingState<F, NodeType>, Object> smc = new SMCAlgorithm<>(transitionDensity, observationDensity, emissions, options);
			Pair<Double, ParticlePopulation<GenericGraphMatchingState<F, NodeType>>> ret = smc.sample();
			logZ = ret.getFirst();
			samples = ret.getSecond().particles;
		}
		return logZ;
	}
	
	public List<GenericGraphMatchingState<F, NodeType>> getSamples()
	{
		return samples;
	}

}
