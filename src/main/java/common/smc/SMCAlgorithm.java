package common.smc;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;

import org.apache.commons.math3.util.Pair;

import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.ParticlePopulation;
import briefj.BriefIO;
import briefj.BriefParallel;

/**
 * An SMC algorithm using multi-threading for proposing and suitable
 * for abstract 'SMC samplers' problems as well as more classical ones.
 * 
 * Also performs adaptive re-sampling by monitoring ESS.
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 *
 * @param <P> The type (class) of the individual particles
 */
public class SMCAlgorithm<L, E>
{
  private final LatentSimulator<L> proposal;
  private final ObservationDensity<L, E> observationDensity;
  private final List<E> emissions;
  private final SMCOptions options;
  private PrintWriter logger = null;
  private List<String> outputLines = null;

  public SMCAlgorithm(
	      LatentSimulator<L> transitionDensity,
	      ObservationDensity<L, E> observationDensity,
	      List<E> emissions,
	      SMCOptions options)
  {
    this.emissions = emissions;
    this.proposal = transitionDensity;
    this.observationDensity = observationDensity;

    this.options = options;
    this.randoms = new Random[options.nParticles];
    SplittableRandom splitRandom = new SplittableRandom(options.random.nextLong());
    for (int i = 0; i < options.nParticles; i++)
      this.randoms[i] = new Random(splitRandom.split().nextLong());

    if (options.loggerFilePath != null) {
    	logger = BriefIO.output(new File(options.loggerFilePath));
    	outputLines = new ArrayList<>();
    }
  }

  /**
   * This is used to ensure that the result is deterministic even in a 
   * multi-threading context: each particle index has its own unique random 
   * stream
   */
  private final Random[] randoms;
  
  /**
   * Compute the SMC algorithm
   * 
   * @return The particle population at the last step
   */
  public Pair<Double, ParticlePopulation<L>> sample()
  {
    ParticlePopulation<L> currentPopulation = propose(null, 0);
    
    int nSMCIterations = proposal.numIterations();
    double logZ = 0.0;
    
    for (int currentIteration = 1; currentIteration < nSMCIterations - 1; currentIteration++)
    {
      currentPopulation = propose(currentPopulation, currentIteration);
      //System.out.println("effective sample size: " + currentPopulation.getRelativeESS());
      if (outputLines != null) {
    	  outputLines.add(currentIteration + ", " + options.essThreshold + ", " + options.nParticles + ", " + options.resamplingScheme + ", " + currentPopulation.getRelativeESS());
      }
      if (currentPopulation.getRelativeESS() < options.essThreshold && currentIteration < nSMCIterations - 2)
        currentPopulation = currentPopulation.resample(options.random, options.resamplingScheme);
      else if (currentIteration == nSMCIterations - 2)
    	  currentPopulation = currentPopulation.resample(options.random, options.resamplingScheme);
      double logZt = currentPopulation.logNormEstimate();
      //System.out.println("logZt: " + logZt);
      logZ += logZt;
    }

    if (outputLines != null && logger != null) {
    	for (String line : outputLines)
    		logger.println(line);
    	logger.close();
    }
    return Pair.create(logZ, currentPopulation);
  }

  /**
   * Calls the proposal options.nParticles times, form the new weights, and return the new population.
   * 
   * If the provided currentPopulation is null, use the initial distribution, otherwise, use the 
   * transition. Both are specified by the proposal object.
   * 
   * @param currentPopulation The population of particles before the proposal
   * @param currentIteration The iteration of the particles used as starting points for the proposal step
   * @return
   */
  private ParticlePopulation<L> propose(final ParticlePopulation<L> currentPopulation, final int currentIteration)
  {
    final boolean isInitial = currentPopulation == null;
    
    final double [] logWeights = new double[options.nParticles];
    @SuppressWarnings("unchecked")
    final L[] particles = (L[]) new Object[options.nParticles];
    E curEmission = emissions.get(currentIteration);
    //E oldEmission = (currentIteration >= 1) ? oldEmission = emissions.get(currentIteration-1) : null;

    BriefParallel.process(options.nParticles, options.nThreads, particleIndex ->
    {
    	L curLatent = null;
    	if (isInitial) { 
    		curLatent = proposal.sampleInitial(randoms[particleIndex]);
    	} else {
  			curLatent = proposal.sampleForwardTransition(randoms[particleIndex], currentPopulation.particles.get(particleIndex));
  		}

	    double logWeight = observationDensity.logDensity(curLatent, curEmission);
	    if (!isInitial)
	    {
		    double logWeightPrev = observationDensity.logDensity(currentPopulation.particles.get(particleIndex), emissions.get(currentIteration));
		    double logWeightCorr = observationDensity.logWeightCorrection(curLatent, currentPopulation.particles.get(particleIndex));
	    	logWeight = Math.log(currentPopulation.getNormalizedWeight(particleIndex)) + logWeight - logWeightPrev + logWeightCorr;
	    }
	    logWeights[particleIndex] = logWeight;
	    particles[particleIndex] = curLatent;
    });
    
    return ParticlePopulation.buildDestructivelyFromLogWeights(
        logWeights, 
        Arrays.asList(particles),
        isInitial ? 0.0 : currentPopulation.logScaling);
  }

}