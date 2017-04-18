package common.smc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

import org.apache.commons.lang3.tuple.Pair;

import briefj.opt.Option;


/**
 * Performs one cycle of importance sampling + resampling in a streaming fashion.
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 *
 * @param <S>
 */
public class StreamingPropagator<S>
{
  public final ProposalWithRestart<S> proposal;
  public final PropagatorOptions options;
  private Consumer<S> processor = null;
  
  public StreamingPropagator(ProposalWithRestart<S> proposal, PropagatorOptions options)
  {
    this.proposal = proposal;
    this.options = options;
  }
  
  public void addProcessor(Consumer<S> newProcessor)
  {
    if (newProcessor == null)
      throw new RuntimeException();
    if (this.processor == null)
      this.processor = newProcessor;
    else
      this.processor = this.processor.andThen(newProcessor);
  }

  public StreamingPropagator(ProposalWithRestart<S> proposal)
  {
    this(proposal, new PropagatorOptions());
  }
  
  public static class PropagationResult<S>
  {
    /**
     * The virtual (implicit) particles.
     */
    public final CompactPopulation population;
    
    /**
     * The concrete samples obtained by resampling the virtual particles.
     */
    public final List<S> samples;

    private PropagationResult(CompactPopulation population, List<S> samples)
    {
      this.population = population;
      this.samples = samples;
    }
  }
  
  /**
   * Performs one cycle of importance sampling + resampling.
   * @return
   */
  public PropagationResult<S> execute()
  {
    CompactPopulation population = new CompactPopulation();
    propose(
        population, 
        options.targetedRelativeESS, 
        options.numberOfConcreteParticles, 
        options.maxNumberOfVirtualParticles);
    if (options.verbose)
      System.out.println(
            "nVirtual=" + population.getNumberOfParticles() + ", "
          + "nConcrete=" + options.numberOfConcreteParticles + ", "
          + "relative_ess=" + (population.ess()/options.numberOfConcreteParticles));
    double [] sortedCumulativeProbabilitiesForFinalResampling = 
        options.resamplingScheme.getSortedCumulativeProbabilities(
            options.resamplingRandom, 
            options.numberOfConcreteParticles);
    List<S> samples = resample(
        population, 
        sortedCumulativeProbabilitiesForFinalResampling);
    return new PropagationResult<>(population, samples);
  }
  
  public PropagationResult<S> executeDPF()
  {
	  CompactPopulation population = new CompactPopulation();
      double c = discretePropose(population);
      System.out.println(
              "nTotal=" + population.getNumberOfParticles() + ", "
            + "nConcrete=" + options.numberOfConcreteParticles);

      List<S> samples = discreteResample(population, c);
      return new PropagationResult<>(population, samples);      
  }

  private List<S> discreteResample(CompactPopulation population, double c)
  {
	  ProposalWithRestart<S> proposal = this.proposal;
	  if (proposal.numberOfCalls() != 0)
	  {
		  proposal = proposal.restart();

		  if (proposal.numberOfCalls() != 0)
			  throw new RuntimeException("restart() incorrectly implemented");
	  }

	  final double logSum = population.getLogSum();
	  
	  final int nParticles = population.getNumberOfParticles();
	  final int popAfterCollapse = options.numberOfConcreteParticles;
	  final List<S> result = new ArrayList<>(popAfterCollapse);
	  final List<Pair<Double, S>> remainingParticles = new ArrayList<>();
	  CompactPopulation sanityCheck = new CompactPopulation();

	  double remainingParticlesNormalization = 0.0;
	  
	  S candidate = null;

	  while (proposal.hasNextLogWeightSamplePair())
	  {
		  int before = proposal.numberOfCalls();
		  Pair<Double, S> nextLogWeightSamplePair = proposal.nextLogWeightSamplePair();
		  if (proposal.numberOfCalls() != before + 1)
			  throw new RuntimeException("The method numberOfCalls() was incorrectly implemented in the proposal");
		  candidate = nextLogWeightSamplePair.getRight();
		  final double normalizedWeight = Math.exp(nextLogWeightSamplePair.getLeft() - logSum);
		  if (c * normalizedWeight >= 1) {
			  result.add(candidate);
		  } else {
			  remainingParticles.add(Pair.of(normalizedWeight, candidate));
			  remainingParticlesNormalization += normalizedWeight;
		  }
		  sanityCheck.insertLogWeight(nextLogWeightSamplePair.getLeft());
	  }

	  // replay the last few calls of the proposal sequence to make sure things were indeed behaving deterministically
	  for (int i = proposal.numberOfCalls(); i < nParticles; i++)
		  sanityCheck.insertLogWeight(proposal.nextLogWeight());
	  if (sanityCheck.getLogSum() != logSum || sanityCheck.getLogSumOfSquares() != population.getLogSumOfSquares()) 
		  throw new RuntimeException("The provided proposal does not behave deterministically: " + sanityCheck.getLogSum() + " vs " + logSum);

	
	  // now, use stratified resampling to determine the particles to be resampling from the remaining list
	  if (remainingParticles.size() == 0)
		  return result;
	  
	  int numDarts = popAfterCollapse - result.size();
	  double [] sortedCumulativeProbabilitiesForFinalResampling = ResamplingScheme.STRATIFIED.getSortedCumulativeProbabilities(options.resamplingRandom, numDarts);
	  double normalizedPartialSum = 0.0;
	  int currIdx = 0;
	  for (int i = 0; i < numDarts; i++)
	  {
	      double nextCumulativeProbability = sortedCumulativeProbabilitiesForFinalResampling[i];
	      // sum normalized weights until we get to the next resampled cumulative probability
	      while (normalizedPartialSum < nextCumulativeProbability)
	      {
	    	  Pair<Double, S> nextLogWeightSamplePair = remainingParticles.get(currIdx++);
	    	  candidate = nextLogWeightSamplePair.getRight();
	    	  double normalizedWeight = nextLogWeightSamplePair.getLeft();
	    	  normalizedPartialSum += normalizedWeight/remainingParticlesNormalization;
	      }
	      // we have found one particle that survived the collapse
	      result.add(candidate);
	  }
	  if (result.size() != popAfterCollapse)
		  throw new RuntimeException("Discrete particle filter resampling is not correctly implemented.");

	  return result;
  }

  /**
   * Perform resampling by replaying randomness to instantiate
   * concrete version of the particles that survive the resampling step.
   * 
   * @param proposal
   * @param sortedCumulativeProbabilities See ResamplingScheme
   * @return The list of resampled, equi-weighted particles
   */
  private List<S> resample(
      CompactPopulation population,
      double [] sortedCumulativeProbabilities)
  {
    ProposalWithRestart<S> proposal = this.proposal;
    if (proposal.numberOfCalls() != 0)
    {
      proposal = proposal.restart();
      
      if (proposal.numberOfCalls() != 0)
        throw new RuntimeException("restart() incorrectly implemented");
    }
    
    final double logSum = population.getLogSum();
    final int nParticles = population.getNumberOfParticles();
    final int popAfterCollapse = sortedCumulativeProbabilities.length;
    final List<S> result = new ArrayList<>(popAfterCollapse);
    CompactPopulation sanityCheck = new CompactPopulation();
    
    double normalizedPartialSum = 0.0;
    S candidate = null;
    for (int i = 0; i < popAfterCollapse; i++)
    {
      double nextCumulativeProbability = sortedCumulativeProbabilities[i];
      // sum normalized weights until we get to the next resampled cumulative probability
      while (normalizedPartialSum < nextCumulativeProbability) 
      {
        int before = proposal.numberOfCalls();
        Pair<Double, S> nextLogWeightSamplePair = proposal.nextLogWeightSamplePair();
        if (proposal.numberOfCalls() != before + 1)
          throw new RuntimeException("The method numberOfCalls() was incorrectly implemented in the proposal");
        candidate = nextLogWeightSamplePair.getRight();
        final double normalizedWeight = Math.exp(nextLogWeightSamplePair.getLeft() - logSum);
        normalizedPartialSum += normalizedWeight;
        sanityCheck.insertLogWeight(nextLogWeightSamplePair.getLeft());
      }
      // we have found one particle that survived the collapse
      result.add(candidate);
    }
    
    // replay the last few calls of the proposal sequence to make sure things were indeed behaving deterministically
    for (int i = proposal.numberOfCalls(); i < nParticles; i++)
      sanityCheck.insertLogWeight(proposal.nextLogWeight());
    if (sanityCheck.getLogSum() != logSum || sanityCheck.getLogSumOfSquares() != population.getLogSumOfSquares()) 
      throw new RuntimeException("The provided proposal does not behave deterministically: " + sanityCheck.getLogSum() + " vs " + logSum);
    
    return result;
  }
  
  /**
   * Grow this population by using a proposal distribution.
   * 
   * The number of particles proposed is determined as follows:
   * First, the proposal will be called at least minNumberOfParticles times.
   * Then, after minNumberOfParticles have been proposed, the growth will continue 
   * until the first of these two condition is met:
   * - maxNumberOfParticles is exceeded
   * - the relative ESS exceeds the targetedRelativeESS (here relative ESS is defined
   *   as ESS divided by minNumberOfParticles, since minNumberOfParticle will 
   *   correspond to the number of 'concrete' particles)
   * 
   * @param proposal
   * @param targetedRelativeESS
   * @param minNumberOfParticles
   * @param maxNumberOfParticles
   */
  private void  propose( 
    CompactPopulation population,
    double targetedRelativeESS,
    int minNumberOfParticles,
    int maxNumberOfParticles)
  {
    while 
        ( 
          population.getNumberOfParticles() < minNumberOfParticles || 
          (
              population.getNumberOfParticles() < maxNumberOfParticles && 
              population.ess() / minNumberOfParticles < targetedRelativeESS
          )
        )
      population.insertLogWeight(proposal.nextLogWeight());
  }

  private double discretePropose(CompactPopulation population)
  {
	  // find c such that sum min(c w_j, 1) = N
	  // 1. get all the weights
	  List<Double> logWeights = new ArrayList<>();
	  double logWeight = 0.0;
	  while (proposal.hasNextLogWeightSamplePair())
	  {
		  logWeight = proposal.nextLogWeight();
		  population.insertLogWeight(logWeight);
	      logWeights.add(logWeight);
	  }
	  
	  // determine if we need to find c, in other words, do we have enough room to store all the particles or do we have to resample?
	  if (logWeights.size() <= options.numberOfConcreteParticles)
	  {
		  // room to store all of the particles, no need to do any resampling
		  return Double.POSITIVE_INFINITY;
	  }
	  
	  // 2. sort the weights
	  Collections.sort(logWeights);

	  // 3. find kappa
	  double logSum = population.getLogSum();
	  List<Double> kappas = new ArrayList<>();
	  
	  // if every weight is such that N*q_j > 1, then there does not exist a kappa, choose kappa = N
	  double kappa = 0.0;
	  double num = options.numberOfConcreteParticles * Math.exp(logWeights.get(logWeights.size() - 1) - logSum);
	  if (num <= 1.0)
	  {
		  kappa = options.numberOfConcreteParticles;
	  }
	  else 
	  {
		  findKappa(options.numberOfConcreteParticles, logWeights, logSum, 0, logWeights.size(), kappas);
		  if (kappas.size() == 0)
			  throw new RuntimeException("Recursion to find kappa is not corretly implemented.");
	
		  kappa = kappas.get(kappas.size() - 1);
	  }

	  double Bkappa = 0.0;
	  int BkappaSize = 0;
	  for (int i = 0; i < logWeights.size(); i++)
	  {
		  double normalizedWeight = Math.exp(logWeights.get(i) - logSum);
		  if (normalizedWeight >= kappa) {
			  break;
		  }
		  Bkappa += normalizedWeight;
		  BkappaSize += 1;
	  }
	  int Akappa = logWeights.size() - BkappaSize;
	  double c = Bkappa > 0 ? (options.numberOfConcreteParticles - Akappa)/Bkappa : 1.0/kappa;
	  return c;
  }

  private static void findKappa(int N, List<Double> sortedWeights, double logSum, int begin, int end, List<Double> kappas)
  {
	  if (begin >= end) {
		  return;
	  }
	  int medianIdx = (end + begin)/2; 
	  double kappa = Math.exp(sortedWeights.get(medianIdx) - logSum);

	  // compute B_{\kappa} and A_{\kappa}
	  double Bkappa = 0.0;
	  for (int i = 0; i < medianIdx; i++)
	  {
		  Bkappa += Math.exp(sortedWeights.get(i) - logSum);
	  }
	  double Akappa = sortedWeights.size() - medianIdx;

	  // check condition (5) 
	  double condition = (Bkappa/kappa + Akappa);

	  if (condition > N) { // the current kappa is too small, try a larger value
		  findKappa(N, sortedWeights, logSum, medianIdx + 1, end, kappas);
	  } else { // the current kappa is a candidate so store it. Continue the search to see if there is a smaller kappa available that satisfies the condition (5)
		  kappas.add(kappa);
		  findKappa(N, sortedWeights, logSum, begin, medianIdx - 1, kappas);
	  }
  }

  public static final class PropagatorOptions
  {
    @Option 
    public boolean verbose = false;

    @Option(gloss = "Number of particles stored in memory.")
    public int numberOfConcreteParticles = DEFAULT_N_CONCRETE_PARTICLES;

    @Option(gloss = "Maximum number of implicit particles, represented via random number generation replay. Only costs CPU, not memory.")
    public int maxNumberOfVirtualParticles = 1000000;

    @Option(gloss = "Virtual particles will be used until that relative effective sampling size is reached (or maxNumberOfVirtualParticles is reached)")
    public double targetedRelativeESS = 0.5;
    
    @Option
    public Random resamplingRandom = new Random(1);
    
    @Option
    public ResamplingScheme resamplingScheme = ResamplingScheme.STRATIFIED;
    
    public static final int DEFAULT_N_CONCRETE_PARTICLES = 1000;
  }
}
