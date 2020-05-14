package common.learning;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.DoubleStream;

import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.apache.commons.math3.util.Pair;

import com.google.common.collect.Collections2;

import bayonet.opt.DifferentiableFunction;
import bayonet.opt.LBFGSMinimizer;
import briefj.BriefParallel;
import briefj.collections.Counter;
import briefj.run.Results;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.graph.GraphNode;
import common.model.Command;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.PruningObservationDensity;
import common.smc.components.SequentialGraphMatchingSampler;
import common.util.OutputHelper;

/**
 * Estimate the parameters given a list of matching
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 */
public class SupervisedLearning<F, NodeType extends GraphNode<?>>
{
	public static boolean parallelize = true;
	public static int numLBFGSIterations = 100;

	public Pair<Double, double[]> MAPviaMCEM(
			Random random, int repId,
			Command<F, NodeType> command,
			List<Pair<List<Set<NodeType>>, List<NodeType>>> instances,
			int maxIter,
			int numConcreteParticles,
			int numImplicitParticles,
			double lambda, 
			double [] initial,
			double tolerance,
			boolean checkGradient,
			boolean useSPF) {

		// prepare static components, outside of the MC-EM loop
		List<List<Object>> emissionsList = new ArrayList<>();
		List<GenericGraphMatchingState<F, NodeType>> initialStates = new ArrayList<>();
		for (int i = 0; i < instances.size(); i++)
		{
			List<Object> emissions = new ArrayList<>();
			emissions.addAll(instances.get(i).getSecond());
			emissionsList.add(emissions);

			initialStates.add(GraphMatchingState.getInitialState(instances.get(i).getSecond()));
		}

		int iter = 0;
		boolean converged = false;

		LBFGSMinimizer minimizer = new LBFGSMinimizer(numLBFGSIterations);
		minimizer.verbose = true;

		double [] w = initial;
		double nllk = 0.0;

		// Declare variables to track MC-EM diagnostic statistics
		List<double []> paramTrajectory = new ArrayList<>();
		List<Double> vars = new ArrayList<>();
		List<Double> means = new ArrayList<>();

		paramTrajectory.add(w);

		while (!converged && iter < maxIter)
		{

			// generate latent sequence permutation and the decisions using SMC sampler 
			// TODO: use parallelization
			ObjectiveFunction2<F, NodeType> objective = new ObjectiveFunction2<>(command);
			List<ObjectiveFunction2<F, NodeType>> objs = new ArrayList<>();
			for (int i = 0; i < instances.size(); i++)
			{
				List<GenericGraphMatchingState<F, NodeType>> samples = null;
				samples = generateSamples(random, instances.get(i), emissionsList.get(i), initialStates.get(i), command, numConcreteParticles, numImplicitParticles, useSPF);

//				if (iter < 10)
//					samples = generateSamples(random, instances.get(i), emissionsList.get(i), initialStates.get(i), command, numConcreteParticles, numImplicitParticles, useSPF);
//				else
//					samples = generateSamples(random, instances.get(i), emissionsList.get(i), initialStates.get(i), command, numConcreteParticles*5, numImplicitParticles*5, useSPF);

				objective.addInstances(samples);
				
				ObjectiveFunction2<F, NodeType> obj2 = new ObjectiveFunction2<>(command);
				obj2.addInstances(samples);
				objs.add(obj2);
			}
			
			// run the gradient checker
			if (checkGradient && iter == 0)
			{
				gradientChecker(objective, initial);
			}

			// take the samples and find the values of the parameters that minimize the objective function
			double [] randomW = new double[w.length];
			for (int i = 0; i < w.length; i++)
				randomW[i] = random.nextDouble();
			//double [] wNew = minimizer.minimize(objective, randomW, tolerance);
			double [] wNew = minimizer.minimize(objective, w, tolerance);
			double nllkNew = objective.valueAt(wNew);
			System.out.println("curr nllk: " + nllkNew);
			System.out.println("wNew: ");
			for (double ww : wNew)
				System.out.println(ww);

			//converged = MonteCarloExpectationMaximization.checkConvergence(w, wNew);
			converged = checkConvergence(nllk, nllkNew, tolerance);
			w = wNew;
			nllk = nllkNew;
			command.updateModelParameters(w);
			paramTrajectory.add(w);

			// compute the variance of Monte Carlo estimator for each of the instance
			double sumOfAvg = 0.0;
			double sumOfVar = 0.0;
			for (int i = 0; i < instances.size(); i++)
			{
				Pair<Double, Double> meanVar = objs.get(i).computeVariance(w);
				sumOfAvg += meanVar.getFirst();
				sumOfVar += meanVar.getSecond();
			}
			means.add(sumOfAvg);
			vars.add(sumOfVar);

			iter++;
		}
		
		File resultsDir = Results.getResultFolder();
		OutputHelper.writeVector(new File(resultsDir, "rep" + repId + "/sumOfMeans.csv"), means);
		OutputHelper.writeVector(new File(resultsDir, "rep" + repId + "/sumOfVars.csv"), vars);
		OutputHelper.writeTableAsCSV(new File(resultsDir, "rep" + repId + "/params.csv"), null, paramTrajectory);
		
		return Pair.create(nllk, w);
	}

	public static boolean checkConvergence(double oldNllk, double newNllk, double tol)
	{
		if (Math.abs(oldNllk - newNllk) < tol)
			return true;
		return false;
	}

	private List<GenericGraphMatchingState<F, NodeType>> generateSamples(
			Random random, 
			Pair<List<Set<NodeType>>, List<NodeType>> instance,
			List<Object> emissions,
			GenericGraphMatchingState<F, NodeType> initial,
			Command<F, NodeType> command,
			int numConcreteParticles,
			int numImplicitParticles,
			boolean useSPF)
	{
		GenericMatchingLatentSimulator<F, NodeType> transitionDensity = new GenericMatchingLatentSimulator<>(command, initial, false, true);
		ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity = new PruningObservationDensity<>(instance.getFirst());
		SequentialGraphMatchingSampler<F, NodeType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, useSPF);
		smc.sample(random, numConcreteParticles, numImplicitParticles);
		return smc.getSamples();
	}

	// returns nllk as well as the parameter estimates
	public Pair<Double, double[]> MAP(
			Command<F, NodeType> command,
			List<Pair<List<Set<NodeType>>, List<NodeType>>> instances, 
			double lambda, 
			double [] initial,
			double tolerance,
			boolean checkGradient)
	{
		ObjectiveFunction<F, NodeType> objective = new ObjectiveFunction<>(command, instances, lambda);

		// run the gradient checker
		if (checkGradient) {
			gradientChecker(objective, initial);
		}

		LBFGSMinimizer minimizer = new LBFGSMinimizer(numLBFGSIterations);
		minimizer.verbose = true;
		double [] w = minimizer.minimize(objective, initial, tolerance);
		double nllk = objective.valueAt(w);
		return Pair.create(nllk, w);
	}
	
	public Pair<Double, double[]> unknownSequenceMAP(Random random,
			Command<F, NodeType> command,
			List<Pair<List<Set<NodeType>>, List<NodeType>>> instances,
			int numSamples,
			double lambda, 
			double [] initial,
			double tolerance,
			boolean checkGradient)
		{
			// sample permutations for each instance -- equivalent to beefing up the number of training data
			List<Pair<List<Set<NodeType>>, List<NodeType>>> shuffledInstances = new ArrayList<>();
			for (Pair<List<Set<NodeType>>, List<NodeType>> instance : instances)
			{
				long numPossible;
				try {
					numPossible = CombinatoricsUtils.factorial(instance.getSecond().size());					
				} catch (MathArithmeticException ex) {
					numPossible = Long.MAX_VALUE;
				}
				if (numPossible <= numSamples) {
					for (List<NodeType> permutation : Collections2.permutations(instance.getSecond()))
					{
						shuffledInstances.add(Pair.create(instance.getFirst(), permutation));
					}
				} else {
  				for (int n = 0; n < numSamples; n++)
  				{
  					List<NodeType> shuffledNodes = new ArrayList<>(instance.getSecond());
  					Collections.shuffle(shuffledNodes, random);
  					shuffledInstances.add(Pair.create(instance.getFirst(), shuffledNodes));
  				}
				}
			}

			Counter<F> logGradients = new Counter<>();

			DifferentiableFunction func = new DifferentiableFunction() {
				
				@Override
				public double valueAt(double[] x) {
					Counter<F> params = new Counter<>();
					for (int i = 0; i < x.length; i++)
					{
						params.setCount(command.getIndexer().i2o(i), x[i]);
						logGradients.setCount(command.getIndexer().i2o(i), 0.0);
					}

					double logDensity = 0.0;
					for (Pair<List<Set<NodeType>>, List<NodeType>> instance : shuffledInstances) {
						Pair<Double, Counter<F>> ret = value(command, params, instance);
						logDensity += ret.getFirst();
						for (F f : params)
						{
							logGradients.incrementCount(f, ret.getSecond().getCount(f));
						}
					}
					
					logDensity /= numSamples;

					double regularization = 0.0;
					for (int j = 0; j < command.getFeatureExtractor().dim(); j++)
					{
						regularization += x[j] * x[j];
					}

					logDensity -= (regularization*lambda / 2.0);
					return -logDensity;
				}
				
				@Override
				public int dimension() {
					return command.getFeatureExtractor().dim();
				}
				
				@Override
				public double[] derivativeAt(double[] x) {
					valueAt(x);
					double [] ret = new double[x.length];
					for (int j = 0; j < x.length; j++)
					{
						ret[j] = logGradients.getCount(command.getIndexer().i2o(j))/numSamples;
						ret[j] -= lambda * x[j];
						ret[j] *= -1;
					}
					return ret;
				}
			};
		
			LBFGSMinimizer minimizer = new LBFGSMinimizer(100);
			minimizer.verbose = true;
			double [] w = minimizer.minimize(func, initial, 1e-6);
			double nllk = func.valueAt(w);
			return Pair.create(nllk, w);
		}
	
	public static <F, NodeType extends GraphNode<?>> Pair<Double, Counter<F>> evaluate(Command<F, NodeType> command, Counter<F> params, Pair<List<Set<NodeType>>, List<NodeType>> instance)
	{
		// compute the log likelihood and the log gradient
		return GraphMatchingState.evaluateDecision(command, params, instance);
	}

	/**
	 * Computes the log-likelihood, returns the log likelihood along with the log gradient
	 * 
	 * @param command
	 * @param instance
	 * @param fe
	 * @param param
	 * @return
	 */
	public static <F, NodeType extends GraphNode<?>> Pair<Double, Counter<F>> value(Command<F, NodeType> command, Counter<F> params, Pair<List<Set<NodeType>>, List<NodeType>> instance)
	{
		return command.constructFinalState(instance.getSecond(), instance.getFirst(), params);
	}

	public static <F, NodeType extends GraphNode<?>> Pair<Double, Counter<F>> valueRandomizedSequence(Random random, int numSamples, Command<F, NodeType> command, Pair<List<Set<NodeType>>, List<NodeType>> instance, GraphFeatureExtractor<F, NodeType> fe, Counter<F> param, boolean parallelize)
	{
		// this part can benefit from some parallelization but it is not so straight forward. here is the first attempt.
		
		if (parallelize) {
			List<Pair<List<Set<NodeType>>, List<NodeType>>> shuffledInstances = new ArrayList<>();
			// have to make a large number of copies in order to parallelize
			for (int n = 0; n < numSamples; n++)
			{
				List<NodeType> shuffledNodes = new ArrayList<>(instance.getSecond());
				Collections.shuffle(shuffledNodes, random);
				shuffledInstances.add(Pair.create(instance.getFirst(), shuffledNodes));
			}

			double [] logLiks = new double[numSamples];
			List<Counter<F>> logGradients = new ArrayList<>(numSamples);
			BriefParallel.process(numSamples, 8, (i) -> {
				Pair<List<Set<NodeType>>, List<NodeType>> shuffledInstance = shuffledInstances.get(i);
				Pair<Double, Counter<F>> ret = value(command, command.getModelParameters(), Pair.create(instance.getFirst(), shuffledInstance.getSecond()));
				logLiks[i] = ret.getFirst();
				logGradients.set(i, ret.getSecond());
			});
			
			// now combine the gradients
			Counter<F> logGradient = new Counter<>();
			for (Counter<F> lg : logGradients)
			{
				for (F f : lg)
				{
					logGradient.incrementCount(f, lg.getCount(f));
				}
			}
			double logLik = DoubleStream.of(logLiks).sum();
			return Pair.create(logLik, logGradient);
		} else {
		
			// below serial implementation
			double logLik = 0.0;
			Counter<F> logGradient = new Counter<>();
			for (int i = 0; i < numSamples; i++)
			{
				List<NodeType> copyNodes = new ArrayList<>(instance.getSecond());
				Collections.shuffle(copyNodes, random);
				Pair<Double, Counter<F>> ret = value(command, command.getModelParameters(), Pair.create(instance.getFirst(), copyNodes));
				logLik += ret.getFirst();
				for (F f: ret.getSecond())
				{
					logGradient.incrementCount(f, ret.getSecond().getCount(f));
				}
			}
	
			logLik /= numSamples;
			for (F f : logGradient)
			{
				logGradient.setCount(f, logGradient.getCount(f)/numSamples);
			}
			return Pair.create(logLik, logGradient);
		}
	}

	public void gradientChecker(DifferentiableFunction objective, double [] initial)
	{
		double h = 1e-7;
		double val1 = objective.valueAt(initial);
		double [] grad1 = objective.derivativeAt(initial);
		double [] grad2 = new double[initial.length];
		for (int k = 0; k < initial.length; k++)
		{
			initial[k] += h;
			double val2 = objective.valueAt(initial);
			initial[k] -= h;
			grad2[k] = (val2 - val1)/h;
		}
		double diff = 0.0;
		for (int k = 0; k < initial.length; k++)
		{
			double val = Math.abs(grad1[k] - grad2[k]);
			System.out.println(grad1[k] + "-" + grad2[k] + "=" + val);
			diff += val;
		}
		diff /= grad2.length;
		System.out.println("Gradient check: " + diff);
		if (diff > 0.01)
		{
			throw new RuntimeException();
		}
	}
	
	/**
	 * Objective function using 
	 * 
	 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
	 *
	 * @param <F>
	 * @param <NodeType>
	 */
	public static class ObjectiveFunction2<F, NodeType extends GraphNode<?>> implements DifferentiableFunction
	{
		private List<Pair<List<Set<NodeType>>, List<NodeType>>> latentDecisions;
		private Command<F, NodeType> command; 
		
		private double logDensity;
		private Counter<F> logGradient = null;
		
		private double [] currX = null;
		private double lambda;

		private SupportSet<double []> support = null;

		public ObjectiveFunction2(Command<F, NodeType> command)
		{
			this.command = command;
			latentDecisions = new ArrayList<>();
			this.logGradient = new Counter<>();
		}
		
		public void addInstances(List<GenericGraphMatchingState<F, NodeType>> samples)
		{
			// construct instances to be used for objective function
			for (GenericGraphMatchingState<F, NodeType> sample : samples)
			{
				latentDecisions.add(Pair.create(sample.getDecisions(), sample.getVisitedNodes()));
			}
		}
		
		public Pair<Double, Double> computeVariance(double [] w)
		{
			command.updateModelParameters(w);
			SummaryStatistics stat = new SummaryStatistics();
			if (parallelize) {
				int N = latentDecisions.size();
				List<Counter<F>> results = new ArrayList<>(N);
				for (int i = 0; i < N; i++) results.add(null);

				BriefParallel.process(N, 8, (i) -> {
					Pair<Double, Counter<F>> ret = evaluate(command, command.getModelParameters(), latentDecisions.get(i));
					stat.addValue(ret.getFirst());
				});


			} else {

				// serial version
				//int i = 0;
				for (Pair<List<Set<NodeType>>, List<NodeType>> instance : latentDecisions)
				{
					//System.out.println("test:" + i);
					//i++;
					Pair<Double, Counter<F>> ret = evaluate(command, command.getModelParameters(), instance);
					stat.addValue(ret.getFirst());
				}
			}

			return Pair.create(stat.getMean(), stat.getVariance());
		}

		private boolean requiresComputation(double [] x)
		{
			if (currX == null) return true;
			
			if (x == currX) 
				return false;
			
			for (int i = 0; i < x.length; i++) {
				if (currX[i] != x[i]) 
					return true;
			}
			return false;
		}
		
		@Override
		public double valueAt(double[] x) {
			
			if (support != null && !support.inSupport(x)) {
				// not in support
				for (F f : logGradient)
				{
					logGradient.setCount(f, Double.MAX_VALUE);
				}
				return Double.MAX_VALUE;
			}
			
			if (!requiresComputation(x))
				return -logDensity;
			
			if (this.currX == null)
				this.currX = new double[x.length];

			logDensity = 0.0;
			Counter<F> params = new Counter<>();
			for (int i = 0; i < x.length; i++)
			{
				currX[i] = x[i];
				logGradient.setCount(command.getIndexer().i2o(i), 0.0);
				params.setCount(command.getIndexer().i2o(i), x[i]);
			}

			if (parallelize) {
				int N = latentDecisions.size();
				List<Counter<F>> results = new ArrayList<>(N);
				for (int i = 0; i < N; i++) results.add(null);
				double [] logLiks = new double[N];

				BriefParallel.process(N, 8, (i) -> {
					Pair<Double, Counter<F>> ret = evaluate(command, params, latentDecisions.get(i));
					results.set(i, ret.getSecond());
					logLiks[i] = ret.getFirst();
				});

				for (Counter<F> ret : results)
				{
					for (F f : logGradient)
					{
						logGradient.incrementCount(f, ret.getCount(f));
					}
				}
				
				logDensity = DoubleStream.of(logLiks).parallel().sum();

			} else {

				// serial version
				//int i = 0;
				for (Pair<List<Set<NodeType>>, List<NodeType>> instance : latentDecisions)
				{
					//System.out.println("test:" + i);
					//i++;
					Pair<Double, Counter<F>> ret = evaluate(command, params, instance);
					logDensity += ret.getFirst();
					for (F f : ret.getSecond())
					{
						logGradient.incrementCount(f, ret.getSecond().getCount(f));
					}
				}
			}

			double regularization = 0.0;
			for (int j = 0; j < command.getFeatureExtractor().dim(); j++)
			{
				regularization += x[j] * x[j];
			}

			logDensity -= (regularization*lambda / 2.0);
			return -logDensity;
		}

		@Override
		public int dimension() {
			return command.getFeatureExtractor().dim();
		}
		
		@Override
		public double[] derivativeAt(double[] x) {
			if (requiresComputation(x))
				valueAt(x);
			
			// convert to double array
			double [] ret = new double[x.length];
			for (int j = 0; j < x.length; j++)
			{
				ret[j] = logGradient.getCount(command.getIndexer().i2o(j));
				ret[j] -= lambda * x[j];
				ret[j] *= -1;
			}
			return ret;
		}
	}

	public static class ObjectiveFunction<F, NodeType extends GraphNode<?>> implements DifferentiableFunction
	{
			private double logDensity;
			private Counter<F> logGradient = null;
			
			private double [] currX = null;
			private double lambda;
			
			private List<Pair<List<Set<NodeType>>, List<NodeType>>> instances;

			private Command<F, NodeType> command;

			private SupportSet<double []> support = null;

			public ObjectiveFunction(Command<F, NodeType> command, List<Pair<List<Set<NodeType>>, List<NodeType>>> instances)
			{
				this.command = command;
				this.instances = instances;
				this.logGradient = new Counter<>();
			}

			public ObjectiveFunction(Command<F, NodeType> command, List<Pair<List<Set<NodeType>>, List<NodeType>>> instances, double lambda)
			{
				this(command, instances);
				this.lambda = lambda;
			}
			
			public ObjectiveFunction(Command<F, NodeType> command, List<Pair<List<Set<NodeType>>, List<NodeType>>> instances, double lambda, SupportSet<double []> support)
			{
				this(command, instances, lambda);
				this.support = support;
			}


			private boolean requiresComputation(double [] x)
			{
				if (currX == null) return true;
				
				if (x == currX) 
					return false;
				
				for (int i = 0; i < x.length; i++) {
					if (currX[i] != x[i]) 
						return true;
				}
				return false;
			}
			
			@Override
			public double valueAt(double[] x) {
				
				if (support != null && !support.inSupport(x)) {
					// not in support
					for (F f : logGradient)
					{
						logGradient.setCount(f, Double.MAX_VALUE);
					}
					return Double.MAX_VALUE;
				}
				
				if (!requiresComputation(x))
					return -logDensity;
				
				if (this.currX == null)
					this.currX = new double[x.length];

				logDensity = 0.0;
				Counter<F> params = new Counter<>();
				for (int i = 0; i < x.length; i++)
				{
					currX[i] = x[i];
					logGradient.setCount(command.getIndexer().i2o(i), 0.0);
					params.setCount(command.getIndexer().i2o(i), x[i]);
				}

				if (parallelize) {
					int N = instances.size();
					List<Counter<F>> results = new ArrayList<>(N);
					for (int i = 0; i < N; i++) results.add(null);
					double [] logLiks = new double[N];
					
					BriefParallel.process(N, 8, (i) -> {
						Pair<Double, Counter<F>> ret = value(command, params, instances.get(i));
						results.set(i, ret.getSecond());
						logLiks[i] = ret.getFirst();
					});

					for (Counter<F> ret : results)
					{
						for (F f : logGradient)
						{
							logGradient.incrementCount(f, ret.getCount(f));
						}
					}
					
					logDensity = DoubleStream.of(logLiks).parallel().sum();

				} else {

					// serial version
					//int i = 0;
					for (Pair<List<Set<NodeType>>, List<NodeType>> instance : instances)
					{
						//System.out.println("test:" + i);
						//i++;
						Pair<Double, Counter<F>> ret = value(command, params, instance);
						logDensity += ret.getFirst();
						for (F f : ret.getSecond())
						{
							logGradient.incrementCount(f, ret.getSecond().getCount(f));
						}
					}
				}

				double regularization = 0.0;
				for (int j = 0; j < command.getFeatureExtractor().dim(); j++)
				{
					regularization += x[j] * x[j];
				}

				logDensity -= (regularization*lambda / 2.0);
				return -logDensity;
			}

			@Override
			public int dimension() {
				return command.getFeatureExtractor().dim();
			}
			
			@Override
			public double[] derivativeAt(double[] x) {
				if (requiresComputation(x))
					valueAt(x);
				
				// convert to double array
				double [] ret = new double[x.length];
				for (int j = 0; j < x.length; j++)
				{
					ret[j] = logGradient.getCount(command.getIndexer().i2o(j));
					ret[j] -= lambda * x[j];
					ret[j] *= -1;
				}
				return ret;
			}
	}

}
