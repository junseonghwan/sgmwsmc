package common.learning;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import bayonet.opt.DifferentiableFunction;
import bayonet.opt.LBFGSMinimizer;
import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.graph.GraphNode;
import common.model.Command;
import common.smc.DiscreteParticleFilter;
import common.smc.DiscreteParticleFilter.DiscreteLatentSimulator;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.GenericDiscreteLatentSimulator;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;

public class BridgeSamplingLearning
{
	public static int LBFGS_ITER = 100;
	
	public static <F, NodeType extends GraphNode<?>> List<GenericGraphMatchingState<F, NodeType>> samplePathsUsingDPF(Random random, List<NodeType> nodes, List<Set<NodeType>> finalState, DiscreteLatentSimulator<GenericGraphMatchingState<F, NodeType>> transitionDensity, ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity, List<Object> emissions, int maxIter, int numParticles, int maxNumVirtualParticles)
	{
		// form map from node to finalState
		Map<NodeType, Set<NodeType>> finalNode2Matching = new HashMap<>();
		for (Set<NodeType> e : finalState)
		{
			for (NodeType v : e)
			{
				finalNode2Matching.put(v, e);				
			}
		}

		DiscreteParticleFilter<GenericGraphMatchingState<F, NodeType>, Object> dpf = new DiscreteParticleFilter<>(transitionDensity, observationDensity, emissions);
		dpf.options.numberOfConcreteParticles = 100;
		dpf.sample();

		// filter out paths that are not in the finalState
		List<GenericGraphMatchingState<F, NodeType>> ret = new ArrayList<>();
		for (GenericGraphMatchingState<F, NodeType> state : dpf.getSamples())
		{
			if (isFinalState(nodes, finalNode2Matching, state.getNode2EdgeView()))
			{
				ret.add(state);
			}
		}
		
		return ret;
	}

	public static <F, NodeType extends GraphNode<?>> List<GenericGraphMatchingState<F, NodeType>> samplePaths(Random random, List<NodeType> nodes, List<Set<NodeType>> finalState, LatentSimulator<GenericGraphMatchingState<F, NodeType>> transitionDensity, ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity, List<Object> emissions, int maxIter, int numParticles, int maxNumVirtualParticles)
	{
		// form map from node to finalState
		Map<NodeType, Set<NodeType>> finalNode2Matching = new HashMap<>();
		for (Set<NodeType> e : finalState)
		{
			for (NodeType v : e)
			{
				finalNode2Matching.put(v, e);				
			}
		}

		SequentialGraphMatchingSampler<F, NodeType> smc = null;
		if (numParticles == maxNumVirtualParticles)
			smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, false);
		else
			smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, true);

		smc.sample(random, numParticles, maxNumVirtualParticles);

		// filter out paths that are not in the finalState
		List<GenericGraphMatchingState<F, NodeType>> ret = new ArrayList<>();
		for (GenericGraphMatchingState<F, NodeType> state : smc.getSamples())
		{
			if (isFinalState(nodes, finalNode2Matching, state.getNode2EdgeView()))
			{
				ret.add(state);
			}
		}
		
		return ret;
	}

	public static <F, NodeType extends GraphNode<?>> boolean isFinalState(List<NodeType> nodes, Map<NodeType, Set<NodeType>> finalNode2Matching, Map<NodeType, Set<NodeType>> sampleNode2Matching)
	{
		for (NodeType u : nodes)
		{
			Set<NodeType> observed = finalNode2Matching.get(u);
			Set<NodeType> sample = sampleNode2Matching.get(u);
			if (!observed.containsAll(sample))
				return false;
		}
		return true;
	}
	
	public static <F, NodeType extends GraphNode<?>> Pair<Double, double []> learnUsingDPF(
		   Random random,
		   Command<F, NodeType> command,
		   List<Pair<List<NodeType>, List<Set<NodeType>>>> instances,
		   int numParticles, int maxVirtualParticles, double lambda, int maxIter, double tol)
  {
		double [] w = new double[command.getFeatureExtractor().dim()];
		for (int i = 0; i < w.length; i++) {
			w[i] = command.getModelParameters().getCount(command.getIndexer().i2o(i));
		}

		List<List<Object>> emissions = new ArrayList<>();
		List<GenericGraphMatchingState<F, NodeType>> initialStates = new ArrayList<>();
		for (Pair<List<NodeType>, List<Set<NodeType>>> instance : instances)
		{
			List<Object> emission = new ArrayList<>(instance.getFirst());
			emissions.add(emission);
			initialStates.add(GraphMatchingState.getInitialState(instance.getFirst()));
		}
		ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity = new ObservationDensity<GenericGraphMatchingState<F,NodeType>, Object>() {
			@Override
			public double logDensity(
					GenericGraphMatchingState<F, NodeType> latent,
					Object emission) {
				return 0;
			}

			@Override
			public double logWeightCorrection(
					GenericGraphMatchingState<F, NodeType> curLatent,
					GenericGraphMatchingState<F, NodeType> oldLatent) {
				return 0;
			}

			@Override
			public boolean cancellationApplied() {
				return false;
			}
		};

		int d = command.getFeatureExtractor().dim();
		double nllk = 0.0;
		double [] prev = new double[d], curr = new double[d];
		for (int iter = 0; iter < maxIter; iter++)
		{
			List<GenericGraphMatchingState<F, NodeType>> allPaths = new ArrayList<>();
			System.out.println("Iter: " + iter);
			for (int i = 0; i < instances.size(); i++)
			{
				System.out.println("Instance i: " + i);
				DiscreteLatentSimulator<GenericGraphMatchingState<F, NodeType>> transitionDensity = new GenericDiscreteLatentSimulator<>(command, initialStates.get(i), false);
				Pair<List<NodeType>, List<Set<NodeType>>> instance = instances.get(i);
				allPaths.addAll(samplePathsUsingDPF(random, instance.getFirst(), instance.getSecond(), transitionDensity, observationDensity, emissions.get(i), maxIter, numParticles, maxVirtualParticles));
				System.out.println("num paths: " + allPaths.size());
			}

			Pair<Double, double []> ret = executeMstep(command, prev, allPaths, null, LBFGS_ITER, lambda, true);
			nllk = ret.getFirst();
			curr = ret.getSecond();

			if (checkConvergence(prev, curr, tol))
			{
				break;
			}
			else
				prev = curr;
			
		}
		return Pair.create(nllk, curr);
  }

	public static <F, NodeType extends GraphNode<?>> Pair<Double, double []> learn(
		   Random random,
		   Command<F, NodeType> command,
		   List<Pair<List<NodeType>, List<Set<NodeType>>>> instances,
		   int numParticles, int maxVirtualParticles, double lambda, int maxIter, double tol, boolean checkGradient, List<String> paramTrajectory)
   {
		double [] w = new double[command.getFeatureExtractor().dim()];
		for (int i = 0; i < w.length; i++) {
			w[i] = command.getModelParameters().getCount(command.getIndexer().i2o(i));
			//w[i] = random.nextGaussian();
		}

		
		List<List<Object>> emissions = new ArrayList<>();
		List<GenericGraphMatchingState<F, NodeType>> initialStates = new ArrayList<>();
		for (Pair<List<NodeType>, List<Set<NodeType>>> instance : instances)
		{
			List<Object> emission = new ArrayList<>(instance.getFirst());
			emissions.add(emission);
			initialStates.add(GraphMatchingState.getInitialState(instance.getFirst()));
		}
		ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity = new ObservationDensity<GenericGraphMatchingState<F,NodeType>, Object>() {
			@Override
			public double logDensity(
					GenericGraphMatchingState<F, NodeType> latent,
					Object emission) {
				return 0;
			}

			@Override
			public double logWeightCorrection(
					GenericGraphMatchingState<F, NodeType> curLatent,
					GenericGraphMatchingState<F, NodeType> oldLatent) {
				return 0;
			}

			@Override
			public boolean cancellationApplied() {
				return false;
			}
		};

		int d = command.getFeatureExtractor().dim();
		double nllk = 0.0;
		double [] prev = w, curr = new double[d];
		int iter = 0;
		for (; iter < maxIter; iter++)
		{
			command.updateModelParameters(prev); // update the model parameters
			if (paramTrajectory != null) {
				for (F f : command.getModelParameters())
				{
					paramTrajectory.add(iter + ", " + f + ", " + command.getModelParameters().getCount(f));
				}
			}

			// simulate the decisions and permutations
			List<GenericGraphMatchingState<F, NodeType>> allPaths = new ArrayList<>();
			System.out.println("Iter: " + iter);
			for (int i = 0; i < instances.size(); i++)
			{
				System.out.println("Instance: " + i);
				LatentSimulator<GenericGraphMatchingState<F, NodeType>> transitionDensity = new GenericMatchingLatentSimulator<>(command, initialStates.get(i), false, true);
				Pair<List<NodeType>, List<Set<NodeType>>> instance = instances.get(i);
				allPaths.addAll(samplePaths(random, instance.getFirst(), instance.getSecond(), transitionDensity, observationDensity, emissions.get(i), maxIter, numParticles, maxVirtualParticles));
				System.out.println("num paths: " + allPaths.size());
			}

			// optimize for the parameters
			Pair<Double, double []> ret = executeMstep(command, prev, allPaths, null, LBFGS_ITER, lambda, checkGradient);
			nllk = ret.getFirst();
			curr = ret.getSecond();
			
			if (checkConvergence(prev, curr, tol))
			{
				break;
			}
			else
				prev = curr;

		}
		
		if (paramTrajectory != null) {
			for (F f : command.getModelParameters())
			{
				paramTrajectory.add(iter + ", " + f + ", " + command.getModelParameters().getCount(f) + ", " + nllk);
			}
		}

		return Pair.create(nllk, curr);
   }

	public static <F, NodeType extends GraphNode<?>> Pair<Double, double []> learnOnEntireBoard(
		   Random random,
		   Command<F, NodeType> command,
		   List<Pair<List<Set<NodeType>>, List<NodeType>>> instances,
		   int numParticles, int maxVirtualParticles, double lambda, int maxIter, double tol, boolean checkGradient, List<String> paramTrajectory)
  {
		double [] w = new double[command.getFeatureExtractor().dim()];
		for (int i = 0; i < w.length; i++) {
			//w[i] = command.getModelParameters().getCount(command.getIndexer().i2o(i));
			w[i] = random.nextGaussian();
		}

		List<List<Object>> emissions = new ArrayList<>();
		List<GenericGraphMatchingState<F, NodeType>> initialStates = new ArrayList<>();
		for (Pair<List<Set<NodeType>>, List<NodeType>> instance : instances)
		{
			List<Object> emission = new ArrayList<>(instance.getSecond());
			emissions.add(emission);
			initialStates.add(GraphMatchingState.getInitialState(instance.getSecond()));
		}
		ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity = new ObservationDensity<GenericGraphMatchingState<F,NodeType>, Object>() {
			@Override
			public double logDensity(
					GenericGraphMatchingState<F, NodeType> latent,
					Object emission) {
				return 0;
			}

			@Override
			public double logWeightCorrection(
					GenericGraphMatchingState<F, NodeType> curLatent,
					GenericGraphMatchingState<F, NodeType> oldLatent) {
				return 0;
			}

			@Override
			public boolean cancellationApplied() {
				return false;
			}
		};

		int d = command.getFeatureExtractor().dim();
		double nllk = 0.0;
		double [] prev = w, curr = new double[d];
		int iter = 0;
		for (; iter < maxIter; iter++)
		{
			command.updateModelParameters(prev); // update the model parameters
			if (paramTrajectory != null) {
				for (F f : command.getModelParameters())
				{
					paramTrajectory.add(iter + ", " + f + ", " + command.getModelParameters().getCount(f));
				}
			}

			// simulate the decisions and permutations
			List<GenericGraphMatchingState<F, NodeType>> allPaths = new ArrayList<>();
			System.out.println("Iter: " + iter);
			for (int i = 0; i < instances.size(); i++)
			{
				System.out.println("Instance: " + i);
				LatentSimulator<GenericGraphMatchingState<F, NodeType>> transitionDensity = new GenericMatchingLatentSimulator<>(command, initialStates.get(i), false, true);
				Pair<List<Set<NodeType>>, List<NodeType>> instance = instances.get(i);
				allPaths.addAll(samplePaths(random, instance.getSecond(), instance.getFirst(), transitionDensity, observationDensity, emissions.get(i), maxIter, numParticles, maxVirtualParticles));
				System.out.println("num paths: " + allPaths.size());
			}

			// optimize for the parameters
			Pair<Double, double []> ret = executeMstep(command, prev, allPaths, null, LBFGS_ITER, lambda, checkGradient);
			nllk = ret.getFirst();
			curr = ret.getSecond();
			
			if (checkConvergence(prev, curr, tol))
			{
				break;
			}
			else
				prev = curr;

		}
		
		if (paramTrajectory != null) {
			for (F f : command.getModelParameters())
			{
				paramTrajectory.add(iter + ", " + f + ", " + command.getModelParameters().getCount(f) + ", " + nllk);
			}
		}

		return Pair.create(nllk, curr);
  }

	
	public static <F, NodeType extends GraphNode<?>> Pair<Double, double []> executeMstep(Command<F, NodeType> command, double [] w, List<GenericGraphMatchingState<F, NodeType>> instances, SupportSet<double []> supportSet, int iter, double lambda, boolean checkGradient)
	{
  		BridgeObjectiveFunction<F, NodeType> objective = new BridgeObjectiveFunction<F, NodeType>(command, instances, lambda, supportSet);
		// check gradient
		// run the gradient checker
		if (checkGradient) {
			double h = 1e-10;
			double val1 = objective.valueAt(w);
			double [] grad1 = objective.derivativeAt(w);
			double [] grad2 = new double[w.length];
			for (int k = 0; k < w.length; k++)
			{
				w[k] += h;
				double val2 = objective.valueAt(w);
				w[k] -= h;
				grad2[k] = (val2 - val1)/h;
			}
			double diff = 0.0;
			for (int k = 0; k < w.length; k++)
			{
				double val = Math.abs(grad1[k] - grad2[k]);
				System.out.println(grad1[k] + "-" + grad2[k] + "=" + val);
				diff += val;
			}
			diff /= grad2.length;
			System.out.println("Gradient check: " + diff);
			if (diff > 0.1)
			{
				throw new RuntimeException();
			}
		}

  		LBFGSMinimizer minimizer = new LBFGSMinimizer(100);
  		minimizer.verbose = true;
  		double [] curr_w = minimizer.minimize(objective, w, 1e-6);
  		double nllk = objective.valueAt(curr_w);
  		System.out.println("curr nllk: " + nllk);
  		return Pair.create(nllk, curr_w);
	}
	
	public static boolean checkConvergence(double [] prev, double [] curr, double tol)
	{
		double absoluteDiff = 0.0;
		for (int i = 0; i < prev.length; i++)
		{
			absoluteDiff += Math.abs(prev[i] - curr[i]);
			System.out.println("w[" + i + "]: " + curr[i]);
		}
		absoluteDiff /= prev.length;
		System.out.println("param diff: " + absoluteDiff);
		if (absoluteDiff < tol)
			return true;
		else
			return false;
	}
	
	public static class BridgeObjectiveFunction<F, NodeType extends GraphNode<?>> implements DifferentiableFunction
	{
		private double logDensity;
		private Counter<F> logGradient = null;
		
		private double [] currX = null;
		private double lambda;

		private List<GenericGraphMatchingState<F, NodeType>> instances;

		private Command<F, NodeType> command;

		private SupportSet<double []> support = null;

		public BridgeObjectiveFunction(Command<F, NodeType> command, List<GenericGraphMatchingState<F, NodeType>> instances, double lambda, SupportSet<double []> support)
		{
			this.command = command;
			this.instances = instances;
			this.lambda = lambda;
			this.support = support;
			this.logGradient = new Counter<>();
		}

		@Override
		public int dimension() 
		{
			return command.getFeatureExtractor().dim();
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
		public double valueAt(double[] x) 
		{
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
			command.setModelParameters(params);
			
			for (GenericGraphMatchingState<F, NodeType> instance : instances)
			{
				Pair<Double, Counter<F>> ret = GraphMatchingState.computeLikelihood(command, instance.getVisitedNodes(), instance.getDecisions());
				logDensity += ret.getFirst();
				for (F f : ret.getSecond())
				{
					logGradient.incrementCount(f, ret.getSecond().getCount(f));
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
		public double[] derivativeAt(double[] x) 
		{
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
 