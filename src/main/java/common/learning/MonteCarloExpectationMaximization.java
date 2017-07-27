package common.learning;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import bayonet.opt.LBFGSMinimizer;
import briefj.BriefIO;
import briefj.collections.Counter;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;
import common.learning.SupervisedLearning.ObjectiveFunction;
import common.model.Command;
import common.processor.Processor;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;

public class MonteCarloExpectationMaximization<F, NodeType extends GraphNode<?>> 
{
	public static double tol = 1e-2;
	public static int numConcreteParticles = 100;
	public static int maxNumVirtualParticles = 10000;
	public static int maxIter = 100;
	public static int burnIn = 50;
	public static boolean plotSurface = false;

	public double execute(Random random,
			Command<F, NodeType> command,
			List<NodeType> nodes,
			List<Set<NodeType>> truth,
			GenericGraphMatchingState<F, NodeType> initialState,
			GenericMatchingLatentSimulator<F, NodeType> transitionDensity,
			ObservationDensity<GenericGraphMatchingState<F, NodeType>, Object> observationDensity, 
			SupportSet<double []> supportSet,
			double lambda,
			Processor<Pair<Double, Counter<F>>> processor)
	{
		int iter = 0;
		boolean converged = false;
		double [] w = new double[command.getFeatureExtractor().dim()];
		for (int i = 0; i < w.length; i++) {
			w[i] = command.getModelParameters().getCount(command.getIndexer().i2o(i));
		}
		//command.updateModelParameters(w);
		List<Object> emissions = new ArrayList<>(nodes.size());
		for (int i = 0; i < nodes.size(); i++) emissions.add(null);

		double nllk = 0.0;

		while (!converged && iter < maxIter)
		{
			// the parameter gets updated inside command so no need to create a new instance each time
			SequentialGraphMatchingSampler<F, NodeType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
			if (iter < burnIn)
				smc.sample(numConcreteParticles, maxNumVirtualParticles);
			else
				smc.sample(numConcreteParticles, maxNumVirtualParticles);
			/*
			GenericDiscreteLatentSimulator<F, NodeType> transition = new GenericDiscreteLatentSimulator<>(command, initialState, true);
			DiscreteParticleFilter<GenericGraphMatchingState<F, NodeType>, Object> dpf = new DiscreteParticleFilter<>(transition, observationDensity, emissions);
			dpf.options.numberOfConcreteParticles = numConcreteParticles;
			dpf.options.verbose = false;
			dpf.sample();
			*/
	  		List<Pair<List<Set<NodeType>>, List<NodeType>>> instances = new ArrayList<>();
	  		if (truth != null) {
	  			MatchingSampleEvaluation<F, NodeType> mse = MatchingSampleEvaluation.evaluate(smc.getSamples(), truth);
	  			//MatchingSampleEvaluation<F, NodeType> mse = MatchingSampleEvaluation.evaluate(dpf.getSamples(), truth);
	  			System.out.println("Consensus: " + mse.consensusMatching.getSecond() + "/" + truth.size());
	  			System.out.println("Best loglik: " + mse.bestLogLikMatching.getSecond().getFirst());
	  			System.out.println("MAP: " + mse.bestLogLikMatching.getSecond().getSecond() + "/" + truth.size());
	  			System.out.println("Best: " + mse.bestAccuracyMatching.getSecond() + "/" + truth.size());
	  			//instances.add(Pair.create(mse.consensusMatching.getFirst(), nodes));
	  			//instances.add(Pair.create(mse.bestLogLikMatching.getFirst().getMatchings(), nodes));
	  		}

	  		for (GenericGraphMatchingState<F, NodeType> sample : smc.getSamples())
	  		//for (GenericGraphMatchingState<F, NodeType> sample : dpf.getSamples())
	  		{
	  			instances.add(Pair.create(sample.getMatchings(), sample.getVisitedNodes()));
	  		}

	  		ObjectiveFunction<F, NodeType> obj = new ObjectiveFunction<>(command, instances, lambda, supportSet);
	  		double prev_nllk = obj.valueAt(w);

	  		double [] initial = new double[w.length];
	  		supportSet.initParam(random, initial);
	  		Pair<Double, double []> ret = evaluateSurface(command, initial, instances, supportSet, iter, lambda, plotSurface);
	  		nllk = ret.getFirst();
	  		System.out.println("nllk change: " + nllk + " - " + prev_nllk + " = " + (nllk - prev_nllk));
	  		System.out.println("relative change: (" + nllk + " - " + prev_nllk + ") /" + nllk + " = " + (nllk - prev_nllk)/nllk);

	  		// check for convergence
  			converged = checkConvergence(w, ret.getSecond());
  			System.arraycopy(ret.getSecond(), 0, w, 0, ret.getSecond().length);
	  		command.updateModelParameters(w);
	  		iter++;

	  		for (F f : command.getModelParameters()) {
	  			System.out.println(f + ": " + command.getModelParameters().getCount(f));
				}
	  		//System.out.println("current nllk: " + nllk);
	  		processor.process(Pair.create(nllk, command.getModelParameters()));
	  		System.out.println("=====");

		}

		return nllk;
	}

	public static boolean checkConvergence(double [] prev, double [] curr)
	{
		double absoluteDiff = 0.0;
		for (int i = 0; i < prev.length; i++)
		{
			absoluteDiff += Math.abs(prev[i] - curr[i]);
		}
		absoluteDiff /= prev.length;
		System.out.println("param diff: " + absoluteDiff);
		if (absoluteDiff < tol)
			return true;
		else
			return false;
	}

	public static int nGrid = 50;
	public static double [] x_dim = {-5, 0.0001};
	public static double [] y_dim = {-5, 0.0001};
	public static String outputFile = "mcem_surface";
	public Pair<Double, double []> evaluateSurface(Command<F, NodeType> command, double [] w, List<Pair<List<Set<NodeType>>, List<NodeType>>> instances, SupportSet<double []> supportSet, int iter, double lambda, boolean evaluateSurface)
	{
		// evaluate the surface over a grid of points

  		ObjectiveFunction<F, NodeType> objective = new ObjectiveFunction<F, NodeType>(command, instances, lambda, supportSet);
  		LBFGSMinimizer minimizer = new LBFGSMinimizer(100);
  		minimizer.verbose = false;
  		double [] curr_w = minimizer.minimize(objective, w, 1e-3);
  		double nllk = objective.valueAt(curr_w);
  		System.out.println("curr nllk: " + nllk);

		if (command.getFeatureExtractor().dim() != 2)
			return Pair.create(nllk, curr_w);


		if (evaluateSurface) {
			PrintWriter writer = BriefIO.output(new File(outputFile + "_" + iter + ".csv"));
			writer.println(curr_w[0] + ", " + curr_w[1] + ", " + nllk);
	  		double x_len = x_dim[1] - x_dim[0];
	  		double y_len = y_dim[1] - y_dim[0];
	  		for (int i = 0; i < nGrid; i++)
	  		{
	  			double x = x_dim[0] + i * x_len/nGrid;
	  			StringBuilder sb = new StringBuilder();
	  			for (int j = 0; j < nGrid; j++)
	  			{
	  					double y = y_dim[0] + j * y_len/nGrid;
  	  	  		double val = objective.valueAt(new double[]{x, y});
  	  	  		sb.append(x + ", " + y + ", " + val + "\n");
	  			}
  	  		writer.print(sb.toString());
	  		}
	  		writer.close();
		}
  		return Pair.create(nllk, curr_w);
	}

}
