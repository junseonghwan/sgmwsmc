package common.evaluation;

import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import briefj.Indexer;
import briefj.collections.Counter;
import common.graph.GraphNode;
import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;

public class LearningUtils
{
	public static <F, NodeType extends GraphNode<F>> Pair<Double, Counter<F>> learnParameters(
			Random random, 
			DecisionModel<F, NodeType> decisionModel, 
			GraphFeatureExtractor<F, NodeType> fe, 
			List<Pair<List<Set<NodeType>>, List<NodeType>>> instances,
			double lambda, 
			double tol)
	{
		Counter<F> params = fe.getDefaultParameters();
		Indexer<F> indexer = new Indexer<>(params.keySet());
		double [] initial = new double[fe.dim()];
		for (int i = 0; i < initial.length; i++)
		{
			initial[i] = random.nextGaussian();
			params.setCount(indexer.i2o(i), initial[i]);
		}

		Command<F, NodeType> command = new Command<>(decisionModel, fe);
		SupervisedLearning<F, NodeType> sl = new SupervisedLearning<>();
		Pair<Double, double[]> ret = sl.MAP(command, instances, lambda, initial, tol, true);
		//System.out.println("-logDensity: " + ret.getFirst());
		/*
		for (int i = 0; i < ret.getSecond().length; i++)
		{
			F f = indexer.i2o(i);
			System.out.println(f + ": " + ret.getSecond()[i]);
		}
		*/

		for (int i = 0; i < ret.getSecond().length; i++)
		{
			F f = indexer.i2o(i);
			params.setCount(f, ret.getSecond()[i]);
		}
		return Pair.create(ret.getFirst(), params);
	}

}
