package common.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Stack;

import org.apache.commons.math3.util.Pair;

import briefj.Indexer;
import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingStateFactory;
import common.graph.GraphNode;

/** 
 * Collection of objects and method invocation shared across experiments
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 * @param <NodeType>
 */
public class Command<F, NodeType extends GraphNode<?>>
{
	private DecisionModel<F, NodeType> decisionModel;
	private GraphFeatureExtractor<F, NodeType> fe;
	private Counter<F> params;
	private Indexer<F> indexer;

	public Command(DecisionModel<F, NodeType> decisionModel, GraphFeatureExtractor<F, NodeType> fe)
	{
		this(decisionModel, fe, fe.getDefaultParameters());
	}
	
	public Command(DecisionModel<F, NodeType> decisionModel, GraphFeatureExtractor<F, NodeType> fe, Counter<F> params)
	{
		this.fe = fe;
		this.decisionModel = decisionModel;
		setModelParameters(params);
	}
	
	public void setModelParameters(Counter<F> params)
	{
		if (params.size() != fe.dim())
			throw new RuntimeException("The parameter dimension and feature dimension does not match.");
			
		this.params = params;
		this.indexer = new Indexer<>(params.keySet());
	}
	
	public Counter<F> getModelParameters()
	{
		return params;
	}
	
	public GraphFeatureExtractor<F, NodeType> getFeatureExtractor()
	{
		return this.fe;
	}
	
	public Indexer<F> getIndexer()
	{
		return indexer;
	}

	public void updateModelParameters(double [] w)
	{
		if (w.length != this.fe.dim())
			throw new RuntimeException("Unable to update parameter because the dimensions do not match.");
		
		for (int i = 0; i < w.length; i++)
		{
			this.params.setCount(indexer.i2o(i), w[i]);
		}
	}

	public void sampleNext(Random random, GenericGraphMatchingState<F, NodeType> state, boolean sequentialSampling, boolean exactSampling)
	{
		state.sampleNextState(random, this, sequentialSampling, exactSampling);
	}
	
	public MultinomialLogisticModel<F, NodeType> getCurrentModel()
	{
		return new MultinomialLogisticModel<>(fe, params);
	}
	
	public static <F, NodeType extends GraphNode<?>> MultinomialLogisticModel<F, NodeType> constructModel(GraphFeatureExtractor<F, NodeType> fe, Counter<F> params)
	{
		return new MultinomialLogisticModel<>(fe, params);
	}
	
	public static <F, NodeType extends GraphNode<?>> MultinomialLogisticModel<F, NodeType> getModel(GraphFeatureExtractor<F, NodeType> fe, Counter<F> params)
	{
		return new MultinomialLogisticModel<>(fe, params);
	}
	
	public List<GenericGraphMatchingState<F, NodeType>> generateNextSamples(GenericGraphMatchingState<F, NodeType> state, boolean sequentialSampling)
	{
		MultinomialLogisticModel<F, NodeType> model = new MultinomialLogisticModel<>(fe, params);
		return state.generateDescendants(model, decisionModel, sequentialSampling);
	}
	
	public DecisionModel<F, NodeType> getDecisionModel()
	{
		return decisionModel;
	}

	// find a set of decisions that lead to the matching
	// note: there may be more than one sequence of decisions that lead to the same matching -- this method will enumerate them (should not be too many)
	// TODO: offer a variant of this method that samples the decision in case the number of paths that lead to the same matching explode (not encountered yet)
	// returns log likelihood and log derivative
	/*
	public Pair<Double, Counter<F>> constructFinalState(List<NodeType> permutation, List<Set<NodeType>> matching, MultinomialLogisticModel<F, NodeType> model)	
	{
		Map<NodeType, Set<NodeType>> finalState = new HashMap<>();
		for (Set<NodeType> edge : matching)
		{
			for (NodeType node : edge)
			{
				finalState.put(node, edge);
			}
		}
		
		List<GraphMatchingState<F, NodeType>> allStates = new ArrayList<>();
		
		GraphMatchingState<F, NodeType> state = GraphMatchingState.getInitialState(permutation);
		Stack<GraphMatchingState<F, NodeType>> stack = new Stack<>();
		stack.push(state);
		while (stack.size() > 0)
		{
			GraphMatchingState<F, NodeType> newState = stack.pop();
			if (newState.hasNextStep())
				stack.addAll(newState.executeMove(model, decisionModel, finalState));
			else
				allStates.add(newState);
		}
		
		// combine the likelihood and log gradients
		double logDensity = 0.0;
		Counter<F> gradientCounter = new Counter<>();
		for (GraphMatchingState<F, NodeType> s : allStates)
		{
			logDensity += s.getLogDensity();
			Counter<F> stateGradient = s.getLogGradient();

			for (F f : stateGradient)
				gradientCounter.incrementCount(f, stateGradient.getCount(f));
		}

		return Pair.create(logDensity, gradientCounter);
		//GraphFeatureVector<F> logGradient = GraphFeatureVector.createFeatureVectorFromCounter(gradientCounter);
		//return Pair.create(logDensity, logGradient);
	}
	*/
	
	public Pair<Double, Counter<F>> constructFinalState(List<NodeType> nodes, List<Set<NodeType>> finalMatching, Counter<F> params)	
	{
		MultinomialLogisticModel<F, NodeType> model = Command.constructModel(this.fe, params);
		
		Map<NodeType, Set<NodeType>> finalState = new HashMap<>();
		for (Set<NodeType> edge : finalMatching)
		{
			for (NodeType node : edge)
			{
				finalState.put(node, edge);
			}
		}

		List<GenericGraphMatchingState<F, NodeType>> allStates = new ArrayList<>();

		GenericGraphMatchingState<F, NodeType> state = GraphMatchingStateFactory.createInitialGraphMatchingState(this, nodes);
		Stack<GenericGraphMatchingState<F, NodeType>> stack = new Stack<>();
		stack.push(state);
		while (stack.size() > 0)
		{
			GenericGraphMatchingState<F, NodeType> newState = stack.pop();
			if (newState.hasNextStep()) {
				stack.addAll(newState.executeMove(model, decisionModel, finalState));
				//stack.add(newState.executeMove(model, decisionModel, finalState).get(0));
			} else
				allStates.add(newState);
		}

		//System.out.println(finalMatching);
		if (allStates.size() == 0) {
			/*
			System.out.println("bug: " + finalMatching);
			allStates = new ArrayList<>();

			state = GraphMatchingStateFactory.createInitialGraphMatchingState(this, nodes);
			stack = new Stack<>();
			stack.push(state);
			while (stack.size() > 0)
			{
				GenericGraphMatchingState<F, NodeType> newState = stack.pop();
				if (newState.hasNextStep()) {
					stack.addAll(newState.executeMove(model, decisionModel, finalState));
					//stack.add(newState.executeMove(model, decisionModel, finalState).get(0));
				} else
					allStates.add(newState);
			}
			*/
			/*
			System.out.println("bug: " + finalMatching);
			throw new RuntimeException();
			*/
		}
		//System.out.println("numStates=" + allStates.size());
		
		// combine the likelihood and log gradients
		double logDensity = 0.0;
		Counter<F> gradientCounter = new Counter<>();
		for (GenericGraphMatchingState<F, NodeType> s : allStates)
		{
			logDensity += s.getLogDensity();
			Counter<F> stateGradient = s.getLogGradient();

			for (F f : stateGradient)
				gradientCounter.incrementCount(f, stateGradient.getCount(f));
		}

		return Pair.create(logDensity, gradientCounter);
		//return Pair.create(allStates.get(0).getLogDensity(), allStates.get(0).getLogGradient());
	}

}
