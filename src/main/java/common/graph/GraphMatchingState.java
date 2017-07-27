package common.graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import bayonet.distributions.Multinomial;
import bayonet.math.NumericalUtils;
import briefj.collections.Counter;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.MultinomialLogisticModel;

// data type representing a general graph matching state 
public class GraphMatchingState<F, NodeType extends GraphNode<?>> implements GenericGraphMatchingState<F, NodeType> 
{
	//protected List<Set<NodeType>> matching;
	protected Set<Set<NodeType>> matching;
	protected Set<NodeType> coveredNodes;
	protected List<NodeType> visitedNodes;
	protected List<NodeType> unvisitedNodes;
	protected Map<NodeType, Set<NodeType>> nodeToMatching;
	protected List<Set<NodeType>> decisions;

	protected double logDensity = 0.0;
	protected Counter<F> logGradient = new Counter<>();

	protected GraphMatchingState()
	{
	}

	private GraphMatchingState(List<NodeType> nodes)
	{
		this.unvisitedNodes = new ArrayList<>(nodes);

		//this.matching = new ArrayList<>();
		this.matching = new HashSet<>();
		this.nodeToMatching = new HashMap<>();

		//this.visitedNodes = new HashSet<>();
		this.visitedNodes = new ArrayList<>();
		this.coveredNodes = new HashSet<>();
		
		this.decisions = new ArrayList<>();
	}

	public static <F, NodeType extends GraphNode<?>> GraphMatchingState<F, NodeType> getInitialState(List<NodeType> nodes)
	{
		GraphMatchingState<F, NodeType> initialState = new GraphMatchingState<>(nodes);
		return initialState;
	}

	@Override
	public List<Set<NodeType>> getDecisions()
	{
		return this.decisions;
	}
	
	public static <F, NodeType extends GraphNode<?>> Pair<Double, Counter<F>> computeLikelihood(Command<F, NodeType> command, List<NodeType> nodes, List<Set<NodeType>> decisions)
	{
		if (nodes.size() != decisions.size())
			throw new RuntimeException("The number of decisions should equal to the number of nodes.");

		List<NodeType> unvisited = new ArrayList<>(nodes);
		Set<NodeType> covered = new HashSet<>();
		Map<NodeType, Set<NodeType>> node2Edge = new HashMap<>();
		
		double logLik = 0.0;
		Counter<F> logGradient = new Counter<>();
		for (int i = 0; i < unvisited.size(); i++)
		{
			NodeType node = unvisited.remove(0);
			Set<NodeType> observedDecision = decisions.get(i);
			Counter<F> phi = null;
			Counter<F> numerator = new Counter<>();

			List<Set<NodeType>> m = new ArrayList<>(new HashSet<>(node2Edge.values())); // current matching state
			List<Set<NodeType>> decisionCandidates = command.getDecisionModel().getDecisions(node, unvisited, covered, m, node2Edge);

			boolean found = false;
			double logNorm = Double.NEGATIVE_INFINITY;
			for (Set<NodeType> decisionCandidate : decisionCandidates)
			{
				Pair<Double, Counter<F>> pair = command.getCurrentModel().logProb(node, decisionCandidate, m);
				logNorm = NumericalUtils.logAdd(logNorm, pair.getFirst());
				for (F f : pair.getSecond())
				{
					numerator.incrementCount(f, Math.exp(pair.getFirst()) * pair.getSecond().getCount(f));
				}

				if (observedDecision.containsAll(decisionCandidate))
				{
					found = true;
					phi = pair.getSecond();
					logLik += pair.getFirst();
				}
			}
			if (!found) {
				throw new RuntimeException("The observed decision sequences is not realizable.");				
			}

			// update the log-likelihood
			logLik -= logNorm;

			// update the gradient of the log-likelihood
			for (F f : phi)
			{
				logGradient.incrementCount(f, phi.getCount(f) - numerator.getCount(f)/Math.exp(logNorm));
			}

			// update the matching state
			if (!node2Edge.containsKey(node))
			{
				observedDecision.add(node);
				covered.addAll(observedDecision);
				for (NodeType u : observedDecision)
				{
					node2Edge.put(u, observedDecision);
				}
			}
		}

		return Pair.create(logLik, logGradient);
	}

	@Override
	public GraphMatchingState<F, NodeType> copyState()
	{
		GraphMatchingState<F, NodeType> copyState = new GraphMatchingState<>();
		//copyState.matching = new ArrayList<>(this.matching);
		copyState.matching = new HashSet<>(this.matching);
		//copyState.visitedNodes = new HashSet<>(this.visitedNodes);
		copyState.visitedNodes = new ArrayList<>(this.visitedNodes);
		copyState.unvisitedNodes = new ArrayList<>(this.unvisitedNodes);
		copyState.coveredNodes = new HashSet<>(this.coveredNodes);
		copyState.nodeToMatching = new HashMap<>(this.nodeToMatching);

		copyState.decisions = new ArrayList<>(this.decisions);

		copyState.logDensity = this.logDensity;
		if (this.logGradient != null) {
    		for (F f : this.logGradient)
    		{
    			copyState.logGradient.setCount(f, this.logGradient.getCount(f));
    		}
		}

		return copyState;
	}
	
	/*
	public static <F, NodeType extends GraphNode<F>> GraphMatchingState<F, NodeType> constructFinalState(List<NodeType> nodesInSequence, List<Set<NodeType>> matching)
	{
		GraphMatchingState<F, NodeType> finalState = new GraphMatchingState<>();
		finalState.matching = new ArrayList<>(matching);
		finalState.visitedNodes = new ArrayList<>(nodesInSequence);
		finalState.unvisitedNodes = new ArrayList<>();
		finalState.coveredNodes = new HashSet<>(nodesInSequence);
		return finalState;
	}
	*/
	
	public static <F, NodeType extends GraphNode<?>> Pair<Double, Counter<F>> evaluateDecision(Command<F, NodeType> command, Counter<F> params, Pair<List<NodeType>, List<Set<NodeType>>> instance)
	{
		DecisionModel<F, NodeType> decisionModel = command.getDecisionModel();
		GraphFeatureExtractor<F, NodeType> fe = command.getFeatureExtractor();
		MultinomialLogisticModel<F, NodeType> model = Command.constructModel(fe, params);

		// observation to be evaluated at
		List<NodeType> permutation = instance.getFirst();
		List<Set<NodeType>> decisions = instance.getSecond();

		// state to manipulate
		GraphMatchingState<F, NodeType> state = GraphMatchingState.getInitialState(permutation);

		for (int i = 0; i < permutation.size(); i++)
		{
			double logNorm = Double.NEGATIVE_INFINITY;
			Counter<F> suff = new Counter<>();
			Counter<F> features = null;

			NodeType node = state.unvisitedNodes.remove(0);
			state.visitedNodes.add(node);

			// get the decision set
			List<Set<NodeType>> decisionSet = decisionModel.getDecisions(node, state);

			for (Set<NodeType> d : decisionSet)
			{
				// compute the quantities needed for evaluating the log-likelihood and gradient of log-likelihood
				Pair<Double, Counter<F>> ret = model.logProb(node, d, state);
				logNorm = NumericalUtils.logAdd(logNorm, ret.getFirst());
				for (F f : ret.getSecond())
				{
					suff.incrementCount(f, Math.exp(ret.getFirst()) * ret.getSecond().getCount(f));
				}

				// the node may already be contained in an edge
				Set<NodeType> e = state.nodeToMatching.containsKey(node) ? state.nodeToMatching.get(node) : null;
				Set<NodeType> newEdge = new HashSet<>();
				newEdge.add(node);
				newEdge.addAll(d);
				if (e != null)
					newEdge.addAll(e);

				if (d.containsAll(decisions.get(i)))
				{
					// update the state:
					if (e != null)
						state.matching.remove(e);

					for (NodeType n : newEdge)
					{
						state.coveredNodes.add(n);
						state.nodeToMatching.put(n, newEdge);
					}

					state.matching.remove(d);
					state.matching.add(newEdge);
					state.logDensity += ret.getFirst();
					features = ret.getSecond();
				}
			}
			
			state.logDensity -= logNorm;
			for (F f : suff)
			{
				state.logGradient.incrementCount(f, features.getCount(f) - suff.getCount(f)/Math.exp(logNorm));
			}

		}

		return Pair.create(state.logDensity, state.logGradient);
	}

	/**
	 * Execute one move towards the finalState by making a decision on the NodeType = this.unvisitedNodes.remove(0);
	 * Returns the possible states that can result from the move.
	 * 
	 * @param model
	 * @param decisionModel
	 * @param finalState is assumed to be in the support set. 
	 * @return
	 */
	@Override
	public List<GenericGraphMatchingState<F, NodeType>> executeMove(MultinomialLogisticModel<F, NodeType> model, DecisionModel<F, NodeType> decisionModel, Map<NodeType, Set<NodeType>> finalState)
	{
		Map<GraphMatchingState<F, NodeType>, Counter<F>> newStates = new HashMap<>();

		NodeType node = unvisitedNodes.remove(0);
		visitedNodes.add(node);

		double logNorm = Double.NEGATIVE_INFINITY;
		Counter<F> numerator = new Counter<>();
		List<Set<NodeType>> decisions = decisionModel.getDecisions(node, this);
		for (Set<NodeType> d : decisions)
		{
			Pair<Double, Counter<F>> pair = model.logProb(node, d, this);
			logNorm = NumericalUtils.logAdd(logNorm, pair.getFirst());
			for (F f : pair.getSecond())
			{
				numerator.incrementCount(f, Math.exp(pair.getFirst()) * pair.getSecond().getCount(f));
			}

			// the node may already be contained in an edge -- get it
			Set<NodeType> e = nodeToMatching.containsKey(node) ? nodeToMatching.get(node) : null;
			Set<NodeType> newEdge = new HashSet<>();
			newEdge.add(node);
			newEdge.addAll(d);
			if (e != null)
				newEdge.addAll(e);

			/*
			if (newEdge.size() == 1) {
				throw new RuntimeException();
			}
			*/
			
			// check if the current state is a sub state of the final state:
			if (finalState.get(node).containsAll(newEdge))
			{
				GraphMatchingState<F, NodeType> newState = this.copyState();
				if (e != null)
					newState.matching.remove(e);

				for (NodeType n : newEdge)
				{
					newState.coveredNodes.add(n);
					newState.nodeToMatching.put(n, newEdge);
				}

				newState.matching.remove(d);
				newState.matching.add(newEdge);
				newState.logDensity += pair.getFirst();
				
				if (decisionModel.pathExists(newState, finalState))
					newStates.put(newState, pair.getSecond());
			}
		}

		for (GraphMatchingState<F, NodeType> newState : newStates.keySet())
		{
			newState.logDensity -= logNorm;
			Counter<F> fv = newStates.get(newState);
			for (F f : fv)
			{
				newState.logGradient.incrementCount(f, fv.getCount(f) - numerator.getCount(f)/Math.exp(logNorm));
			}
		}

		if (decisions.size() == 0)
			newStates.put(this, null);

		//Set<GenericGraphMatchingState<F, NodeType>> ret = new HashSet<>(newStates.keySet());
		List<GenericGraphMatchingState<F, NodeType>> ret = new ArrayList<>(newStates.keySet());
		return ret;
	}
			
	/** 
	 * Unnormalized log density
	 * @return
	 */
	@Override
	public double getLogDensity()
	{
		return logDensity;
	}

	@Override
	public Counter<F> getLogGradient()
	{
		return logGradient;
	}

	@Override
	public boolean hasNextStep()
	{
		return (unvisitedNodes.size() > 0);
	}
	
	@Override
	public NodeType getExampleNode()
	{
		if (visitedNodes.size() > 0) {
			for (NodeType node : visitedNodes)
				return node;
		}
		else if (unvisitedNodes.size() > 0)
			return unvisitedNodes.get(0);
		throw new RuntimeException("Bug in the code: Incorrect state representation encountered!");
	}

	/**
	 * Generate a list of descendants. It differs from this.executeMove() in that executeMove returns the states that can potentially lead to the final state.
	 * 
	 * @param model
	 * @param decisionModel
	 * @param sequential
	 * @return
	 */
	@Override
	public List<GenericGraphMatchingState<F, NodeType>> generateDescendants(MultinomialLogisticModel<F, NodeType> model, DecisionModel<F, NodeType> decisionModel, boolean sequential)
	{
		// generate the list of decisions
		List<GenericGraphMatchingState<F, NodeType>> descendants = new ArrayList<>();
		if (sequential) {
			descendants.addAll(constructNextState(0, decisionModel, model));
		} else {
			for (int i = 0; i < unvisitedNodes.size(); i++)
			{
				descendants.addAll(constructNextState(i, decisionModel, model));
			}
		}
		return descendants;
	}
	
	/**
	 * leaves the current state (this object) invariant
	 * 
	 * @param idx
	 * @param decisionModel
	 * @param model
	 * @return
	 */
	private List<GraphMatchingState<F, NodeType>> constructNextState(int idx, DecisionModel<F, NodeType> decisionModel, MultinomialLogisticModel<F, NodeType> model)
	{
		List<GraphMatchingState<F, NodeType>> ret = new ArrayList<>();

		NodeType node = unvisitedNodes.get(idx);
		List<Set<NodeType>> decisions = decisionModel.getDecisions(node, this);
		double logSum = Double.NEGATIVE_INFINITY;
		for (Set<NodeType> d : decisions)
		{
			GraphMatchingState<F, NodeType> copy = this.copyState();
			d.add(node);
			copy.matching.add(d);
			copy.unvisitedNodes.remove(idx);
			copy.visitedNodes.add(node);
			for (NodeType n : d)
				copy.coveredNodes.add(n);
			ret.add(copy);
			double logWeight = model.logProb(node, d, copy).getFirst();
			logSum = NumericalUtils.logAdd(logSum, logWeight);
			copy.logDensity += logWeight;
		}

		if (decisions.size() == 0) {
			GraphMatchingState<F, NodeType> copy = this.copyState();
			if (!copy.covers(node)) {
				Set<NodeType> e = new HashSet<>();
				e.add(node);
				copy.matching.add(e);
			}
			copy.unvisitedNodes.remove(idx);
			copy.visitedNodes.add(node);
			copy.coveredNodes.add(node);
			ret.add(copy);
		} else {
			for (GraphMatchingState<F, NodeType> copy : ret)
			{
				copy.logDensity -= logSum;
			}
		}

		return ret;
	}

	/**
	 * perform one move from the current state by sampling the next move -- use for SMC sampling
	 * return the logProb of the next state reached
	 * 
	 * @param random
	 * @param model
	 * @param decisionModel
	 * @param sequential
	 * @param exactSampling
	 * @return
	 */
	@Override
	public double sampleNextState(Random random, Command<F, NodeType> command, boolean sequential, boolean exactSampling)
	{
		// sample a node that will be making the decision
		int idx = 0;
		//double logProbSigma = 0.0;
		if (!sequential) {
			idx = random.nextInt(unvisitedNodes.size());
			//logProbSigma = -Math.log(unvisitedNodes.size());
		}

		NodeType node = unvisitedNodes.remove(idx);
		Pair<Double, Double> ret = sampleDecision(random, command.getDecisionModel(), command.getCurrentModel(), node, exactSampling);
		double logProb = ret.getFirst();
		this.logDensity += (logProb - ret.getSecond());
		//this.logDensity += logProb;
		visitedNodes.add(node);
		//coveredNodes.add(node); // will not add if node has been already added in previous iteration
		return (logProb - ret.getSecond());
	}

	double logForwardProposal = 0.0;
	// samples a decision and returns the unnormalized log probability of sampling that decision as well as the log normalization 
	private Pair<Double, Double> sampleDecision(Random random, DecisionModel<F, NodeType> decisionModel, MultinomialLogisticModel<F, NodeType> model, NodeType node, boolean exactSampling)
	{
		List<Set<NodeType>> decisions = decisionModel.getDecisions(node, this);
		//System.out.println("# decisions: " + decisions.size());
		/*
		if (decisions.size() == 0) {
			if (!covers(node)) {
				Set<NodeType> e = new HashSet<>();
				e.add(node);
				matching.add(e);
			}
			return Pair.create(0.0, 0.0);
		}
		*/

		
		int idx = 0;
		double logProb = 0.0;
		double logNorm = 0.0;
		if (exactSampling) {
			double [] logProbs = new double[decisions.size()];
			for (int i = 0; i < decisions.size(); i++)
			{
				logProbs[i] = model.logProb(node, decisions.get(i), this).getFirst();
			}
			logNorm = NumericalUtils.logAdd(logProbs);

	  		double [] probs = new double[logProbs.length];
	  		System.arraycopy(logProbs, 0, probs, 0, logProbs.length);
	  		try {
	  			Multinomial.expNormalize(probs);
	  		} catch (Exception ex) {
	  			System.out.println("exception");
	  		}
	  		idx = Multinomial.sampleMultinomial(random, probs);
	  		logProb = logProbs[idx];
	  		logForwardProposal = logProbs[idx] - logNorm;
		} else {
			idx = random.nextInt(decisions.size());
			logForwardProposal = -Math.log(decisions.size());
			logProb = model.logProb(node, decisions.get(idx), this).getFirst();
		}

		// insert the chosen decision
		this.decisions.add(decisions.get(idx));
		
		Set<NodeType> edge = decisions.get(idx);

		if (edge.size() == 0) {
			// this is do nothing decision, which means to either form a singleton or the node already belongs to an edge
			// in the latter case, we need to get the edge that this node belongs to
  			if (nodeToMatching.containsKey(node)) {
  				edge = nodeToMatching.get(node);
  			}
		} else if (matching.contains(edge)) {
			boolean edgeRemoved = matching.remove(edge);
			if (!edgeRemoved)
				throw new RuntimeException();
		}

		Set<NodeType> newEdge = new HashSet<>(edge);
		newEdge.add(node);

		matching.add(newEdge);
		for (NodeType n : newEdge)
		{
			coveredNodes.add(n);
			nodeToMatching.put(n, newEdge);
		}

		return Pair.create(logProb, logNorm);
	}

	public double sampleNextUnCoveredNode(Random random, Command<F, NodeType> command, boolean sequential, boolean exactSampling)
	{
		
		int idx = 0;
		NodeType node = null;
		while (true)
		{
			if (!sequential) {
				idx = random.nextInt(unvisitedNodes.size());
			}

			node = unvisitedNodes.remove(idx);
			if (!coveredNodes.contains(node))
				break;
		}

		Pair<Double, Double> ret = sampleDecision(random, command.getDecisionModel(), command.getCurrentModel(), node, exactSampling);
		double logProb = ret.getFirst();
		this.logDensity += (logProb - ret.getSecond());
		//this.logDensity += logProb;
		visitedNodes.add(node);
		//coveredNodes.add(node); // will not add if node has been already added in previous iteration
		return (logProb - ret.getSecond());

	}

	public GraphMatchingState<F, NodeType> localMove(Random random)
	{
		// make a copy
		GraphMatchingState<F, NodeType> nextState = copyState();

		// select two nodes at random
		NodeType u = nextState.visitedNodes.remove(random.nextInt(nextState.visitedNodes.size()));
		NodeType v = nextState.visitedNodes.remove(random.nextInt(nextState.visitedNodes.size()));

		// get the edges containing u and v
		Set<NodeType> eu = nextState.nodeToMatching.get(u);
		Set<NodeType> ev = nextState.nodeToMatching.get(v);
		nextState.matching.remove(eu);
		nextState.matching.remove(ev);

		Set<NodeType> e = new HashSet<>(eu);
		e.addAll(ev);

		// form new edges
		Set<NodeType> e1 = new HashSet<>();
		e1.add(u);
		e1.add(v);
		Set<NodeType> e2 = new HashSet<>();
		for (NodeType node : e) {
			if (node != u && node != v) e2.add(node);
		}

		nextState.nodeToMatching.put(u, e1);
		nextState.nodeToMatching.put(v, e1);
		for (NodeType node : e2) {
			nextState.nodeToMatching.put(node, e2);
		}
		
		nextState.visitedNodes.add(u);
		nextState.visitedNodes.add(v);

		nextState.matching.add(e1);
		if (e2.size() > 0)
			nextState.matching.add(e2);
		
		return nextState;
	}
	
	@Override
	public List<NodeType> getVisitedNodes()
	{
		//return new ArrayList<>(visitedNodes);
		return visitedNodes;
	}
	
	@Override
	public Set<NodeType> getVisitedNodesAsSet()
	{
		return new HashSet<>(visitedNodes);
	}


	@Override
	public List<NodeType> getUnvisitedNodes()
	{
		return unvisitedNodes;
	}
	
	@Override
	public Set<NodeType> getCoveredNodes()
	{
		return coveredNodes;
	}

	@Override
	public List<Set<NodeType>> getMatchings()
	{
		List<Set<NodeType>> m = new ArrayList<>(matching);
		return m;
	}
	
	@Override
	public boolean covers(NodeType node)
	{
		return coveredNodes.contains(node);
	}
	
	@Override
	public void shuffleNodes(Random random)
	{
		Collections.shuffle(unvisitedNodes, random);
	}

	@Override
	public Map<NodeType, Set<NodeType>> getNode2EdgeView()
	{
		/*
		Map<NodeType, Set<NodeType>> nodeToEdge = new HashMap<>();
		for (Set<NodeType> edge : matching)
		{
			for (NodeType node : edge)
			{
				if (nodeToEdge.containsKey(node))
					throw new RuntimeException("Matching violation! There is a bug in the matching sampler.");
				
				nodeToEdge.put(node, edge);
			}
		}
		return nodeToEdge;
		*/
		return nodeToMatching;
	}
	
	private List<Set<NodeType>> sortedMatching = null;
	public List<Set<NodeType>> sortMatching()
	{
		if (sortedMatching == null) {
  		sortedMatching = new ArrayList<>();
  		if (matching.size() == 0)
  			return sortedMatching; // empty matching, nothing to sort
  		
  		List<NodeType> representativeNodes = new ArrayList<>();
  		Map<NodeType, Set<NodeType>> nodeToEdge = new HashMap<>();
  		for (Set<NodeType> edge : matching)
  		{
  			List<NodeType> nodes = new ArrayList<>(edge);
  			Collections.sort(nodes);
  			NodeType repNode = nodes.get(0);
  			representativeNodes.add(repNode);
  			nodeToEdge.put(repNode, new HashSet<>(nodes));
  		}
  
  		Collections.sort(representativeNodes);
  		for (NodeType node : representativeNodes)
  		{
  			sortedMatching.add(nodeToEdge.get(node));
  		}
		}
		
		return sortedMatching;
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		// sort each edge in the matching by the node that they contain
		List<Set<NodeType>> sortedMatching = this.sortMatching();
		for (Set<NodeType> edge : sortedMatching)
		{
			sb.append("{");
			for (NodeType node : edge)
			{
				sb.append(node.toString() + ";");
			}
			sb.append("}");
			sb.append("\n");
		}

		return sb.toString();
	}
	
	@Override
	public int hashCode()
	{
		return toString().hashCode();
	}

	@Override
	public boolean equals(Object o)
	{
		return (this.hashCode() == o.hashCode());
	}
	
	@Override
	public double getLogForwardProposal()
	{
		return logForwardProposal;
	}

}
