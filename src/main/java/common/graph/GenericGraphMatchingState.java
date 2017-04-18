package common.graph;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import briefj.collections.Counter;
import common.model.Command;
import common.model.DecisionModel;
import common.model.MultinomialLogisticModel;

public interface GenericGraphMatchingState<F, NodeType extends GraphNode<?>>
{
	public boolean hasNextStep();
	public List<GenericGraphMatchingState<F, NodeType>> executeMove(MultinomialLogisticModel<F, NodeType> model, DecisionModel<F, NodeType> decisionModel, Map<NodeType, Set<NodeType>> finalState);
	public double getLogDensity();
	public Counter<F> getLogGradient();
	public NodeType getExampleNode();
	public double sampleNextState(Random random, Command<F, NodeType> command, boolean sequential, boolean exactSampling);
	public GenericGraphMatchingState<F, NodeType> copyState();
	public List<NodeType> getUnvisitedNodes();
	public List<NodeType> getVisitedNodes();
	public Set<NodeType> getVisitedNodesAsSet();
	public Set<NodeType> getCoveredNodes();
	public List<Set<NodeType>> getDecisions();
	public List<Set<NodeType>> getMatchings();
	public boolean covers(NodeType node);
	public void shuffleNodes(Random random);
	public Map<NodeType, Set<NodeType>> getNode2EdgeView();
	public List<GenericGraphMatchingState<F, NodeType>> generateDescendants(MultinomialLogisticModel<F, NodeType> model, DecisionModel<F, NodeType> decisionModel, boolean sequential);
	public double getLogForwardProposal();
}
