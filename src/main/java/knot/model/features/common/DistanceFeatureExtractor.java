package knot.model;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import knot.data.Knot;
import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.model.GraphFeatureExtractor;

public class DistanceFeatureExtractor<KnotType extends Knot> implements GraphFeatureExtractor<String, KnotType> {

	public static final String TWO_MATCHING_DISTANCE_1 = "TWO_MATCHING_DISTANCE_1";
	public static final String TWO_MATCHING_DISTANCE_2 = "TWO_MATCHING_DISTANCE_2";
	public static List<String> featureNames = new ArrayList<>();
	static {
		Field [] fields = DistanceFeatureExtractor.class.getFields();
		for (Field field : fields)
		{
			if (Modifier.isFinal(field.getModifiers()))
			featureNames.add(field.getName());
		}
	}

	@Override
  public int dim() {
	  return featureNames.size();
  }

	@Override
	public Counter<String> extractFeatures(KnotType node, Set<KnotType> decision, GenericGraphMatchingState<String, KnotType> matchingState) {

		// compute the features based on the distance of the node to the nodes in decision
		// ignore the context features for now
		Counter<String> features = initializeCounter();
		List<KnotType> decisions = new ArrayList<>(decision);
		for (KnotType otherNode : decisions)
		{
			computeDistance(node, otherNode, features);
		}

	  return features;
	}
	
	@Override
	public Counter<String> extractFeatures(Set<KnotType> e, List<Set<KnotType>> matchingState) {
		// compute the features based on the distance of the node to the nodes in decision
		// ignore the context features for now
		Counter<String> features = initializeCounter();
		List<KnotType> decisions = new ArrayList<>(e);
		if (decisions.size() != 2)
			throw new RuntimeException();
		computeDistance(decisions.get(0), decisions.get(1), features);

		return features;
	}

	public static <KnotType extends Knot> void computeDistance(KnotType node, KnotType otherNode, Counter<String> features)
	{
		Counter<String> nodeFeatures = node.getNodeFeatures();
		Counter<String> otherFeatures = otherNode.getNodeFeatures();
		int ind1 = (node.getPartitionIdx() % 2) == 0 ? 1 : 0;
		int ind2 = (otherNode.getPartitionIdx() % 2) == 0 ? 1 : 0;

		double diffX = Math.pow(nodeFeatures.getCount("x") - otherFeatures.getCount("x"), 2);
		double diffY = Math.pow(nodeFeatures.getCount("y") - otherFeatures.getCount("y"), 2);
		double diffZ = Math.pow(nodeFeatures.getCount("z") - otherFeatures.getCount("z"), 2);
		if (ind1 * ind2 == 1)
			features.setCount(TWO_MATCHING_DISTANCE_1, Math.sqrt(diffX + diffY + diffZ) * ind1 * ind2);
		else {
			//features.setCount(TWO_DISTANCE_2, Math.sqrt(diffX + diffY + diffZ) * (ind1 ^ ind2));
			features.setCount(TWO_MATCHING_DISTANCE_2, Math.sqrt(diffX + diffY + diffZ));
		}

		// TODO: handle distance features for three matching
	}

	@Override
  public Counter<String> getDefaultParameters() {
		return initializeCounter();
  }
	
	private Counter<String> initializeCounter() {
	  Counter<String> init = new Counter<>();
	  for (String featureName : featureNames)
	  {
	  	init.setCount(featureName, 0.0);
	  }
	  return init;		
	}

}
