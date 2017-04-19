package knot.model.features.common;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.model.GraphFeatureExtractor;
import knot.data.Knot;

/**
 * Distance based feature extractor.
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 * @param <KnotType>
 */
public class ThreeMatchingDistanceFeatureExtractor<KnotType extends Knot> implements GraphFeatureExtractor<String, KnotType>
{
	public static final String TWO_MATCHING_DISTANCE_1 = "TWO_MATCHING_DISTANCE_1";
	public static final String TWO_MATCHING_DISTANCE_2 = "TWO_MATCHING_DISTANCE_2";
	public static final String THREE_MATCHING_DISTANCE_1 = "THREE_MATCHING_DISTANCE_1";
	public static final String THREE_MATCHING_DISTANCE_2 = "THREE_MATCHING_DISTANCE_2";

	public static double NORMALIZATION_CONSTANT = 300; // TODO: Do this in R as a pre-processing step and remove this constant
	public static List<String> featureNames = new ArrayList<>();
	static {
		Field [] fields = ThreeMatchingDistanceFeatureExtractor.class.getFields();
		for (Field field : fields)
		{
			if (Modifier.isFinal(field.getModifiers()))
			featureNames.add(field.getName());
		}
	}

	@Override
	public Counter<String> extractFeatures(KnotType node,
	    Set<KnotType> decision,
	    GenericGraphMatchingState<String, KnotType> matchingState) {

		Counter<String> f = initializeCounter();

		if (decision.size() == 1) {
			DistanceFeatureExtractor<KnotType> fe = new DistanceFeatureExtractor<>();
			Counter<String> twoMatchingDistanceFE = fe.extractFeatures(node, decision, matchingState);
			f.setCount(TWO_MATCHING_DISTANCE_1, twoMatchingDistanceFE.getCount(TWO_MATCHING_DISTANCE_1)/NORMALIZATION_CONSTANT);
			f.setCount(TWO_MATCHING_DISTANCE_2, twoMatchingDistanceFE.getCount(TWO_MATCHING_DISTANCE_2)/NORMALIZATION_CONSTANT);
		} else if (decision.size() == 2) {
			// 3-matching feature: the maximal distance
			List<KnotType> nodes = new ArrayList<>(decision);
			nodes.add(node);

			double maxDist = 0.0;
			double minDist = Double.MAX_VALUE;
			for (int i = 0; i < nodes.size(); i++)
			{
				for (int j = i + 1; j < nodes.size(); j++)
				{
					double ij = computeDistance(nodes.get(i), nodes.get(j));
					if (ij > maxDist) {
						maxDist = ij;
					}
					if (ij < minDist) {
						minDist = ij;
					}
				}
				f.setCount(THREE_MATCHING_DISTANCE_1, minDist/NORMALIZATION_CONSTANT);
				f.setCount(THREE_MATCHING_DISTANCE_2, maxDist/NORMALIZATION_CONSTANT);
			}
		} else {
			f.setCount(TWO_MATCHING_DISTANCE_1, 10);
		}

		return f;
	}

	@Override
	public Counter<String> extractFeatures(Set<KnotType> e, List<Set<KnotType>> matchingState) {
		Counter<String> f = initializeCounter();

		if (e.size() == 2) {
			DistanceFeatureExtractor<KnotType> fe = new DistanceFeatureExtractor<>();
			Counter<String> twoMatchingDistanceFE = fe.extractFeatures(e, matchingState);
			f.setCount(TWO_MATCHING_DISTANCE_1, twoMatchingDistanceFE.getCount(TWO_MATCHING_DISTANCE_1)/NORMALIZATION_CONSTANT);
			f.setCount(TWO_MATCHING_DISTANCE_2, twoMatchingDistanceFE.getCount(TWO_MATCHING_DISTANCE_2)/NORMALIZATION_CONSTANT);
		} else if (e.size() == 3) {
			// 3-matching feature: the maximal distance
			List<KnotType> nodes = new ArrayList<>(e);

			double maxDist = 0.0;
			double minDist = Double.MAX_VALUE;
			for (int i = 0; i < nodes.size(); i++)
			{
				for (int j = i + 1; j < nodes.size(); j++)
				{
					double ij = computeDistance(nodes.get(i), nodes.get(j));
					if (ij > maxDist) {
						maxDist = ij;
					} 
					if (ij < minDist) {
						minDist = ij;
					}
				}
				f.setCount(THREE_MATCHING_DISTANCE_1, minDist/NORMALIZATION_CONSTANT);
				f.setCount(THREE_MATCHING_DISTANCE_2, maxDist/NORMALIZATION_CONSTANT);
			}
		} else {
			f.setCount(TWO_MATCHING_DISTANCE_1, 10);
		}

		return f;
	}
	
	public static <KnotType extends Knot> double computeDistance(KnotType knot1, KnotType knot2)
	{
		Counter<String> f1 = knot1.getNodeFeatures();
		Counter<String> f2 = knot2.getNodeFeatures();

		double diffX = Math.pow(f1.getCount("x") - f2.getCount("x"), 2);
		double diffY = Math.pow(f1.getCount("y") - f2.getCount("y"), 2);
		double diffZ = Math.pow(f1.getCount("z") - f2.getCount("z"), 2);

		return Math.sqrt(diffX + diffY + diffZ);
	}

	@Override
	public Counter<String> getDefaultParameters() {
		return initializeCounter();
	}

	@Override
	public int dim() {
	  return featureNames.size();
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
