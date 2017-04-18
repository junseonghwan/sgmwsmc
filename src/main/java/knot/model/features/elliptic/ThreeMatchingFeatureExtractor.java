package knot.model.features.elliptic;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import knot.data.EllipticalKnot;
import knot.model.features.rectangular.SizeFeatureExtractor;
import common.graph.GenericGraphMatchingState;
import common.model.GraphFeatureExtractor;

public class ThreeMatchingFeatureExtractor implements GraphFeatureExtractor<String, EllipticalKnot> 
{
	public static final String TWO_MATCHING_SHARED_AXIS = "TWO_MATCHING_SHARED_AXIS";
	public static final String THREE_MATCHING_SHARED_AXIS = "THREE_MATCHING_SHARED_AXIS";
	public static List<String> featureNames = new ArrayList<>();
	static {
		Field [] fields = SizeFeatureExtractor.class.getFields();
		for (Field field : fields)
		{
			if (Modifier.isFinal(field.getModifiers()) && Modifier.isStatic(field.getModifiers()))
				featureNames.add(field.getName());
		}
	}

	@Override
	public Counter<String> extractFeatures(EllipticalKnot node,
			Set<EllipticalKnot> decision,
			GenericGraphMatchingState<String, EllipticalKnot> matchingState) {
		
		Counter<String> f = new Counter<>();
		
		List<EllipticalKnot> knots = new ArrayList<>(decision);
		knots.add(node);

		for (int i = 0; i < knots.size(); i++) {
			for (int j = i+1; j < knots.size(); j++) {
				if (sharesAxis(knots.get(i), knots.get(j)) == 1) {
					if (knots.size() == 2)
						f.setCount(TWO_MATCHING_SHARED_AXIS, 1);
					else
						f.setCount(THREE_MATCHING_SHARED_AXIS, 1);
					break;
				}
			}
		}

		return f;
	}

	@Override
	public Counter<String> extractFeatures(Set<EllipticalKnot> e, List<Set<EllipticalKnot>> matchingState) {
		Counter<String> f = new Counter<>();
		
		List<EllipticalKnot> knots = new ArrayList<>(e);

		for (int i = 0; i < knots.size(); i++) {
			for (int j = i+1; j < knots.size(); j++) {
				if (sharesAxis(knots.get(i), knots.get(j)) == 1) {
					if (knots.size() == 2)
						f.setCount(TWO_MATCHING_SHARED_AXIS, 1);
					else
						f.setCount(THREE_MATCHING_SHARED_AXIS, 1);
					break;
				}
			}
		}

		return f;
	}

	public static int sharesAxis(EllipticalKnot k1, EllipticalKnot k2)
	{
		if (k1.getNodeFeatures().getCount("area_over_axis") > 0 && k2.getNodeFeatures().getCount("area_over_axis") > 0) {
			return 1;
		}
		return 0;
	}

	@Override
	public Counter<String> getDefaultParameters() {
	  Counter<String> init = new Counter<>();
	  for (String featureName : featureNames)
	  {
	  	init.setCount(featureName, 0.0);
	  }
	  return init;
	}

	@Override
	public int dim() {
	  return featureNames.size();
	}

}
