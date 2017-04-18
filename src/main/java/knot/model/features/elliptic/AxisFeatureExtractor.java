package knot.model.features.elliptic;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.model.GraphFeatureExtractor;
import knot.data.EllipticalKnot;
import knot.model.ThreeMatchingDistanceFeatureExtractor;

public class AxisFeatureExtractor implements GraphFeatureExtractor<String, EllipticalKnot>
{
	//public static final String TWO_MATCHING_AXIS_SHARED = "TWO_MATCHING_AXIS_SHARED";
	public static final String THREE_MATCHING_AXIS_SHARED = "THREE_MATCHING_AXIS_SHARED";

	public static List<String> featureNames = new ArrayList<>();
	static {
		Field [] fields = AxisFeatureExtractor.class.getFields();
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

		Counter<String> f = getDefaultParameters();
		List<EllipticalKnot> knots = new ArrayList<>(decision);
		knots.add(node);
		if (knots.size() > 3) throw new RuntimeException(">= 4-matching not allowed");

		/*
		for (int i = 0; i < knots.size(); i++)
		{
			for (int j = i + 1; j < knots.size(); j++)
			{
				if (sharesAxis(knots.get(i), knots.get(j))) {
					if (knots.size() == 2)
						f.setCount(TWO_MATCHING_AXIS_SHARED, 1);
					else if (knots.size() == 3)
						f.setCount(THREE_MATCHING_AXIS_SHARED, 1);
			
					break;
				}
			}
		}
		*/
		/*
		double overlap = knots.get(i).getNodeFeatures().getCount("x") - knots.get(j).getNodeFeatures().getCount("x");
		overlap += knots.get(i).getNodeFeatures().getCount("var_x") + knots.get(j).getNodeFeatures().getCount("var_x");
		f.setCount(THREE_MATCHING_AXIS_SHARED, overlap/ThreeMatchingDistanceFeatureExtractor.NORMALIZATION_CONSTANT);
		*/

		return f;
	}
	
	@Override
	public Counter<String> extractFeatures(Set<EllipticalKnot> e, List<Set<EllipticalKnot>> matchingState) {
		Counter<String> f = getDefaultParameters();
		List<EllipticalKnot> knots = new ArrayList<>(e);
		if (knots.size() >= 4) throw new RuntimeException(">= 4-matching not allowed");

		for (int i = 0; i < knots.size(); i++)
		{
			for (int j = i + 1; j < knots.size(); j++)
			{
				if (sharesAxis(knots.get(i), knots.get(j))) {
					break;
				}
			}
		}
		return f;
	}

	public static boolean sharesAxis(EllipticalKnot k1, EllipticalKnot k2)
	{
		Set<Integer> boundaryAxes0 = new HashSet<>();
		boundaryAxes0.add((int)k1.getNodeFeatures().getCount("boundary_axis0"));
		boundaryAxes0.add((int)k1.getNodeFeatures().getCount("boundary_axis1"));

		Set<Integer> boundaryAxes1 = new HashSet<>();
		boundaryAxes1.add((int)k2.getNodeFeatures().getCount("boundary_axis0"));
		boundaryAxes1.add((int)k2.getNodeFeatures().getCount("boundary_axis1"));

		for (int axisIdx : boundaryAxes1)
		{
			if (axisIdx != 0 && boundaryAxes0.contains(axisIdx)) {
				return true;
			}
		}

		return false;
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
