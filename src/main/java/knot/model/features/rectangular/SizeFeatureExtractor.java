package knot.model.features.rectangular;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import knot.data.RectangularKnot;
import common.graph.GenericGraphMatchingState;
import common.model.GraphFeatureExtractor;

public class SizeFeatureExtractor implements GraphFeatureExtractor<String, RectangularKnot>
{
	public static double MARGIN = 3.0;
	
	/*
	public static final String SIZE = "SIZE";
	public static final String DIMENSION_COMPATIBILITY = "DIMENSION_COMPATIBILITY";
	*/
	
	public static final String W_DIFF = "W_DIFF";
	public static final String H_DIFF = "H_DIFF";


	public static List<String> featureNames = new ArrayList<>();
	static {
		Field [] fields = SizeFeatureExtractor.class.getFields();
		for (Field field : fields)
		{
			if (Modifier.isFinal(field.getModifiers()))
			featureNames.add(field.getName());
		}
	}

	@Override
	public Counter<String> extractFeatures(RectangularKnot node,
			Set<RectangularKnot> decision,
			GenericGraphMatchingState<String, RectangularKnot> matchingState) {

		Counter<String> f = new Counter<>();

		Counter<String> f1 = node.getNodeFeatures();
		double w1 = f1.getCount("w");
		double h1 = f1.getCount("h");
		//double a1 = w1*h1;
		//double diffMax = 0.0;
		//int ind = 0;
		for (RectangularKnot knot : decision)
		{
			Counter<String> f2 = knot.getNodeFeatures();
			double w2 = f2.getCount("w");
			double h2 = f2.getCount("h");
			/*
			double a2 = w2*h2;
			double diff = Math.abs(w1/a1 - w2/a2) + Math.abs(h1/a1 - h2/a2);
			if (diff > diffMax) {
				diffMax = diff;
				ind = (w1 <= w2 + MARGIN) && (h1 <= h2 + MARGIN) ? 1 : 0;
				ind = ind | ((w2 <= w1 + MARGIN) && (h2 <= h1 + MARGIN) ? 1 : 0);
			}
			*/
			f.incrementCount(W_DIFF, Math.abs(w1 - w2)/Math.max(w1, w2));
			f.incrementCount(H_DIFF, Math.abs(h1 - h2)/Math.max(h1, h2));
		}

		/*
		f.incrementCount(SIZE, diffMax);
		f.incrementCount(DIMENSION_COMPATIBILITY, ind);
		*/
		return f;
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

	@Override
	public Counter<String> extractFeatures(Set<RectangularKnot> e,
			List<Set<RectangularKnot>> matchingState) {
		throw new RuntimeException();
	}

}
