package knot.model.features.rectangular;

import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import knot.data.RectangularKnot;
import knot.model.features.common.DistanceFeatureExtractor;
import common.graph.GenericGraphMatchingState;
import common.model.GraphFeatureExtractor;

/**
 * Feature extractor for RectangularKnot type. This is no longer in use as the knots are represented as ellipse now.
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 */
public class DistanceSizeFeatureExtractor implements GraphFeatureExtractor<String, RectangularKnot>
{

	@Override
	public Counter<String> extractFeatures(RectangularKnot node,
			Set<RectangularKnot> decision,
			GenericGraphMatchingState<String, RectangularKnot> matchingState) {
		DistanceFeatureExtractor<RectangularKnot> fe1 = new DistanceFeatureExtractor<>();
		SizeFeatureExtractor fe2 = new SizeFeatureExtractor();
		Counter<String> f1 = fe1.extractFeatures(node, decision, matchingState);
		Counter<String> f2 = fe2.extractFeatures(node, decision, matchingState);

		Counter<String> f = new Counter<>();
		for (String feature : f1)
		{
			f.setCount(feature, f1.getCount(feature));
		}
		for (String feature : f2)
		{
			f.setCount(feature, f2.getCount(feature));
		}
		
		return f;
	}

	@Override
	public Counter<String> getDefaultParameters() {
		Counter<String> p1 = (new DistanceFeatureExtractor<RectangularKnot>()).getDefaultParameters();
		Counter<String> p2 = (new SizeFeatureExtractor()).getDefaultParameters();
		Counter<String> p = new Counter<>();
		for (String f : p1)
		{
			p.setCount(f, p1.getCount(f));
		}
		for (String f : p2)
		{
			p.setCount(f, p2.getCount(f));
		}
		return p;
	}

	@Override
	public int dim() {
		return (new DistanceFeatureExtractor<RectangularKnot>()).dim() + (new SizeFeatureExtractor()).dim();
	}

	@Override
	public Counter<String> extractFeatures(Set<RectangularKnot> e,
			List<Set<RectangularKnot>> matchingState) {
		throw new RuntimeException();
	}

}
