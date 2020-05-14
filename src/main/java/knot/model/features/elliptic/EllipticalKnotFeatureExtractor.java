package knot.model.features.elliptic;

import java.util.List;
import java.util.Set;

import common.graph.GenericGraphMatchingState;
import common.model.GraphFeatureExtractor;
import knot.data.EllipticalKnot;
import knot.model.features.common.ThreeMatchingDistanceFeatureExtractor;
import briefj.collections.Counter;

/**
 * Feature extractor for elliptical knots.
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 */
public class EllipticalKnotFeatureExtractor implements GraphFeatureExtractor<String, EllipticalKnot> 
{
	private ThreeMatchingDistanceFeatureExtractor<EllipticalKnot> distanceFE = new ThreeMatchingDistanceFeatureExtractor<>();
	private AreaFeatureExtractor areaFE = new AreaFeatureExtractor();
	//private AxisFeatureExtractor axisFE = new AxisFeatureExtractor();

	private Counter<String> mean = null;
	private Counter<String> sd = null;
	
	@Override
	public Counter<String> extractFeatures(EllipticalKnot node,
			Set<EllipticalKnot> decision,
			GenericGraphMatchingState<String, EllipticalKnot> matchingState) {

		Counter<String> f = new Counter<>();

		Counter<String> distanceFeatrues = distanceFE.extractFeatures(node, decision, matchingState);
		Counter<String> areaFeatrues = areaFE.extractFeatures(node, decision, matchingState);
		//Counter<String> axisFeatrues = axisFE.extractFeatures(node, decision, matchingState);

		for (String feature : distanceFeatrues)
		{
			//System.out.println("feature: " + feature + ", " + mean.getCount(feature));
			if (mean != null && sd != null)
				f.setCount(feature, (distanceFeatrues.getCount(feature) - mean.getCount(feature))/sd.getCount(feature));
			else
				f.setCount(feature, distanceFeatrues.getCount(feature));
		}
		for (String feature : areaFeatrues)
		{
			//System.out.println("feature: " + feature + ", " + mean.getCount(feature));
			if (mean != null && sd != null)
				f.setCount(feature, (areaFeatrues.getCount(feature) - mean.getCount(feature))/sd.getCount(feature));
			else
				f.setCount(feature, areaFeatrues.getCount(feature));
		}
		/*
		for (String feature : axisFeatrues)
			f.setCount(feature, axisFeatrues.getCount(feature));
			*/

		return f;
	}

	@Override
	public Counter<String> extractFeatures(Set<EllipticalKnot> e, List<Set<EllipticalKnot>> matchingState) {
		Counter<String> f = new Counter<>();

		Counter<String> distanceFeatrues = distanceFE.extractFeatures(e, matchingState);
		Counter<String> areaFeatrues = areaFE.extractFeatures(e, matchingState);
		//Counter<String> axisFeatrues = axisFE.extractFeatures(e, matchingState);

		for (String feature : distanceFeatrues)
			f.setCount(feature, distanceFeatrues.getCount(feature));
		for (String feature : areaFeatrues)
			f.setCount(feature, areaFeatrues.getCount(feature));
		/*
		for (String feature : axisFeatrues)
			f.setCount(feature, axisFeatrues.getCount(feature));
			*/

		return f;
	}

	@Override
	public Counter<String> getDefaultParameters() {
		Counter<String> p1 = distanceFE.getDefaultParameters();
		Counter<String> p2 = areaFE.getDefaultParameters();
		//Counter<String> p3 = axisFE.getDefaultParameters();
		Counter<String> p = new Counter<>();
		for (String f : p1)
		{
			p.setCount(f, p1.getCount(f));
		}
		for (String f : p2)
		{
			p.setCount(f, p2.getCount(f));
		}
		/*
		for (String f : p3)
		{
			p.setCount(f, p3.getCount(f));
		}
		*/
		return p;
	}
	@Override
	public int dim() {
		//return distanceFE.dim() + areaFE.dim() + axisFE.dim();
		return distanceFE.dim() + areaFE.dim();
	}

	@Override
	public void setStandardization(Counter<String> mean, Counter<String> sd) {
		this.mean = mean;
		this.sd = sd;
	}

}
