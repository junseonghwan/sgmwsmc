package common.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphNode;

public class CanonicalFeatureExtractor<F, NodeType extends GraphNode<F>> implements GraphFeatureExtractor<F, NodeType>
{
	Counter<F> defaultParam;
	
	private CanonicalFeatureExtractor(NodeType node)
	{
		defaultParam = new Counter<>();
		Counter<F> features = node.getNodeFeatures();
		for (F f : features)
		{
			defaultParam.setCount(f, 0.0);
		}
	}
	
	public static <F, NodeType extends GraphNode<F>> CanonicalFeatureExtractor<F, NodeType> constructCanonicalFeaturesFromExample(NodeType nodeExample)
	{
		return new CanonicalFeatureExtractor<F, NodeType>(nodeExample);
	}

	@Override
  public Counter<F> extractFeatures(NodeType node, Set<NodeType> d, GenericGraphMatchingState<F, NodeType> matchingState) {
	  Counter<F> f1 = node.getNodeFeatures();
	  NodeType otherNode = null;
	  
	  if (d.size() == 1) {
		  otherNode = (new ArrayList<>(d)).get(0);
	  } else if (d.size() == 2 && d.contains(node)) {
		  List<NodeType> nodes = new ArrayList<>(d); 
		  otherNode = nodes.get(0);
		  if (otherNode.equals(node)) 
			  otherNode = nodes.get(1);
	  } else 
		  return getDefaultParameters();

  	Counter<F> f2 = otherNode.getNodeFeatures();
  	Counter<F> f = new Counter<>(); 
  	for (F feature : f1)
  	{
  		f.incrementCount(feature, -Math.abs(f1.getCount(feature) - f2.getCount(feature)));
  	}

  	return f;
  }
	
	@Override
	public Counter<F> extractFeatures(Set<NodeType> e, List<Set<NodeType>> matchingState) {
		
		if (e.size() <= 1)
			return getDefaultParameters();
		
		List<NodeType> knots = new ArrayList<>(e);
		Counter<F> f1 = knots.get(0).getNodeFeatures();
		Counter<F> f2 = knots.get(1).getNodeFeatures();
	  Counter<F> f = new Counter<>(); 
	  	for (F feature : f1)
	  	{
	  		f.incrementCount(feature, -Math.abs(f1.getCount(feature) - f2.getCount(feature)));
	  	}

	  	return f;
	}


	@Override
  public Counter<F> getDefaultParameters() {
		return defaultParam;
  }

	@Override
  public int dim() {
	  return defaultParam.size();
  }

}
