package image.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import briefj.collections.Counter;
import image.data.ImageNode;
import common.graph.BipartiteMatchingState;
import common.graph.GenericGraphMatchingState;
import common.model.CanonicalFeatureExtractor;
import common.model.GraphFeatureExtractor;

public class ImageFeatureExtractor implements GraphFeatureExtractor<String, ImageNode>
{
	private CanonicalFeatureExtractor<Integer, ImageNode> fe;

	public ImageFeatureExtractor(ImageNode example) 
	{
		fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(example);
	}

	@Override
	public Counter<String> extractFeatures(ImageNode node, Set<ImageNode> decision, GenericGraphMatchingState<String, ImageNode> matching) 
	{
		if (!(matching instanceof BipartiteMatchingState))
			throw new RuntimeException("At the moment, ImageFeatureExtractor can only work with bipartite matching.");

		BipartiteMatchingState<String, ImageNode> matchingState = (BipartiteMatchingState<String, ImageNode>)matching;
		
		// compute the canonical features
		Counter<Integer> scf = fe.extractFeatures(node, decision, null);
		Counter<String> features = new Counter<>();
		for (Integer f : scf)
		{
			features.setCount("scf" + f, scf.getCount(f));
		}

		// incorporate the quadratic features
		// 1. iterate over adjacency matrix for node, for adjacent node, check if it is covered.
		// if already covered but matched with a node not in the adjacency of the otherNode, -1
		// if already covered and matched with a node in the adjacency of the otherNode, 1
		// if not covered (then obviously not matched yet), then 1 since it can still lead to correct matching in the future
		// 2. repeat for the otherNode

		// node2Edge map view of the matching state
		Map<ImageNode, Set<ImageNode>> node2Edge = matchingState.getNode2EdgeView();
		// get the node being proposed for matching
		ImageNode otherNode = (new ArrayList<ImageNode>(decision)).get(0);

		Counter<Integer> adj = node.getAdjacency();
		Counter<Integer> adjOther = otherNode.getAdjacency();

		Set<ImageNode> coveredNodes = matchingState.getCoveredNodes();
		double incorrect = 0.0;
		for (ImageNode coveredNode : coveredNodes)
		{
			if (coveredNode.getPartitionIdx() == node.getPartitionIdx())
				incorrect += adjacent(coveredNode, adj, adjOther, node2Edge);
			else
				incorrect += adjacent(coveredNode, adjOther, adj, node2Edge);
		}
		/*
		if (incorrect < -5 && node.getIdx() == otherNode.getIdx()) {
			System.out.println("num incorrect: " + incorrect);
			System.out.println(node.toString() + ", " + otherNode.toString());
			System.out.println(matchingState.toString());
		}
		*/
		features.setCount("adj", incorrect);

		/*
		Set<ImageNode> p1 = new HashSet<>(matchingState.getVisitedNodes());
		p1.addAll(matchingState.getUnvisitedNodes());
		Set<ImageNode> p2 = new HashSet<>(matchingState.getPartition2());

		int total = 0;
		for (Integer f : adj) {
			total += adj.getCount(f);
			total += adjOther.getCount(f);
		}

		double sum1 = 0.0;
		for (ImageNode node1 : p1)
		{
			sum1 += adjacent(node1, adj, adjOther, node2Edge);
		}
		double sum2 = 0.0;
		for (ImageNode node2 : p2)
		{
			sum2 += adjacent(node2, adjOther, adj, node2Edge);
		}

		features.setCount("adj", (sum1 + sum2)/total);
		*/
		return features;
  }
	
	public int adjacent(ImageNode i, Counter<Integer> adj, Counter<Integer> adjOther, Map<ImageNode, Set<ImageNode>> node2Edge)
	{
		// check if node i is in adj
		if (adj.getCount(i.getIdx()) == 1) {
			if (!node2Edge.containsKey(i))
				return 0; // i is not covered yet, so we don't have information on whether its partner will be in adjOther 

			// node i is matched, check if i's match is in adjOther
			Set<ImageNode> e = node2Edge.get(i);
			if (e.size() > 2)
				throw new RuntimeException("|e| > 2: BipartiteMatchingState is not correctly implemented.");

			for (ImageNode iprime : e)
			{
				if (iprime.equals(i)) continue;
				if (adjOther.getCount(iprime.getIdx()) == 1)
				{
					return 0;
				}
				else
				{
					// matched with a node not in adjOther
					return -1;
				}
			}
		}
		return 0; // not adjacent 
	}

	@Override
  public Counter<String> getDefaultParameters() {
	  Counter<Integer> param = fe.getDefaultParameters();
	  Counter<String> defaultParam = new Counter<>();
	  for (Integer f : param)
	  {
		  defaultParam.setCount("scf" + f, param.getCount(f));
	  }
	  defaultParam.setCount("adj", 0.0);
	  return defaultParam;
  }

	@Override
  public int dim() {
	  return fe.dim() + 1;
  }

	@Override
	public Counter<String> extractFeatures(Set<ImageNode> e,
			List<Set<ImageNode>> matchingState) {
		throw new RuntimeException("Not implemented.");
	}

}
