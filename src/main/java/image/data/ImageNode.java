package image.data;

import briefj.collections.Counter;
import common.graph.GraphNode;

/**
 * Representation of a key point (vertex) of CMU House image data
 * Features are indexed by integers: 
 * 1, ..., 60: indexes one of 60 shape context features
 * getAdjacency() returns the adjacency matrix obtained via Delaunay triangulation for deriving the edge features
 * 
 * TODO: Create a more general class to represent a key point (features should perhaps be represented using String) 
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 */
public class ImageNode implements GraphNode<Integer>
{
	private int pidx, idx;
	private Counter<Integer> nodeFeatures;
	private Counter<Integer> adjacency;

	public ImageNode(int pidx, int idx, Counter<Integer> features)
	{
		this.pidx = pidx;
		this.idx = idx;
		this.nodeFeatures = features;
	}
	
	public ImageNode(int pidx, int idx, Counter<Integer> features, Counter<Integer> adj)
	{
		this(pidx, idx, features);
		this.adjacency = adj;
	}

	@Override
	public int compareTo(GraphNode<Integer> o) {
		if (this.getIdx() < o.getIdx())
			return -1;
		else if (this.getIdx() > o.getIdx())
			return 1;
		return 0;
	}

	@Override
	public int getIdx() {
	  return idx;
	}
	
	@Override
	public int getPartitionIdx() {
	  return pidx;
	}

	@Override
	public Counter<Integer> getNodeFeatures() {
	  return nodeFeatures;
	}
	
	public Counter<Integer> getAdjacency() {
		return adjacency;
	}

	@Override
	public String toString() {
		return "(" + getPartitionIdx() + ", " + getIdx() + ")";
	}

	@Override
	public int hashCode()
	{
		return this.toString().hashCode();
	}
	
	@Override
	public boolean equals(Object o)
	{
		return (this.hashCode() == o.hashCode());
	}

}
