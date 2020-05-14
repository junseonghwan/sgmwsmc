package registration.experiments;

import briefj.collections.Counter;
import common.graph.GraphNode;

public class PointCloud implements GraphNode<Double> {

	private int pidx, idx;
	private Counter<Double> nodeFeatures;
	private Counter<Integer> adjacency;

	public PointCloud(int pidx, int idx, Counter<Double> features)
	{
		this.pidx = pidx;
		this.idx = idx;
		this.nodeFeatures = features;
	}
	
	public PointCloud(int pidx, int idx, Counter<Double> features, Counter<Integer> adj)
	{
		this(pidx, idx, features);
		this.adjacency = adj;
	}
	
	public Counter<Integer> getAdjacency() {
		return adjacency;
	}
	
	@Override
	public int compareTo(GraphNode<Double> o) {
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
	public Counter<Double> getNodeFeatures() {
		return nodeFeatures;
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
