package simulation.experiments;

import java.util.ArrayList;
import java.util.List;

import briefj.collections.Counter;
import common.graph.GraphNode;

public class SimpleNode implements GraphNode<String>
{
	private int pidx, idx;
	private Counter<String> features = new Counter<>();
	public static List<String> featureNames = new ArrayList<>(); 
	public SimpleNode(int pidx, int idx, double [] features)
	{
		this.pidx = pidx;
		this.idx = idx;
		for (int i = 0; i < features.length; i++)
		{
			this.features.setCount("f" + i, features[i]);
			if (!featureNames.contains("f" + i))
				featureNames.add("f" + i);
		}
	}
	
	@Override
  public int compareTo(GraphNode<String> o) {
		if (this.getPartitionIdx() < o.getPartitionIdx())
			return -1;
		else if (this.getPartitionIdx() > o.getPartitionIdx())
			return 1;
		else {
			if (this.idx < o.getIdx())
				return -1;
			else if (this.idx > o.getIdx())
				return 1;
			else
				return 0;
		}
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
  public Counter<String> getNodeFeatures() {
	  return features;
  }

	@Override
	public String toString()
	{
		return "(" + pidx + ", " + idx + ")";
	}
}
