package knot.data;

import java.util.ArrayList;
import java.util.List;

import briefj.collections.Counter;
import common.graph.GraphNode;

public class RectangularKnot implements Knot
{
  private int idx;
  private int pidx;
  Counter<String> nodeFeatures = new Counter<>();
  
  public static final List<String> featureNames = new ArrayList<>();
  
  static {
  	featureNames.add("x");
  	featureNames.add("y");
  	featureNames.add("z");
  	featureNames.add("w");
  	featureNames.add("h");
  }
	
	public RectangularKnot(int pidx, int idx, double x, double y, double z, double w, double h)
	{
		this.pidx = pidx;
		this.idx = idx;
	  nodeFeatures.setCount("x", x);
	  nodeFeatures.setCount("y", y);
	  nodeFeatures.setCount("z", z);
	  nodeFeatures.setCount("w", w);
	  nodeFeatures.setCount("h", h);
	}

	@Override
  public int getIdx() {
	  return this.idx;
  }

	@Override
  public int getPartitionIdx() {
	  return this.pidx;
  }

	@Override
  public Counter<String> getNodeFeatures() {
	  return nodeFeatures;
  }
	
	@Override
	public String toString()
	{
		return "(" + pidx + ", " + idx + ")"; 
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

	@Override
  public int compareTo(GraphNode<String> o) {
		/*
		if (this.getPartitionIdx() < o.getPartitionIdx())
			return -1;
		else if (this.getPartitionIdx() > o.getPartitionIdx())
			return 1;
		else {
			if (this.getIdx() < o.getIdx())
				return -1;
			else if (this.getIdx() > o.getIdx())
				return 1;
			else return 0;
		}
		*/
		RectangularKnot other = (RectangularKnot)o; 
		if (this.nodeFeatures.getCount("x") < other.nodeFeatures.getCount("x"))
			return -1;
		else if (this.nodeFeatures.getCount("x") > other.nodeFeatures.getCount("x"))
			return 1;
		else 
			return 0;
  }
}
