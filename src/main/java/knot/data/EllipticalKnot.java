package knot.data;

import java.util.ArrayList;
import java.util.List;

import briefj.collections.Counter;
import common.graph.GraphNode;

public class EllipticalKnot implements Knot
{
	private int idx;
	private int pidx;
	private Counter<String> nodeFeatures = new Counter<>();

	public static List<String> featureNames = new ArrayList<>();

	static {
		featureNames.add("x");
		featureNames.add("y");
		featureNames.add("z");
		featureNames.add("n");
		featureNames.add("var_x");
		featureNames.add("var_y");
		featureNames.add("cov_xy");
		featureNames.add("boundary_axis0");
		featureNames.add("boundary_axis1");
		featureNames.add("yaxis");
		featureNames.add("zaxis");
		featureNames.add("area_over_axis");
	}

	public EllipticalKnot(int pidx, int idx, double x, double y, double z, double n, double var_x, double var_y, double cov, int boundary_axis0, int boundary_axis1, double yaxis, double zaxis, double area_over_axis)
	{
		this.pidx = pidx;
		this.idx = idx;
		nodeFeatures.setCount("x", x);
		nodeFeatures.setCount("y", y);
		nodeFeatures.setCount("z", z);
		nodeFeatures.setCount("n", n);
		nodeFeatures.setCount("var_x", var_x);
		nodeFeatures.setCount("var_y", var_y);
		nodeFeatures.setCount("cov_xy", cov);
		nodeFeatures.setCount("boundary_axis0", boundary_axis0);
		nodeFeatures.setCount("boundary_axis1", boundary_axis1);
		nodeFeatures.setCount("yaxis", yaxis);
		nodeFeatures.setCount("zaxis", zaxis);
		nodeFeatures.setCount("area_over_axis", area_over_axis);
	}

	@Override
	public int compareTo(GraphNode<String> o) {
		EllipticalKnot other = (EllipticalKnot)o; 
		if (this.nodeFeatures.getCount("x") < other.nodeFeatures.getCount("x"))
			return -1;
		else if (this.nodeFeatures.getCount("x") > other.nodeFeatures.getCount("x"))
			return 1;
		else 
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
	public Counter<String> getNodeFeatures() {
		return nodeFeatures;
	}
	
	@Override
	public String toString()
	{
		return "(" + pidx + ", " + idx + ")"; 
	}

}
