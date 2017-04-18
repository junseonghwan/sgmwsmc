package common.graph;

import briefj.collections.Counter;

/**
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 * @param <F> denotes the type for the canonical features.
 */
public interface GraphNode<F> extends Comparable<GraphNode<F>>
{
	public int getIdx();
	public int getPartitionIdx();
	public Counter<F> getNodeFeatures();
	
}
