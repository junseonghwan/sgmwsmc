package common.learning;

import java.util.Random;

/**
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 * @param <T> Type of variable (for example, Double if the parameters are real
 */
public interface SupportSet<T> 
{
	public boolean inSupport(T t);
	public void initParam(Random random, double [] w);
}
