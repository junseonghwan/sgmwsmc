package scratch;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import knot.data.KnotDataReader;
import knot.data.RectangularKnot;

public class Scratch 
{

	public static void main(String [] args)
	{
		Random random = new Random(1);
		List<RectangularKnot> knots = KnotDataReader.simulateKnots(random, 2, 3);
		Set<RectangularKnot> set = new HashSet<>(knots);
		System.out.println(set.containsAll(knots));
		
		RectangularKnot knot1 = new RectangularKnot(0, 0, 0, 0, 0, 0, 0);
		for (RectangularKnot knot : set)
		{
			if (knot1.equals(knot))
				System.out.println("found!");
		}
		System.out.println(set.contains(knot1));
		
		Map<String, Integer> map = new HashMap<>();
		map.put("zbc", 1);
		map.put("abcd", 3);
		map.put("bcd", 2);
		map.put("rews", 2);
		for (String key : map.keySet())
		{
			System.out.println(key);
		}
	}
}
