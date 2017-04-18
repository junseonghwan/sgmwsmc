package knot.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import knot.model.features.elliptic.AreaFeatureExtractor;

import org.apache.commons.math3.util.Pair;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import common.evaluation.MatchingSampleEvaluation;

import briefj.BriefIO;

// provide utilities for reading in the lumber data 
public class KnotDataReader 
{
	public static final double [] Y_DIM = new double[]{200.0, 500.0};
	public static final double [] Z_DIM = new double[]{350.0, 500.0};
	public static final int BOARD_LENGTH = 168; // inches
	public static final int BOARD_WIDTH = 4; // inches
	public static final int BOARD_HEIGHT = 2; // inches

	public static Pair<List<Set<RectangularKnot>>, List<RectangularKnot>> readRectangularKnots(String file, boolean reverseSequence)
	{
		Map<Integer, Set<RectangularKnot>> label2Edge = new HashMap<>();
		List<RectangularKnot> knots = new ArrayList<>();
		
		// open the file, read in the data -- each row of the file is a knot
		for (String line : BriefIO.readLines(file))
		{
			String [] row = line.split(",");
			int pidx = Integer.parseInt(row[0].trim()); 
			int idx = Integer.parseInt(row[1].trim());
			double x = Double.parseDouble(row[3].trim());
			double y = Double.parseDouble(row[4].trim());
			double z = Double.parseDouble(row[5].trim());
			double w = Double.parseDouble(row[6].trim());
			double h = Double.parseDouble(row[7].trim());
			y *= (BOARD_WIDTH/(Y_DIM[1] - Y_DIM[0]));
			z *= (BOARD_HEIGHT/(Z_DIM[1] - Z_DIM[0]));

			int label = 0;
			if (row.length == 9) label = Integer.parseInt(row[8].trim());
			if (label == 0) continue;

			RectangularKnot knot = new RectangularKnot(pidx, idx, x, y, z, w, h);
			knots.add(knot);

			if (label2Edge.containsKey(label))
			{
				label2Edge.get(label).add(knot);
			}
			else
			{
				label2Edge.put(label, Sets.newHashSet(knot));
			}
		}

		if (!reverseSequence)
			Collections.sort(knots);
		else
			Collections.reverse(knots);	
		return Pair.create(new ArrayList<>(label2Edge.values()), knots);
	}

	public static Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>> readEllipticalKnots(String file, boolean reverseSequence)
	{
		System.out.println("Processing " + file);
		Map<Integer, Set<EllipticalKnot>> label2Edge = new HashMap<>();
		List<EllipticalKnot> knots = new ArrayList<>();

		// open the file, read in the data -- each row of the file is a knot
		int lineno = 0;
		for (String line : BriefIO.readLines(file))
		{
			lineno++;
			if (lineno == 1) continue;
			String [] row = line.split(",");
			double x = Double.parseDouble(row[0].trim());
			double y = Double.parseDouble(row[1].trim());
			double z = Double.parseDouble(row[2].trim());
			double varX = Double.parseDouble(row[3].trim());
			double varY = Double.parseDouble(row[4].trim());
			double cov = Double.parseDouble(row[5].trim());
			double n = Double.parseDouble(row[6].trim());
			int pidx = Integer.parseInt(row[8].trim()); 
			int idx = Integer.parseInt(row[9].trim());
			int label = Integer.parseInt(row[10].trim());
			int boundary_axis0 = Integer.parseInt(row[11].trim());
			int boundary_axis1 = Integer.parseInt(row[12].trim());
			double yaxis = Double.parseDouble(row[13].trim());
			double zaxis = Double.parseDouble(row[14].trim());
			double area_over_axis = Double.parseDouble(row[15].trim());

			EllipticalKnot knot = new EllipticalKnot(pidx, idx, x, y, z, n, varX, varY, cov, boundary_axis0, boundary_axis1, yaxis, zaxis, area_over_axis);
			knots.add(knot);

			if (label2Edge.containsKey(label))
			{
				label2Edge.get(label).add(knot);
			}
			else
			{
				label2Edge.put(label, Sets.newHashSet(knot));
			}
		}

		if (!reverseSequence)
			Collections.sort(knots);
		else
			Collections.reverse(knots);	
		return Pair.create(new ArrayList<>(label2Edge.values()), knots);
	}

	public static Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>> readEllipticalKnotsTwoMatchings(String file, boolean reverseSequence)
	{
		System.out.println("Processing " + file);
		Map<Integer, Set<EllipticalKnot>> label2Edge = new HashMap<>();

		// open the file, read in the data -- each row of the file is a knot
		int lineno = 0;
		for (String line : BriefIO.readLines(file))
		{
			lineno++;
			if (lineno == 1) continue;
			String [] row = line.split(",");
			double x = Double.parseDouble(row[0].trim());
			double y = Double.parseDouble(row[1].trim());
			double z = Double.parseDouble(row[2].trim());
			double varX = Double.parseDouble(row[3].trim());
			double varY = Double.parseDouble(row[4].trim());
			double cov = Double.parseDouble(row[5].trim());
			double n = Double.parseDouble(row[6].trim());
			int pidx = Integer.parseInt(row[8].trim()); 
			int idx = Integer.parseInt(row[9].trim());
			int label = Integer.parseInt(row[10].trim());
			int boundary_axis0 = Integer.parseInt(row[11].trim());
			int boundary_axis1 = Integer.parseInt(row[12].trim());
			double yaxis = Double.parseDouble(row[13].trim());
			double zaxis = Double.parseDouble(row[14].trim());
			double area_over_axis = Double.parseDouble(row[15].trim());

			EllipticalKnot knot = new EllipticalKnot(pidx, idx, x, y, z, n, varX, varY, cov, boundary_axis0, boundary_axis1, yaxis, zaxis, area_over_axis);

			if (label2Edge.containsKey(label))
			{
				Set<EllipticalKnot> e = label2Edge.get(label);
				e.add(knot);
				if (e.size() == 3) {
					Set<EllipticalKnot> eprime = chooseTwoMatching(e);
					label2Edge.put(label, eprime);
				}
			}
			else
			{
				label2Edge.put(label, Sets.newHashSet(knot));
			}
		}

		List<Set<EllipticalKnot>> truth = new ArrayList<>(label2Edge.values());
		List<EllipticalKnot> knots = MatchingSampleEvaluation.getNodes(truth);
		if (!reverseSequence)
			Collections.sort(knots);
		else
			Collections.reverse(knots);
		
		return Pair.create(truth, knots);
	}
	
	public static Set<EllipticalKnot> chooseTwoMatching(Set<EllipticalKnot> e)
	{
		if (e.size() != 3) throw new RuntimeException();
		List<EllipticalKnot> d = new ArrayList<>(e);
		Collections.sort(d, new Comparator<EllipticalKnot>() {
			@Override
			public int compare(EllipticalKnot o1, EllipticalKnot o2) {
				double a1 = AreaFeatureExtractor.computeArea(o1)[0];
				double a2 = AreaFeatureExtractor.computeArea(o2)[0];
				if (a1 < a2)
					return -1;
				else if (a2 > a1)
					return 1;
				return 0;
			}
		});
		/*
		System.out.println(AreaFeatureExtractor.computeArea(d.get(0)));
		System.out.println(AreaFeatureExtractor.computeArea(d.get(1)));
		System.out.println(AreaFeatureExtractor.computeArea(d.get(2)));
		*/
		Set<EllipticalKnot> ret = new HashSet<>();
		ret.add(d.get(1));
		ret.add(d.get(2));
		return ret;
	}

	public static List<Segment> readSegmentedBoard(String file)
	{
		System.out.println("Processing " + file);

		// open the file, read in the data -- each row of the file is a knot
		List<Segment> segments = new ArrayList<>();
		int lineno = 0;
		for (String line : BriefIO.readLines(file))
		{
			lineno++;
			if (lineno == 1) continue;
			String [] row = line.split(",");
			double x = Double.parseDouble(row[0].trim());
			double y = Double.parseDouble(row[1].trim());
			double z = Double.parseDouble(row[2].trim());
			double varX = Double.parseDouble(row[3].trim());
			double varY = Double.parseDouble(row[4].trim());
			double cov = Double.parseDouble(row[5].trim());
			double n = Double.parseDouble(row[6].trim());
			int pidx = Integer.parseInt(row[8].trim()); 
			int idx = Integer.parseInt(row[9].trim());
			int label = Integer.parseInt(row[10].trim());
			int boundary_axis0 = Integer.parseInt(row[11].trim());
			int boundary_axis1 = Integer.parseInt(row[12].trim());
			double yaxis = Double.parseDouble(row[13].trim());
			double zaxis = Double.parseDouble(row[14].trim());
			double area_over_axis = Double.parseDouble(row[15].trim());
			int segment = Integer.parseInt(row[16].trim());

			EllipticalKnot knot = new EllipticalKnot(pidx, idx, x, y, z, n, varX, varY, cov, boundary_axis0, boundary_axis1, yaxis, zaxis, area_over_axis);
			if (segments.size() < segment)
				segments.add(new Segment(segment));

			segments.get(segment-1).addNode(label, knot);
		}

		return segments;
	}

	public static ImmutableList<RectangularKnot> simulateKnots(Random random, int numPartitions, int numNodesPerPartition)
	{
		List<RectangularKnot> nodes = new ArrayList<>();
		for (int pidx = 0; pidx < numPartitions; pidx++)
		{
			for (int idx = 0; idx < numNodesPerPartition; idx++)
			{
				double x = random.nextDouble()*100;
				double y = random.nextDouble()*20;
				double z = random.nextDouble()*10;
				double w = random.nextDouble()*2;
				double h = random.nextDouble()*2;
				nodes.add(new RectangularKnot(pidx, idx, x, y, z, w, h));
			}
		}
		return ImmutableList.copyOf(nodes);
	}

	public static class Segment
	{
		public int id;
		public Map<Integer, Set<EllipticalKnot>> label2Edge;
		public List<EllipticalKnot> knots;

		public Segment(int id)
		{
			this.id = id;
			this.label2Edge = new HashMap<>();
			this.knots = new ArrayList<>();
		}
		
		public void addNode(int label, EllipticalKnot knot)
		{
			knots.add(knot);
			if (label2Edge.containsKey(label))
			{
				label2Edge.get(label).add(knot);
			}
			else
			{
				label2Edge.put(label, Sets.newHashSet(knot));
			}
		}
	}
}
