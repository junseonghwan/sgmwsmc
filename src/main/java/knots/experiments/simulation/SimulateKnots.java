package knots.experiments.simulation;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import bayonet.distributions.Normal;
import bayonet.distributions.Uniform;
import briefj.BriefIO;
import knot.data.RectangularKnot;

public class SimulateKnots 
{

	public static Pair<List<Set<RectangularKnot>>, List<RectangularKnot>> simulateKnots(Random random, int N, double xmax, double ymax, double zmax, double wmin, double wmax, double hmin, double hmax, double deltaMean, double deltaVar, double corr)
	{
		List<Set<RectangularKnot>> matching = new ArrayList<>(N);
		List<RectangularKnot> knots = new ArrayList<>();

		int idx = 0;
		for (int n = 0; n < N; n++)
		{
			List<Integer> surfaces = new ArrayList<>(4);
			surfaces.add(0); surfaces.add(1); surfaces.add(2); surfaces.add(3);

			Set<RectangularKnot> edge = new HashSet<>();
			// sample which of the two wide surfaces to place the first know
			int s0 = random.nextInt(2) * 2;
			surfaces.remove(s0);
			// sample the coordinates
			double [] xyz0 = new double[3];
			xyz0[0] = random.nextDouble() * xmax;
			xyz0[1] = random.nextDouble() * ymax;
			xyz0[2] = s0 == 0 ? 0 : zmax;
			// sample the width and height
			double w0 = Uniform.generate(random, wmin, wmax);
			double h0 = Uniform.generate(random, hmin, hmax);
			RectangularKnot k0 = new RectangularKnot(s0, idx++, xyz0[0], xyz0[1], xyz0[2], w0, h0);
			edge.add(k0);
			
			RectangularKnot splittedKnot = checkKnotBoundaryAndSplit(s0, idx, xyz0, w0, h0, ymax, zmax);
			if (splittedKnot != null) {
				edge.add(splittedKnot);
				surfaces.remove(new Integer(splittedKnot.getPartitionIdx()));
			}

			// sample the opposite knot face
			// sample the surface
			int s1 = surfaces.remove(random.nextInt(surfaces.size()));
			double [] xyz1 = new double[3];

			// sample x1 | x
			xyz1[0] = xyz0[0] + Normal.generate(random, deltaMean, deltaVar); // shift the x-location by delta_x ~ N(mu, var)
			// sample or determine the y and z coordinates
			if (s1 == 0) {
				xyz1[1] = random.nextDouble() * ymax; 
				xyz1[2] = 0;
			} else if (s1 == 1) {
				xyz1[1] = ymax;
				xyz1[2] = random.nextDouble() * zmax; 
			}	else if (s1 == 2) {
				xyz1[1] = random.nextDouble() * ymax; 
				xyz1[2] = zmax;
			} else if (s1 == 3) {
				xyz1[1] = 0;
				xyz1[2] = random.nextDouble() * zmax; 
			}

			double w1 = Normal.generate(random, corr * w0, (1 - corr * corr) * 5);
			double h1 = Normal.generate(random, corr * h0, (1 - corr * corr) * 5);
			RectangularKnot k1 = new RectangularKnot(s1, idx++, xyz1[0], xyz1[1], xyz1[2], w1, h1);
			edge.add(k1);
			
			if (edge.size() < 4) {
				matching.add(edge);
				knots.addAll(edge);
				//System.out.println(edge);
			}
		}
		
		return Pair.create(matching, knots);
	}
	
	private static RectangularKnot checkKnotBoundaryAndSplit(int surface, int idx, double [] xyz, double w, double h, double ymax, double zmax)
	{
		if (surface == 0 || surface == 2) {
  		if ((xyz[1] + h/2) > ymax) {
  			//System.out.println("3-matching");
  			// create a knot face for the part that is extending beyond the edge of the board
  			double offset = (xyz[1] + h/2 - ymax)/2;
  			double z = surface == 0 ? offset : zmax - offset;
  			RectangularKnot k1 = new RectangularKnot(1, idx++, xyz[0], ymax, z, w, offset * 2);
  			return k1;
  		} else if ((xyz[1] - h/2) < 0) {
  			//System.out.println("3-matching");
  			double offset = (xyz[1] + h/2)/2;
  			double z = surface == 0 ? offset : zmax - offset;
  			RectangularKnot k1 = new RectangularKnot(3, idx++, xyz[0], 0, z, w, offset * 2);
  			return k1;
  		}
		}
		return null;
	}

	
	public static void main(String [] args)
	{
		Random random = new Random(1);
		int N = 20;
		double xmax = 5000;
		double ymax = 300;
		double zmax = 150;
		double wmin = 20;
		double wmax = 100;
		double hmin = 20;
		double hmax = 50;
		double deltaMean = 30;
		double deltaVar = 10;
		double corr = 0.6;
		Pair<List<Set<RectangularKnot>>, List<RectangularKnot>> ret = SimulateKnots.simulateKnots(random, N, xmax, ymax, zmax, wmin, wmax, hmin, hmax, deltaMean, deltaVar, corr);
		// output the knot and plot it using R (or GNU plot) for visualization
		// print the matching
		for (Set<RectangularKnot> e : ret.getFirst())
		{
			System.out.println(e + "{");
			for (RectangularKnot knot : e)
			{
				System.out.print(knot.getNodeFeatures().getCount("x") + ", ");
				System.out.print(knot.getNodeFeatures().getCount("y") + ", ");
				System.out.print(knot.getNodeFeatures().getCount("z") + ", ");
				System.out.print(knot.getNodeFeatures().getCount("w") + ", ");
				System.out.print(knot.getNodeFeatures().getCount("h") + " ");
				System.out.println();
			}
			System.out.println("}");
		}
		
		List<String> output = new ArrayList<>();
		List<RectangularKnot> knots = ret.getSecond();
		Collections.sort(knots);
		for (RectangularKnot knot : knots)
		{
			StringBuilder sb = new StringBuilder();
			sb.append(knot.getNodeFeatures().getCount("x") + ", ");
			sb.append(knot.getNodeFeatures().getCount("y") + ", ");
			sb.append(knot.getNodeFeatures().getCount("z") + ", ");
			sb.append(knot.getNodeFeatures().getCount("w") + ", ");
			sb.append(knot.getNodeFeatures().getCount("h") + " ");
			output.add(sb.toString());
		}
		
		PrintWriter writer = BriefIO.output(new File("output/simulated-knots/knots.csv"));
		for (String line : output)
		{
			writer.println(line);
		}
		writer.close();
	}
}
