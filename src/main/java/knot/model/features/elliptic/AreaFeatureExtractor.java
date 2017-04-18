package knot.model.features.elliptic;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;

import briefj.collections.Counter;
import common.graph.GenericGraphMatchingState;
import common.model.GraphFeatureExtractor;
import knot.data.EllipticalKnot;

public class AreaFeatureExtractor implements GraphFeatureExtractor<String, EllipticalKnot> 
{
	public static final String TWO_MATCHING_AREA_DIFF = "TWO_MATCHING_AREA_DIFF";
	//public static final String TWO_MATCHING_ANGLE_DIFF = "TWO_MATCHING_ANGLE_DIFF";
	public static final String THREE_MATCHING_AREA_DIFF = "THREE_MATCHING_AREA_DIFF";
	//public static final String NUM_POINTS_DIFF = "NUM_POINTS_DIFF";
	
	public static double NORM_CONST = 1000;
	public static double NORM_CONST2 = 500;
	public static double CONFIDENCE_LEVEL = 0.975;
	public static double SQRT_CRITICAL_VALUE;
	
	public static List<String> featureNames = new ArrayList<>();
	static {
		Field [] fields = AreaFeatureExtractor.class.getFields();
		for (Field field : fields)
		{
			if (Modifier.isFinal(field.getModifiers()) && Modifier.isStatic(field.getModifiers()))
				featureNames.add(field.getName());
		}
		
		ChiSquaredDistribution chiSq = new ChiSquaredDistribution(2);
		SQRT_CRITICAL_VALUE = Math.sqrt(chiSq.inverseCumulativeProbability(CONFIDENCE_LEVEL));
	}

	@Override
	public Counter<String> extractFeatures(EllipticalKnot node,
	    Set<EllipticalKnot> decision,
	    GenericGraphMatchingState<String, EllipticalKnot> matchingState) {
		
		Counter<String> f = getDefaultParameters();
		
		if (decision.size() == 1) {
			// 2-matching -- compare the area and the number of points
  			double [] a1 = computeArea(node);
  			double [] a2 = null;
  			double n2 = 0;
  			for (EllipticalKnot otherNode : decision) {
  				a2 = computeArea(otherNode);
  				n2 = otherNode.getNodeFeatures().getCount("n");

  				if (node.getPartitionIdx() % 2 == 0 && otherNode.getPartitionIdx() % 2 == 0)
  					f.setCount(TWO_MATCHING_AREA_DIFF, Math.abs(a1[0] - a2[0])/NORM_CONST);
  				else {
  					if (AxisFeatureExtractor.sharesAxis(node, otherNode)) {
  						//f.setCount(TWO_MATCHING_AREA_DIFF_2, 1.0);
//  						f.setCount(TWO_MATCHING_ANGLE_DIFF, Math.abs(a1[1] - a2[1])/NORM_CONST);
  					} else {
  						f.setCount(TWO_MATCHING_AREA_DIFF, Math.abs(a1[0] - a2[0])/NORM_CONST);
  					}
  				}
	  			if (a2[0] == 0.0) throw new RuntimeException("Area = 0? Either computation failed or knot detection failed.");
	  			//f.setCount(TWO_MATCHING_AREA_DIFF, Math.abs(a1 - a2)/NORM_CONST);
	  			//f.setCount(NUM_POINTS_DIFF, Math.abs(node.getNodeFeatures().getCount("n") - n2));
			}
		} else if (decision.size() == 2) {
			// 3-matching -- compute area for each of the knots, then add up the two smaller ones and compare with the larger one
			List<EllipticalKnot> knots = new ArrayList<>(decision);
			knots.add(node);
			
			List<Pair<Double, Double>> pairs = new ArrayList<>();
			for (int i = 0; i < knots.size(); i++)
			{
				double area = computeArea(knots.get(i))[0];
				double n = knots.get(i).getNodeFeatures().getCount("n");
				pairs.add(Pair.create(area, n));
				for (int j = i+1; j < knots.size(); j++)
				{
					// compute the distance and combine the areas for two nearer knots
				}
			}

			Collections.sort(pairs, new Comparator<Pair<Double, Double>>() {
				@Override
				public int compare(Pair<Double, Double> o1, Pair<Double, Double> o2) {
					double a1 = o1.getFirst();
					double a2 = o2.getFirst();
					if (a1 < a2) {
						return -1;
					} else if (a1 > a2) {
						return 1;
					}
					return 0;
				}
			});

			//double a1 = pairs.get(0).getFirst() + pairs.get(1).getFirst();
			double a1 = pairs.get(1).getFirst();
			double a2 = pairs.get(2).getFirst();

			/*
			double n1 = pairs.get(0).getSecond() + pairs.get(1).getSecond();
			double n2 = pairs.get(2).getSecond();
			*/

			f.setCount(THREE_MATCHING_AREA_DIFF, Math.abs(a1 - a2)/NORM_CONST);
			//f.setCount(NUM_POINTS_DIFF, Math.abs(n1 - n2)/NORM_CONST2);
		}

		return f;
	}
	
	@Override
	public Counter<String> extractFeatures(Set<EllipticalKnot> e, List<Set<EllipticalKnot>> matchingState) {
		Counter<String> f = getDefaultParameters();
		
		List<EllipticalKnot> d = new ArrayList<>(e);
		if (e.size() == 2) {
			// 2-matching -- compare the area and the number of points
			double [] a1 = computeArea(d.get(0));
			EllipticalKnot otherNode = d.get(1);
			double [] a2 = computeArea(otherNode);
			double n2 = otherNode.getNodeFeatures().getCount("n");
			if (a2[0] == 0.0) throw new RuntimeException("Area = 0? Either computation failed or knot detection failed.");
			if (d.get(0).getPartitionIdx() % 2 == 0 && d.get(1).getPartitionIdx() % 2 == 0)
				f.setCount(TWO_MATCHING_AREA_DIFF, Math.abs(a1[0] - a2[0])/NORM_CONST);
			else {
					if (AxisFeatureExtractor.sharesAxis(d.get(0), d.get(1))) {
  						//f.setCount(TWO_MATCHING_AREA_DIFF_2, 1.0);
  						//f.setCount(TWO_MATCHING_ANGLE_DIFF, Math.abs(a1[1] - a2[1])/NORM_CONST);
  					} else {
  						f.setCount(TWO_MATCHING_AREA_DIFF, Math.abs(a1[0] - a2[0])/NORM_CONST);
  					}

				//f.setCount(TWO_MATCHING_ANGLE_DIFF_2, Math.abs(a1[1] - a2[1])/NORM_CONST);
			}
			//f.setCount(NUM_POINTS_DIFF, Math.abs(node.getNodeFeatures().getCount("n") - n2));
		} else if (e.size() == 3) {
			// 3-matching -- compute area for each of the knots, then add up the two smaller ones and compare with the larger one
			List<EllipticalKnot> knots = new ArrayList<>(e);
			
			List<Pair<Double, Double>> pairs = new ArrayList<>();
			for (int i = 0; i < knots.size(); i++)
			{
				double [] ret = computeArea(knots.get(i));
				double n = knots.get(i).getNodeFeatures().getCount("n");
				pairs.add(Pair.create(ret[0], n));
			}

			Collections.sort(pairs, new Comparator<Pair<Double, Double>>() {
				@Override
				public int compare(Pair<Double, Double> o1, Pair<Double, Double> o2) {
					double a1 = o1.getFirst();
					double a2 = o2.getFirst();
					if (a1 < a2) {
						return -1;
					} else if (a1 > a2) {
						return 1;
					}
					return 0;
				}
			});

			double a1 = pairs.get(0).getFirst() + pairs.get(1).getFirst();
			double a2 = pairs.get(2).getFirst();

			double n1 = pairs.get(0).getSecond() + pairs.get(1).getSecond();
			double n2 = pairs.get(2).getSecond();

			f.setCount(THREE_MATCHING_AREA_DIFF, Math.abs(a1 - a2)/NORM_CONST);
			//f.setCount(NUM_POINTS_DIFF, Math.abs(n1 - n2)/NORM_CONST2);
		}

		return f;
	}

	public static RealVector e1 = new ArrayRealVector(new double[]{1.0, 0.0}); 
	public static double [] computeArea(EllipticalKnot knot)
	{
		double [][] S = new double[2][2];
		S[0][0] = knot.getNodeFeatures().getCount("var_x");
		S[0][1] = S[1][0] = knot.getNodeFeatures().getCount("cov_xy");
		S[1][1] = knot.getNodeFeatures().getCount("var_y");
		EigenDecomposition eigen = new EigenDecomposition(MatrixUtils.createRealMatrix(S));
		double [] lambdas = eigen.getRealEigenvalues();
		double area = Math.PI * Math.sqrt(lambdas[0]) * SQRT_CRITICAL_VALUE * Math.sqrt(lambdas[1]) * SQRT_CRITICAL_VALUE; 
		// compute the rotation angle
		double dot = eigen.getEigenvector(0).dotProduct(e1);
		// the eigen vector has length 1 so does e1
		double angle = Math.acos(dot);
		return new double[]{area, angle};
	}

	@Override
	public Counter<String> getDefaultParameters() {
	  Counter<String> init = new Counter<>();
	  for (String featureName : featureNames)
	  {
	  	init.setCount(featureName, 0.0);
	  }
	  return init;
	}

	@Override
	public int dim() {
	  return featureNames.size();
	}

}
