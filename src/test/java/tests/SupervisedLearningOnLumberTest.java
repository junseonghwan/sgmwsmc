package tests;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.KnotDataReader;
import knot.data.RectangularKnot;
import knot.model.KnotDoubletonDecisionModel;
import knot.model.features.common.DistanceFeatureExtractor;

import org.apache.commons.math3.util.Pair;

import briefj.BriefIO;
import briefj.collections.Counter;
import briefj.run.Mains;
import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;

public class SupervisedLearningOnLumberTest  implements Runnable
{
	public String dataDirectory = "data/16Oct2015/";
	public int [] lumbers = {4, 8, 17, 18, 20, 24};
	public double lambda = 1.0;
	public boolean outputToFile = false;

	public void run()
	{
		Random random = new Random(1609161609);

		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances = new ArrayList<>();
		for (int lumber : lumbers)
		{
			String dataPath = dataDirectory + "Board " + lumber + "/labelledMatching.csv";
			Pair<List<Set<RectangularKnot>>, List<RectangularKnot>> instance = KnotDataReader.readRectangularKnots(dataPath, true);
			instances.add(instance);
		}

		//GraphFeatureExtractor<String, RectangularKnot> fe = new DistanceSizeFeatureExtractor();
		GraphFeatureExtractor<String, RectangularKnot> fe = new DistanceFeatureExtractor<>();
		DecisionModel<String, RectangularKnot> decisionModel = new KnotDoubletonDecisionModel<>();
		Command<String, RectangularKnot> command = new Command<>(decisionModel, fe);
		
		SupervisedLearning<String, RectangularKnot> sl = new SupervisedLearning<>();
		double [] initial = new double[fe.dim()];
		for (int j = 0; j < initial.length; j++)
		{
			initial[j] = -random.nextDouble();
			System.out.println(initial[j]);
		}

		Pair<Double, double[]> ret = sl.MAP(command, instances, lambda, initial, 1e-6, true);
		command.updateModelParameters(ret.getSecond());
		System.out.println("min: " + ret.getFirst());
		for (String f : command.getModelParameters())
		{
			System.out.println(f + ": " + command.getModelParameters().getCount(f));
		}
		
		// let's evaluate the likelihood over a grid of points to see how it looks
		if (outputToFile)
		{
			writeToFile(command, instances);
		}
	}
	
	public void writeToFile(Command<String, RectangularKnot> command,  List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances)
	{
		File tempFile = new File("output/lumber-log-density" + instances.size() + ".txt");
		PrintWriter writer = BriefIO.output(tempFile);
		double min = -1, max = -0.01;
		int gridSize = 100;
		double stepSize = (max - min)/gridSize;
		double [] x = new double[2];
		Counter<String> param = command.getModelParameters();
		for (int i = 0; i <= gridSize; i++)
		{
			x[0] = min + stepSize * i;
			for (int j = 0; j <= gridSize; j++)
			{
				x[1] = min + stepSize * j; 
				param.setCount(0 + "", x[0]);
				param.setCount(1 + "", x[1]);
				// evaluate
				double val = 0.0;
				for (Pair<List<Set<RectangularKnot>>, List<RectangularKnot>> instance : instances)
				{
					val += SupervisedLearning.value(command, param, instance).getFirst();
				}
				writer.println(x[0] + ", " + x[1] + ", " + -val);
			}
		}
		writer.close();
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new SupervisedLearningOnLumberTest());
	}
}
