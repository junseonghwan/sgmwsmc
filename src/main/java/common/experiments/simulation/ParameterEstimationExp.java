package common.experiments.simulation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.Pair;

import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class ParameterEstimationExp implements Runnable {

	@Option public static final int numRep = 10;
	@Option public static final int numData = 10;
	@Option public static final int numPartitions = 4;
	@Option public static final int numNodesPerPartition = 10;
	@Option public static final int numFeatures = 10;
	@Option public static final double sigma_var = 1.4;
	@Option public static final double nu_var = 1.1;
	
	@Option public static final Random random = new Random(999);
	@Option public static final double tol = 1e-6;
	@Option public static int gridSize = 100;
	@Option public static boolean outputSurface = false;
	@Option public static boolean appendOutput = true;

	@Option public static final String outputPath = "output/simulation/";
	@Option public static final String knownPathExpOutputFileName = "known_sequence_param_estimation_exp.csv";
	@Option public static final String surfaceDataFileName = "known_sequence_param_estimation_surface2";

	public static final double lambda = 1.0/(2*sigma_var);
	public static final String staticOutput = numData + ", " + numPartitions + ", " + numNodesPerPartition + ", " + numFeatures + ", " + sigma_var + ", " + nu_var;
	
	public static DecisionModel<String, SimpleNode> decisionModel = new PairwiseMatchingModel<>();
	public static GraphFeatureExtractor<String, SimpleNode> fe = null; 
	public static Command<String, SimpleNode> command = null;


	@Override
	public void run() 
	{
		// generate parameters \theta ~ MVN(0, \Sigma) or \theta_j ~ N(0, \sigma_j^2)
		// generate nodes and the features f(v)_j ~ N(0, \nu_j^2), j = 1, ..., numFeatures
		// assign the nodes to partitions
		// generate matching
		// estimate the parameters -- MAP estimate
		// experiment with the known vs unknown sequence, number of sequences for unknown sequence case, number of data points, variances \sigma_j's

		// generate parameters
		double [] w = SimulationUtils.sampleParameters(random, numFeatures, 0.0, sigma_var);
		DescriptiveStatistics stat = new DescriptiveStatistics();
		List<String> outputLines = new ArrayList<>();
		for (int i = 0; i < numRep; i++)
		{
			// evaluate the surface only for the first replication
			stat.addValue(correctSequenceExperiments(random, w, outputSurface && (i == 0), outputLines));			
		}
		System.out.println("rmse_mean: " + stat.getMean() + ", rmse_var: " + stat.getVariance());

		// append to the file
		File outputFile = new File(outputPath + knownPathExpOutputFileName);
		PrintWriter writer = null;
		if (outputFile.exists() && appendOutput) {
			try {
				writer = new PrintWriter(new FileOutputStream(outputFile, true));
			} catch (Exception ex) {
				System.out.println(ex.getMessage());
				throw new RuntimeException();
			}
		} else {
			writer = BriefIO.output(outputFile);
			writer.println("I, K, N, d, sigma_var, nu_var, rmse, nllk_at_truth, nllk_at_map");
		}
		for (String line : outputLines)
		{
			writer.println(line);
		}
		writer.close();		
	}

	// return RMSE
	public double correctSequenceExperiments(Random random, double [] w, boolean outputSurface, List<String> outputLines)
	{
		// generate the data
		List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances = new ArrayList<>();		
		for (int n = 0; n < numData; n++)
		{
			List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, sigma_var);
			Collections.shuffle(nodes);

			if (command == null)
			{
				// initialize the model settings
				fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
				command = new Command<>(decisionModel, fe);
				command.updateModelParameters(w);
			}

			// generate a matching
			GraphMatchingState<String, SimpleNode> state = GraphMatchingState.getInitialState(nodes);
			while (state.hasNextStep())
			{
				state.sampleNextState(random, command, true, true);
			}
			//System.out.println("instance1=" + state);
			System.out.println(nodes);
			System.out.println(state.getVisitedNodes());
			instances.add(Pair.create(state.getMatchings(), state.getVisitedNodes()));
		}

		SupervisedLearning<String, SimpleNode> sl = new SupervisedLearning<>();
		double [] initial = new double[numFeatures];
		for (int j = 0; j < initial.length; j++)
		{
			initial[j] = random.nextGaussian();
			System.out.println("initial[" + j + "]: " + initial[j]);
		}

		double valAtTheta = 0.0;
		for (Pair<List<Set<SimpleNode>>, List<SimpleNode>> instance : instances)
		{
			valAtTheta += SupervisedLearning.value(command, command.getModelParameters(), instance).getFirst();
		}

		Pair<Double, double[]> ret = sl.MAP(command, instances, lambda, initial, 1e-6, true);

		double valAtEstimate = ret.getFirst();

		System.out.println("nllk@MAP: " + valAtEstimate + ", nllk@theta: " + -valAtTheta);
		System.out.println("Estimate, Truth");
		double rmse = 0.0;
		for (int i = 0; i < fe.dim(); i++)
		{
			System.out.println(ret.getSecond()[i] + ", " + w[i]);
			rmse += Math.pow(ret.getSecond()[i] - w[i], 2.0);
		}
		rmse /= fe.dim();
		rmse = Math.sqrt(rmse);
		System.out.println("rmse: " + rmse);

		String outputLine = staticOutput + ", " + rmse + ", " + -valAtTheta + ", " + ret.getFirst(); 
		outputLines.add(outputLine);
		
		// let's evaluate the likelihood over a grid of points to see how it looks
		if (outputSurface)
		{
			List<String> lines = SimulationUtils.evaluateSurfaceOverRegularGrid(sl, command, instances, fe, gridSize);

			PrintWriter writer = null;
			File file = new File(outputPath + surfaceDataFileName + "_" + numFeatures + ".csv");
			if (file.exists()) {
				try {
					writer = new PrintWriter(new FileOutputStream(file, true));
				} catch (Exception ex) {
					System.out.println(ex);
					throw new RuntimeException();
				}
			} else {
				writer = BriefIO.output(file);
				String header = "";
				for (int d = 0; d < numFeatures; d++) {
					header += "f" + d;
					if (d < numFeatures - 1)
						header += ", ";
				}
					
				writer.println("I, K, N, d, sigma_var, nu_var, " + header + ", nllk, nllk_truth, nllk_map");
			}

			for (String line : lines) {
				writer.println(staticOutput + ", " + line + ", " + -valAtTheta + ", " + valAtEstimate);
			}
			writer.close();
		}

		return rmse;
	}

	public static void main(String[] args) {
		Mains.instrumentedRun(args, new ParameterEstimationExp());
	}

}
