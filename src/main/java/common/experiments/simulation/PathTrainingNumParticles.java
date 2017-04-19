package common.experiments.simulation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.Pair;

import bayonet.distributions.Normal;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;
import common.graph.GraphMatchingState;
import common.learning.BridgeSamplingLearning;
import common.learning.SupervisedLearning;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;

public class PathTrainingNumParticles implements Runnable 
{

	@Option public static int numRep = 1;
	@Option public static int numData = 10;
	@Option public static int numPartitions = 4;
	@Option public static int numNodesPerPartition = 3;
	@Option public static int numFeatures = 2;

	@Option public static double tol = 1e-4;
	@Option public static int maxIter = 100;
	@Option public static int numParticles = 1000;
	
	@Option public static final double sigma_var = 1.4;
	public static final double lambda = 1/(2*sigma_var);
	@Option public Random rand = new Random(12);
	
	public static String staticOutputString = numData + ", " + numParticles + ", " + numPartitions + ", " + numNodesPerPartition + ", " + numFeatures;
	public static DecisionModel<String, SimpleNode> decisionModel = new PairwiseMatchingModel<>();
	public static Command<String, SimpleNode> command = null;
	public static GraphFeatureExtractor<String, SimpleNode> fe = null;

	public static boolean output = false;
	public static final String outputPath = "output/simulation/";
	public static final String unknownPathExpOutputFileName = "em_param_estimation.csv";

	@Override
	public void run() 
	{
		List<String> outputLines = new ArrayList<>();
		DescriptiveStatistics rmseMonteCarlo = new DescriptiveStatistics();
		DescriptiveStatistics nllkMonteCarlo = new DescriptiveStatistics();
		DescriptiveStatistics rmseKnownSeq = new DescriptiveStatistics();
		DescriptiveStatistics nllkKnownSeq = new DescriptiveStatistics();
		for (int i = 0; i < numRep; i++)
		{
			Random random = new Random(rand.nextLong());
			
			// generate parameters
			double [] w = new double[numFeatures];
			for (int n = 0; n < numFeatures; n++) {
				w[n] = Normal.generate(random, 0.0, sigma_var);
				System.out.println("w[" + n + "]: " + w[n]);
			}
			
			// generate the data
			List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances = generateData(random, w);

			double nllkAtTheta = 0.0;
			for (Pair<List<Set<SimpleNode>>, List<SimpleNode>> instance : instances)
			{
				nllkAtTheta += SupervisedLearning.value(command, command.getModelParameters(), instance).getFirst();
			}

			// use EM-type training to estimate the parameters
			double [] retMonteCarlo = estimateParametersUnknownSequence(random, w, instances, outputLines);
			rmseMonteCarlo.addValue(retMonteCarlo[0]);
			nllkMonteCarlo.addValue(retMonteCarlo[1]);

			// keep the true sequence, estimate the parameters and compute the likelihood at the MAP
			double [] retKnownSeq = estimateParamKnownSequence(random, w, command, instances, outputLines);
			rmseKnownSeq.addValue(retKnownSeq[0]);
			nllkKnownSeq.addValue(retKnownSeq[1]);

			System.out.println("nllk: " + retMonteCarlo[1] + ", " + retKnownSeq[1] + ", " + -nllkAtTheta);
		}
		System.out.println(rmseMonteCarlo.getMean() + ", " + rmseMonteCarlo.getStandardDeviation());
		//System.out.println(rmseKnownSeq.getMean() + ", " + rmseKnownSeq.getStandardDeviation());

		System.out.println(nllkMonteCarlo.getMean() + ", " + nllkMonteCarlo.getStandardDeviation());
		//System.out.println(nllkKnownSeq.getMean() + ", " + nllkKnownSeq.getStandardDeviation());

		if (output) {
			PrintWriter writer = null;
			File outputFile = new File(outputPath + unknownPathExpOutputFileName);
			if (!outputFile.exists()) {
				writer = BriefIO.output(outputFile);
				writer.println("numData, numMCSamples, numPartitions, maxNodesPerPartition, numFeatures, RMSE, nllk, type");
			} else {
				try {
					writer = new PrintWriter(new FileOutputStream(outputFile, true));
				} catch (FileNotFoundException ex) {
					throw new RuntimeException();
				}
			}
			for (String line : outputLines)
			{
				writer.println(line);
			}
			writer.close();
		}
	}
	
	private List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> generateData(Random random, double [] w)
	{
		List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances = new ArrayList<>();

		for (int n = 0; n < numData; n++)
		{
			List<SimpleNode> nodes = SimulationUtils.generateSimpleNodes(random, numPartitions, numNodesPerPartition, numFeatures, 0.0, sigma_var);
			Collections.shuffle(nodes, random); // randomize the sequence 
			
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
			
			instances.add(Pair.create(state.getMatchings(), nodes)); 
		}
		return instances;
	}
	
	public static double [] estimateParametersUnknownSequence(Random random, double [] theta0, List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances, List<String> outputLines)
	{
		// estimate the parameters
		Pair<Double, double []> ret = BridgeSamplingLearning.learnOnEntireBoard(random, command, instances, numParticles, numParticles, lambda, maxIter, tol, false, null);
		
		double valAtEstimate = ret.getFirst();

		System.out.println("Estimate, Truth");
		double rmse = 0.0;
		for (int i = 0; i < fe.dim(); i++)
		{
			System.out.println(ret.getSecond()[i] + ", " + theta0[i]);
			rmse += Math.pow(ret.getSecond()[i] - theta0[i], 2.0);
		}
		rmse /= fe.dim();
		rmse = Math.sqrt(rmse);
		System.out.println("rmse: " + rmse);

		String outputLine = staticOutputString + ", " + rmse + ", " + ret.getFirst() + ", UNKNOWN_SEQ";
		outputLines.add(outputLine);

		return new double[]{rmse, valAtEstimate};
	}

	public static double [] estimateParamKnownSequence(Random random, double [] theta0,  Command<String, SimpleNode> command, List<Pair<List<Set<SimpleNode>>, List<SimpleNode>>> instances, List<String> outputLines)
	{
		if (command == null)
			throw new RuntimeException("Must define Command");
		
		SupervisedLearning<String, SimpleNode> sl = new SupervisedLearning<>();
		double [] initial = new double[numFeatures];
		for (int j = 0; j < initial.length; j++)
		{
			initial[j] = random.nextGaussian();
			System.out.println("initial[" + j + "]: " + initial[j]);
		}

		Pair<Double, double[]> ret = sl.MAP(command, instances, lambda, initial, 1e-6, true);

		double valAtEstimate = ret.getFirst();

		System.out.println("Estimate, Truth");
		double rmse = 0.0;
		for (int i = 0; i < fe.dim(); i++)
		{
			System.out.println(ret.getSecond()[i] + ", " + theta0[i]);
			rmse += Math.pow(ret.getSecond()[i] - theta0[i], 2.0);
		}
		rmse /= fe.dim();
		rmse = Math.sqrt(rmse);
		System.out.println("rmse: " + rmse);

		String outputLine = staticOutputString + ", " + rmse + ", " + ret.getFirst() + ", KNOWN_SEQ";
		outputLines.add(outputLine);

		return new double[]{rmse, valAtEstimate};
	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new PathTrainingNumParticles());
	}

}
