package image.experiments;

import image.data.ImageDataReader;
import image.data.ImageNode;
import image.model.ImageFeatureExtractor;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import common.evaluation.MatchingSampleEvaluation;
import common.graph.BipartiteMatchingState;
import common.graph.GenericGraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.BipartiteDecisionModel;
import common.model.Command;
import common.model.DecisionModel;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.BriefParallel;
import briefj.opt.Option;
import briefj.run.Mains;

public class EdgeFeatureExperiment implements Runnable
{
	@Option(required=true) public static int k = 25;
	@Option(required=true) public static double lambda = 1.0;
	@Option(required=true) public static boolean isTrainingSetDivisibleByK = true;
	@Option(required=true) public static int repNum = 0;
	public static String outputDirName = "output/image/edge-feature-performance/";
	public static String outputFileName = "edge_feature_performance";
	public static String settingsFile = "edge_feature_settings";

	@Override
	public void run()
	{
		long seed = System.currentTimeMillis();
		Random random = new Random(seed);
		List<Pair<List<Set<ImageNode>>, List<ImageNode>>> trainingInstances = ImageDataReader.prepareData(random, k, isTrainingSetDivisibleByK);
		SupervisedLearning<String, ImageNode> sl = new SupervisedLearning<>();
		ImageFeatureExtractor fe = new ImageFeatureExtractor(trainingInstances.get(0).getSecond().get(0));
		DecisionModel<String, ImageNode> decisionModel = new BipartiteDecisionModel<>();
		Command<String, ImageNode> command = new Command<>(decisionModel, fe);
		double [] w = sl.MAP(command, trainingInstances, lambda, new double[fe.dim()], 1e-6, false).getSecond();
		command.updateModelParameters(w);
		for (String f : command.getModelParameters())
			System.out.println(f + ": " + command.getModelParameters().getCount(f));

		// now predict
		List<Pair<List<Set<ImageNode>>, List<ImageNode>>> testingInstances = ImageDataReader.prepareData(random, k, !isTrainingSetDivisibleByK);

		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < testingInstances.get(0).getSecond().size(); i++) emissions.add(null);

		int [][] output = new int[testingInstances.size()][3];
		BriefParallel.process(testingInstances.size(), 8, i -> {
	  		BipartiteMatchingState<String, ImageNode> initialState = BipartiteMatchingState.getInitial(testingInstances.get(i).getSecond()); 
	  		output[i][0] = initialState.getUnvisitedNodes().get(0).getPartitionIdx();
	  		output[i][1] = initialState.getPartition2().get(0).getPartitionIdx();
	  		GenericMatchingLatentSimulator<String, ImageNode> transitionDensity = new GenericMatchingLatentSimulator<>(command, initialState, false, true);
	  		ObservationDensity<GenericGraphMatchingState<String, ImageNode>, Object> observationDensity = new ExactProposalObservationDensity<>(command);
	  		SequentialGraphMatchingSampler<String, ImageNode> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
	  		smc.sample(random, 1000, 1000);
	  		MatchingSampleEvaluation<String, ImageNode> me = MatchingSampleEvaluation.evaluate(smc.getSamples(), testingInstances.get(i).getFirst());
	  		output[i][2] = me.consensusMatching.getSecond();
	  		System.out.println(output[i][0] + ", " + output[i][1] + ", " + output[i][2]);
		});

		if (!isTrainingSetDivisibleByK) {
			outputFileName = outputDirName + outputFileName + "_" + k + "_large_training_" + repNum + ".csv";
			settingsFile = outputDirName + settingsFile + "_" + k + "_large_training_" + repNum + ".txt";
		} else {
			outputFileName = outputDirName + outputFileName + "_" + k + "_" + repNum + ".csv";
			settingsFile = outputDirName + settingsFile + "_" + k + "_" + repNum + ".txt";
		}

		BriefFiles.createParentDirs(new File(outputFileName));
		PrintWriter writer = BriefIO.output(new File(outputFileName));
		double sum = 0.0;
		for (int i = 0; i < testingInstances.size(); i++)
		{
			sum += output[i][2];
			writer.println(output[i][0] + ", " + output[i][1] + ", " + output[i][2]);
		}
		System.out.println("Average performance: " + sum/testingInstances.size());
		writer.close();

		PrintWriter settings = BriefIO.output(new File(settingsFile));
		settings.println("lambda=" + lambda);
		settings.println("seed=" + seed);
		settings.println("k=" + k);
		settings.println("isTrainingSetDivisibleByK=" + isTrainingSetDivisibleByK);
		settings.close();

	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new EdgeFeatureExperiment());
	}

}
