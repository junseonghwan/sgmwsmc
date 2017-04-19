package knot.experiments.simulation;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.EllipticalKnot;
import knot.data.KnotDataReader.Segment;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.KnotPairwiseMatchingDecisionModel;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

import org.apache.commons.math3.util.Pair;

import common.evaluation.LearningUtils;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.RandomProposalObservationDensity;
import common.smc.components.SequentialGraphMatchingSampler;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;

/**
 * Experiments to validate the data simulation procedure. 
 * Estimate the parameters using the simulated data. Predict on the real data. Evaluate the performance.
 *  
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 */
public class ValidateBoardGeneration implements Runnable
{
	@Option public static double lambda = 0.1;
	@Option public static double tol = 1e-6;
	@Option Random rand = new Random(1);
	@Option public static boolean exactSampling = true;
	@Option public static boolean sequentialMatching = true;
	@Option public static int numConcreteParticles = 1000;
	@Option public static int maxNumVirtualParticles = 10000;

	@Option
	public static String paramOutputName = "simulated_param_estimate.csv";
	@Option
	public static String outputPath = "output/knot-matching/";
	@Option public static String trainingDataDirectory = "data/simmatching/";
	@Option
	public static int numTrainingData = 100;
	@Option
	public static String [] testingDataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	@Option
	public static String fileName = "enhanced_matching_segmented";
	@Option
	public static String [] TRAINING_BOARDS = {};
	@Option
	public static String [] TESTING_BOARDS = {};
	static {
		if (TRAINING_BOARDS.length == 0 && numTrainingData > 0) {
			TRAINING_BOARDS = new String[numTrainingData];
			for (int i = 0; i < numTrainingData; i++) {
				TRAINING_BOARDS[i] = i + 1 + "";
			}
		}
		if (TESTING_BOARDS.length == 0) {
			TESTING_BOARDS = new String[numTrainingData];
			List<String> boards = new ArrayList<>();
			for (String dataDirectory : testingDataDirectories)
			{
				List<File> dirs = BriefFiles.ls(new File(dataDirectory));
				for (int i = 0; i < dirs.size(); i++)
				{
					String board = dirs.get(i).getName();
					if (board.charAt(0) == '.') continue;
					boards.add(dataDirectory + "" + board);
				}
			}
			TESTING_BOARDS = boards.toArray(new String[boards.size()]);
		}
	}

	@Override
	public void run()
	{
		// read the training data and estimate the parameters
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<List<Segment>> trainingInstances = KnotExpUtils.readSegmentedSimulatedBoard(trainingDataDirectory, fileName, TRAINING_BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> trainingData = KnotExpUtils.unpack(trainingInstances);

		List<List<Segment>> testingInstances = KnotExpUtils.readSegmentedBoard(null, TESTING_BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> testingData = KnotExpUtils.unpack(testingInstances);

		DecisionModel<String, EllipticalKnot> decisionModel = new KnotPairwiseMatchingDecisionModel();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();

		long seed = rand.nextLong();
		Random random = new Random(seed);
		
		List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> trainingInstaces = KnotExpUtils.pack(trainingData);
		Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(random, decisionModel, fe, trainingInstaces, lambda, tol);
		
		// predict on the testing set
		List<String> lines = new ArrayList<>();
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe, ret.getSecond());
		ObservationDensity<GenericGraphMatchingState<String, EllipticalKnot>, Object> observationDensity = null;
		if (exactSampling)
			observationDensity = new ExactProposalObservationDensity<>(command);
		else
			observationDensity  = new RandomProposalObservationDensity<>(command);

		List<MatchingSampleEvaluation<String, EllipticalKnot>> evals = new ArrayList<>();

		for (int i = 0; i < testingInstances.size(); i++)
		{
			int numCorrect = 0;
			int numTotal = 0;
			int numNonTrivialSegments = 0;
			int bestMatchingCorrect = 0;
			double jaccardIndex = 0.0;
			int numNodes = 0;

			List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> heldOut = testingData.get(i);
			for (Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>> segment : heldOut)
			{
				GraphMatchingState<String, EllipticalKnot> initial = GraphMatchingState.getInitialState(segment.getSecond());
				LatentSimulator<GenericGraphMatchingState<String, EllipticalKnot>> transitionDensity = new GenericMatchingLatentSimulator<String, EllipticalKnot>(command, initial, sequentialMatching, exactSampling);
	
				List<Object> emissions = new ArrayList<>();
				for (int j = 0; j < segment.getSecond().size(); j++) emissions.add(null);
	
				// draw samples using SMC
				System.out.println("Evaluating board " + TESTING_BOARDS[i]);
				long start = System.currentTimeMillis();
				SequentialGraphMatchingSampler<String, EllipticalKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
				smc.sample(numConcreteParticles, maxNumVirtualParticles);
				List<GenericGraphMatchingState<String, EllipticalKnot>> samples = smc.getSamples();
				MatchingSampleEvaluation<String, EllipticalKnot> eval = MatchingSampleEvaluation.evaluate(samples, segment.getFirst());
				long end = System.currentTimeMillis();
				evals.add(eval);
	
				double logLikAtTruth = SupervisedLearning.value(command, ret.getSecond(), segment).getFirst();
				System.out.println("===== Segment evaluation Summary =====");
				System.out.println(segment.getFirst());
				System.out.println("Consensus matching accuracy: " + eval.consensusMatching.getSecond() + ", size: " + eval.consensusMatching.getFirst().size());
				//System.out.println(eval.consensusMatching.getFirst().toString());
				System.out.println("Maximum likelihood matching: " + eval.bestLogLikMatching.getSecond().getFirst() + ", accuracy: " + eval.bestLogLikMatching.getSecond().getSecond());
				//System.out.println(eval.bestLogLikMatching.getFirst().toString());
				System.out.println("Best accuracy: " + eval.bestAccuracyMatching.getSecond());
				System.out.println("Average correctness: " + eval.avgAccuracy);
				System.out.println("Avg Jaccard Index: " + eval.avgJaccardIndex);
				System.out.println("Total # of matching: " + segment.getFirst().size());
				System.out.println("loglik@truth: " + logLikAtTruth);
				System.out.println("Time (s): " + (end-start)/1000.0);
				System.out.println("===== End segment evaluation Summary =====");
	
				bestMatchingCorrect += eval.bestAccuracyMatching.getSecond();
				numCorrect += eval.bestLogLikMatching.getSecond().getSecond();
				numTotal += segment.getFirst().size();
				jaccardIndex += eval.avgJaccardIndex;
				numNodes += segment.getSecond().size();
				numNonTrivialSegments += segment.getFirst().size() > 1 ? 1 : 0;
			}
	
			if (lines != null) {
				String line = TESTING_BOARDS[i] + ", " + numCorrect + ", " + bestMatchingCorrect + ", " + numTotal + ", " + jaccardIndex + ", " + numNodes + ", " + numNonTrivialSegments;
				lines.add(line);
			}
	
			System.out.println("===== Board " + TESTING_BOARDS[i] + " evaluation Summary =====");
			System.out.println(numCorrect + "/" + numTotal);
			System.out.println("===== End evaluation Summary =====");
		}

		PrintWriter writer = BriefIO.output(new File(outputPath + "segmented_sim_data_validation_exp.csv"));
		writer.println("board, prediction, best, total, jaccard, num_nodes, segments");
		for (String line : lines) {
			writer.println(line);
		}
		writer.close();
		
		// output the parameters: 
		writer = BriefIO.output(new File(outputPath + paramOutputName));
		for (String f : ret.getSecond())
		{
			writer.println(f + ", " + ret.getSecond().getCount(f));
		}
		writer.close();


	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new ValidateBoardGeneration());
	}

}
