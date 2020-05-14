package tests;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import briefj.BriefFiles;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;
import common.evaluation.LearningUtils;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.RandomProposalObservationDensity;
import common.smc.components.SequentialGraphMatchingSampler;
import knot.data.EllipticalKnot;
import knot.data.Knot;
import knot.data.KnotDataReader;
import knot.data.KnotDataReader.Segment;
import knot.experiments.elliptic.SegmentedExperimentsRealData;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

public class MatchingPerformanceTester implements Runnable 
{

	@Option public static int numConcreteParticles = 200;
	@Option public static int maxVirtualParticles = 1000;
	@Option public static double lambda = 0.05;
	@Option public static double tol = 1e-6;
	@Option Random rand = new Random(1);
	@Option public static boolean exactSampling = true;
	@Option public static boolean sequentialMatching = true;
	public static String [] BOARDS = {};
	public static String [] TESTING_BOARDS = {"data/16Mar2016/54003"};
	@Option public static String fileName = "enhanced_matching_segmented";

	public static String [] dataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	//public static String outputPath = "output/knot-matching/";

	static {
		if (BOARDS.length == 0) {
			List<String> boards = new ArrayList<>();
			for (String dataDirectory : dataDirectories)
			{
				List<File> dirs = BriefFiles.ls(new File(dataDirectory));
				for (int i = 0; i < dirs.size(); i++)
				{
					String board = dirs.get(i).getName();
					if (board.charAt(0) == '.') continue;
					boards.add(dataDirectory + "" + board);
				}
			}
			BOARDS = boards.toArray(new String[boards.size()]);
		}
	}

	@Override
	public void run() 
	{
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<List<Segment>> trainingInstances = readSegmentedBoard(fileName, BOARDS, TESTING_BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> trainingData = SegmentedExperimentsRealData.unpack(trainingInstances);

		List<List<Segment>> testingInstances = KnotExpUtils.readSegmentedBoard(fileName, TESTING_BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> testingData = SegmentedExperimentsRealData.unpack(testingInstances);
		//List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> testingData = KnotExpUtils.unpack(testingInstances);

		//DecisionModel<String, EllipticalKnot> decisionModel = new KnotPairwiseMatchingDecisionModel();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();

		// 1. train the model parameters
		// 2. make a deterministic pass on the segments to match any obvious cases
		// 3. run SMC on any remaining segments

		// leave-one-out cross validation
		List<String> lines = new ArrayList<>();
		long seed = rand.nextLong();
		//System.out.println("seed: " + seed);
		Random random = new Random(seed);
		evaluateSegmentedMatching(random, decisionModel, fe, trainingData, testingData, lambda, tol, sequentialMatching, exactSampling, lines);
	}

	public static List<List<Segment>> readSegmentedBoard(String filename, String [] boards, String [] testBoards, boolean reverseSequence)
	{
		List<List<Segment>> instances = new ArrayList<>();
		for (String board : boards)
		{
			if (isTestBoard(board, testBoards))
				continue;

			//String dataPath = dataDir + "Board " + lumber + "/knotdetection/matching/enhanced_matching_segmented.csv";
			String dataPath = board + "/enhanced_matching_segmented.csv";
			//System.out.println(dataPath);
			instances.add(KnotDataReader.readSegmentedBoard(dataPath));
		}

		return instances;
	}
	
	public static boolean isTestBoard(String board, String [] testBoards)
	{
		for (String testBoard : testBoards)
		{
			if (testBoard.equalsIgnoreCase(board))
				return true;
		}
		return false;
	}
	

	
	public static <KnotType extends Knot> List<MatchingSampleEvaluation<String, KnotType>> evaluateSegmentedMatching(Random random, 
			DecisionModel<String, KnotType> decisionModel, 
			GraphFeatureExtractor<String, KnotType> fe, 
			List<List<Pair<List<Set<KnotType>>, List<KnotType>>>> trainingInstances,
			List<List<Pair<List<Set<KnotType>>, List<KnotType>>>> testingInstances,
			double lambda, double tol, boolean sequentialMatching, boolean exactSampling, List<String> lines)
	{
		// train the parameters
		List<Pair<List<Set<KnotType>>, List<KnotType>>> trainingInstaces = KnotExpUtils.pack(trainingInstances);
		Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(random, decisionModel, fe, trainingInstaces, lambda, tol);
		// output the parameters
		for (String f : ret.getSecond())
		{
			System.out.println(f + ", " + ret.getSecond().getCount(f));
		}
		
		Command<String, KnotType> command = new Command<>(decisionModel, fe, ret.getSecond());

		// set up SMC
		ObservationDensity<GenericGraphMatchingState<String, KnotType>, Object> observationDensity = null;
		if (exactSampling)
			observationDensity = new ExactProposalObservationDensity<>(command);
		else
			observationDensity  = new RandomProposalObservationDensity<>(command);

		List<MatchingSampleEvaluation<String, KnotType>> evals = new ArrayList<>();

		for (int i = 0; i < testingInstances.size(); i++)
		{
			List<Pair<List<Set<KnotType>>, List<KnotType>>> heldOut = testingInstances.get(i);

			int numCorrect = 0;
			int numTotal = 0;
			int numNonTrivialSegments = 0;
			int bestMatchingCorrect = 0;
			double jaccardIndex = 0.0;
			int numNodes = 0;
			for (Pair<List<Set<KnotType>>, List<KnotType>> segment : heldOut)
			{
				GraphMatchingState<String, KnotType> initial = GraphMatchingState.getInitialState(segment.getSecond());
				LatentSimulator<GenericGraphMatchingState<String, KnotType>> transitionDensity = new GenericMatchingLatentSimulator<String, KnotType>(command, initial, sequentialMatching, exactSampling);

				List<Object> emissions = new ArrayList<>();
				for (int j = 0; j < segment.getSecond().size(); j++) emissions.add(null);

				// draw samples using SMC
				System.out.println("Evaluating board " + TESTING_BOARDS[i]);
				long start = System.currentTimeMillis();
				SequentialGraphMatchingSampler<String, KnotType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
				smc.sample(random, numConcreteParticles, maxVirtualParticles, null);
				List<GenericGraphMatchingState<String, KnotType>> samples = smc.getSamples();
				MatchingSampleEvaluation<String, KnotType> eval = MatchingSampleEvaluation.evaluate(samples, segment.getFirst());
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

		return evals;
	}

	
	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new MatchingPerformanceTester());
	}

}
