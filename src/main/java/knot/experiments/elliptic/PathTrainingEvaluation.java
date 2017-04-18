package knot.experiments.elliptic;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.learning.BridgeSamplingLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.PairwiseMatchingModel;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import knot.data.EllipticalKnot;
import knot.data.KnotDataReader.Segment;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

public class PathTrainingEvaluation implements Runnable {

	@Option public static int numParticles = 10;
	@Option public static int maxIter = 100;
	
	@Option public static double lambda = 1;
	@Option public static double tol = 1e-2;
	@Option Random rand = new Random(1);
	@Option public static boolean exactSampling = true;
	@Option public static boolean sequentialMatching = false;
	public static String [] BOARDS = {};

	public static String [] dataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	@Option public static String fileName = "enhanced_matching_segmented.csv";

	public static String outputPath = "output/knot-matching/";
	public static String paramOutputPath = outputPath + "real_param_estimate.csv";
	public static String performanceOutputPath = outputPath + "segmented_real_boards_em_training.csv";

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
	public void run() {
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<List<Segment>> segments = KnotExpUtils.readSegmentedBoard(fileName, BOARDS, false);
		List<List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>>> data = PathTraining.unpack(segments, true);

		EllipticalKnotFeatureExtractor fe = new EllipticalKnotFeatureExtractor();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe);
		
		ObservationDensity<GenericGraphMatchingState<String, EllipticalKnot>, Object> observationDensity = new ObservationDensity<GenericGraphMatchingState<String,EllipticalKnot>, Object>() {

			@Override
			public double logDensity(
			    GenericGraphMatchingState<String, EllipticalKnot> latent,
			    Object emission) {
				return 0;
			}

			@Override
			public double logWeightCorrection(
			    GenericGraphMatchingState<String, EllipticalKnot> curLatent,
			    GenericGraphMatchingState<String, EllipticalKnot> oldLatent) {
				return 0;
			}

			@Override
			public boolean cancellationApplied() {
				return false;
			}
		};
		
		// store the lines to be outputted to a file
		List<String> performanceOutputLines = new ArrayList<>();
		List<String> parametersOutput = new ArrayList<>();
		

		// perform leave-one-out cross validation
		for (int i = 0; i < data.size(); i++)
		{
			List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>> instance = data.remove(i);
			
			// prepare the data for training
			List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>> instances = new ArrayList<>();
			for (List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>> datum : data)
			{
				instances.addAll(datum);
			}
			
			Pair<Double, double []> ret = BridgeSamplingLearning.learn(rand, command, instances, numParticles, numParticles, lambda, maxIter, tol, false, null);
			command.updateModelParameters(ret.getSecond());

			// predict on the real data
			int numCorrect = 0;
			int numTotal = 0;
			int bestMatchingCorrect = 0;
			double jaccardIndex = 0.0;
			int numNodes = 0;
			for (Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>> segment : instance)
			{
				GenericGraphMatchingState<String, EllipticalKnot> initialState = GraphMatchingState.getInitialState(segment.getFirst());
				LatentSimulator<GenericGraphMatchingState<String, EllipticalKnot>> transitionDensity = new GenericMatchingLatentSimulator<>(command, initialState, false, true);

				List<Object> emissions = new ArrayList<>(segment.getFirst());
				SequentialGraphMatchingSampler<String, EllipticalKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, false);
				smc.sample(100, 100, null);
				List<GenericGraphMatchingState<String, EllipticalKnot>> samples = smc.getSamples();
				MatchingSampleEvaluation<String, EllipticalKnot> eval = MatchingSampleEvaluation.evaluate(samples, segment.getSecond());
				System.out.println("Maximum likelihood matching: " + eval.bestLogLikMatching.getSecond().getFirst() + ", accuracy: " + eval.bestLogLikMatching.getSecond().getSecond());
				System.out.println("Best accuracy: " + eval.bestAccuracyMatching.getSecond());

				bestMatchingCorrect += eval.bestAccuracyMatching.getSecond();
				numCorrect += eval.bestLogLikMatching.getSecond().getSecond();
				numTotal += segment.getSecond().size();
				jaccardIndex += eval.avgJaccardIndex;
				numNodes += segment.getFirst().size();
			}
			
			// add the data back
			data.add(i, instance);
			
			String line = BOARDS[i] + ", " + numCorrect + ", " + bestMatchingCorrect + ", " + numTotal + ", " + jaccardIndex + ", " + numNodes;
			performanceOutputLines.add(line);
			
			for (String f : command.getModelParameters())
			{
				String str = i + ", " + f + ", " + command.getModelParameters().getCount(f);
				parametersOutput.add(str);
			}

			System.out.println("===== Board " + BOARDS[i] + " evaluation Summary =====");
			System.out.println(numCorrect + "/" + numTotal);
			System.out.println("===== End evaluation Summary =====");
		}

		
		if (paramOutputPath != null)
		{
			PrintWriter writer = BriefIO.output(new File(paramOutputPath));
			for (String paramLine : parametersOutput)
			{
				writer.println(paramLine);
			}
			writer.close();
		}
		
		if (performanceOutputPath != null)
		{
			PrintWriter writer = BriefIO.output(new File(performanceOutputPath));
			for (String line : performanceOutputLines)
			{
				writer.println(line);
			}
			writer.close();
			
		}

	}

	public static void main(String[] args) {
		Mains.instrumentedRun(args, new PathTrainingEvaluation());
	}

}
