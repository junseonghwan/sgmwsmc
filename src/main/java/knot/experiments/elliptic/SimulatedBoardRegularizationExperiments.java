package knot.experiments.elliptic;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import common.evaluation.LearningUtils;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
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
import knot.data.KnotDataReader.Segment;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;

public class SimulatedBoardRegularizationExperiments implements Runnable 
{
	@Option public static Random rand = new Random(1);
	@Option public static String [] dataDirectories = {"data/simmatching/"};
	@Option public static String inputFileName = "enhanced_matching_segmented";
	@Option public static double tol = 1e-6;
	@Option public static boolean exactSampling = true;
	@Option public static boolean sequentialMatching = false;
	@Option public static int numConcreteParticles = 1000;
	@Option public static int maxNumVirtualParticles = 10000;
	@Option public static String outputPath = "output/knot-matching/simmatching/";
	@Option public static String resultsFile = "simmatching_regularization_results.csv";
	@Option public static String detailsFile = "simmatching_regularization_details.csv";
	@Option public static String paramFile = "simmatching_regularization_param.csv";

	public static String [] BOARDS = {};
	public static List<Double> lambdas = new ArrayList<>();
	public static double lmin = 0;
	public static double lmax = 2;
	public static double interval = 0.05;
	@Option public static int K = 5; // K-fold CV 
	@Option public static int numRep = 5;
	
	static {
		double lambda = lmin; 
		while (lambda < lmax)
		{
			lambdas.add(lambda);
			lambda += interval;
		}
	}

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
					if (board.indexOf("segmented") >= 0)
						boards.add(dataDirectory + "" + board);
				}
			}
			BOARDS = boards.toArray(new String[boards.size()]);
		}
	}

	@Override
	public void run()
	{
		// read the data
		List<List<Segment>> instances = KnotExpUtils.readSegmentedBoard(null, BOARDS, false);

		// declare the decision model and the feature extractor to be used
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();

		// use cross validation to select lambda
		// 1. for loop over the lambdas
		// 2. in the for loop, form K groups (folds) randomly
		// 3. hold out one group, estimate the parameters, evaluate the error (loss) on the held out group -- take the sum over each group
		// 4. output K, lambda, sum of the errors
		List<String> lines = new ArrayList<>();
		List<String> detailedLines = new ArrayList<>();
		List<String> paramsOutputLines = new ArrayList<>();
		for (int n = 0; n < numRep; n++)
		{
			for (double lambda : lambdas)
			{
				// randomly allocate data points to groups
				List<List<List<Segment>>> groups = KnotExpUtils.generateFoldsSegmentedBoardVersion(rand, K, instances);
	
				double jaccardLoss = 0.0;
				double zeroOneLoss = 0.0;
				for (int k = 0; k < groups.size(); k++)
				{
					System.out.println(lambda + ", " + n + ", " + k);

					List<List<Segment>> heldOut = groups.remove(k);
					List<List<Segment>> remaining = new ArrayList<>();
					for (List<List<Segment>> group : groups)
						remaining.addAll(group);
					
					List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> unpackedTrainingInstances = KnotExpUtils.unpack(remaining);
					List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> trainingInstances = KnotExpUtils.pack(unpackedTrainingInstances);
					Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(rand, decisionModel, fe, trainingInstances, lambda, tol);
				
					Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe, ret.getSecond());
					ObservationDensity<GenericGraphMatchingState<String, EllipticalKnot>, Object> observationDensity = null;
					if (exactSampling)
						observationDensity = new ExactProposalObservationDensity<>(command);
					else
						observationDensity  = new RandomProposalObservationDensity<>(command);
		
					int numCorrect = 0;
					int numTotal = 0;
					double jaccardIndex = 0.0;
					int numNodes = 0;
					for (Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>> segment : KnotExpUtils.pack(KnotExpUtils.unpack(heldOut)))
					{
						GraphMatchingState<String, EllipticalKnot> initial = GraphMatchingState.getInitialState(segment.getSecond());
						LatentSimulator<GenericGraphMatchingState<String, EllipticalKnot>> transitionDensity = new GenericMatchingLatentSimulator<String, EllipticalKnot>(command, initial, sequentialMatching, exactSampling);
		
						List<Object> emissions = new ArrayList<>();
						for (int j = 0; j < segment.getSecond().size(); j++) emissions.add(null);
		
						// draw samples using SMC
						SequentialGraphMatchingSampler<String, EllipticalKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
						if (segment.getSecond().size() <= 3) {
							smc.sample(10, maxNumVirtualParticles);
						} else {
							smc.sample(numConcreteParticles, maxNumVirtualParticles);
						}
						List<GenericGraphMatchingState<String, EllipticalKnot>> samples = smc.getSamples();
						MatchingSampleEvaluation<String, EllipticalKnot> eval = MatchingSampleEvaluation.evaluate(samples, segment.getFirst());
		
						//double logLikAtTruth = SupervisedLearning.value(command, ret.getSecond(), segment).getFirst();
						//System.out.println("===== Segment evaluation Summary =====");
						//System.out.println(segment.getFirst());
						//System.out.println("Consensus matching accuracy: " + eval.consensusMatching.getSecond() + ", size: " + eval.consensusMatching.getFirst().size());
						//System.out.println(eval.consensusMatching.getFirst().toString());
						//System.out.println("Maximum likelihood matching: " + eval.bestLogLikMatching.getSecond().getFirst() + ", accuracy: " + eval.bestLogLikMatching.getSecond().getSecond());
						//System.out.println(eval.bestLogLikMatching.getFirst().toString());
						//System.out.println("Best accuracy: " + eval.bestAccuracyMatching.getSecond());
						//System.out.println("Average correctness: " + eval.avgAccuracy);
						//System.out.println("Avg Jaccard Index: " + eval.avgJaccardIndex);
						//System.out.println("Total # of matching: " + segment.getFirst().size());
						//System.out.println("loglik@truth: " + logLikAtTruth);
						//System.out.println("Time (s): " + (end-start)/1000.0);
						//System.out.println("===== End segment evaluation Summary =====");
		
						numTotal += segment.getFirst().size();
						numNodes += segment.getSecond().size();
						try {
  						numCorrect += eval.bestLogLikMatching.getSecond().getSecond();
  						jaccardIndex += eval.avgJaccardIndex;
  						//numNonTrivialSegments += segment.getFirst().size() > 1 ? 1 : 0;
						} catch (Exception ex) {
							System.out.println(ex.getMessage());
						}
					}

					double jaccardLossK = 1 - (jaccardIndex / numNodes);
					double zeroOneLossK = 1 - ((double)numCorrect / numTotal);
					jaccardLoss += jaccardLossK;
					zeroOneLoss += zeroOneLossK;
					groups.add(heldOut);

					String detailedOutput1 = lambda + ", " + n + ", " + k + ", " + jaccardLossK + ", " + zeroOneLossK;
					detailedLines.add(detailedOutput1);
					for (String f : ret.getSecond())
					{
						paramsOutputLines.add(lambda + ", " + n + ", " + k + ", " + f + ", " + ret.getSecond().getCount(f));
					}
				}
				String line = lambda + ", " + n + ", " + jaccardLoss + ", " + zeroOneLoss;
				//System.out.println(line);
				lines.add(line);
			}
		}

		// output to file
		PrintWriter writer = BriefIO.output(new File(outputPath + resultsFile));
		writer.println("lambda, n, jaccard, zeroOne");
		for (String line : lines) {
			writer.println(line);
		}
		writer.close();
		
		writer = BriefIO.output(new File(outputPath + detailsFile));
		writer.println("lambda, n, k, jaccard, zeroOne");
		for (String line : detailedLines) {
			writer.println(line);
		}
		writer.close();

		writer = BriefIO.output(new File(outputPath + paramFile));
		writer.println("lambda, n, k, feature_name, feature_value");
		for (String line : paramsOutputLines) {
			writer.println(line);
		}
		writer.close();

	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new SimulatedBoardRegularizationExperiments());
	}

}
