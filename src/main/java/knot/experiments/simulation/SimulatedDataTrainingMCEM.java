package knot.experiments.simulation;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.Pair;

import knot.data.EllipticalKnot;
import knot.data.KnotDataReader.Segment;
import knot.experiments.elliptic.RealDataTrainingMCEM;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import common.util.OutputHelper;
import briefj.BriefFiles;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;
import briefj.run.Results;

public class SimulatedDataTrainingMCEM implements Runnable 
{
	@Option(required=false) public static int fold = 2;
	@Option(required=false) public static boolean useSPF = true;
	@Option(required=false) public static int numConcreateParticles = 100;
	@Option(required=false) public static int maxImplicitParticles = 1_000;
	@Option(required=false) public static double lambda = 10.0;
	@Option(required=false) public static int maxEMIter = 10;
	@Option(required=false) public static boolean parallelize = true;
	@Option(required=false) public static int maxLBFGSIter = 100;
	@Option(required=false) Random random = new Random(123);

	@Option(required=false) public static double tol = 1e-10;
	@Option(required=false) public static boolean exactSampling = true;
	@Option(required=false) public static boolean sequentialMatching = true;

	@Option public static Random rand = new Random(1);
	@Option public static String dataDirectory = "data/simmatching/";
	@Option public static String inputFileName = "enhanced_matching_segmented";

	@Option public static String outputPath = "output/knot-matching/simmatching/";
	@Option public static String resultsFile = "simmatching_regularization_results.csv";
	@Option public static String detailsFile = "simmatching_regularization_details.csv";
	@Option public static String paramFile = "simmatching_regularization_param.csv";

	public static String [] BOARDS = {};
	public static int NUM_BOARDS = 100;
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
//			List<File> dirs = BriefFiles.ls(new File(dataDirectory));
//			for (int i = 0; i < dirs.size(); i++)
//			{
//				String board = dirs.get(i).getName();
//				if (board.charAt(0) == '.') continue;
//				if (board.indexOf("segmented") >= 0)
//					boards.add(dataDirectory + "" + board);
//			}
			for (int n = 1; n <= NUM_BOARDS; n++)
			{
				String board = dataDirectory + "/" + inputFileName + n + ".csv"; 
				boards.add(board);
			}

			BOARDS = boards.toArray(new String[boards.size()]);
		}
	}


	@Override
	public void run()
	{
		SupervisedLearning.parallelize = parallelize;
		SupervisedLearning.numLBFGSIterations = maxLBFGSIter;

		// read the data
		List<List<Segment>> instances = KnotExpUtils.readSegmentedBoard("", BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = RealDataTrainingMCEM.unpack(instances);

		//DecisionModel<String, EllipticalKnot> decisionModel = new KnotPairwiseMatchingDecisionModel();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe);

		// compute the features on each of these data points and standardization
		List<SummaryStatistics> standardizations = new ArrayList<>(fe.dim());
		for (int p = 0; p < fe.dim(); p++) standardizations.add(new SummaryStatistics());
		
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> trainingFold = new ArrayList<>();
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> testingFold = new ArrayList<>();
		for (int i = 0; i < data.size(); i++)
		{
			if (i < 50) {
				trainingFold.add(data.get(i));
			} else {
				testingFold.add(data.get(i));
			}
		}
		
		if (fold == 1)
		{
			List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> temp = trainingFold;
			trainingFold = testingFold;
			testingFold = temp;
		}
		
		for (List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> datum : trainingFold)
		{
			for (int j = 0; j < datum.size(); j++)
			{
				for (Set<EllipticalKnot> e : datum.get(j).getFirst())
				{
					Counter<String> phi = fe.extractFeatures(e, datum.get(j).getFirst());
					for (String f : phi)
					{
						int idx = command.getIndexer().o2i(f);
						standardizations.get(idx).addValue(phi.getCount(f));						
					}
				}
			}

		}
		
		Counter<String> mean = new Counter<>();
		Counter<String> sd = new Counter<>();
		for (int i = 0; i < standardizations.size(); i++)
		{
			String f = command.getIndexer().i2o(i);
			mean.setCount(f, standardizations.get(i).getMean());
			sd.setCount(f, standardizations.get(i).getStandardDeviation());
		}
		// this is to set the standardization to be used
		fe.setStandardization(mean, sd);

		// 1. train the model parameters
		// 2. make a deterministic pass on the segments to match any obvious cases
		// 3. run SMC on any remaining segments

		double [][] performance = new double[50][6];
		for (int j = 0; j < fe.dim(); j++)
		{
			System.out.println("feature " + j + ": " + command.getIndexer().i2o(j) + ", " + standardizations.get(j).getMean() + ", " + standardizations.get(j).getStandardDeviation());
		}

		// estimate the parameters
		SupervisedLearning<String, EllipticalKnot> sl = new SupervisedLearning<>();
		double [] initial = new double[fe.dim()];
		System.out.println("Data split: " + trainingFold.size() + ", " + testingFold.size());
		Pair<Double, double[]> ret = sl.MAPviaMCEM(random, fold, command, KnotExpUtils.pack(trainingFold), maxEMIter, numConcreateParticles, maxImplicitParticles, lambda, initial, tol, false, useSPF);
		System.out.println(ret.getFirst());
		for (double w : ret.getSecond())
			System.out.print(w + ", ");
		System.out.println();

		for (int i = 0; i < 50; i++)
		{
			System.out.println("Testing instance " + i + ": " + BOARDS[50+i]);
			
			List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> datum = testingFold.get(i);

			// evaluate the left out sample
			command.updateModelParameters(ret.getSecond()); // update the params
			// compute the total accuracy for this board
			for (int j = 0; j < datum.size(); j++)
			{
				GraphMatchingState<String, EllipticalKnot> initialState = GraphMatchingState.getInitialState(datum.get(j).getSecond());
				GenericMatchingLatentSimulator<String, EllipticalKnot> transitionDensity = new GenericMatchingLatentSimulator<>(command, initialState, false, true);
				ExactProposalObservationDensity<String, EllipticalKnot> observationDensity = new ExactProposalObservationDensity<>(command);
				List<Object> emissions = new ArrayList<>();
				emissions.addAll(datum.get(j).getSecond());

				SequentialGraphMatchingSampler<String, EllipticalKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, useSPF);
				long start = System.currentTimeMillis();
				smc.sample(random, 500, 1_000, null);
				long end = System.currentTimeMillis();

				// output the prediction accuracy
				MatchingSampleEvaluation<String, EllipticalKnot> eval = MatchingSampleEvaluation.evaluate(smc.getSamples(), datum.get(j).getFirst());
				int numMatchings = datum.get(j).getFirst().size();
				performance[i][0] += eval.avgAccuracy;
				performance[i][1] += eval.bestLogLikMatching.getSecond().getSecond();
				performance[i][2] += eval.avgJaccardIndex;
				performance[i][3] += numMatchings;
				performance[i][4] += emissions.size();
				performance[i][5] += (end - start)/1000.0;
				System.out.println("Eval: " + eval.avgAccuracy + "/" + numMatchings + ", " + eval.bestLogLikMatching.getSecond().getSecond() + "/" + numMatchings + ", " + eval.avgJaccardIndex + "/" + emissions.size());
				
				if (eval.bestLogLikMatching.getSecond().getSecond() < numMatchings) {
					// single sample prediction error: 
					System.out.println("===Incorrect prediction===");
					System.out.println("Truth:");
					List<Set<EllipticalKnot>> truth = datum.get(j).getFirst(); 
					for (Set<EllipticalKnot> edgeTruth : truth) {
						System.out.println(edgeTruth);
					}
					System.out.println("Predicted:");
					List<Set<EllipticalKnot>> predicted = eval.bestLogLikMatching.getFirst().getMatchings();
					for (Set<EllipticalKnot> predictedEdge : predicted) {
						System.out.println(predictedEdge);
					}
				}
			}
			command.updateModelParameters(initial); // re-set the params
			data.add(i, datum);
		}

		File resultsDir = Results.getResultFolder();
		String [] headers = new String[]{"AvgAccuracy", "PredictionAccuracy", "AvgJaccardIndex", "NumMatchings", "NumNodes", "PredictionTimes"};
		OutputHelper.write2DArrayAsCSV(new File(resultsDir, "simulatedDataPerformance" + fold + ".csv"), headers, performance);
	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new SimulatedDataTrainingMCEM());
	}

}
