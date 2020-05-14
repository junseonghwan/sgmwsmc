package knot.experiments.elliptic;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.EllipticalKnot;
import knot.data.KnotDataReader.Segment;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.Pair;

import briefj.BriefFiles;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;
import briefj.run.Results;
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

public class RealDataPredictionTiming implements Runnable 
{
	@Option(required=true) public static boolean useSPF = true;
	@Option(required=true) public static int targetESS = 100;
	@Option(required=true) public static int numConcreteParticles = 100;
	@Option(required=true) public static int maxImplicitParticles = 1_000;
	@Option(required=true) public static double lambda = 10.0;
	@Option(required=true) public static int maxEMIter = 50;
	@Option(required=true) public static boolean parallelize = true;
	@Option(required=true) public static int maxLBFGSIter = 100;
	@Option(required=false) Random random = new Random(123);

	@Option(required=false) public static double tol = 1e-10;
	@Option(required=false) public static boolean exactSampling = true;
	@Option(required=false) public static boolean sequentialMatching = true;
	public static String [] BOARDS = {};

	public static String [] dataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	@Option public static String fileName = "enhanced_matching_segmented.csv";

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
		SupervisedLearning.parallelize = parallelize;
		SupervisedLearning.numLBFGSIterations = maxLBFGSIter;

		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<List<Segment>> instances = KnotExpUtils.readSegmentedBoard(fileName, BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = unpack(instances);
		
		//DecisionModel<String, EllipticalKnot> decisionModel = new KnotPairwiseMatchingDecisionModel();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe);

		// compute the features on each of these data points and standardization
		List<SummaryStatistics> standardizations = new ArrayList<>(fe.dim());
		for (int p = 0; p < fe.dim(); p++) standardizations.add(new SummaryStatistics());
		for (int i = 0; i < data.size(); i++)
		{
			for (int j = 0; j < data.get(i).size(); j++)
			{
				for (Set<EllipticalKnot> e : data.get(i).get(j).getFirst())
				{
					Counter<String> phi = fe.extractFeatures(e, data.get(i).get(j).getFirst());
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
		fe.setStandardization(mean, sd);

		// 1. train the model parameters
		// 2. make a deterministic pass on the segments to match any obvious cases
		// 3. run SMC on any remaining segments

		double [][] performance = new double[data.size()][9];
		for (int j = 0; j < fe.dim(); j++)
		{
			System.out.println("feature " + j + ": " + command.getIndexer().i2o(j) + ", " + standardizations.get(j).getMean() + ", " + standardizations.get(j).getStandardDeviation());
		}

		// train using all of the data
		SupervisedLearning<String, EllipticalKnot> sl = new SupervisedLearning<>();
		double [] initial = new double[fe.dim()];
		Pair<Double, double[]> ret = sl.MAPviaMCEM(random, 0, command, KnotExpUtils.pack(data), maxEMIter, numConcreteParticles, maxImplicitParticles, lambda, initial, tol, false, useSPF);
		System.out.println(ret.getFirst());
		for (double w : ret.getSecond())
			System.out.print(w + ", ");
		System.out.println();
		command.updateModelParameters(ret.getSecond()); // update the params

		for (int i = 0; i < data.size(); i++)
		{
			List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> datum = data.get(i);

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
				smc.sample(random, targetESS, 10_000, null);
				MatchingSampleEvaluation<String, EllipticalKnot> eval = MatchingSampleEvaluation.evaluate(smc.getSamples(), datum.get(j).getFirst());
				long end = System.currentTimeMillis();

				// output the prediction accuracy
				int numMatchings = datum.get(j).getFirst().size();
				performance[i][0] += eval.avgAccuracy;
				performance[i][1] += eval.bestLogLikMatching.getSecond().getSecond();
				performance[i][2] += eval.avgJaccardIndex;
				performance[i][3] += numMatchings;
				performance[i][4] += emissions.size();
				performance[i][5] += (end - start)/1000.0;
				performance[i][6] += eval.bestLogLik;
				performance[i][7] += eval.logLiks.getMean();
				performance[i][8] += eval.logLiks.getStandardDeviation();
				System.out.println("Eval: " + eval.avgAccuracy + "/" + numMatchings + ", " + eval.bestLogLikMatching.getSecond().getSecond() + "/" + numMatchings + ", " + eval.avgJaccardIndex + "/" + emissions.size());
			}
		}

		File resultsDir = Results.getResultFolder();
		String [] headers = new String[]{"AvgAccuracy", "PredictionAccuracy", "AvgJaccardIndex", "NumMatchings", "NumNodes", "PredictionTimes"};
		OutputHelper.write2DArrayAsCSV(new File(resultsDir, "realDataPerformance.csv"), headers, performance);
	}
	
	public static List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> unpack(List<List<Segment>> instances)
	{
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = new ArrayList<>();
		for (List<Segment> instance : instances)
		{
			List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> datum = new ArrayList<>(); 
			for (Segment segment : instance)
			{
				datum.add(Pair.create(new ArrayList<>(segment.label2Edge.values()), segment.knots));
			}
			data.add(datum);
		}
		return data;
	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new RealDataPredictionTiming());
	}

}
