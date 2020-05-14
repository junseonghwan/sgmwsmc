package knot.experiments.simulation;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.EllipticalKnot;
import knot.data.KnotDataReader.Segment;
import knot.experiments.elliptic.RealDataTrainingMCEM;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.Pair;

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

public class SimulatedDataESS implements Runnable 
{
	@Option(required=true) public static boolean skipTwoKnotGraph = true;
	@Option(required=true) public static boolean useSPF = true;
	@Option(required=true) public static int numConcreteParticles = 100;
	@Option(required=true) public static int maxImplicitParticles = 1000;
	@Option(required=true) public static double lambda = 10.0;
	@Option(required=true) public static int maxEMIter = 5;
	@Option(required=true) public static boolean parallelize = true;
	@Option(required=true) public static int maxLBFGSIter = 100;
	@Option(required=false) Random random = new Random(123);

	@Option(required=false) public static double tol = 1e-10;
	@Option(required=false) public static boolean exactSampling = true;
	@Option(required=false) public static boolean sequentialMatching = true;

	@Option public static Random rand = new Random(1);
	@Option public static String [] dataDirectories = {"data/simmatching/"};

	public static final int [] targetESSs = new int[]{100, 200, 400, 600, 800, 1000};

	public static String [] BOARDS = {};

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
					{
						boards.add(dataDirectory + "" + board);
					}
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

		// read the data
		List<List<Segment>> instances = KnotExpUtils.readSegmentedBoard("", BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = RealDataTrainingMCEM.unpack(instances);

		//DecisionModel<String, EllipticalKnot> decisionModel = new KnotPairwiseMatchingDecisionModel();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe);

		List<SummaryStatistics> standardizations = new ArrayList<>(fe.dim());
		for (int p = 0; p < fe.dim(); p++) standardizations.add(new SummaryStatistics());
		for (List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> datum : data)
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
		fe.setStandardization(mean, sd);

		// estimate the parameters using all of the data
		SupervisedLearning<String, EllipticalKnot> sl = new SupervisedLearning<>();
		double [] initial = new double[fe.dim()];
		Pair<Double, double[]> ret = sl.MAPviaMCEM(random, 0, command, KnotExpUtils.pack(data), maxEMIter, numConcreteParticles, maxImplicitParticles, lambda, initial, tol, false, useSPF);
		command.updateModelParameters(ret.getSecond());

		// check the size of the segmented graphs
		List<double []> performance = new ArrayList<>();
		for (List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> board : data)
		{
			for (int targetESS : targetESSs)
			{
				double [] arr = new double[10];
				for (Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>> segment : board)
				{
					if (skipTwoKnotGraph && segment.getSecond().size() == 2) continue;
	
					GraphMatchingState<String, EllipticalKnot> initialState = GraphMatchingState.getInitialState(segment.getSecond());
					GenericMatchingLatentSimulator<String, EllipticalKnot> transitionDensity = new GenericMatchingLatentSimulator<>(command, initialState, false, true);
					ExactProposalObservationDensity<String, EllipticalKnot> observationDensity = new ExactProposalObservationDensity<>(command);
					List<Object> emissions = new ArrayList<>();
					emissions.addAll(segment.getSecond());
	
					SequentialGraphMatchingSampler<String, EllipticalKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, useSPF);
					long start = System.currentTimeMillis();
					smc.sample(random, targetESS, 10_000, null);
					MatchingSampleEvaluation<String, EllipticalKnot> eval = MatchingSampleEvaluation.evaluate(smc.getSamples(), segment.getFirst());
					long end = System.currentTimeMillis();
					
					int numMatchings = segment.getFirst().size();
	
					arr[0] += eval.avgAccuracy;
					arr[1] += eval.bestLogLikMatching.getSecond().getSecond();
					arr[2] += eval.avgJaccardIndex;
					arr[3] += numMatchings;
					arr[4] += emissions.size();
					arr[5] += (end - start)/1000.0;
					arr[6] += eval.bestLogLik;
					arr[7] += eval.logLiks.getMean();
					arr[8] += eval.logLiks.getStandardDeviation();

					System.out.println("Eval: " + eval.avgAccuracy + "/" + numMatchings + ", " + eval.bestLogLikMatching.getSecond().getSecond() + "/" + numMatchings + ", " + eval.avgJaccardIndex + "/" + emissions.size());
				}
				arr[9] = targetESS;
				performance.add(arr);
			}
		}
		
		// output performance
		File resultsDir = Results.getResultFolder();
		String [] header = new String[]{"AvgAccuracy", "PredictionAccuracy", "AvgJaccard", "NumMatchings", "NumKnots", "Time", "TargetESS"};
		OutputHelper.writeLinesOfDoubleArr(new File(resultsDir, "performances.csv"),  header, performance); 
	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new SimulatedDataESS());
	}

}
