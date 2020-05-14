package knot.experiments.elliptic;

import java.io.File;
import java.io.PrintWriter;
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

import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;

public class TrainAndPredict  implements Runnable 
{
	@Option(required=false) public static boolean useSPF = true;
	@Option(required=false) public static int targetESS = 100;
	@Option(required=false) public static int numConcreteParticles = 100;
	@Option(required=false) public static int maxImplicitParticles = 1_000;
	@Option(required=false) public static double lambda = 10.0;
	@Option(required=false) public static int maxEMIter = 10;
	@Option(required=false) public static boolean parallelize = true;
	@Option(required=false) public static int maxLBFGSIter = 100;
	@Option(required=false) Random random = new Random(123);

	@Option(required=false) public static double tol = 1e-10;
	@Option(required=false) public static boolean exactSampling = true;
	@Option(required=false) public static boolean sequentialMatching = true;
	public static String [] BOARDS = {};
	public static String [] TEST_BOARDS = {};

	public static String [] dataDirectories = {"data/training10/"};
	public static String [] testDataDirectories = {"data/testing10/"};
	public static String outputDir = "data/output10/";
	//@Option public static String fileName = "enhanced_matching_segmented.csv";

	static {
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
		
		List<String> testBoards = new ArrayList<>();
		for (String dataDirectory : testDataDirectories)
		{
			List<File> dirs = BriefFiles.ls(new File(dataDirectory));
			for (int i = 0; i < dirs.size(); i++)
			{
				String board = dirs.get(i).getName();
				if (board.charAt(0) == '.') continue;
				testBoards.add(dataDirectory + "" + board);
			}
		}
		TEST_BOARDS = testBoards.toArray(new String[testBoards.size()]);
	}

	@Override
	public void run() 
	{
		SupervisedLearning.parallelize = parallelize;
		SupervisedLearning.numLBFGSIterations = maxLBFGSIter;
		
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		//List<List<Segment>> instances = KnotExpUtils.readSegmentedBoard(fileName, BOARDS, false);
		List<List<Segment>> trainingInstances = KnotExpUtils.readTestBoards(BOARDS, false);
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> trainingData = unpack(trainingInstances);
		
		List<List<Segment>> testInstances = KnotExpUtils.readTestBoards(TEST_BOARDS, false);
		System.out.println(testInstances.size());
		
		//DecisionModel<String, EllipticalKnot> decisionModel = new KnotPairwiseMatchingDecisionModel();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe);

		// compute the features on each of these data points and standardization
		List<SummaryStatistics> standardizations = new ArrayList<>(fe.dim());
		for (int p = 0; p < fe.dim(); p++) standardizations.add(new SummaryStatistics());
		for (int i = 0; i < trainingData.size(); i++)
		{
			for (int j = 0; j < trainingData.get(i).size(); j++)
			{
				for (Set<EllipticalKnot> e : trainingData.get(i).get(j).getFirst())
				{
					Counter<String> phi = fe.extractFeatures(e, trainingData.get(i).get(j).getFirst());
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
		// 2. run SMC on test data

		for (int j = 0; j < fe.dim(); j++)
		{
			System.out.println("feature " + j + ": " + command.getIndexer().i2o(j) + ", " + standardizations.get(j).getMean() + ", " + standardizations.get(j).getStandardDeviation());
		}

		SupervisedLearning<String, EllipticalKnot> sl = new SupervisedLearning<>();
		double [] initial = new double[fe.dim()];
		Pair<Double, double[]> ret = sl.MAPviaMCEM(random, 0, command, KnotExpUtils.pack(trainingData), maxEMIter, numConcreteParticles, maxImplicitParticles, lambda, initial, tol, false, useSPF);
		System.out.println(ret.getFirst());
		for (double w : ret.getSecond())
			System.out.print(w + ", ");
		System.out.println();

		// evaluate the left out sample
		command.updateModelParameters(ret.getSecond()); // update the params

		// run prediction on the test data
		for (int j = 0; j < testInstances.size(); j++)
		{
			Segment segment = testInstances.get(j).get(0); // the test data are not segmented.
			GraphMatchingState<String, EllipticalKnot> initialState = GraphMatchingState.getInitialState(segment.knots);
			GenericMatchingLatentSimulator<String, EllipticalKnot> transitionDensity = new GenericMatchingLatentSimulator<>(command, initialState, false, true);
			ExactProposalObservationDensity<String, EllipticalKnot> observationDensity = new ExactProposalObservationDensity<>(command);
			List<Object> emissions = new ArrayList<>();
			emissions.addAll(segment.knots);

			SequentialGraphMatchingSampler<String, EllipticalKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, useSPF);
			long start = System.currentTimeMillis();
			//smc.sample(random, targetESS, 10_000, null);
			smc.sample(random, targetESS, 1000, null);
			long end = System.currentTimeMillis();
			double runTime = (end - start)/1000.0;
			System.out.println("Run time: " + runTime);

			List<GenericGraphMatchingState<String, EllipticalKnot>> samples = smc.getSamples();
			double bestLogDensity = Double.NEGATIVE_INFINITY;
			GenericGraphMatchingState<String, EllipticalKnot> bestSample = samples.get(0);
			for (GenericGraphMatchingState<String, EllipticalKnot> sample : samples)
			{
				if (sample.getLogDensity() > bestLogDensity) {
					bestLogDensity = sample.getLogDensity();
					bestSample = sample;
				}
			}

			// output a CSV file
			String fileName = TEST_BOARDS[j].split("/")[2];
			String outputFileName = outputDir + "/" + fileName;
			PrintWriter writer = BriefIO.output(new File(outputFileName));
			int idx = 0;
			for (Set<EllipticalKnot> matching : bestSample.getMatchings()) {
				for (EllipticalKnot knot : matching) {
					writer.println(knot.getPartitionIdx() + "," + knot.getIdx() + "," + idx);
				}
				idx++;
			}
			writer.close();
		}
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

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new TrainAndPredict());
	}
}
