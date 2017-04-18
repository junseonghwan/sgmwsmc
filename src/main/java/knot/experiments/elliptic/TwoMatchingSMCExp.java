package knot.experiments.elliptic;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.EllipticalKnot;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

import org.apache.commons.math3.util.Pair;

import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.learning.BridgeSamplingLearning;
import common.model.Command;
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.GibbsObservationDensity;
import common.smc.components.SequentialGraphMatchingSampler;
import briefj.BriefFiles;
import briefj.opt.Option;
import briefj.run.Mains;

public class TwoMatchingSMCExp implements Runnable
{
	@Option public static double lambda = 0.05;
	@Option public static double tol = 1e-3;
	@Option public static int maxIter = 100;
	@Option public static boolean sequentialSampling = false;
	@Option public static boolean exactSampling = true;
	@Option public static int [] numConcreteParticles = {1000};
	@Option public static Random random = new Random(1);
	@Option public static boolean append = true;
	@Option public static boolean output = true;
	@Option public static int numReps = 1;
	@Option public static int numTrainingParticles = 10;

	public static String [] BOARDS = {};

	public static String [] dataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	@Option public static String fileName = "enhanced_matching_segmented";
	@Option public static String smcOutputPath = "output/knot-matching/smc_output.csv";

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

	private double [][] runSMCExp(int numTrainingParticles, int numParticles, List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances)
	{
		// get the MAP estimate via leave-one-out CV
		DoubletonDecisionModel<String, EllipticalKnot> decisionModel = new DoubletonDecisionModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe);

		double [][] results = new double[instances.size()][3];
		for (int i = 0; i < instances.size(); i++)
		{
			// leave-one-out cross validation -- remove the i-th instance
			Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>> heldOut = instances.remove(i);
			
			//Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(random, decisionModel, fe, instances, lambda, tol);
			Pair<Double, double []> ret = BridgeSamplingLearning.learnOnEntireBoard(random, command, instances, numTrainingParticles, numTrainingParticles, lambda, maxIter, tol, false, null);
			command.updateModelParameters(ret.getSecond());

			ObservationDensity<GenericGraphMatchingState<String, EllipticalKnot>, Object> observationDensity = new GibbsObservationDensity<>(command);
			List<Object> emissions = new ArrayList<>();
			for (int j = 0; j < heldOut.getSecond().size(); j++) emissions.add(null);

			long start = System.currentTimeMillis();
			LatentSimulator<GenericGraphMatchingState<String, EllipticalKnot>> transitionDensity = new GenericMatchingLatentSimulator<>(command, GraphMatchingState.getInitialState(heldOut.getSecond()), sequentialSampling, exactSampling);
			SequentialGraphMatchingSampler<String, EllipticalKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, false);
			smc.sample(numParticles, numParticles, null);
			long end = System.currentTimeMillis();
			double time = (end - start)/1000.0;
			List<GenericGraphMatchingState<String, EllipticalKnot>> samples = smc.getSamples();

			MatchingSampleEvaluation<String, EllipticalKnot> me = MatchingSampleEvaluation.evaluate(samples, heldOut.getFirst());
			results[i][0] = me.bestLogLikMatching.getSecond().getSecond();
			results[i][1] = me.avgJaccardIndex;
			results[i][2] = time;

			// add the instance back at the i-th location
			instances.add(i, heldOut);
		}
		
		return results;
	}

	private void runSMC(List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances)
	{
		List<String> smcOutputLines = new ArrayList<>();

		for (int nrep = 0; nrep < numReps; nrep++)
		{
			for (int numParticles : numConcreteParticles)
			{
				double [][] smcRet = runSMCExp(numTrainingParticles, numParticles, instances);
				for (int i = 0; i < instances.size(); i++)
				{
					String line = (i+1) + ", " + nrep + ", " + numParticles + ", " + smcRet[i][0] + ", " + instances.get(i).getFirst().size() + ", " + smcRet[i][1] + ", " + smcRet[i][2];
					System.out.println(line);
					smcOutputLines.add(line);
				}
			}
		}

		if (output) {
			TwoMatchingMCMCExp.outputData(smcOutputPath, smcOutputLines, append);
		}
	}

	@Override
	public void run()
	{
		List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances = KnotExpUtils.readEllipticalData(fileName, BOARDS, false, 2);
		runSMC(instances);		
	}

	public static void main(String[] args)
	{
		Mains.instrumentedRun(args, new TwoMatchingSMCExp());
	}

}
