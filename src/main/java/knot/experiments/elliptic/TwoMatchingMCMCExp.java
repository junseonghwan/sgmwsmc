package knot.experiments.elliptic;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.EllipticalKnot;
import knot.experiments.rectangular.KnotExpUtils;
import knot.model.features.elliptic.EllipticalKnotFeatureExtractor;

import org.apache.commons.math3.util.Pair;

import common.evaluation.LearningUtils;
import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.Command;
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.GibbsObservationDensity;
import common.smc.components.RandomProposalObservationDensity;
import common.smc.components.SequentialGraphMatchingSampler;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;

/**
 * Implements local move MCMC for two-matching on quadripartite graph for knot matching application.
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 */
public class TwoMatchingMCMCExp implements Runnable {

	@Option public static int thinning = 10;
	@Option public static int burnIn = 100;
	@Option public static int T = 20000;
	@Option public static double lambda = 0.05;
	@Option public static double tol = 1e-6;
	@Option public static boolean sequentialSampling = false;
	@Option public static boolean exactSampling = true;
	@Option public static int [] numConcreteParticles = {20, 40, 60, 80, 100};
	@Option public static int maxVirtualParticles = 10000;
	@Option public static Random random = new Random(1);
	@Option public static boolean append = false;
	@Option public static boolean output = true;
	@Option public static int numReps = 1;

	public static String [] BOARDS = {};

	public static String [] dataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	@Option public static String fileName = "enhanced_matching_segmented";
	@Option public static String mcmcOutputPath = "output/knot-matching/mcmc_output.csv";
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
	
	public static List<Pair<GenericGraphMatchingState<String, EllipticalKnot>, Pair<Integer, Long>>> runMCMC(Random random, int chainLength, Command<String, EllipticalKnot> command, Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>> instance)
	{
		List<Pair<GenericGraphMatchingState<String, EllipticalKnot>, Pair<Integer, Long>>> states = new ArrayList<>();
		// initialize matching randomly -- sample from sequential decision model or sample from SMC uniformly (w/o parameters)
		GraphMatchingState<String, EllipticalKnot> currState = sampleInitialState(random, command, instance.getSecond());
		
		int numAccepted = 0;
		for (int t = 0; t < chainLength; t++)
		{
			/*
			if (t % 100 == 0)
				System.out.println("Iter: " + t);
				*/
			// make a local move
			GraphMatchingState<String, EllipticalKnot> mstar = localMove(random, currState);

			// compute the acceptance ratio
			double a = computeAcceptanceRatio(command, currState, mstar);
			//System.out.println(a);

			if (random.nextDouble() < a) {
				// accept
				currState = mstar;
				numAccepted++;
				
				// compute the accuracy
				//System.out.println("New sample accuracy: " + MatchingSampleEvaluation.computeAccuracy(currState, instance.getFirst()));
			}

			if (t > burnIn && t % thinning == 0)
			{
				states.add(Pair.create(currState, Pair.create(t, System.currentTimeMillis())));
			}
		}

		System.out.println("prob accepted: " + numAccepted/(double)chainLength);

		return states;
	}

	public static double computeAcceptanceRatio(Command<String, EllipticalKnot> command, GraphMatchingState<String, EllipticalKnot> mcurr, GraphMatchingState<String, EllipticalKnot> mstar)
	{
		// check that mstar is in the support set
		if (!inSupport(mstar))
			return 0.0;

		double likStar = Math.exp(GibbsObservationDensity.computeLogLikelihood(command, mstar));
		double likCurr = Math.exp(GibbsObservationDensity.computeLogLikelihood(command, mcurr));
		if (likCurr == 0.0)
			return 1.0;
		return likStar/likCurr;
	}
	
	public static boolean inSupport(GraphMatchingState<String, EllipticalKnot> mstar)
	{
		for (Set<EllipticalKnot> edge : mstar.getMatchings())
		{
			Set<Integer> pidx = new HashSet<>();
			for (EllipticalKnot node : edge)
			{
				if (pidx.contains(node.getPartitionIdx()))
					return false;
				else
					pidx.add(node.getPartitionIdx());
			}
		}

		return true;
	}

	public static GraphMatchingState<String, EllipticalKnot> sampleInitialState(Random random, Command<String, EllipticalKnot> command, List<EllipticalKnot> nodes)
	{
		GraphMatchingState<String, EllipticalKnot> initial = GraphMatchingState.getInitialState(nodes);

		// sample initial state uniformly from SMC
		Counter<String> params = command.getModelParameters();
		command.setModelParameters(command.getFeatureExtractor().getDefaultParameters());
		ObservationDensity<GenericGraphMatchingState<String, EllipticalKnot>, Object> observationDensity = null;
		if (exactSampling)
			observationDensity = new ExactProposalObservationDensity<>(command);
		else
			observationDensity = new RandomProposalObservationDensity<>(command);

		LatentSimulator<GenericGraphMatchingState<String, EllipticalKnot>> transitionDensity = new GenericMatchingLatentSimulator<String, EllipticalKnot>(command, initial, sequentialSampling, exactSampling);

		List<Object> emissions = new ArrayList<>();
		for (int j = 0; j < nodes.size(); j++) emissions.add(null);

		// draw samples using SMC
		SequentialGraphMatchingSampler<String, EllipticalKnot> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
		smc.sample(random, 10, 10, null);
		List<GenericGraphMatchingState<String, EllipticalKnot>> samples = smc.getSamples();
		GraphMatchingState<String, EllipticalKnot> sample = (GraphMatchingState<String, EllipticalKnot>) samples.get(random.nextInt(samples.size()));
		if (!MatchingSampleEvaluation.validateMatchingSamples(samples, nodes)) {
			throw new RuntimeException("Invalid sample found");
		}

		command.setModelParameters(params);
		return sample;
	}

	public static GraphMatchingState<String, EllipticalKnot> localMove(Random random, GraphMatchingState<String, EllipticalKnot> state)
	{
		return state.localMove(random);
	}

	private List<String> runMCMCExp(int nrep, int chainLength, List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances)
	{
		// get the MAP estimate via leave-one-out CV
		DoubletonDecisionModel<String, EllipticalKnot> decisionModel = new DoubletonDecisionModel<>();
		GraphFeatureExtractor<String, EllipticalKnot> fe = new EllipticalKnotFeatureExtractor();

		List<String> lines = new ArrayList<>();
		//double [][] results = new double[instances.size()][3];
		for (int i = 0; i < instances.size(); i++)
		{
			Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>> heldOut = instances.remove(i);
			Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(random, decisionModel, fe, instances, lambda, tol);

			Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe, ret.getSecond());
			long start = System.currentTimeMillis();
			List<Pair<GenericGraphMatchingState<String, EllipticalKnot>, Pair<Integer, Long>>> samples = runMCMC(random, chainLength, command, heldOut);

			// find the matching with highest likelihood
			processSamples(command, samples, heldOut.getFirst(), nrep, i, start, lines);
						
			/*
			results[i][0] = val.getFirst();
			results[i][1] = val.getSecond();
			results[i][2] = time;
			*/

			instances.add(i, heldOut);
		}

		return lines;
	}

	public void processSamples(Command<String, EllipticalKnot> command, List<Pair<GenericGraphMatchingState<String, EllipticalKnot>, Pair<Integer, Long>>> samples, List<Set<EllipticalKnot>> truth, int nrep, int boardIdx, long startTime, List<String> lines)
	{
		List<EllipticalKnot> knots = MatchingSampleEvaluation.getNodes(truth);

		// find the sample with highest likelihood, use that as prediction
		double maxLogLik = Double.NEGATIVE_INFINITY;
		int accuracyAtPrediction = 0;
		//GraphMatchingState<String, EllipticalKnot> prediction = null;
		//List<Pair<Integer, Double>> results = new ArrayList<>();
		double jaccard = 0.0;
		int count = 0;
		for (Pair<GenericGraphMatchingState<String, EllipticalKnot>, Pair<Integer, Long>> pair : samples)
		{
			count += 1;
			GraphMatchingState<String, EllipticalKnot> sample = (GraphMatchingState<String, EllipticalKnot>)pair.getFirst();

			double logLik = GibbsObservationDensity.computeLogLikelihood(command, sample);
			int accuracy = MatchingSampleEvaluation.computeAccuracy(sample, truth);
			jaccard += (MatchingSampleEvaluation.jaccardIndex(knots, truth, sample) / knots.size());
			int iter = pair.getSecond().getFirst();
			double time = (pair.getSecond().getSecond().doubleValue() - startTime)/1000.0;
			if (logLik > maxLogLik)
			{
				maxLogLik = logLik;
				//prediction = sample;
				accuracyAtPrediction = accuracy;
				System.out.println("max log lik: " + maxLogLik);
				System.out.println("accuracy: " + accuracy);
			}
				
			lines.add(boardIdx + ", " + nrep + ", " + iter + ", " + T + ", " + accuracyAtPrediction + ", " + truth.size() + ", " + jaccard + ", " + count + ", " + time);				
		}

		//return Pair.create(MatchingSampleEvaluation.computeAccuracy(prediction, truth), MatchingSampleEvaluation.jaccardIndex(knots, truth, samples));

		//return results;
	}

	
	private void runMCMC(List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances)
	{
		List<String> mcmcOutputLines = new ArrayList<>();
		for (int nrep = 0; nrep < numReps; nrep++)
		{
			mcmcOutputLines.addAll(runMCMCExp(nrep, T, instances));
			/*
			double [][] mcmcRet = runMCMCExp(chainLength, instances);
			for (int i = 0; i < instances.size(); i++)
			{
				String line = (i+1) + ", " + nrep + ", " + chainLength + ", " + mcmcRet[i][0] + ", " + instances.get(i).getFirst().size() + ", " + mcmcRet[i][1] + ", " + mcmcRet[i][2];
				System.out.println(line);
				mcmcOutputLines.add(line);
			}
			*/
		}
		if (output) {
			outputData(mcmcOutputPath, mcmcOutputLines, append);
		}
	}


	@Override
	public void run() {
		List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances = KnotExpUtils.readEllipticalData(fileName, BOARDS, false, 2);

		runMCMC(instances);
	}
	
	public static void outputData(String outputPath, List<String> lines, boolean append)
	{
		PrintWriter writer = null;
		FileOutputStream fstream = null;
		if (append) {
			
			try {
				fstream = new FileOutputStream(new File(outputPath), append);
				writer = new PrintWriter(fstream);
			} catch (Exception ex) {
				
			}
			
		} else {
			writer = BriefIO.output(new File(outputPath));
		}
		
		for (String line : lines)
		{
			writer.println(line);
		}
		writer.close();

		if (fstream != null) {
			try {
				fstream.close();
			} catch (Exception ex) {
				throw new RuntimeException();
			}
		}
	}

	public static void main(String[] args) {
		Mains.instrumentedRun(args, new TwoMatchingMCMCExp());
	}

}
