package validation.smc;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import briefj.BriefIO;
import briefj.collections.Counter;
import briefj.opt.Option;
import briefj.run.Mains;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.model.PairwiseMatchingModel;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import knot.data.KnotDataReader;
import knot.data.RectangularKnot;

public class UniformMatchingOvercountingExperiments implements Runnable
{
	@Option
	public static boolean sequential = false;
	@Option
	public static boolean exactSampling = true;
	@Option
	public static int numPartitions = 4;
	@Option
	public static int numNodesPerPartitions = 3;
	@Option
	public static int numReps = 1;
	@Option
	public static Random random = new Random(20170208);
	@Option
	public static String output_file = "output/overcounting/uniform_" + numPartitions + "_" + numNodesPerPartitions + ".csv";

	public double runExperiments(Random random, int numParticles, boolean overcountingCorrected)
	{
		List<RectangularKnot> knots = KnotDataReader.simulateKnots(random, numPartitions, numNodesPerPartitions);
		GraphFeatureExtractor<String, RectangularKnot> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(knots.get(0));
		Counter<String> params = fe.getDefaultParameters();
		for (String f : params)
		{
			params.setCount(f, 0.0); // uniform so all parameters are set to 0
		}
		DecisionModel<String, RectangularKnot> decisionModel = new PairwiseMatchingModel<>();
		Command<String, RectangularKnot> command = new Command<>(decisionModel, fe, params);

		GraphMatchingState<String, RectangularKnot> initial = GraphMatchingState.getInitialState(knots);
		GenericMatchingLatentSimulator<String, RectangularKnot> transitionDensity = new GenericMatchingLatentSimulator<String, RectangularKnot>(command, initial, sequential, exactSampling);
		ObservationDensity<GenericGraphMatchingState<String, RectangularKnot>, Object> observationDensity = null;
		if (overcountingCorrected)
			observationDensity = new ExactProposalObservationDensity<>(command);
		else
			observationDensity = new ExactProposalObservationDensityWithoutOvercountingCorrection<>(command);
		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < knots.size(); i++) emissions.add(null);

		SequentialGraphMatchingSampler<String, RectangularKnot> sgm = new SequentialGraphMatchingSampler<String, RectangularKnot>(transitionDensity, observationDensity, emissions, true);
		sgm.sample(numParticles, numParticles);

		// check to see if SMC samples contain state and see if the estimate by the SMC is accurate
		Counter<GenericGraphMatchingState<String, RectangularKnot>> population = new Counter<>();
		for (GenericGraphMatchingState<String, RectangularKnot> sample : sgm.getSamples())
		{
			if (!population.containsKey(sample))
				System.out.println(sample);
			population.incrementCount(sample, 1.0);
		}

		System.out.println("num states=" + population.size());
		double expected = 1.0/population.size();
		double err = 0.0;
		for (GenericGraphMatchingState<String, RectangularKnot> state : population)
		{
			double p = (double)population.getCount(state)/numParticles;
			err += Math.pow(p - expected, 2.0);
			System.out.println("prob= " + p);
			System.out.println(state);
		}
		double rmse = Math.sqrt(err/population.size());
		System.out.println("Avg Error: " + rmse);
		return rmse;
	}
	
	@Override
	public void run()
	{
		final int [] numParticles = {100, 200, 400, 800, 1000, 2000, 4000, 8000, 10000, 20000, 40000, 80000, 100000};
		List<String> lines = new ArrayList<>();
		for (int i = 0; i < numParticles.length; i++)
		{
			DescriptiveStatistics corrected = new DescriptiveStatistics();
			DescriptiveStatistics notCorrected = new DescriptiveStatistics();
			for (int n = 0; n < numReps; n++)
			{
				long seed = random.nextLong();
				Random rand = new Random(seed);
				double rmseCorrect = runExperiments(rand, numParticles[i], true);
				double rmseInCorrect = runExperiments(rand, numParticles[i], false);
				System.out.println("RMSE: " + seed + ", " + rmseCorrect + ", " + rmseInCorrect);
				corrected.addValue(rmseCorrect);
				notCorrected.addValue(rmseInCorrect);
			}
			lines.add(numParticles[i] + ", " + "Yes, " + corrected.getMean() + ", " + corrected.getStandardDeviation());
			lines.add(numParticles[i] + ", " + "No, " + notCorrected.getMean() + ", " + notCorrected.getStandardDeviation());
		}

		PrintWriter writer = BriefIO.output(new File(output_file));
		writer.println("N, OvercountingCorrected, RMSE, sd_RMSE");
		for (String line : lines)
		{
			writer.println(line);
		}
		writer.close();
		
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new UniformMatchingOvercountingExperiments());
	}
}
