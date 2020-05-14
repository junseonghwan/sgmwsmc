package knot.experiments.rectangular;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

import knot.data.EllipticalKnot;
import knot.data.Knot;
import knot.data.KnotDataReader;
import knot.data.KnotDataReader.Segment;
import knot.data.RectangularKnot;

import org.apache.commons.math3.util.Pair;

import briefj.BriefIO;
import briefj.collections.Counter;
import common.evaluation.LearningUtils;
import common.evaluation.MatchingSampleEvaluation;
import common.experiments.overcounting.ExactProposalObservationDensityWithoutOvercountingCorrection;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.graph.GraphNode;
import common.learning.SupervisedLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.LatentSimulator;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.RandomProposalObservationDensity;
import common.smc.components.SequentialGraphMatchingSampler;

public class KnotExpUtils 
{
	public static final String DEFAULT_DATA_DIR = "data/16Oct2015/";
	public static int numConcreteParticles = 1000;
	public static int maxNumVirtualParticles = 10000;
	public static String outputPath = "output/knot-matching/";
	public static double lambda = 1.0; 
	public static double tol = 1e-6; 
	public static boolean exactSampling = true;
	public static boolean sequentialMatching= true;
	
	public static List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> prepareData(String dataDir, int [] lumbers, boolean reverseSequence)
	{
		List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances = new ArrayList<>();
		for (int lumber : lumbers)
		{
			String dataPath = dataDir + "Board " + lumber + "/labelledMatching.csv";
			instances.add(KnotDataReader.readRectangularKnots(dataPath, reverseSequence));
		}

		return instances;
	}
	
	public static List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> readEllipticalData(String dataDir, int [] lumbers, boolean reverseSequence)
	{
		List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances = new ArrayList<>();
		for (int lumber : lumbers)
		{
			//String dataPath = dataDir + "Board " + lumber + "/knotdetection/matching/enhanced_matching.csv";
			String dataPath = dataDir + "Board " + lumber + "/enhanced_matching.csv";
			instances.add(KnotDataReader.readEllipticalKnots(dataPath, reverseSequence));
		}

		return instances;
	}

	public static List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> readEllipticalData(String filename, String [] boards, boolean reverseSequence, int maxMatchingSize)
	{
		List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances = new ArrayList<>();
		for (String board : boards)
		{
			//String dataPath = dataDir + "Board " + lumber + "/knotdetection/matching/enhanced_matching.csv";
			//String dataPath = dataDir + "Board " + lumber + "/enhanced_matching.csv";
			String dataPath = board + "/enhanced_matching.csv";

			if (maxMatchingSize == 2) {
				instances.add(KnotDataReader.readEllipticalKnotsTwoMatchings(dataPath, reverseSequence));
			}
		}

		return instances;
	}

	public static List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> readGeneratedEllipticalData(String dataDir, int [] lumbers, boolean reverseSequence)
	{
		List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> instances = new ArrayList<>();
		for (int lumber : lumbers)
		{
			String dataPath = dataDir + "enhanced_matching" + lumber + ".csv";
			instances.add(KnotDataReader.readEllipticalKnots(dataPath, reverseSequence));
		}

		return instances;
	}

	public static List<List<Segment>> readSegmentedSimulatedBoard(String dataDir, String filename, String [] lumbers, boolean reverseSequence)
	{
		List<List<Segment>> instances = new ArrayList<>();
		for (String lumber : lumbers)
		{
			String dataPath = dataDir + "enhanced_matching_segmented" + lumber + ".csv";
			instances.add(KnotDataReader.readSegmentedBoard(dataPath));
		}

		return instances;
	}

	public static List<List<Segment>> readSegmentedBoard(String filename, String [] boards, boolean reverseSequence)
	{
		List<List<Segment>> instances = new ArrayList<>();
		int idx = 1;
		for (String board : boards)
		{
			System.out.print(idx++ + ": ");
			
			//String dataPath = dataDir + "Board " + lumber + "/knotdetection/matching/enhanced_matching_segmented.csv";
			String dataPath = board;
			if (filename == null) {
				dataPath = board + "/enhanced_matching_segmented.csv";
			} else {
				if (filename != "") {
					dataPath = board + "/" + filename;					
				}
			}
				
			//System.out.println(dataPath);
			instances.add(KnotDataReader.readSegmentedBoard(dataPath));
		}

		return instances;
	}
	
	public static List<List<Segment>> readTestBoards(String [] boards, boolean reverseSequence)
	{
		List<List<Segment>> instances = new ArrayList<>();
		int idx = 1;
		for (String board : boards)
		{
			System.out.print(idx++ + ": ");
			
			//String dataPath = dataDir + "Board " + lumber + "/knotdetection/matching/enhanced_matching_segmented.csv";
			String dataPath = board;				
			//System.out.println(dataPath);
			instances.add(KnotDataReader.readSegmentedTestBoard(dataPath));
		}

		return instances;
	}


	public static List<MatchingSampleEvaluation<String, RectangularKnot>> evaluate(Random random, 
			DecisionModel<String, RectangularKnot> decisionModel, 
			GraphFeatureExtractor<String, RectangularKnot> fe, 
			List<Pair<List<Set<RectangularKnot>>, List<RectangularKnot>>> instances,
			int [] lumbers)
	{
		return evaluate(random, decisionModel, fe, instances, lumbers, lambda, tol, sequentialMatching, exactSampling, null);
	}

	public static <KnotType extends Knot> List<MatchingSampleEvaluation<String, KnotType>> evaluateOvercounting(Random random, 
			DecisionModel<String, KnotType> decisionModel, 
			GraphFeatureExtractor<String, KnotType> fe, 
			List<Pair<List<Set<KnotType>>, List<KnotType>>> instances,
			int [] lumbers, double lambda, double tol, 
			boolean sequentialMatching, boolean overcountingCorrected, List<String> lines)
	{
		List<MatchingSampleEvaluation<String, KnotType>> evals = new ArrayList<>();

		for (int i = 0; i < instances.size(); i++)
		{
			Pair<List<Set<KnotType>>, List<KnotType>> heldOut = instances.remove(i);
			Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(random, decisionModel, fe, instances, lambda, tol);
			Command<String, KnotType> command = new Command<>(decisionModel, fe, ret.getSecond());

			GraphMatchingState<String, KnotType> initial = GraphMatchingState.getInitialState(heldOut.getSecond());
			LatentSimulator<GenericGraphMatchingState<String, KnotType>> transitionDensity = new GenericMatchingLatentSimulator<String, KnotType>(command, initial, sequentialMatching, exactSampling);
			ObservationDensity<GenericGraphMatchingState<String, KnotType>, Object> observationDensity = null;
			if (overcountingCorrected)
				observationDensity = new ExactProposalObservationDensity<>(command);
			else
				observationDensity  = new ExactProposalObservationDensityWithoutOvercountingCorrection<>(command);

			List<Object> emissions = new ArrayList<>();
			for (int j = 0; j < heldOut.getSecond().size(); j++) emissions.add(null);

			// draw samples using SMC
			System.out.println("Evaluating board " + lumbers[i]);
			long start = System.currentTimeMillis();
			SequentialGraphMatchingSampler<String, KnotType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
			smc.sample(random, numConcreteParticles, maxNumVirtualParticles, null);
			List<GenericGraphMatchingState<String, KnotType>> samples = smc.getSamples();
			MatchingSampleEvaluation<String, KnotType> eval = MatchingSampleEvaluation.evaluate(samples, heldOut.getFirst());
			long end = System.currentTimeMillis();
			evals.add(eval);

			double logLikAtTruth = SupervisedLearning.value(command, ret.getSecond(), heldOut).getFirst();

			System.out.println("===== Evaluation Summary =====");
			System.out.println(heldOut.getFirst());
			System.out.println("Consensus matching accuracy: " + eval.consensusMatching.getSecond() + ", size: " + eval.consensusMatching.getFirst().size());
			//System.out.println(eval.consensusMatching.getFirst().toString());
			System.out.println("Maximum likelihood matching: " + eval.bestLogLikMatching.getSecond().getFirst() + ", accuracy: " + eval.bestLogLikMatching.getSecond().getSecond());
			//System.out.println(eval.bestLogLikMatching.getFirst().toString());
			//System.out.println("Best accuracy: " + eval.bestAccuracyMatching.getSecond());
			System.out.println("Average correctness: " + eval.avgAccuracy);
			System.out.println("Total # of matching: " + heldOut.getFirst().size());
			System.out.println("loglik@truth: " + logLikAtTruth);
			System.out.println("Time (s): " + (end-start)/1000.0);
			System.out.println("===== End Evaluation Summary =====");

			if (lines != null) {
				String line = lumbers[i] + ", " + eval.consensusMatching.getSecond() + ", " + eval.bestLogLikMatching.getSecond().getSecond() + ", " + heldOut.getFirst().size() + ", " + (end - start)/1000.0;
				lines.add(line);
			}

			instances.add(i, heldOut);
		}

		return evals;

	}

	
	public static <KnotType extends Knot> List<MatchingSampleEvaluation<String, KnotType>> evaluate(Random random, 
			DecisionModel<String, KnotType> decisionModel, 
			GraphFeatureExtractor<String, KnotType> fe, 
			List<Pair<List<Set<KnotType>>, List<KnotType>>> instances,
			int [] lumbers, double lambda, double tol, boolean sequentialMatching, boolean exactSampling, List<String> lines)
	{
		List<MatchingSampleEvaluation<String, KnotType>> evals = new ArrayList<>();

		for (int i = 0; i < instances.size(); i++)
		{
			Pair<List<Set<KnotType>>, List<KnotType>> heldOut = instances.remove(i);
			Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(random, decisionModel, fe, instances, lambda, tol);
			Command<String, KnotType> command = new Command<>(decisionModel, fe, ret.getSecond());
			ObservationDensity<GenericGraphMatchingState<String, KnotType>, Object> observationDensity = null;
			if (exactSampling)
				observationDensity = new ExactProposalObservationDensity<>(command);
			else
				observationDensity  = new RandomProposalObservationDensity<>(command);

			GraphMatchingState<String, KnotType> initial = GraphMatchingState.getInitialState(heldOut.getSecond());
			LatentSimulator<GenericGraphMatchingState<String, KnotType>> transitionDensity = new GenericMatchingLatentSimulator<String, KnotType>(command, initial, sequentialMatching, exactSampling);

			List<Object> emissions = new ArrayList<>();
			for (int j = 0; j < heldOut.getSecond().size(); j++) emissions.add(null);

			// draw samples using SMC
			System.out.println("Evaluating board " + lumbers[i]);
			long start = System.currentTimeMillis();
			SequentialGraphMatchingSampler<String, KnotType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
			smc.sample(random, numConcreteParticles, maxNumVirtualParticles, null);
			List<GenericGraphMatchingState<String, KnotType>> samples = smc.getSamples();
			MatchingSampleEvaluation<String, KnotType> eval = MatchingSampleEvaluation.evaluate(samples, heldOut.getFirst());
			long end = System.currentTimeMillis();
			evals.add(eval);

			double logLikAtTruth = SupervisedLearning.value(command, ret.getSecond(), heldOut).getFirst();

			System.out.println("===== Evaluation Summary =====");
			System.out.println(heldOut.getFirst());
			System.out.println("Consensus matching accuracy: " + eval.consensusMatching.getSecond() + ", size: " + eval.consensusMatching.getFirst().size());
			//System.out.println(eval.consensusMatching.getFirst().toString());
			System.out.println("Maximum likelihood matching: " + eval.bestLogLikMatching.getSecond().getFirst() + ", accuracy: " + eval.bestLogLikMatching.getSecond().getSecond());
			//System.out.println(eval.bestLogLikMatching.getFirst().toString());
			//System.out.println("Best accuracy: " + eval.bestAccuracyMatching.getSecond());
			System.out.println("Average correctness: " + eval.avgAccuracy);
			System.out.println("Total # of matching: " + heldOut.getFirst().size());
			System.out.println("loglik@truth: " + logLikAtTruth);
			System.out.println("Time (s): " + (end-start)/1000.0);
			System.out.println("===== End Evaluation Summary =====");

			if (lines != null) {
				String line = lumbers[i] + ", " + eval.consensusMatching.getSecond() + ", " + eval.bestLogLikMatching.getSecond().getSecond() + ", " + heldOut.getFirst().size() + ", " + (end - start)/1000.0;
				lines.add(line);
			}

			instances.add(i, heldOut);
		}

		return evals;
	}

	public static <KnotType extends Knot> List<MatchingSampleEvaluation<String, KnotType>> evaluateSegmentedMatching(Random random, 
			DecisionModel<String, KnotType> decisionModel, 
			GraphFeatureExtractor<String, KnotType> fe, 
			List<List<Pair<List<Set<KnotType>>, List<KnotType>>>> instances, 
			String [] lumbers, double lambda, double tol, boolean sequentialMatching, boolean exactSampling, List<String> lines)
	{
		return evaluateSegmentedMatching(random, decisionModel, fe, instances, lumbers, lambda, tol, sequentialMatching, exactSampling, lines, null);
	}

	public static <KnotType extends Knot> List<MatchingSampleEvaluation<String, KnotType>> evaluateSegmentedMatching(Random random, 
			DecisionModel<String, KnotType> decisionModel, 
			GraphFeatureExtractor<String, KnotType> fe, 
			List<List<Pair<List<Set<KnotType>>, List<KnotType>>>> instances, 
			String [] lumbers, double lambda, double tol, boolean sequentialMatching, boolean exactSampling, List<String> lines, String paramOutputPath)
	{
		List<MatchingSampleEvaluation<String, KnotType>> evals = new ArrayList<>();
		List<String> params = new ArrayList<>();

		for (int i = 0; i < instances.size(); i++)
		{
			List<Pair<List<Set<KnotType>>, List<KnotType>>> heldOut = instances.remove(i);
			List<Pair<List<Set<KnotType>>, List<KnotType>>> trainingInstaces = pack(instances);
			Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(random, decisionModel, fe, trainingInstaces, lambda, tol);
			if (paramOutputPath != null) {
				for (String f : ret.getSecond()) {
					params.add(i + ", " + f + ", "+ ret.getSecond().getCount(f));
				}
			}
			
			Command<String, KnotType> command = new Command<>(decisionModel, fe, ret.getSecond());
			ObservationDensity<GenericGraphMatchingState<String, KnotType>, Object> observationDensity = null;
			if (exactSampling)
				observationDensity = new ExactProposalObservationDensity<>(command);
			else
				observationDensity  = new RandomProposalObservationDensity<>(command);

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
				System.out.println("Evaluating board " + lumbers[i]);
				long start = System.currentTimeMillis();
				SequentialGraphMatchingSampler<String, KnotType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
				smc.sample(random, numConcreteParticles, maxNumVirtualParticles, null);
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
				String line = lumbers[i] + ", " + numCorrect + ", " + bestMatchingCorrect + ", " + numTotal + ", " + jaccardIndex + ", " + numNodes + ", " + numNonTrivialSegments;
				lines.add(line);
			}

			System.out.println("===== Board " + lumbers[i] + " evaluation Summary =====");
			System.out.println(numCorrect + "/" + numTotal);
			System.out.println("===== End evaluation Summary =====");

			instances.add(i, heldOut);
		}
		
		if (paramOutputPath != null)
		{
			PrintWriter writer = BriefIO.output(new File(paramOutputPath));
			for (String paramLine : params)
			{
				writer.println(paramLine);
			}
			writer.close();
		}

		return evals;
	}

	public static <KnotType extends Knot> List<MatchingSampleEvaluation<String, KnotType>> evaluateSMCPerformance(Random random, 
			DecisionModel<String, KnotType> decisionModel, 
			GraphFeatureExtractor<String, KnotType> fe, 
			List<List<Pair<List<Set<KnotType>>, List<KnotType>>>> instances,
			String [] boards, double lambda, double tol, boolean sequentialMatching, boolean exactSampling, List<String> lines)
	{
		List<MatchingSampleEvaluation<String, KnotType>> evals = new ArrayList<>();

		for (int i = 0; i < instances.size(); i++)
		{
			List<Pair<List<Set<KnotType>>, List<KnotType>>> heldOut = instances.remove(i);
			List<Pair<List<Set<KnotType>>, List<KnotType>>> trainingInstaces = pack(instances);
			Pair<Double, Counter<String>> ret = LearningUtils.learnParameters(random, decisionModel, fe, trainingInstaces, lambda, tol);
			Command<String, KnotType> command = new Command<>(decisionModel, fe, ret.getSecond());
			ObservationDensity<GenericGraphMatchingState<String, KnotType>, Object> observationDensity = null;
			if (exactSampling)
				observationDensity = new ExactProposalObservationDensity<>(command);
			else
				observationDensity  = new RandomProposalObservationDensity<>(command);

			int numCorrect = 0;
			int numTotal = 0;
			int numNonTrivialSegments = 0;
			int bestMatchingCorrect = 0;
			for (Pair<List<Set<KnotType>>, List<KnotType>> segment : heldOut)
			{
				GraphMatchingState<String, KnotType> initial = GraphMatchingState.getInitialState(segment.getSecond());
				LatentSimulator<GenericGraphMatchingState<String, KnotType>> transitionDensity = new GenericMatchingLatentSimulator<String, KnotType>(command, initial, sequentialMatching, exactSampling);

				List<Object> emissions = new ArrayList<>();
				for (int j = 0; j < segment.getSecond().size(); j++) emissions.add(null);

				// draw samples using SMC
				System.out.println("Evaluating board " + boards[i]);
				long start = System.currentTimeMillis();
				SequentialGraphMatchingSampler<String, KnotType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, false);
				smc.sample(random, numConcreteParticles, maxNumVirtualParticles, null);
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
				System.out.println("Total # of matching: " + segment.getFirst().size());
				System.out.println("loglik@truth: " + logLikAtTruth);
				System.out.println("Time (s): " + (end-start)/1000.0);
				System.out.println("===== End segment evaluation Summary =====");

				bestMatchingCorrect += eval.bestAccuracyMatching.getSecond();
				numCorrect += eval.bestLogLikMatching.getSecond().getSecond();
				numTotal += segment.getFirst().size();
				numNonTrivialSegments += segment.getFirst().size() > 1 ? 1 : 0;
			}

			if (lines != null) {
				String line = boards[i] + ", " + numCorrect + ", " + bestMatchingCorrect + ", " + numTotal + ", " + numNonTrivialSegments;
				lines.add(line);
			}

			System.out.println("===== Board " + boards[i] + " evaluation Summary =====");
			System.out.println(numCorrect + "/" + numTotal);
			System.out.println("===== End evaluation Summary =====");

			instances.add(i, heldOut);
		}

		return evals;
	}
	
	public static <KnotType extends Knot> List<Pair<List<Set<KnotType>>, List<KnotType>>> pack(List<List<Pair<List<Set<KnotType>>, List<KnotType>>>> instances)
	{
		List<Pair<List<Set<KnotType>>, List<KnotType>>> packedInstances = new ArrayList<>();
		for (List<Pair<List<Set<KnotType>>, List<KnotType>>> instance : instances)
		{
			packedInstances.addAll(instance);
		}
		return packedInstances;
	}

	public static List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> unpack(List<List<Segment>> instances)
	{
		List<List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>>> data = new ArrayList<>();
		for (List<Segment> instance : instances)
		{
			List<Pair<List<Set<EllipticalKnot>>, List<EllipticalKnot>>> datum = new ArrayList<>(); 
			for (Segment segment : instance)
			{
				//if (segment.knots.size() > 3)
					datum.add(Pair.create(new ArrayList<>(segment.label2Edge.values()), segment.knots));
			}
			data.add(datum);
		}
		return data;
	}
	
	public static List<List<List<Segment>>> generateFoldsSegmentedBoardVersion(Random random, int K, List<List<Segment>> data)
	{
		List<List<List<Segment>>> folds = new ArrayList<>();
		double N = data.size();
		int numDataPerGroup = (int)Math.floor(N/K);
		List<Integer> idxs = new ArrayList<>();
		for (int i = 0; i < N; i++) idxs.add(i);
		Collections.shuffle(idxs, random); // shuffle the indices
		for (int k = 0; k < K; k++)
		{
			List<List<Segment>> fold = new ArrayList<>();
			for (int j = 0; j < numDataPerGroup; j++)
				fold.add(data.get(idxs.remove(idxs.size()-1)));
			folds.add(fold);
		}
		
		if (idxs.size() == 1) {
			folds.get(random.nextInt(K)).add(data.get(idxs.remove(0)));
		} else if (idxs.size() > 1)
			throw new RuntimeException("Bug in K-fold allocation code.");
		
		return folds;
	}

	public static <F, KnotType extends GraphNode<?>> List<GenericGraphMatchingState<F, KnotType>> runSMC(Random random, List<KnotType> nodes, DecisionModel<F, KnotType> decisionModel, GraphFeatureExtractor<F, KnotType> fe, double [] w, boolean sequentialMatching, boolean exactSampling, boolean useStreaming, String outputFilePath)
	{
		Command<F, KnotType> command = new Command<>(decisionModel, fe);
		command.updateModelParameters(w);
		
		GraphMatchingState<F, KnotType> initial = GraphMatchingState.getInitialState(nodes);
		LatentSimulator<GenericGraphMatchingState<F, KnotType>> transitionDensity = new GenericMatchingLatentSimulator<F, KnotType>(command, initial, sequentialMatching, exactSampling);
		ObservationDensity<GenericGraphMatchingState<F, KnotType>, Object> observationDensity = null;
		if (exactSampling) {
			observationDensity = new ExactProposalObservationDensity<>(command);
		} else {
			observationDensity = new RandomProposalObservationDensity<>(command);
		}

		List<Object> emissions = new ArrayList<>();
		for (int j = 0; j < nodes.size(); j++) emissions.add(null);

		// draw samples using SMC
		SequentialGraphMatchingSampler<F, KnotType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, useStreaming);
		smc.sample(random, numConcreteParticles, maxNumVirtualParticles, outputFilePath);
		return smc.getSamples();
	}
	
	public static <F, KnotType extends GraphNode<?>> List<GenericGraphMatchingState<F, KnotType>> runSMC(Random random, List<KnotType> nodes, Command<F, KnotType> command, ObservationDensity<GenericGraphMatchingState<F, KnotType>, Object> observationDensity, int numConcreteParticles, int maxVirtualParticles, boolean sequentialMatching, boolean exactSampling, boolean useStreaming, String outputFilePath)
	{
		GraphMatchingState<F, KnotType> initial = GraphMatchingState.getInitialState(nodes);
		LatentSimulator<GenericGraphMatchingState<F, KnotType>> transitionDensity = new GenericMatchingLatentSimulator<F, KnotType>(command, initial, sequentialMatching, exactSampling);

		List<Object> emissions = new ArrayList<>();
		for (int j = 0; j < nodes.size(); j++) emissions.add(null);

		// draw samples using SMC
		SequentialGraphMatchingSampler<F, KnotType> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions, useStreaming);
		smc.sample(random, numConcreteParticles, maxVirtualParticles, outputFilePath);
		return smc.getSamples();
	}

}
