package registration.experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import briefj.BriefIO;
import briefj.collections.Counter;
import briefj.run.Mains;
import common.graph.BipartiteMatchingState;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.learning.MonteCarloExpectationMaximization;
import common.model.BipartiteDecisionModel;
import common.model.CanonicalFeatureExtractor;
import common.model.Command;
import common.model.DoubletonDecisionModel;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import image.data.ImageNode;
import knot.data.EllipticalKnot;
import knot.data.RectangularKnot;
import knot.model.KnotDoubletonDecisionModel;
import knot.model.features.common.DistanceFeatureExtractor;
import knot.model.features.rectangular.DistanceSizeFeatureExtractor;

public class ImageRegistrationExp implements Runnable
{
	public static String graph1 = "/Users/seonghwanjun/Dropbox/Research/repo/sgmwsmc/pointCloud/normal_1.txt";
	public static String graph2 = "/Users/seonghwanjun/Dropbox/Research/repo/sgmwsmc/pointCloud/normal_2.txt";

	public List<ImageNode> readNodeFeatures(String file, int pidx)
	{
		List<ImageNode> ret = new ArrayList<>();
		int idx = 0;
		// read in points from two graphs
		for (String line : BriefIO.readLines(file)) {
			String [] row = line.split("\t");
			double x = Double.parseDouble(row[0]);
			double y = Double.parseDouble(row[1]);
			double z = Double.parseDouble(row[2]);
			Counter<Integer> features = new Counter<>();
			features.setCount(0, x);
			features.setCount(1, y);
			features.setCount(2, z);
			ImageNode node = new ImageNode(pidx, idx++, features);
			ret.add(node);
		}
		return ret;
	}
	
	@Override
	public void run()
	{
		List<ImageNode> nodes = readNodeFeatures(graph1, 0);
		nodes.addAll(readNodeFeatures(graph2, 1));

		Random random = new Random(1);
		DoubletonDecisionModel<Integer, ImageNode> decisionModel = new DoubletonDecisionModel<>();
		CanonicalFeatureExtractor<Integer, ImageNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));

		Command<Integer, ImageNode> command = new Command<>(decisionModel, fe);

		List<Object> emissions = new ArrayList<>();
		for (int j = 0; j < nodes.size(); j++) emissions.add(null);

		GraphMatchingState<Integer, ImageNode> initialState = GraphMatchingState.getInitialState(nodes);
  		GenericMatchingLatentSimulator<Integer, ImageNode> transitionDensity = new GenericMatchingLatentSimulator<>(command, initialState, false, false);
  		ObservationDensity<GenericGraphMatchingState<Integer, ImageNode>, Object> observationDensity = new ExactProposalObservationDensity<>(command);
  		SequentialGraphMatchingSampler<Integer, ImageNode> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
  		smc.sample(random, 100, 100);

		MonteCarloExpectationMaximization<Integer, ImageNode> mcem = new MonteCarloExpectationMaximization<>();
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new ImageRegistrationExp());
	}

}
