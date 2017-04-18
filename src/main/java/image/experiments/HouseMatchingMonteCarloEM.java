package image.experiments;

import image.data.ImageDataReader;
import image.data.ImageNode;
import image.model.ImageFeatureExtractor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.learning.MonteCarloExpectationMaximization;
import common.model.Command;
import common.model.DecisionModel;
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.RandomProposalObservationDensity;
import briefj.run.Mains;

public class HouseMatchingMonteCarloEM implements Runnable
{
	public static Random random = new Random(12);
	public static final String IMAGE_DIR = "data/house/";
	
	@Override
	public void run()
	{
		// 1. read a pair of images
		// 2. use MC-EM: i) use SMC to draw matching between the two images, ii) find parameters to maximize the likelihood, iii) repeat until some convergence criterion is met
		List<ImageNode> nodes001 = ImageDataReader.readData(IMAGE_DIR, "001");
		List<ImageNode> nodes002 = ImageDataReader.readData(IMAGE_DIR, "111");
		List<ImageNode> nodes = new ArrayList<>(nodes001.size() + nodes002.size());
		nodes.addAll(nodes001);
		nodes.addAll(nodes002);
		GraphMatchingState<String, ImageNode> initial = GraphMatchingState.getInitialState(nodes);
		
		DecisionModel<String, ImageNode> decisionModel = new DoubletonDecisionModel<>();
		GraphFeatureExtractor<String, ImageNode> fe = new ImageFeatureExtractor(nodes001.get(0));
		Command<String, ImageNode> command = new Command<>(decisionModel, fe); 
		GenericMatchingLatentSimulator<String, ImageNode> transitionDensity = new GenericMatchingLatentSimulator<>(command, initial, false, true);
		ObservationDensity<GenericGraphMatchingState<String, ImageNode>, Object> observationDensity = new RandomProposalObservationDensity<>(command);
		
		MonteCarloExpectationMaximization<String, ImageNode> mcem = new MonteCarloExpectationMaximization<>();
		//mcem.execute(random, decisionModel, fe, transitionDensity, observationDensity, nodes, initial);
	}
	
	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new HouseMatchingMonteCarloEM());
	}

}
