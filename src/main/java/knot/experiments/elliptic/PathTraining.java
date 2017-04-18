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

import org.apache.commons.math3.util.Pair;

import common.learning.BridgeSamplingLearning;
import common.model.Command;
import common.model.DecisionModel;
import common.model.PairwiseMatchingModel;
import briefj.BriefFiles;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class PathTraining implements Runnable
{
	@Option public static int numParticles = 10;
	@Option public static int maxIter = 200;
	
	@Option public static double lambda = 1.0;
	@Option public static double tol = 1e-6;
	@Option Random rand = new Random(1);
	@Option public static boolean exactSampling = true;
	@Option public static boolean sequentialMatching = false;
	public static String [] BOARDS = {};

	public static String [] dataDirectories = {"data/21Oct2015/", "data/16Mar2016/"};
	@Option public static String fileName = "enhanced_matching_segmented.csv";
	public static String outputPath = "output/knot-matching/";
	public static String outputParamTrajectoryPath = outputPath + "realParamTrejectory.csv";

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
		System.out.println(BriefFiles.currentDirectory().getAbsolutePath());
		List<List<Segment>> segments = KnotExpUtils.readSegmentedBoard(fileName, BOARDS, false);
		List<List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>>> data = unpack(segments, false);
		List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>> instances = new ArrayList<>();
		for (List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>> datum : data)
		{
			instances.addAll(datum);
		}

		EllipticalKnotFeatureExtractor fe = new EllipticalKnotFeatureExtractor();
		DecisionModel<String, EllipticalKnot> decisionModel = new PairwiseMatchingModel<>();
		Command<String, EllipticalKnot> command = new Command<>(decisionModel, fe);

		List<String> lines = new ArrayList<>();

		BridgeSamplingLearning.learn(rand, command, instances, numParticles, numParticles, lambda, maxIter, tol, false, lines);
		
		if (outputParamTrajectoryPath != null) 
		{
			PrintWriter writer = BriefIO.output(new File(outputParamTrajectoryPath));
			for (String line : lines)
			{
				writer.println(line);
			}
			writer.close();
		}
	}

	public static List<List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>>> unpack(List<List<Segment>> instances, boolean unpackAll)
	{
		List<List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>>> data = new ArrayList<>();
		for (List<Segment> instance : instances)
		{
			List<Pair<List<EllipticalKnot>, List<Set<EllipticalKnot>>>> datum = new ArrayList<>(); 
			for (Segment segment : instance)
			{
				if (!unpackAll && segment.knots.size() < 3)
						continue;
				
				datum.add(Pair.create(segment.knots, new ArrayList<>(segment.label2Edge.values())));					

			}
			data.add(datum);
		}
		return data;
	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new PathTraining());
	}

}
