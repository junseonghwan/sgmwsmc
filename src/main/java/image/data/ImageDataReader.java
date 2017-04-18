package image.data;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.util.Pair;

import briefj.BriefIO;
import briefj.collections.Counter;

public class ImageDataReader 
{
	public static String IMAGE_DIR = "data/house/";
	
	public static List<Pair<List<Set<ImageNode>>, List<ImageNode>>> prepareData(Random random, int k, boolean divisibleByK)
	{
		List<List<ImageNode>> images = new ArrayList<>();
		for (int i = 1; i <= 111; i++)
		{
			String imageNum = i + "";
			while (imageNum.length() < 3) {
				imageNum = "0" + imageNum;
			}
			if ((i-1) % k == 0 && divisibleByK) {
				List<ImageNode> nodes = ImageDataReader.readData(IMAGE_DIR, imageNum);
				images.add(nodes);
			} else if ((i-1) % k != 0 && !divisibleByK) {
				List<ImageNode> nodes = ImageDataReader.readData(IMAGE_DIR, imageNum);
				images.add(nodes);
			}
		}

		List<Pair<List<Set<ImageNode>>, List<ImageNode>>> instances = new ArrayList<>();
		for (int i = 0; i < images.size(); i++)
		{
			List<ImageNode> p1 = images.get(i);
			for (int j = i+1; j < images.size(); j++)
			{
				List<ImageNode> p2 = images.get(j);

				List<Set<ImageNode>> matching = new ArrayList<>();
				for (int l = 0; l < p1.size(); l++)
				{
					Set<ImageNode> e = new HashSet<>();
					e.add(p1.get(l));
					e.add(p2.get(l));
					matching.add(e);
				}
				List<ImageNode> nodes = new ArrayList<>(p1);
				Collections.shuffle(nodes, random);
				nodes.addAll(p2);
				instances.add(Pair.create(matching, nodes));
			}
		}

		return instances;
	}


	public static List<ImageNode> readData(String dataDir, String imageNum)
	{
		int pidx = Integer.parseInt(imageNum);
		List<Counter<Integer>> scfs = new ArrayList<>(); // store shape context features
		List<Counter<Integer>> adjs = new ArrayList<>(); // store adjacency matrix
		
		// read the shape context features
		for (String line : BriefIO.readLines(new File(dataDir + "/house" + imageNum + ".scf")))
		{
			String [] scf = line.split("\\s+");
			Counter<Integer> features = new Counter<>();
			for (int i = 0; i < scf.length; i++)
			{
				features.setCount(i + 1, Integer.parseInt(scf[i]));
			}
			scfs.add(features);
		}

		// read adjacency
		for (String line : BriefIO.readLines(new File(dataDir + "/house" + imageNum + ".adj")))
		{
			String [] adj = line.split("\\s+");
			Counter<Integer> adjFeatures = new Counter<Integer>();
			for (int i = 0; i < adj.length; i++)
			{
				adjFeatures.setCount(i+1, Integer.parseInt(adj[i]));
			}
			adjs.add(adjFeatures);
		}
		
		if (scfs.size() != adjs.size())
			throw new RuntimeException("There is an error in the data. The adjcency and scfs do not have the same length.");

		List<ImageNode> ret = new ArrayList<>();
		for (int idx = 0; idx < scfs.size(); idx++)
		{
			ImageNode node = new ImageNode(pidx, idx+1, scfs.get(idx), adjs.get(idx));
			ret.add(node);
		}
		
		return ret;
	}
	
}
