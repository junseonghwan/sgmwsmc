package tests;

import java.util.Collections;
import java.util.List;

import image.data.ImageDataReader;
import image.data.ImageNode;
import image.model.ImageFeatureExtractor;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.Sets;

import briefj.collections.Counter;
import common.graph.BipartiteMatchingState;

public class ImageFeatureExtractorTest 
{
	@Test
	public void test1()
	{
		// load the house data
		List<ImageNode> i1 = ImageDataReader.readData(ImageDataReader.IMAGE_DIR, "001");
		List<ImageNode> i2 = ImageDataReader.readData(ImageDataReader.IMAGE_DIR, "002");
		BipartiteMatchingState<String, ImageNode> matchingState = BipartiteMatchingState.getInitial(i1, i2);
		ImageFeatureExtractor fe = new ImageFeatureExtractor(i1.get(0));

		// compute the feature manually and compare to the computation using the feature extractor

		// correctly match a pair of nodes
		Counter<String> features = fe.extractFeatures(i1.get(0), Sets.newHashSet(i2.get(0)), matchingState);
		Assert.assertTrue(features.getCount("adj") == 0.0);
		
		matchingState.move(0, 0);
		
		// correctly match a pair of nodes
		features = fe.extractFeatures(i1.get(1), Sets.newHashSet(i2.get(1)), matchingState);
		Assert.assertTrue(features.getCount("adj") == 0.0);
	}
	
	@Test
	public void test2()
	{
		// load the house data
		List<ImageNode> i1 = ImageDataReader.readData(ImageDataReader.IMAGE_DIR, "001");
		List<ImageNode> i2 = ImageDataReader.readData(ImageDataReader.IMAGE_DIR, "002");
		BipartiteMatchingState<String, ImageNode> matchingState = BipartiteMatchingState.getInitial(i1, i2);
		ImageFeatureExtractor fe = new ImageFeatureExtractor(i1.get(0));

		// incorrectly match pair of nodes
		int idx1 = 0, idx2 = 1;
		Counter<String> features = fe.extractFeatures(i1.get(idx1), Sets.newHashSet(i2.get(idx2)), matchingState);
		Assert.assertTrue(features.getCount("adj") == 0.0);
		
		matchingState.move(idx1, idx2);
		
		// incorrectly match another pair of nodes
		idx1 = 1; idx2 = 0;
		features = fe.extractFeatures(i1.get(idx1), Sets.newHashSet(i2.get(idx2)), matchingState);
		
		Assert.assertTrue(features.getCount("adj") == 0.0);

	}

	@Test
	public void test3()
	{
		// load the house data
		List<ImageNode> i1 = ImageDataReader.readData(ImageDataReader.IMAGE_DIR, "001");
		List<ImageNode> i2 = ImageDataReader.readData(ImageDataReader.IMAGE_DIR, "111");
		BipartiteMatchingState<String, ImageNode> matchingState = BipartiteMatchingState.getInitial(i1, i2);
		ImageFeatureExtractor fe = new ImageFeatureExtractor(i1.get(0));

		// randomize the nodes
		Collections.shuffle(i1);
		for (int i = 0; i < i1.size(); i++)
		{
			ImageNode node1 = i1.get(i);
			ImageNode node2 = i2.get(node1.getIdx()-1);
			Counter<String> features = fe.extractFeatures(node1, Sets.newHashSet(node2), matchingState);
			System.out.println(node1 + ", " + node2);
			System.out.println(features.getCount("adj"));
			matchingState.move(node1, node2);
		}

	}

}
