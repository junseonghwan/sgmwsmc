package ranking.experiments;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;

import ranking.data.Document;
import ranking.model.DocumentRankingModel;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Pair;

import bayonet.distributions.Multinomial;
import bayonet.opt.DifferentiableFunction;
import bayonet.opt.LBFGSMinimizer;
import blang.variables.RealVector;
import briefj.BriefIO;
import briefj.opt.Option;
import briefj.run.Mains;

public class TD2003 implements Runnable
{
	@Option
	public static String dataLoc = "/Users/seonghwanjun/Dropbox/Research/ranking/TD2003/QueryLevelNorm/";
	@Option
	//public static String outputPath = "/Users/sjun/Dropbox/Research/ranking/TD2003/Scores/";
	public static String outputPath = "/Users/seonghwanjun/Dropbox/Research/ranking/TD2003/Scores/";
	@Option
	public static int [] folds = {5};
	@Option
	public static int k = 16;
	@Option
	public static int B = 2000;
	@Option
	public static int M = 5;
	public static int R = 2;
	public static int numFeatures = 64;
	public static String trainFile = "train.txt";
	public static String testFile = "test.txt";
	public static String validationFile = "vali.txt";

	@Override
	public void run()
	{
		// randomly initialize the param values
		Random random = new Random(20160510);
		for (int fold : folds)
		{
			System.out.println("Running Fold" + fold);
			runExperiment(new Random(random.nextLong()), fold);
			System.out.println("===============");
		}
	}
	
	public void runExperiment(Random random, int fold)
	{
		double [] startingPoint = new double[numFeatures];
		randomInit(new Random(random.nextLong()), startingPoint);
		double [] lambdas = {0.5, 0.75, 1, 1.25, 1.5, 1.75, 2};
		//double [] lambdas = {1};
		double [] bestParams = null;
		double bestNDCG = Double.NEGATIVE_INFINITY;
		// parse the data
		Map<String, List<Document>> data = new HashMap<>();
		parseData(fold, trainFile, data);

		for (double lambda : lambdas)
		{
			System.out.println("lambda=" + lambda);
			double [] params = train(new Random(random.nextLong()), data, startingPoint, lambda, B, M, fold);
			// predict on the validation set
			double eval = test(fold, params, validationFile, false);
			System.out.println("NDCG@" + k + "=" + eval);
			for (int i = 0; i < params.length; i++) {
				System.out.print(params[i] + ", ");
			}
			if (eval > bestNDCG) {
				bestParams = params;
				bestNDCG = eval;
			}
			System.out.println("");
			System.out.println("===========\n"); 
		}

		System.out.println("best param=");
		for (int i = 0; i < bestParams.length; i++) {
			System.out.print(bestParams[i] + ", ");
		}
		System.out.println("===========\n"); 

		// predict on the testing set
		evaluate(fold, bestParams);

	}

	public void evaluate(int fold, double [] params)
	{
		System.out.println("Testing...");
		double eval = test(fold, params, testFile, true);
		System.out.println("===\n" + "Average of NDCG" + k + "=" + eval);
	}

	public double bestVal = Double.POSITIVE_INFINITY;
	
	private void formTrainingData(Random random, String qid, List<Map<Integer, Document>> labelledData, List<List<Document>> docList)
	{
		for (int r = 0; r < R; r++)
		{
			if (docList.get(r).size() == 0) {
				System.out.println("qid=" + qid + " doesn't have any relevant document.");
				return;
			}
		}

		for (int b = 0; b < B; b++)
		{
			List<Document> subSample = new ArrayList<>();

			for (int r = 0; r < R; r++)
			{
				subSample.add(docList.get(r).remove(random.nextInt(docList.get(r).size())));
			}

			// fill up the remaining by sampling documents at random
			int count = M - subSample.size();
			while (count > 0)
			{
				double [] probs = new double[R];
				int N = 0;
				for (int r = 0; r < R; r++) {
					N += docList.get(r).size();
					probs[r] = docList.get(r).size();
				}
				if (N == 0) break;
				for (int r = 0; r < R; r++) {
					probs[r] /= N;
				}
				
				int relClass = Multinomial.sampleMultinomial(random, probs);
				subSample.add(docList.get(relClass).remove(random.nextInt(docList.get(relClass).size())));
				count -= 1;
			}

			Map<Integer, Document> matching = new HashMap<>();
			int rank = 0;
			Collections.sort(subSample);
			double prevRank = Double.POSITIVE_INFINITY;
			for (Document doc : subSample)
			{
				if (doc.rel() > prevRank) throw new RuntimeException();
				prevRank = doc.rel();
				matching.put(rank, doc);
				rank++;
			}

			labelledData.add(matching);

			putDocumentsByRelevance(subSample, docList);
		}
	}
	
	public double [] train(Random random, Map<String, List<Document>> data, double [] startingPoint, double lambda, int B, int M, int fold)
	{
		// get training instances
		List<Map<Integer, Document>> labelledData = new ArrayList<>();

		for (String qid : data.keySet())
		{
			List<Document> documents = data.get(qid);
			List<List<Document>> docList = sortDocumentsByRelevance(documents);

			// formulate a labelled data
			formTrainingData(random, qid, labelledData, docList);
		}

		DocumentRankingModel model = new DocumentRankingModel();

		// optimize the parameters -- L-BFGS
		DifferentiableFunction f = new DifferentiableFunction() {
			
	    	double logLik = 0.0;
			double [] gradient;
			double [] currX;

			@Override
			public double valueAt(double[] point) {
				
				logLik = 0.0;
				gradient = new double[point.length];
				currX = point;
				for (Map<Integer, Document> matching : labelledData)
				{
					Pair<Double, RealVector> ret = model.logLikelihood(point, matching);
					logLik += ret.getFirst().doubleValue();
					for (int k = 0; k < gradient.length; k++)
					{
						gradient[k] += ret.getSecond().getVector()[k];
					}
				}
				
				int N = labelledData.size();
				logLik /= N;
				
				double reg = 0.0;
				for (int k = 0; k < point.length; k++)
				{
					double val = point[k];
					reg += val * val;
					gradient[k] /= N;
					gradient[k] -= lambda * point[k];
					gradient[k] *= -1;
				}
				reg *= lambda/2.0;
				
				double val = -logLik + reg;
				if (val < bestVal)
				{
					bestVal = val;
				}
				return val;
			}
			
			@Override
			public int dimension() {
				return numFeatures;
			}
			
			@Override
			public double[] derivativeAt(double[] x) {
				if (!alreadyComputed(x))
				{
					valueAt(x);
				}
				return gradient;
			}
			
			private boolean alreadyComputed(double [] x)
			{
				if (currX == null) return false;

				boolean alreadyComputed = true;
				for (int k = 0; k < x.length; k++)
				{
					if (currX[k] != x[k]) 
					{
						alreadyComputed = false;
						break;
					}
				}
				return alreadyComputed;
			}
		};
		
		// gradient check:
		/*
		double h = 1e-5;
		double val1 = f.valueAt(startingPoint);
		double [] grad1 = f.derivativeAt(startingPoint);
		double [] grad2 = new double[startingPoint.length];
		for (int k = 0; k < startingPoint.length; k++)
		{
			startingPoint[k] += h;
			double val2 = f.valueAt(startingPoint);
			startingPoint[k] -= h;
			grad2[k] = (val2 - val1)/h;
		}
		double diff = 0.0;
		for (int i = 0; i < grad1.length; i++)
		{
			double val = Math.abs(grad1[i] - grad2[i]);
			System.out.println(grad1[i] + "-" + grad2[i] + "=" + val);
			diff += val;
		}
		diff /= grad2.length;
		System.out.println("gradient check: " + diff);
		if (diff > 0.1)
		{
			throw new RuntimeException();
		}
		*/

		LBFGSMinimizer minimizer = new LBFGSMinimizer();
		double [] param = minimizer.minimize(f, startingPoint, 0.01);
		return param;
	}

	public void putDocumentsByRelevance(List<Document> documents, List<List<Document>> docList)
	{
		for (Document doc : documents)
		{
			docList.get((int)doc.rel()).add(doc);
		}
	}

	public List<List<Document>> sortDocumentsByRelevance(List<Document> documents)
	{
		List<List<Document>> docList = new ArrayList<>();
		for (int r = 0; r < R; r++)
			docList.add(new ArrayList<>());

		for (Document document : documents)
		{
			for (int r = 0; r < R; r++)
			{
				if (document.rel() == r) {
					docList.get(r).add(document);
					continue;
				}
			}
		}

		return docList;
	}

	public double test(int fold, double [] params, String fileName, boolean out)
	{
		Map<String, List<Document>> testingData = new HashMap<>();
		List<String> qIdsInOrder = parseData(fold, fileName, testingData);

		double val = 0.0;

		PrintWriter writer = null;
		if (out)
			writer = BriefIO.output(new File(outputPath + "/Fold" + fold + "/test-scores.txt"));
		for (String qId : qIdsInOrder)
		{
			List<Document> documents = testingData.get(qId);
			List<Pair<Document, Double>> docScorePair = new ArrayList<>();
			// evaluate the score
			for (Document document : documents)
			{
				double score = DocumentRankingModel.score(document, params);
				docScorePair.add(Pair.create(document, score));
				if (out)
					writer.println(score);
			}
			// sort the documents based on the score to obtain the rankings
			Collections.sort(docScorePair, new Comparator<Pair<Document, Double>>() {
				@Override
		        public int compare(Pair<Document, Double> o1, Pair<Document, Double> o2) {
					double val1 = o1.getSecond().doubleValue();
					double val2 = o2.getSecond().doubleValue();
					if (val1 < val2) return 1;
					if (val1 > val2) return -1;
			        return 0;
		        }
			});

			List<Document> predicted = new ArrayList<>();
			docScorePair.forEach(new Consumer<Pair<Document, Double>>() {

				@Override
				public void accept(Pair<Document, Double> t) {
					predicted.add(t.getFirst());
				}

			});
			double [] DCG = DCG(predicted);
			Collections.sort(documents);
			double [] IDCG = DCG(documents);
			double [] NDCG = new double[k];
			if (out)
				System.out.println("qid=" + qId);
			for (int i = 0; i < k; i++)
			{
				if (IDCG[i] != 0)
					NDCG[i] = DCG[i] / IDCG[i]; 
				if (out)
					System.out.println("DCG, IDCG, NDCG@" + (i+1) + "=" + DCG[i] + ", " + IDCG[i] + ", " + NDCG[i]);
				val += NDCG[i];
			}
		}

		val /= k;
		val /= testingData.keySet().size();
		if (out)
			writer.close();
		return val;
	}
	
	public static double scoreMap(double rel)
	{
		if (rel == 2.0) return 3.0;
		else if (rel == 1.0) return 1.0;
		else return 0.0;
	}

	public static double [] DCG(List<Document> documents)
	{
		double [] DCG = new double[k];
		DCG[0] = scoreMap(documents.get(0).rel());
		for (int i = 1; i < k; i++)
		{
			Document doc = documents.get(i);
			double rel = scoreMap(doc.rel());
			if (i < 2) {
				DCG[i] = DCG[i-1] + rel;
			} else {
				DCG[i] = DCG[i-1] + (rel * FastMath.log(2.0)/FastMath.log(i + 1));
			}
		}
		return DCG;
	}

	public static void randomInit(Random random, double [] params) 
	{
		for (int i = 0; i < params.length; i++)
		{
			params[i] = random.nextGaussian();
		}
	}

	public static List<String> parseData(int fold, String filename, Map<String, List<Document>> data)
	{
		List<String> qidsInOrder = new ArrayList<>();
		Set<String> qIds = new HashSet<>();
		for (String line : BriefIO.readLines(new File(dataLoc + "Fold" + fold + "/" + filename)))
		{
			String [] row = line.split("#");
			String [] dat = row[0].split("\\s+");
			double relevance = Double.parseDouble(dat[0]);
			String qId = dat[1].split(":")[1].trim();
			if (!qIds.contains(qId)) {
				qidsInOrder.add(qId);
				qIds.add(qId);
			}
			List<Double> features = new ArrayList<>();
			//System.out.print(relevance + " qid:" + qId + " ");
			for (int i = 2; i < dat.length; i++)
			{
					String [] split = dat[i].split(":");
					//int featureId = Integer.parseInt(split[0]);
					double featureVal = Double.parseDouble(split[1]);
					features.add(featureVal);
					//System.out.print(featureId + ":" + featureVal + " ");
			}

			String docId = row[1].split("=")[1].trim();
			//System.out.println(" docid = " + docId);
			Document doc = new Document(qId, docId, relevance, features);
			if (!data.containsKey(qId)) {
				data.put(qId, new ArrayList<>());
			}
			data.get(qId).add(doc);
		}
		return qidsInOrder;
	}

	public static void main(String[] args) 
	{
		Mains.instrumentedRun(args, new TD2003());
	}

}
