package ranking.model;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.util.Pair;
import org.junit.Assert;

import ranking.data.Document;
import bayonet.math.NumericalUtils;
import blang.variables.RealVector;

/**
 * Implements the bipartite matching model for ranking the documents.
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 * @param <Node>
 */
public class DocumentRankingModel 
{

	/**
	 * Given the correct matching, compute the likelihood
	 * @param params: parameters to be used for the multinomial model
	 * @param matching: rank -> document map
	 * @param documents: list of all documents available
	 * @return
	 */
  public Pair<Double, RealVector> logLikelihood(double [] params, Map<Integer, Document> matching) 
	{
		double logLik = 0.0;
		double [] gradient = new double[params.length];
		Set<Document> uncoveredDocuments = new HashSet<>(matching.values());
		int numDocuments = uncoveredDocuments.size();
		
		int M = matching.size();

		for (int i = 0; i < numDocuments; i++)
		{
			// for each document, \phi_k(di) - sum_{i'} exp(score)*\phi_k/\sum_{i'} exp(score_{i'})
			// note that logNorm = log(sum_{i'} exp(score_{i'})) so take exp gives us what we want
			double [] numerator = new double[params.length];

			Document correctDocument = matching.get(i);
			double logNorm = Double.NEGATIVE_INFINITY;
			for (Document candidateDoc : uncoveredDocuments) 
			{
				// compute the score: exp<phi(document), params)>
				double score = score(candidateDoc, params) * (M - i);
				logNorm = NumericalUtils.logAdd(logNorm, score);
				for (int k = 0; k < params.length; k++)
				{
					numerator[k] += Math.exp(score) * candidateDoc.features().get(k) * (M - i);
				}
			}
			logLik += ((M-i) * score(correctDocument, params) - logNorm);

			double denominator = Math.exp(logNorm);
			for (int k = 0; k < params.length; k++)
			{
				gradient[k] += ((M-i)*correctDocument.features().get(k) - numerator[k]/denominator); 
			}
			
			if (!uncoveredDocuments.remove(correctDocument))
				throw new RuntimeException("Set does not contain the corrrect document. Bug.");
		}
		
		Assert.assertTrue(uncoveredDocuments.size() == 0);

		return Pair.create(logLik, new RealVector(gradient));
	}
	
	public static double score(Document document, double [] params)
	{
		List<Double> features = document.features();
		if (features.size() != params.length) throw new RuntimeException("# features != # params");
		double score = 0.0;
		for (int i = 0; i < features.size(); i++) {
			score += features.get(i) * params[i];
		}
		return score;
	}

}
