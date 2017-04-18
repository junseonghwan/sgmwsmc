package ranking.data;

import java.util.List;

public class Document implements Comparable<Document>
{
	private List<Double> features;
	private String qId;
	private String docId;
	private double relevance;
	private int rank;

	public Document(String qId, String docId, double relevance, List<Double> features)
	{
		this.qId = qId;
		this.docId = docId;
		this.relevance = relevance;
		this.features = features;
	}
	
	public List<Double> features()
	{
		return features;
	}
	
	public String queryId() { return qId; }
	public String documentId() { return docId; }
	public double rel() { return relevance; }
	public int rank() { return rank; }
	public void setRank(int rank) { this.rank = rank; }

	public boolean equals(Document other) {
		if (this.qId == other.qId && this.docId == other.docId)
			return true;
		return false;
	}

	@Override
  public int compareTo(Document o) 
  {
	// reverse
	if (this.relevance < o.relevance) return 1;
	else if (this.relevance > o.relevance) return -1;
	return 0;
  }
	
	@Override
	public String toString()
	{
		return "(qId, docId, rel)=(" + qId + ", " + docId + ", " + relevance + ")"; 
	}

}
