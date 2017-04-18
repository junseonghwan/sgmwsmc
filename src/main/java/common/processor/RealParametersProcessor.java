package common.processor;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.util.Pair;

import briefj.BriefIO;
import briefj.Indexer;
import briefj.collections.Counter;

/**
 * Parameter vector processor -- represented by Counter<F>
 * 
 * @author Seong-Hwan Jun (s2jun.uw@gmail.com)
 *
 * @param <F>
 */
public class RealParametersProcessor<F> implements Processor<Pair<Double, Counter<F>>>
{
	Indexer<F> indexer;
	Map<Integer, List<Double>> vals = new HashMap<>();
	List<Double> objectiveFunction = new ArrayList<>();
	String processorName;
	String outputPath;
	
	public RealParametersProcessor(String outputPath, String processorName)
	{
		this.outputPath = outputPath;
		this.processorName = processorName;
	}
	
	@Override
  public void process(Pair<Double, Counter<F>> pair) 
	{
		objectiveFunction.add(pair.getFirst());
		Counter<F> t = pair.getSecond();
		if (indexer == null) {
			indexer = new Indexer<>(t.keySet());
		}

		for (F f : t)
		{
			int idx = indexer.o2i(f);
			if (!vals.containsKey(idx))
					vals.put(idx, new ArrayList<>());

			double val = t.getCount(f);
			vals.get(idx).add(val);
		}
  }

	@Override
  public void output() {
		for (int idx : vals.keySet())
		{
			List<Double> params = vals.get(idx);
			F f = indexer.i2o(idx);
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < params.size(); i++)
			{
				sb.append(params.get(i) + "\n");
			}
			PrintWriter writer = BriefIO.output(new File(outputPath + "/" + f.toString() + "_" + processorName));
			writer.println(sb.toString());
			writer.close();
		}
		
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < objectiveFunction.size(); i++)
		{
			sb.append(objectiveFunction.get(i) + "\n");
		}
		PrintWriter writer = BriefIO.output(new File(outputPath + "/objective_" + processorName));
		writer.println(sb.toString());
		writer.close();

  }

}
