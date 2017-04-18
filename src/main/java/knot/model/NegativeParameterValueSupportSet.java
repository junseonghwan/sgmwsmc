package knot.model;

import java.util.Random;

import bayonet.distributions.Uniform;
import knot.data.RectangularKnot;
import common.learning.SupportSet;
import common.model.Command;

public class NegativeParameterValueSupportSet implements SupportSet<double []>
{
	private Command<String, RectangularKnot> command;

	public NegativeParameterValueSupportSet(Command<String, RectangularKnot> command) 
	{
		this.command = command;
  }

	@Override
	public boolean inSupport(double [] t) {
		for (Double w : t)
		{
			if (w >= 0.0)
				return false;
		}
		int idx1 = command.getIndexer().o2i("TWO_DISTANCE_1");
		int idx2 = command.getIndexer().o2i("TWO_DISTANCE_2");
		if (t[idx1] < t[idx2])
			return false;
		return true;
	}

	@Override
  public void initParam(Random random, double[] w) {
		for (int i = 0; i < w.length; i++) {
			w[i] = -random.nextDouble()*3;
		}
		int idx1 = command.getIndexer().o2i("TWO_DISTANCE_1");
		int idx2 = command.getIndexer().o2i("TWO_DISTANCE_2");
		w[idx1] = Uniform.generate(random, w[idx2], 0.0);
		System.out.println(w[idx1] + ",  " + w[idx2]);
  }

}
