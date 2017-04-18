package knot.experiments.rectangular;

import briefj.run.Mains;

public class SimulatedDataExperiments implements Runnable
{
	@Override
	public void run()
	{
		// generate the nodes
		// generate the matching using some sequence
		// 1. estimate the parameters using the correct sequence
		// 2. estimate the parameters using incorrect sequence
		// 3. estimate the parameters using randomized node visitation (this will require most work to get it to work)
		// evaluate using MSE by repeating the experiment many times.
	}

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new SimulatedDataExperiments());
	}

}
