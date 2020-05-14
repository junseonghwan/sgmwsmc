package tests;

import briefj.run.Mains;

public class TestMonteCarloEStep implements Runnable {

	@Override
	public void run() {
		// simulate the data
		// for each replication, draw (\sigma^i, d_{\sigma^i}) ~ SeqDec(. | \theta) for i = 1, ..., I
		// approximate the desired expectation (complete log likelihood) and compute the variance
		// check the variance for each replication
		
	}

	public static void main(String[] args) {
		Mains.instrumentedRun(args, new TestMonteCarloEStep());
	}

}
