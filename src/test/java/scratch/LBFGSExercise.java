package scratch;

import java.util.Random;

import bayonet.distributions.Normal;
import bayonet.opt.DifferentiableFunction;
import bayonet.opt.LBFGSMinimizer;
import briefj.run.Mains;

public class LBFGSExercise implements Runnable
{
	
	public void run() {
		// let's do an optimization of a simple linear regression (y = a + bx + e)
		Random random = new Random(1);
		int N = 100;
		
		double [] param = new double[]{1.0, -5.0};
		double [] x = new double[N];
		double [] y = new double[N];
		for (int i = 0; i < N; i++) {
			x[i] = random.nextGaussian()*5 + 10;
			y[i] = param[0] + param[1]*x[i] + random.nextGaussian();
		}
		
		DifferentiableFunction func = new DifferentiableFunction() {
			
			@Override
			public double valueAt(double[] param) {
				// compute the log-likelihood
				double logLik = 0.0;
				for (int i = 0; i < N; i++) {
					logLik += Normal.logDensity(y[i], param[0] + param[1]*x[i], 1.0);
				}
				return -logLik;
			}
			
			@Override
			public int dimension() {
				return 2;
			}
			
			@Override
			public double[] derivativeAt(double[] param) {
				double [] deriv = new double[2];
				for (int i = 0; i < N; i++) {
					deriv[0] += (y[i] - (param[0] + param[1] * x[i]));
					deriv[1] += (y[i] - (param[0] + param[1] * x[i]))*x[i];
				}
				deriv[0] *= -1;
				deriv[1] *= -1;
				
				return deriv;
			}
		};
		
    System.out.println("Optimization using LBFGS");
    LBFGSMinimizer minimizer = new LBFGSMinimizer(100);
    minimizer.verbose = true;
    double [] learnedWeights = new double[2];
    learnedWeights = minimizer.minimize(func, new double[2], 1e-6);
    for (int i = 0; i < learnedWeights.length; i++) {
    	System.out.print(learnedWeights[i] + " ");
    }

	}
	
	public static void main(String [] args) {
		Mains.instrumentedRun(args, new LBFGSExercise());
	}

}
