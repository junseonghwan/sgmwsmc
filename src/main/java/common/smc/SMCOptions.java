package common.smc;

import java.util.Random;

import common.smc.ResamplingScheme;
import briefj.opt.Option;


/**
 * Command line options for SMC
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 */
public class SMCOptions
{
  @Option(gloss = "The ratio under which we should perform resampling.")
  public double essThreshold = 1.0;
  
  @Option(gloss = "Type of resampling to use")
  public ResamplingScheme resamplingScheme = ResamplingScheme.STRATIFIED;

  @Option(gloss = "Seed for the SMC algorithm")
  public Random random = new Random(1);
  
  @Option(gloss = "Number of particles")
  public int nParticles = 100;

  @Option(gloss = "Number of parallel threads")
  public int nThreads = 1;

  @Option
  public String loggerFilePath = null;
}