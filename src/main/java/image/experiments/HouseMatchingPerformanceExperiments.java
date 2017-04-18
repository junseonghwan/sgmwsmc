package image.experiments;

import image.data.ImageDataReader;
import image.data.ImageNode;
import image.model.ImageFeatureExtractor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import common.evaluation.MatchingSampleEvaluation;
import common.graph.GenericGraphMatchingState;
import common.graph.GraphMatchingState;
import common.model.Command;
import common.model.DecisionModel;
import common.model.DoubletonDecisionModel;
import common.model.GraphFeatureExtractor;
import common.smc.DiscreteParticleFilter;
import common.smc.StreamingParticleFilter.ObservationDensity;
import common.smc.components.ExactProposalObservationDensity;
import common.smc.components.GenericDiscreteLatentSimulator;
import common.smc.components.GenericMatchingLatentSimulator;
import common.smc.components.SequentialGraphMatchingSampler;
import briefj.collections.Counter;
import briefj.run.Mains;

public class HouseMatchingPerformanceExperiments implements Runnable
{
	public static Random random = new Random(12);
	public static final String IMAGE_DIR = "data/house/";

  @Override
	public void run()
	{
		//double [] w = {0.38220360549308424, 0.0, 0.06741070369032731, 0.0, 0.42218418635439214, 0.5793657949163368, 0.8775216362207414, 0.0, 0.5748040098203717, 0.0, 0.18421140175336442, 0.2083055429176498, 0.7259361094882406, 0.18720310148468558, 0.6655766495780162, 0.2871003089705772, 0.9588771848058002, 0.5742164333225235, 0.7951080669586099, 0.13691110220901603, 0.5376169236636799, 0.16151916134323638, -0.1138700334508131, 0.2503759707230171, 1.0150148352884687, 0.47195562863589796, -0.20419539650542828, 0.018762835458636492, 0.05444088236706155, 0.6622331552253776, 0.9128374660318759, 0.1333674920957525, 0.48454504674894683, 0.6951665338775892, 1.4038452208009473, 0.6699394291522328, 1.3172730923731195, 0.008910682268680225, 0.9943988541588857, 0.17684385169582026, 0.7292609231694166, -0.06230651461076829, 0.09462398117537209, 0.1952982465797781, 0.9208365053110374, 0.913098680710732, 0.6007597217712193, 0.9807621729237181, 1.3125888573440483, 0.9737182854477734, 1.0209235770977632, 0.9169466633151538, 0.875529709770531, 0.43374081727025465, 0.47955483346193756, 0.8222251914897373, -0.08290509666208766, 1.3945542448746, 0.5482061780324562, 0.24560152614504036};
		double [] w = {1.1687421537906029, 0.7983606162786142, 1.0792539677681912, 4.7259809686852385, 0.12920028844909923, 2.451260167464478, 0.11219785160944055, 0.0, 0.0, 0.5485447196644516, 0.03187661134181194, 0.4301401451002841, 5.119261978865087, 0.15831878085236195, 0.9665437170359402, -0.7296672986311913, 0.6421429390586643, -0.06249606423373143, 0.0, -0.4787322217930798, -0.025943188189330862, 0.06193559879567092, 0.07361595864729771, -0.1358893700051512, -0.004216001843933465, 2.6598706383854815, -0.010144303469621509, 1.0041702830702266, 0.8339454759633946, 0.9836169264602198, -0.12393359461565903, 0.21496519051264049, 0.8511838526984469, 0.7789583684970268, 0.3066017408716993, 0.8010215160987884, 0.9427989949519227, -0.36973899087326806, 0.37736371464726415, -0.010755773159882468, 0.0, -0.4868995607724972, 1.6354278681529546, 1.4022337600600419, -0.6972731215335931, 2.4986850845136974, 2.7697771551068837, -0.19777665873719996, -0.3967380408451989, 0.8993042650090122, 0.10039376382100196, 3.7525611353413546, 4.752919025215715, 4.739464042411727, 3.0421323501356925, 4.623431160635732, 1.1022710080629268, 0.9778450531875151, 3.9878108567510013, 0.8786376290607072, 7.689718078526511};
		//double [] w = {0.7749063002826303, 0.461665722208063, 0.8904400873048842, 0.0, 1.2971067559446157, 0.5322119429807135, 0.3653945868196163, 0.18328680532315947, 1.4664874727761552, 0.0, 0.15794614196607343, 1.2465107059739315, 0.16661908364074318, 0.3630421756182321, -0.5541559898866536, 0.9194437346879102, 1.4058022389505131, -0.16342605507997118, 0.49156163663911495, 0.6441409790160193, 0.7146989763145902, 0.8793219363772233, -0.4528950659171299, 0.7024638289088807, 1.201282248207238, 0.545055171274069, -0.10097622221622955, 0.36318238464639735, 0.13586092130673585, -0.44670597693971864, 1.0533282651580071, 0.6307190895341935, 0.45225760592065634, 0.8106651105942739, 0.8082924846649173, 0.03523756520871328, 1.445838338343845, 0.4712918605309991, 1.4956248049394683, 0.1252603902526835, 1.0058025048140682, 0.6281506067590504, 0.02676651631629632, -0.11195335549063612, 1.3895064393397427, 0.36903630142433186, 0.281037001970552, 0.788318078392901, 0.19225743933341152, 0.8750939318992849, 0.40948251074202874, 0.9634514153077331, 0.9266427732187729, 0.326800630174327, 0.5371827715923103, 1.1114999555035578, 0.2120986623311969, 1.450106297073555, 0.28289462412513555, 0.7263329153028729};
		List<ImageNode> nodes001 = ImageDataReader.readData(IMAGE_DIR, "001");
		List<ImageNode> nodes002 = ImageDataReader.readData(IMAGE_DIR, "111");
		List<ImageNode> nodes = new ArrayList<>(nodes001.size() + nodes002.size());
 		nodes.addAll(nodes001);
		nodes.addAll(nodes002);
		GraphMatchingState<String, ImageNode> initial = GraphMatchingState.getInitialState(nodes);
		DecisionModel<String, ImageNode> decisionModel = new DoubletonDecisionModel<>();
		//GraphFeatureExtractor<Integer, ImageNode> fe = CanonicalFeatureExtractor.constructCanonicalFeaturesFromExample(nodes.get(0));
		GraphFeatureExtractor<String, ImageNode> fe = new ImageFeatureExtractor(nodes.get(0));
		Counter<String> params = fe.getDefaultParameters();
		for (int i = 0; i < w.length - 1; i++)
		{
			params.setCount("scf" + i, w[i]);
		}
		params.setCount("adj", w[60]);
		
		Command<String, ImageNode> command = new Command<>(decisionModel, fe); 
		GenericMatchingLatentSimulator<String, ImageNode> transitionDensity = new GenericMatchingLatentSimulator<>(command, initial, true, true);
		ObservationDensity<GenericGraphMatchingState<String, ImageNode>, Object> observationDensity = new ExactProposalObservationDensity<>(command);
		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < nodes.size(); i++) emissions.add(null);
		SequentialGraphMatchingSampler<String, ImageNode> smc = new SequentialGraphMatchingSampler<>(transitionDensity, observationDensity, emissions);
		smc.sample(1000, 1000);

		List<Set<ImageNode>> truth = new ArrayList<>();
		for (int idx = 0; idx < nodes001.size(); idx++)
		{
			Set<ImageNode> e = new HashSet<>();
			e.add(nodes001.get(idx));
			e.add(nodes002.get(idx));
			truth.add(e);
		}
		MatchingSampleEvaluation<String, ImageNode> me = MatchingSampleEvaluation.evaluate(smc.getSamples(), truth);
		System.out.println(me.bestLogLikMatching.getSecond().getSecond() + ", " + me.consensusMatching.getSecond() + ", " + me.bestAccuracyMatching.getSecond());
		System.out.println(me.bestLogLikMatching.getSecond().getFirst());
		
		runDPF();
	}
  
  public void runDPF()
  {
		double [] w = {1.1687421537906029, 0.7983606162786142, 1.0792539677681912, 4.7259809686852385, 0.12920028844909923, 2.451260167464478, 0.11219785160944055, 0.0, 0.0, 0.5485447196644516, 0.03187661134181194, 0.4301401451002841, 5.119261978865087, 0.15831878085236195, 0.9665437170359402, -0.7296672986311913, 0.6421429390586643, -0.06249606423373143, 0.0, -0.4787322217930798, -0.025943188189330862, 0.06193559879567092, 0.07361595864729771, -0.1358893700051512, -0.004216001843933465, 2.6598706383854815, -0.010144303469621509, 1.0041702830702266, 0.8339454759633946, 0.9836169264602198, -0.12393359461565903, 0.21496519051264049, 0.8511838526984469, 0.7789583684970268, 0.3066017408716993, 0.8010215160987884, 0.9427989949519227, -0.36973899087326806, 0.37736371464726415, -0.010755773159882468, 0.0, -0.4868995607724972, 1.6354278681529546, 1.4022337600600419, -0.6972731215335931, 2.4986850845136974, 2.7697771551068837, -0.19777665873719996, -0.3967380408451989, 0.8993042650090122, 0.10039376382100196, 3.7525611353413546, 4.752919025215715, 4.739464042411727, 3.0421323501356925, 4.623431160635732, 1.1022710080629268, 0.9778450531875151, 3.9878108567510013, 0.8786376290607072, 7.689718078526511};

		List<ImageNode> nodes001 = ImageDataReader.readData(IMAGE_DIR, "001");
		List<ImageNode> nodes002 = ImageDataReader.readData(IMAGE_DIR, "111");
		List<ImageNode> nodes = new ArrayList<>(nodes001.size() + nodes002.size());
 		nodes.addAll(nodes001);
		nodes.addAll(nodes002);

		GraphMatchingState<String, ImageNode> initial = GraphMatchingState.getInitialState(nodes);
		DecisionModel<String, ImageNode> decisionModel = new DoubletonDecisionModel<>();
		GraphFeatureExtractor<String, ImageNode> fe = new ImageFeatureExtractor(nodes.get(0));
		
		Counter<String> params = fe.getDefaultParameters();
		for (int i = 0; i < w.length - 1; i++)
		{
			params.setCount("scf" + i, w[i]);
		}
		params.setCount("adj", w[60]);

		Command<String, ImageNode> command = new Command<>(decisionModel, fe); 

  	GenericDiscreteLatentSimulator<String, ImageNode> transitionDensity = new GenericDiscreteLatentSimulator<>(command, initial, true);
		ObservationDensity<GenericGraphMatchingState<String, ImageNode>, Object> observationDensity = new ObservationDensity<GenericGraphMatchingState<String,ImageNode>, Object>() {

			@Override
			public double logDensity(
					GenericGraphMatchingState<String, ImageNode> latent,
					Object emission) {
				return latent.getLogDensity();
			}

			@Override
			public double logWeightCorrection(
					GenericGraphMatchingState<String, ImageNode> curLatent,
					GenericGraphMatchingState<String, ImageNode> oldLatent) {
				return 0;
			}

			@Override
			public boolean cancellationApplied() {
				return false;
			}
		};

		List<Object> emissions = new ArrayList<>();
		for (int i = 0; i < nodes.size(); i++) emissions.add(null);
		DiscreteParticleFilter<GenericGraphMatchingState<String, ImageNode>, Object> dpf = new DiscreteParticleFilter<>(transitionDensity, observationDensity, emissions);
		double logZ = dpf.sample();
		System.out.println("logZ= " + logZ);

		List<Set<ImageNode>> truth = new ArrayList<>();
		for (int idx = 0; idx < nodes001.size(); idx++)
		{
			Set<ImageNode> e = new HashSet<>();
			e.add(nodes001.get(idx));
			e.add(nodes002.get(idx));
			truth.add(e);
		}
		
		MatchingSampleEvaluation<String, ImageNode> me = MatchingSampleEvaluation.evaluate(dpf.getPropagationResults().samples, truth);
		System.out.println(me.bestLogLikMatching.getSecond().getSecond() + ", " + me.consensusMatching.getSecond() + ", " + me.bestAccuracyMatching.getSecond());
		System.out.println(me.bestLogLikMatching.getSecond().getFirst());

  }

	public static void main(String [] args)
	{
		Mains.instrumentedRun(args, new HouseMatchingPerformanceExperiments());
	}
}
