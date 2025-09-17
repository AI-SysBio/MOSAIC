# MOSAIC: Scalable Stochastic Simulation with Heterogeneous and Non-Markovian Agents

<img align="right" src="https://raw.githubusercontent.com/AI-SysBio/MOSAIC/master/Figures/MOSAIC.png" width=400>

Discrete stochastic processes are widespread in both nature and human-made systems, with applications across physics, biochemistry, epidemiology, social patterns and finance, just to name a few. In the majority of these systems, the dynamics cannot be properly described with memoryless (or Markovian) interactions, and thus require the use of numerical tools for analyzing these non-Markovian dynamics. This repository contains the implementattion of a general and scalable framework to simulate non-Markovian stochastic systems with arbitrary inter-event time distribution and accuracy. The algorithm is referred to as the Rejection Gillespie algorithm for non-Markovian Reactions (REGIR) [1].

&nbsp;


### Simulating a non-Markovian system




First, you need to load `MOSAIC/MOSAIC.py` file provided in the repository. Then, you can run a non-Markovian simulation with the toy example below

	import MOSAIC as mosaic

	#Set the simulation parameters:
	class param:
		Tend = 10		#Length of the simulation
		N_simulations = 20	#The simulation results should be averaged over many trials
		unit = 'h'		#Unit of time (used for plotting only)
		timepoints = 100	#Number of timepoints to record (used for plotting only)

	r1 = 1
	r2 = 4
	r3 = 0.03
	alpha1 = 20
	alpha2 = 5
      
	#Define the reaction chanels:
	reaction1 = mosaic.Reaction_channel(param,rate=r1, shape_param=alpha1, distribution = 'Gamma')
	reaction1.reactants = ['A']
	reaction1.products = ['B']	
	reaction2 = mosaic.Reaction_channel(param,rate=r2, shape_param=alpha2, distribution = 'Weibull')
	reaction2.reactants = ['B']
	reaction2.products = ['C','A']	
	reaction3 = mosaic.Reaction_channel(param,rate=r3)
	reaction3.reactants = ['A','B']
	reaction3.products = []
		
	
	#Define the initial population of reactants:
	N_init = {'A':300,'B':0,'C':0}

	#Initialize and run the Gillepsie simulation:
	G_simul = gil.Gillespie_simulation(N_init,param)
	G_simul.reaction_channel_list = [reaction1, reaction2, reaction3]
	populations = G_simul.run_simulations(param.Tend, verbose = True)
	
	#Plot the results:
	G_simul.plot_inter_event_time_distribution()
	G_simul.plot_populations()

The algorithm runs for a few seconds and output the following figures (note that you can disables all printing by passing the argument `verbose = False` when running the simulation):
<p align="center">
  <img src="https://raw.githubusercontent.com/AI-SysBio/MOSAIC/master/Figures/MOSAIC_test.png" width=800>
</p>

The oscillations resulting from the markovian dynamics are clearly visible. If you check carefully, you will notice that the *theoretical distributions* do not match exactly the *simulated distributions*, even if you increase the number of simulations. This happens because the entities A and B are reactants of two reaction channels at the same time, and the *theoretical distribution* only represent the inter-event time distribution that **the reaction channel would have if it was the only process interaction with that reactant**. In practice, these kind of situations will occur frequently in non-Markovian systems, so do not worry if the simulated and theoretical distributions do not match exactly. The accuracy of REGIR was rigourously demonstrated in [1] (see the `/REGIR/Benchmark` folder).
      
### Implemented distributions
With the current implementation, each available distribution are characterised by their rate and a shape parameter as follow:

      Exponential:
          - rate: 1/mean
          - shape parameter: None
      
      Normal:
          - rate: 1/mean
          - shape: std/mean
      
      LogNormal:
          - rate: 1/mean
          - shape: std/mean
          
      Gamma:
          - rate: 1/mean
          - shape: α >= 1 (https://en.wikipedia.org/wiki/Gamma_distribution)
          
      Weibull:
          - rate: 1/mean
          - shape: k >= 1 (https://en.wikipedia.org/wiki/Weibull_distribution)
          
      Cauchy:
          - rate: 1/median
          - shape: γ (https://en.wikipedia.org/wiki/Cauchy_distribution)
      

Keep in mind that non-Markovian simulations are only available for reaction channels with a single reactant, as the definition of inter-event time distribution is ambigious for channels with multiple reactants. If a channel is defined without or with more than one reactant, it will be considered as a Poisson process. 



### Reproducing Results from the publication

[1] Pélissier, A, Phan, M, et al. "Unifying non-Markovian Dynamics and Agent Heterogeneity in Scalable Stochastic Simulation". *Submitted to Proceedings of the National Academy of Sciences* (2022) [[Preprint]](https://arxiv.org/abs/2212.05059)

Models for the three application presented in the paper can be found in `MOSAIC/Application_Immune_response`, `MOSAIC/Application_delay` and `MOSAIC/Application_Temporal_network`.

        
        

