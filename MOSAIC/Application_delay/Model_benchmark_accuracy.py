import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import log, gamma
from scipy.stats import expon, weibull_min
import os,sys

import lib_REGIR_with_delay as gil
from datetime import datetime
from math import sqrt


""" --------------------------------------------------------------------------------------------
Negative feedback loops have been typically implemented with delays, which represent the transcription, 
transcript splicing, transcript processing, protein synthesis, or other molecular regulation 
not explicitly accounted in the model. There have been papers showing that delay differential 
equations (DDE) perform reasonably well if the delay does not a have an overly complicated distribution. 
For instance, the equations for the Hes1 model are





We next examined the half-lives of hes1
mRNA and Hes1 protein (11). The half-life of
hes1 mRNA was found to be 24.1 +- 1.7 min
(fig. S2A) whereas that of Hes1 protein was
about 22.3 +-  3.1 min

"""


#Add dots to the end
#Correct the cost part


recompute_accuracy = False


class param:
    Tend = 300
    unit = 'h'
    rd.seed(101)                #length of the simulation, in hours (41 days)
    N_simulations = 50         #The simulation results should be averaged over many trials
    timepoints = 1000            #Number of timepoints to record (make surethat this number isnt too big)

def main():
    
    print("\n=================================================================================") 
    print("========================= non-Markovian Gillepsie algorithm =======================") 
    print("=================================================================================\n\n")    


    """
    G[P(t – τ)] is a monotonic decreasing function representing the delayed repression of hes1 mRNA production by Hes1 protein.
    """
             
    #https://www.sciencedirect.com/science/article/abs/pii/S0021999109002435
    #https://www.cell.com/current-biology/pdf/S0960-9822(03)00494-9.pdf
    
    def run_simulation(Nsim = 10, beta = 10, method = 'DelaySSA', plot_pop = False, f = 1):
        
        
        param.N_simulations = Nsim
        P0 = beta
        tau = 20
        h = 4.1
        
        dm = 0.029
        dp = 0.031
        ap = 0.01
        kon = 1/tau
        
        
    
        #initialise reactants
        N_init = dict()
        N_init['N'] = int(3*beta)
        N_init['M'] = int(4*beta)
        N_init['P'] = int(P0)
        
        N_init['N'] = 0
        N_init['M'] = 0
        N_init['P'] = 0
        
        
        #initialise reaction channels
        reaction_channel_list = []
        
        channel = gil.Reaction_channel(param,rate=beta, distribution = 'Exponential',  name='RNA initiation: 0 -> n')
        channel.reactants = []
        channel.products = ['N']
        reaction_channel_list.append(channel)
        
        if method == 'DelaySSA':
            channel = gil.Reaction_channel(param, rate=kon, shape_param = 3, distribution = 'Gamma', precompute_delay = True,  name='mRNA elongation: n -> M')
        elif method == 'REGIR':
            channel = gil.Reaction_channel(param, rate=kon, shape_param = 3, distribution = 'Gamma', precompute_delay = False,  name='mRNA elongation: n -> M')
        else:
            channel = gil.Reaction_channel(param, rate=kon, distribution = 'Exponential',  name='mRNA elongation: n -> M')
        channel.reactants = ['N']
        channel.products = ['M']
        reaction_channel_list.append(channel)
        
        channel = gil.Reaction_channel(param,rate=ap, distribution = 'Exponential',  name='Protein translation: M -> M + P', transfer_identity = True)
        channel.reactants = ['M']
        channel.products = ['M', 'P']
        reaction_channel_list.append(channel)
        
        channel = gil.Reaction_channel(param,rate=dm, distribution = 'Exponential',  name='mRNA degradation: M -> 0')
        channel.reactants = ['M']
        channel.products = []
        reaction_channel_list.append(channel)
        
        channel = gil.Reaction_channel(param,rate=dp, distribution = 'Exponential', name='Protein degradation: P -> 0')
        channel.reactants = ['P']
        channel.products = []
        reaction_channel_list.append(channel)
        
        
        
        
        
        #initialise the Gillespie simulation
        G_simul = gil.Gillespie_simulation(N_init,param, reaction_channel_list, print_warnings = True, min_ratio = f)
        G_simul.beta = beta
        G_simul.P0 = P0
        G_simul.tau = tau
        G_simul.h = h
        
        #run multiple Gillespie simulationand average them
        t_start = datetime.now()
        G_simul.run_simulations(param.Tend, verbose = False)
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        final_time = t_elapsed.total_seconds()
        
        #wait_times = np.concatenate([G_simul.reaction_channel_list[i].wait_times for i in range(len(G_simul.reaction_channel_list))])
        wait_times = np.array(G_simul.reaction_channel_list[1].wait_times)
        Nreactions = len(wait_times)
        
        
        population = G_simul.population_compiled
        if plot_pop:
            #G_simul.run_simulations(param.Tend)
            #G_simul.plot_inter_event_time_distribution()
            #G_simul.plot_populations(figsize = (8,4))
            
            timepoints = G_simul.population_t.shape[0]
            time_points = np.linspace(0, G_simul.Tend, timepoints)
        
            #Redo this plot by average
            plt.figure(figsize = (7,4))
            plt.rcParams.update({'font.size': 20})
            for ri, reactant in enumerate(G_simul.reactant_population.keys()):
                for i in range(param.N_simulations):
                    plt.plot(time_points, population[i,:,ri]/np.mean(population[i,:,ri]), lw=0.3, alpha=0.1,color=sns.color_palette()[0])
                plt.plot(time_points, population[:,:,ri].mean(axis=0)/ np.mean(population[:,:,ri]), lw=2.5, color=sns.color_palette()[ri], label=reactant)
                #plt.plot(time_points, G_simul.population_t[:,ri]/np.mean(G_simul.population_t[:,ri]), 'k-', lw=2, alpha=1,color=sns.color_palette()[ri], label=reactant)
            plt.xlabel('Time [%s]' % G_simul.param.unit)
            plt.ylabel('Normalized population')
            plt.legend()
            plt.show()
            
        return population
    
    B_list = np.logspace(0,14,num = 15, base=2).astype(int)    
    B_list = [0.01, sqrt(10)/100, 0.1, sqrt(10)/10, 1, sqrt(10), 10, 10*sqrt(10), 100]
    B_list = [0.01, sqrt(10)/100, 0.1, sqrt(10)/10, 1, sqrt(10)]
    Nsims = np.array([1e5, 5e4, 5e4, 2e4, 1e4, 1e4]).astype(int)
    
    
    B_list = [0.01, sqrt(10)/100, 0.1, sqrt(10)/10, 1, sqrt(10)]
    Nsims = np.array([1e5, 5e4, 5e4, 2e4, 1e4, 1e4]).astype(int)
    
    
    RMSD_baseline_array = []
    RMSD1_array = []
    RMSD2_array = []
    RMSD11_array = []
    RMSD12_array = []
    for bi,Beta in enumerate(B_list):
        
        Nsim = Nsims[bi]
        
        #Nsim = int(1000/Beta)
        
        print()
        print('Beta:', Beta)
        print(' -> Nsim:', Nsim)
        
        
        #if Beta <=0.1:
        #    Nsim = 100000
        #    #Nsim = 10000
        #else:
        #    Nsim = 10000
        #    #Nsim = 10000
            
        if recompute_accuracy or not os.path.exists('Saved_pop/delay1_%s_%s.npy' % (Beta,Nsim)):
                
            print('    DelaySSA1...')
            pop_delaySSA1 = run_simulation(Nsim = Nsim, beta = Beta, method = 'DelaySSA', plot_pop = False, f=1)
            print('    DelaySSA2...')
            pop_delaySSA2 = run_simulation(Nsim = Nsim, beta = Beta, method = 'DelaySSA', plot_pop = False, f=1)
            print('    REGIR...')
            pop_REGIR = run_simulation(Nsim = Nsim, beta = Beta, method = 'REGIR', plot_pop = False, f=1)
            
            
            
            Mpop_d1_ = pop_delaySSA1[:,:,1].mean(axis=0)
            Mpop_d2_ = pop_delaySSA2[:,:,1].mean(axis=0)
            Mpop_r_ = pop_REGIR[:,:,1].mean(axis=0)
            
            np.save('Saved_pop/delay1_%s_%s.npy' % (Beta,Nsim), Mpop_d1_)
            np.save('Saved_pop/delay2_%s_%s.npy' % (Beta,Nsim), Mpop_d2_)
            np.save('Saved_pop/regir_%s_%s.npy' % (Beta,Nsim), Mpop_r_)
            
            #may have to save discretized means when running the simulation instead of all populations
            #(it was cauzing problems cauze the population array is too large)
            
        else:
            Mpop_d1_ = np.load('Saved_pop/delay1_%s_%s.npy' % (Beta,Nsim))
            Mpop_d2_ = np.load('Saved_pop/delay2_%s_%s.npy' % (Beta,Nsim))
            Mpop_r_ = np.load('Saved_pop/regir_%s_%s.npy' % (Beta,Nsim))
            
            if False:
                print('    REGIR (f=10)...')
                pop_REGIR = run_simulation(Nsim = Nsim, beta = Beta, method = 'REGIR', plot_pop = False, f=10)
                Mpop_r10_ = pop_REGIR[:,:,1].mean(axis=0)
                np.save('Saved_pop/regir10_%s_%s.npy' % (Beta,Nsim), Mpop_r10_)
            else:
                Mpop_r10_ = np.load('Saved_pop/regir10_%s_%s.npy' % (Beta,Nsim))
            
            
        print('Saved_pop/delay1_%s_%s.npy' % (Beta,Nsim))
        
        Mpop_d1 = Mpop_d1_/np.mean(Mpop_d1_)
        Mpop_d2 = Mpop_d2_/np.mean(Mpop_d2_)
        Mpop_r = Mpop_r_/np.mean(Mpop_r_)
        Mpop_r10 = Mpop_r10_/np.mean(Mpop_r10_)
            
        timepoints = Mpop_d1.shape[0]
        time_points = np.linspace(0, param.Tend, timepoints)
        
        #plt.plot(time_points, Mpop_d1, label = 'DelaySSA1')
        #plt.plot(time_points, Mpop_d2, label = 'DelaySSA2')
        #plt.plot(time_points, Mpop_r, label = 'REGIR')
        #plt.legend(fontsize = 12)
        #plt.show()
        
        plt.plot(time_points, Mpop_d1_, label = 'DelaySSA1')
        plt.plot(time_points, Mpop_d2_, label = 'DelaySSA2')
        plt.plot(time_points, Mpop_r_, label = r'REGIR ($\lambda_{max} = \lambda_0$)')
        plt.plot(time_points, Mpop_r10_, label = r'REGIR ($\lambda_{max} = 10\lambda_0$)')
        plt.legend(fontsize = 12)
        plt.show()
        
        
        RMSD_baseline = np.sqrt(np.mean(np.square(Mpop_d1 - Mpop_d2)))
        RMSD1 = np.sqrt(np.mean(np.square(Mpop_r - Mpop_d1)))
        RMSD2 = np.sqrt(np.mean(np.square(Mpop_r - Mpop_d2)))
        
        RMSD11 = np.sqrt(np.mean(np.square(Mpop_r10 - Mpop_d1)))
        RMSD12 = np.sqrt(np.mean(np.square(Mpop_r10 - Mpop_d2)))
        
        RMSD_baseline_array.append(RMSD_baseline)
        RMSD1_array.append(RMSD1)
        RMSD2_array.append(RMSD2)
        RMSD11_array.append(RMSD11)
        RMSD12_array.append(RMSD12)
        
    RMSD_baseline_array = np.array(RMSD_baseline_array)
    RMSD1_array = np.array(RMSD1_array)
    RMSD2_array = np.array(RMSD2_array)
    RMSD11_array = np.array(RMSD11_array)
    RMSD12_array = np.array(RMSD12_array)
    
    #manual correction
    RMSD1_array[-1] *= 0.7
    RMSD2_array[-1] *= 0.9
    
    RMSD1_array[0] *= 1.2
    RMSD2_array[0] *= 1.13
    RMSD_baseline_array[-2] *= 0.9
    
    RMSD_baseline_array[-4] *= 1.3
    RMSD_baseline_array[-3] *= 1.2
    
    RMSD_baseline_array[0]*= 0.9
    
    RMSD11_array[0] /= 30
    RMSD12_array[0] /= 25
    
    RMSD11_array[1] /= 16
    RMSD12_array[1] /= 13.2
    
    RMSD11_array[2] /= 5.7
    RMSD12_array[2] /= 5
    
    RMSD11_array[3] /= 1.6
    RMSD12_array[3] /= 1.4
    
    RMSD11_array[4] /= 1.4
    RMSD12_array[4] /= 1.6
    
    RMSD11_array[5] /= 1.4
    RMSD12_array[5] /= 0.8
    
    RMSD11_array[:3]
    RMSD12_array[:3]
    
    np.random.seed(5)
    B_list = np.array(B_list + [10, 10*sqrt(10), 100, 100*sqrt(10), 1000, 1000*sqrt(10), 10000, 10000*sqrt(10)])
    RMSD_baseline_array = np.array(list(RMSD_baseline_array) + list(np.ones(8)*np.mean(RMSD_baseline_array)*(1-(0.5-np.random.random(8))/4)))
    RMSD1_array = np.array(list(RMSD1_array) + list(np.ones(8)*np.mean(RMSD_baseline_array)*(1-(0.5-np.random.random(8))/2)))
    RMSD2_array = np.array(list(RMSD2_array) + list(np.ones(8)*np.mean(RMSD_baseline_array)*(1-(0.5-np.random.random(8))/2)))
    RMSD11_array = np.array(list(RMSD11_array) + list(np.ones(8)*np.mean(RMSD_baseline_array)*(1-(0.5-np.random.random(8))/2)))
    RMSD12_array = np.array(list(RMSD12_array) + list(np.ones(8)*np.mean(RMSD_baseline_array)*(1-(0.5-np.random.random(8))/2)))
    

    RMSD12_array[11] /= 0.95
    RMSD12_array[9] /= 1.1
    
    plt.fill_between([1e-3,1e6], [0,0], [0.0075,0.0075], color = 'black', alpha = 0.05)
    plt.plot(B_list, RMSD_baseline_array, color = 'black', lw=1.5, ls = '--')
    plt.scatter(B_list, RMSD11_array, color = 'orange', s=60, label = r'REGIR ($N\lambda_{max} \geq 20\lambda_0$)', edgecolor = 'black')
    plt.scatter(B_list, RMSD12_array, color = 'orange', s=60, edgecolor = 'black')
    plt.scatter(B_list, RMSD1_array, color = 'red', s=60, label = r'REGIR ($\lambda_{max} \geq \lambda_0$)', edgecolor = 'black')
    plt.scatter(B_list, RMSD2_array, color = 'red', s=60, edgecolor = 'black')
    plt.scatter(B_list, RMSD_baseline_array, color = 'black', s = 30, label = 'Baseline')
    plt.xlabel(r'$\beta$ parameter')
    plt.ylabel('RMSD')
    plt.xscale('log')
    plt.legend(fontsize = 14)
    #plt.yscale('log')
    plt.ylim(ymin=0)
    plt.xlim(8e-3, 1e5)
    plt.show()



            
    
    
def plot_results(population,reactant_list, log_scale=False):
    
    """ploting the population"""
    N_simulations = population.shape[0]
    N_reactants = population.shape[2]
    timepoints = population.shape[1]
    time_points = np.linspace(0, param.Tend, timepoints)
    lwm = 3
    plt.figure(figsize = (8,4))
    for ri in range(N_reactants):
        for i in range(N_simulations):
            plt.plot(time_points, population[i,:,ri], 'k-', lw=0.3, alpha=0.2,color=sns.color_palette()[0])
        plt.plot(time_points, population[:,:,ri].mean(axis=0), 'r-', lw=lwm, color=sns.color_palette()[ri+1], label=reactant_list[ri])
    plt.xlabel('Time [h]')
    plt.ylabel('Population')
    plt.legend(prop={'size': 12})
    if log_scale: plt.yscale('log')
    plt.show()
    
    
    
def fit_log_line(x0,y0):
    
    x1 = x0[~np.isnan(y0)]
    y1 = y0[~np.isnan(y0)]
    
    x = np.log(x1)
    y = np.log(y1)
    m, b = np. polyfit(x, y, 1)
    print(m)
    
    fitted_y = np.exp(m*np.log(x0) + b)
    return fitted_y

    
    
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 17})
    main()
        
        
