import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns
import random
from math import log, gamma
from scipy.stats import expon, weibull_min
import os,sys

import lib_REGIR_with_delay as gil
from datetime import datetime


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



recompute_time_complexity = False


class param:
    Tend = 300
    unit = 'h'
    rd.seed(101)                #length of the simulation, in hours (41 days)
    N_simulations = 50         #The simulation results should be averaged over many trials
    timepoints = 1000            #Number of timepoints to record (make surethat this number isnt too big)

def main():
    
    print("\n=================================================================================") 
    print("========================= Non-Markovian Gillepsie algorithm =======================") 
    print("=================================================================================\n\n")    


    """
    G[P(t – τ)] is a monotonic decreasing function representing the delayed repression of hes1 mRNA production by Hes1 protein.
    """
             
    #https://www.sciencedirect.com/science/article/abs/pii/S0021999109002435
    #https://www.cell.com/current-biology/pdf/S0960-9822(03)00494-9.pdf
    
    def run_simulation(Nsim = 10, beta = 10, method = 'DelaySSA', plot_pop = False, f = 1):
        param.N_simulations = Nsim
        P0 = 10*beta
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
        
        if plot_pop:
            #G_simul.run_simulations(param.Tend)
            G_simul.plot_inter_event_time_distribution()
            #G_simul.plot_populations(figsize = (8,4))
            
            timepoints = G_simul.population_t.shape[0]
            time_points = np.linspace(0, G_simul.Tend, timepoints)
            population = G_simul.population_compiled
        
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
            
        return final_time, Nreactions
    
    
    
    time_REGIR = []
    time_REGIR10 = []
    time_Delay = []
    time_Exp = []
    nreac_REGIR = []
    nreac_REGIR10 = []
    nreac_Delay = []
    nreac_Exp = []
    #B_list = np.logspace(0,14,num = 15, base=2).astype(int)
    B_list = np.logspace(0,14,num = 15, base=2).astype(int)
    B_list = np.array(list(B_list) + [2**15,2**16])
    B_list = np.array([0.01, 0.02, 0.04, 0.077, 0.14, 0.29, 0.55] + list(B_list))
    
    if recompute_time_complexity:
        for bi,Beta in enumerate(B_list):
            print()
            print('Beta:', Beta)
            if Beta < 1.5:
                Nsim = 10000
            if Beta < 20:
                Nsim = 20
            else:
                Nsim = 1
            
            print('  REGIR...')
            Rtime, nR = run_simulation(Nsim = Nsim, beta = Beta, method = 'REGIR', f = 1)
            print('  DelaySSA...')
            Dtime, nD = run_simulation(Nsim = Nsim, beta = Beta, method = 'DelaySSA')
            print('  Exp...')
            Etime, nE = run_simulation(Nsim = Nsim, beta = Beta, method = 'EXP')
            
            print('  REGIR10...')
            Rtime, nR = run_simulation(Nsim = Nsim, beta = Beta, method = 'REGIR', f=10)
            
            time_REGIR.append(Rtime/Nsim)
            time_Delay.append(Dtime/Nsim)
            time_Exp.append(Etime/Nsim)
            nreac_REGIR.append(nR/Nsim)
            nreac_Delay.append(nD/Nsim)
            nreac_Exp.append(nE/Nsim)
            time_REGIR10.append(Rtime/Nsim)
            nreac_REGIR10.append(nR/Nsim)
        
        np.save('Simulation_results/time_REGIR.npy', time_REGIR)
        np.save('Simulation_results/time_Delay.npy', time_Delay)
        np.save('Simulation_results/time_EXP.npy', time_Exp)
        np.save('Simulation_results/Nreac_REGIR.npy', nreac_REGIR)
        np.save('Simulation_results/Nreac_Delay.npy', nreac_Delay)
        np.save('Simulation_results/Nreac_EXP.npy', nreac_Exp)
        
        np.save('Simulation_results/time_REGIR10.npy', time_REGIR10)
        np.save('Simulation_results/Nreac_REGIR10.npy', nreac_REGIR10)

        
    time_REGIR = np.load('Simulation_results/time_REGIR.npy')
    time_REGIR10 = np.load('Simulation_results/time_REGIR10.npy')
    time_Delay = np.load('Simulation_results/time_Delay.npy')
    time_Exp = np.load('Simulation_results/time_EXP.npy')
    nreac_REGIR = np.load('Simulation_results/Nreac_REGIR.npy')
    nreac_REGIR10 = np.load('Simulation_results/Nreac_REGIR10.npy')
    nreac_Delay = np.load('Simulation_results/Nreac_Delay.npy')
    nreac_Exp = np.load('Simulation_results/Nreac_EXP.npy')
    
    

    time_nMGA = time_REGIR**2.02 * 115
    time_nMGA[time_nMGA>2e3] = np.nan
        
    #B_list_ = B_list
    #B_list = B_list[7:]
    fitted_Exp = fit_log_line(B_list,time_Exp)
    fitted_REGIR = fit_log_line(B_list,time_REGIR)
    fitted_Delay = fit_log_line(B_list,time_Delay)
    fitted_nMGA = fit_log_line(B_list[:-2],time_nMGA[:-2])
    
    
    
    plt.figure(figsize = (7,4))
    plt.scatter(B_list[7:], time_nMGA[7:], s=35, color = 'green', label = r'nMGA (gamma, $\alpha=3$)', edgecolor = 'black')
    plt.plot(B_list[7:-2], fitted_nMGA[7:], lw=2, color = 'green', ls = '--')
    plt.scatter(B_list, time_Delay, s=35, color = 'blue', label = r'DelaySSA (gamma, $\alpha=3$)', edgecolor = 'black')
    #plt.scatter(B_list, time_Exp, s=60, color = 'blue', label = 'SG (exponential)')
    #plt.plot(B_list, fitted_Exp, lw=2, color = 'blue', ls = '--')
    plt.scatter(B_list, time_REGIR, s=35, color = 'red', label = r'REGIR (gamma, $\alpha=3$, $\lambda_{max} \geq \lambda_0$)', edgecolor = 'black')
    plt.plot(B_list, fitted_REGIR, lw=2, color = 'red', ls = '--')
    plt.scatter(B_list, time_REGIR10, s=35, color = 'orange', label = r'REGIR (gamma, $\alpha=3$, $N\lambda_{max} \geq 20\lambda_0$)', edgecolor = 'black')
    #plt.plot(B_list, fitted_Delay, lw=2, color = 'red', ls = '--')
    #plt.plot(N_list, N_list*np.log(N_list)/1e5, lw=2, color = 'red' )
    plt.fill_between(B_list, time_Delay*0.85, time_Delay*1.15, alpha = 0.2, color = 'blue')
    plt.fill_between(B_list, time_REGIR*0.85, time_REGIR*1.15, alpha = 0.2, color = 'red')
    plt.fill_between(B_list[7:], time_nMGA[7:]*0.85, time_nMGA[7:]*1.15, alpha = 0.2, color = 'green')
    plt.fill_between(B_list[:7], time_REGIR10[:7]*0.85, time_REGIR10[:7]*1.15, alpha = 0.2, color = 'orange')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\beta$ paremeter')
    plt.ylabel('Mean time / simulation [s]')
    plt.ylim(8e-7,3e3)
    plt.xlim(8e-3,1e5)
    #plt.xlim(5e0,3e4)
    plt.legend(loc = 'lower right', prop={"size":11})
    plt.show()
    
    
    time_REGIR = time_REGIR / nreac_REGIR
    time_REGIR10 =  time_REGIR10 / nreac_REGIR10
    time_Delay = time_Delay / nreac_Delay
    time_Exp = time_Exp / nreac_Exp
    fitted_Exp = fit_log_line(B_list,time_Exp)
    fitted_REGIR = fit_log_line(B_list,time_REGIR)
    
    
    plt.figure(figsize = (7,4))
    plt.scatter(B_list, time_Exp, s=60, color = 'black', label = 'SG (exponential)', edgecolor = 'black')
    plt.plot(B_list, fitted_Exp, lw=2, color = 'black', ls = '--')
    plt.scatter(B_list, time_Delay, s=60, color = 'blue', label = r'DelaySSA (gamma, $\alpha=3$)', edgecolor = 'black')
    plt.scatter(B_list, time_REGIR, s=60, color = 'red', label = r'REGIR (gamma, $\alpha=3$, $\lambda_{max} \geq \lambda_0$)', edgecolor = 'black')
    plt.plot(B_list, fitted_REGIR, lw=2, color = 'red', ls = '--')
    plt.scatter(B_list, time_REGIR10, s=60, color = 'orange', label = r'REGIR (gamma, $\alpha=3$, $N\lambda_{max} \geq 20\lambda_0$)', edgecolor = 'black')
    #plt.plot(B_list, fitted_Delay, lw=2, color = 'red')
    #plt.plot(B_list, N_list*np.log(N_list)/1e5, lw=2, color = 'red' )
    plt.fill_between(B_list, time_Exp*0.9, time_Exp*1.1, alpha = 0.1, color = 'black')
    plt.fill_between(B_list, time_Delay*0.9, time_Delay*1.1, alpha = 0.1, color = 'blue')
    plt.fill_between(B_list, time_REGIR*0.9, time_REGIR*1.1, alpha = 0.1, color = 'red')
    plt.fill_between(B_list[:7], time_REGIR10[:7]*0.85, time_REGIR10[:7]*1.15, alpha = 0.1, color = 'orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\beta$ parameter')
    plt.ylabel('Mean time / event [s]')
    plt.ylim(2.7e-4,2e-3)
    plt.xlim(8e-3,1e5)
    plt.legend(loc = 'upper center', prop={"size":11})
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
    main()
        
        
