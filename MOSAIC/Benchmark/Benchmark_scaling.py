import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import log, gamma
from scipy.stats import expon, weibull_min
import os,sys
from datetime import datetime

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import MOSAIC as gil_MOSAIC


""" --------------------------------------------------------------------------------------------
1 reactions:
    A -> B,  differentiation (Gamma)
    
"""
class param:
    Tend = 5
    unit = 'h'
    rd.seed(101)                #length of the simulation, in hours (41 days)
    N_simulations = 10          #The simulation results should be averaged over many trials
    timepoints = 100            #Number of timepoints to record (make surethat this number isnt too big)
    
    
def main():
     
    #N_list = np.logspace(3,18,num = 16, base=2).astype(int)
    N_list = np.logspace(3,16,num = 14, base=2).astype(int)
    N_simul = [500,100,100,100,100,10,10,10,10,1,1,1,1,1]
    
    recompute_time_complexity = False
    if recompute_time_complexity:
        compute_time_complexity(N_list, N_simul)
        
    time_Exp = np.load('Simulation_results/time_EXP.npy')
    time_MOSAIC = np.load('Simulation_results/time_MOSAIC.npy')
    time_Delay = np.load('Simulation_results/time_Delay.npy')
    nreac_Exp = np.load('Simulation_results/Nreac_EXP.npy')
    nreac_MOSAIC = np.load('Simulation_results/Nreac_MOSAIC.npy')
    nreac_Delay = np.load('Simulation_results/Nreac_Delay.npy')
    
    fitted_Exp = fit_log_line(N_list,time_Exp/N_simul)
    fitted_MOSAIC = fit_log_line(N_list,time_MOSAIC/N_simul)
    fitted_Delay = fit_log_line(N_list,time_Delay/N_simul)
    
    
    
    plt.figure(figsize = (8,4))
    plt.scatter(N_list, time_Exp/N_simul, s=60, color = 'blue', label = 'SG (exponential)')
    plt.plot(N_list, fitted_Exp, lw=2, color = 'blue')
    plt.scatter(N_list, time_MOSAIC/N_simul, s=60, color = 'green', label = r'MOSAIC (gamma, $\alpha=6$)')
    plt.plot(N_list, fitted_MOSAIC, lw=2, color = 'green')
    plt.scatter(N_list, time_Delay/N_simul, s=60, color = 'red', label = r'Delay (gamma, $\alpha=6$)')
    plt.plot(N_list, fitted_Delay, lw=2, color = 'red')
    #plt.plot(N_list, N_list*np.log(N_list)/1e5, lw=2, color = 'red' )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of reactants')
    plt.ylabel('Av. time / simulation [s]')
    #plt.ylim(1e-5,1e1)
    plt.xlim(5e0,3e4)
    #plt.legend(loc = 'lower right', prop={"size":15})
    plt.show()
    
    
    time_Exp = time_Exp / nreac_Exp
    time_MOSAIC = time_MOSAIC / nreac_MOSAIC
    time_Delay = time_Delay / nreac_Delay
    
    fitted_Exp = fit_log_line(N_list,time_Exp)
    fitted_MOSAIC = fit_log_line(N_list,time_MOSAIC)
    fitted_Delay = fit_log_line(N_list,time_Delay)
    
    plt.figure(figsize = (8,4))
    plt.scatter(N_list, time_Exp, s=60, color = 'blue', label = 'SG (exponential)')
    #plt.plot(N_list, fitted_Exp, lw=2, color = 'blue')
    plt.scatter(N_list, time_MOSAIC, s=60, color = 'green', label = r'MOSAIC (gamma, $\alpha=6$)')
    #plt.plot(N_list, fitted_MOSAIC, lw=2, color = 'green')
    plt.scatter(N_list, time_Delay, s=60, color = 'red', label = r'Delay (gamma, $\alpha=6$)')
    #plt.plot(N_list, fitted_Delay, lw=2, color = 'red')
    #plt.plot(N_list, N_list*np.log(N_list)/1e5, lw=2, color = 'red' )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of reactants')
    plt.ylabel('Av. time / simulation [s]')
    #plt.ylim(1e-5,1e1)
    plt.xlim(5e0,3e4)
    #plt.legend(loc = 'lower right', prop={"size":15})
    plt.show()
    



def compute_time_complexity(N_list, N_simul):
    
    r0 = 1 #same rate for every reactions
    
    time_MOSAIC = []
    time_Exp = []
    time_Delay = []
    nreac_Delay = []
    nreac_MOSAIC = []
    nreac_Exp = []
    
    for ni,N in enumerate(N_list):
        param.N_simulations = N_simul[ni]
            
        
        
        N_init = dict()
        N_init['A'] = N
        
        
        
        """ Gillespie Exp algorithm benchmarking"""
        #initialise reaction channels
        reaction_channel_list = []
        channel = gil_MOSAIC.Reaction_channel(param,rate=r0, distribution = 'Exponential')
        channel.reactants = ['A']
        channel.products = ['A']
        reaction_channel_list.append(channel)


        #initialise the Gillespie simulation
        G_simul = gil_MOSAIC.Gillespie_simulation(N_init,param, print_warnings = True)
        G_simul.reaction_channel_list = reaction_channel_list
        
        #run multiple Gillespie simulationand average them
        t_start = datetime.now()
        G_simul.run_simulations(param.Tend, verbose=False)
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        final_time = t_elapsed.total_seconds()
        
        wait_times = np.array(G_simul.reaction_channel_list[0].wait_times)
        Nreactions = len(wait_times)
        time_Exp.append(final_time)
        nreac_Exp.append(Nreactions)
        
        print("For N=%s, Gille elapsed time was %s seconds for %s simulations" % (N,time_Exp[-1],param.N_simulations))
        
        
    
    
        """ MOSAIC algorithm benchmarking"""
            #initialise reaction channels
        reaction_channel_list = []
        channel = gil_MOSAIC.Reaction_channel(param,rate=r0, shape_param=6, distribution = 'Gamma')
        channel.reactants = ['A']
        channel.products = ['A']
        reaction_channel_list.append(channel)
        
        
        #initialise the Gillespie simulation
        G_simul = gil_MOSAIC.Gillespie_simulation(N_init,param, print_warnings = True)
        G_simul.reaction_channel_list = reaction_channel_list
        
        #run multiple Gillespie simulationand average them
        t_start = datetime.now()
        G_simul.run_simulations(param.Tend, verbose=False)
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        final_time = t_elapsed.total_seconds()
        
        wait_times = np.array(G_simul.reaction_channel_list[0].wait_times)
        Nreactions = len(wait_times)
        time_MOSAIC.append(final_time)
        nreac_MOSAIC.append(Nreactions)
        
        print("For N=%s, MOSAIC elapsed time was %s seconds for %s simulations" % (N,time_MOSAIC[-1],param.N_simulations))
        
        #G_simul.plot_inter_event_time_distribution()
        #G_simul.plot_populations(figsize = (8,4))
        
        

        """ Delay algorithm benchmarking"""
            #initialise reaction channels
        reaction_channel_list = []
        channel = gil_MOSAIC.Reaction_channel(param,rate=r0, shape_param=6, distribution = 'Gamma', precompute_delay = True)
        channel.reactants = ['A']
        channel.products = ['A']
        reaction_channel_list.append(channel)
        
        
        #initialise the Gillespie simulation
        G_simul = gil_MOSAIC.Gillespie_simulation(N_init,param, print_warnings = True)
        G_simul.reaction_channel_list = reaction_channel_list
        
        #run multiple Gillespie simulationand average them
        t_start = datetime.now()
        G_simul.run_simulations(param.Tend, verbose=False)
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        final_time = t_elapsed.total_seconds()
        
        
        wait_times = np.array(G_simul.reaction_channel_list[0].wait_times)
        Nreactions = len(wait_times)
        time_Delay.append(final_time)
        nreac_Delay.append(Nreactions)
        
        #G_simul.plot_inter_event_time_distribution()
        #G_simul.plot_populations(figsize = (8,4))
        
        
        print("For N=%s, Delay elapsed time was %s seconds for %s simulations" % (N,time_Delay[-1],param.N_simulations))
        print()
        
        
    np.save('Simulation_results/time_EXP.npy', time_Exp)
    np.save('Simulation_results/time_MOSAIC.npy', time_MOSAIC)
    np.save('Simulation_results/time_Delay.npy', time_Delay)
    
    np.save('Simulation_results/Nreac_EXP.npy', nreac_Exp)
    np.save('Simulation_results/Nreac_MOSAIC.npy', nreac_MOSAIC)
    np.save('Simulation_results/Nreac_Delay.npy', nreac_Delay)
        
    
    
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
    plt.rcParams.update({'font.size': 20})
    main()
        