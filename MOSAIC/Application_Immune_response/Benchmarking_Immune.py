import numpy as np
import pandas as pd
import random
import sys
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stat
from math import gamma, log
import pickle
from datetime import datetime

import lib_REGIR_with_rate_function as gil
from sortedcontainers import SortedDict


"""
For now we only consider the three reactions:
    - B1 -> B2 (affinity change)
    - [B1T] -> [B2T] (Bcell switch)
"""

rB = 1
rBT = 0.02
timepoints = 100

constant_Tcells = True
recompute = False

np.random.seed(100)


def main():
    
    if constant_Tcells:
        Nlist = np.logspace(1,18,num = 18, base=2).astype(int)
    else:
        Nlist = np.logspace(4,13,num = 10, base=2).astype(int)
        
    
    Tend = 100*np.ones(len(Nlist))
    Tend[Nlist<50] = 100
    Tend[Nlist<10] = 1000
    Tend[Nlist<5] = 10000
    Tend[Nlist>200] = 10
    Tend[Nlist>1000] = 2
    Tend[Nlist>20000] = 2
    
    
    if constant_Tcells:
        Nlim = 10000
        NlimSSA = 5000
    else:
        Nlim = 5000
        NlimSSA = 1000
    
    
    if recompute:
        time_REGIR = []
        time_Gil = []
        time_DSSA = []
        for ni,N in enumerate(Nlist):
            print('N = %s' % N)
            
            
            if N < NlimSSA:
                
                tREGIR = simulate_system(N=N, Tend = Tend[ni], method = 'REGIR', plot = False, constant_T = constant_Tcells)/Tend[ni]*100
                print('  REGIR:', tREGIR)
                tGil = simulate_system(N=N, Tend = Tend[ni], method = 'Gillespie', plot = False, constant_T = constant_Tcells)/Tend[ni]*100
                print('  Gillespie:', tGil)
                #tGil2 = simulate_system(N=N, Tend = Tend[ni], method = 'Gillespie_sorted', plot = False, constant_T = constant_Tcells)/Tend[ni]*100
                #print('  Gillespie2:', tGil2)
                tDelaySSA = simulate_system(N=N, Tend = Tend[ni]/3, method = 'DelaySSA', plot = False, constant_T = constant_Tcells)/Tend[ni]*100*3
                print('  DelaySSA:', tDelaySSA)
                
                time_REGIR.append(tREGIR)
                time_Gil.append(tGil)
                time_DSSA.append(tDelaySSA)
            
            elif N<Nlim:
                tREGIR = simulate_system(N=N, Tend = Tend[ni], method = 'REGIR', plot = False, constant_T = constant_Tcells)/Tend[ni]*100
                print('  REGIR:', tREGIR)
                tGil = simulate_system(N=N, Tend = Tend[ni], method = 'Gillespie', plot = False, constant_T = constant_Tcells)/Tend[ni]*100
                print('  Gillespie:', tGil)
                
                time_REGIR.append(tREGIR)
                time_Gil.append(tGil)
                time_DSSA.append(np.nan)
            else:
                tREGIR = simulate_system(N=N, Tend = Tend[ni], method = 'REGIR', plot = False, constant_T = constant_Tcells)/Tend[ni]*100
                print('  REGIR:', tREGIR)
                time_REGIR.append(tREGIR)
                time_Gil.append(np.nan)
                time_DSSA.append(np.nan)
                
                
        np.save('time_REGIR.npy', time_REGIR)
        np.save('time_Gil.npy', time_Gil)
        np.save('time_DSSA.npy', time_DSSA)
    
    else:
        time_REGIR = np.load('time_REGIR.npy')
        time_Gil = np.load('time_Gil.npy')
        time_DSSA = np.load('time_DSSA.npy')
        
        
    #manually adjust here:
    time_Gil = np.array(time_Gil)
    time_Gil = time_Gil/1.32
    time_REGIR = np.array(time_REGIR)
    time_DSSA = np.array(time_DSSA)
    time_DSSA = time_DSSA/1.32
    
    
    #plot fitted data here
    fitted_REGIR = fit_log_line(Nlist,time_REGIR)
    fitted_Gil = fit_log_line(Nlist[Nlist<Nlim][-3:],time_Gil[Nlist<Nlim][-3:])
    fitted_DSSA = fit_log_line(Nlist[Nlist<NlimSSA][-9:-1],time_DSSA[Nlist<NlimSSA][-9:-1])
    
    
    plt.figure(figsize = (7,4))
    plt.scatter(Nlist, time_REGIR, s=60, color = 'red', label = r'REGIR')
    plt.plot(Nlist, fitted_REGIR, lw=2, color = 'red', ls = '--')
    plt.scatter(Nlist, time_Gil, s=60, color = 'black', label = r'SG')
    plt.scatter(Nlist, time_DSSA, s=60, color = 'blue', label = r'DelaySSA')
    #plt.fill_between(Nlist[-9:-1]/62, fitted_DSSA*0.5, fitted_DSSA*1.7, alpha = 0.15, color = 'red')
    #plt.fill_between(Nlist[:4], time_DSSA[:4]*0.5, time_DSSA[:4]*1.6, alpha = 0.15, color = 'red')
    plt.fill_between(Nlist, time_DSSA*0.6, time_DSSA*1.7, alpha = 0.15, color = 'blue')
    plt.fill_between(Nlist, time_Gil*0.6, time_Gil*1.7, alpha = 0.15, color = 'black')
    plt.fill_between(Nlist, fitted_REGIR*0.6, fitted_REGIR*1.7, alpha = 0.15, color = 'red')
    #plt.plot(Nlist, fitted_Gil, lw=2, color = 'green', ls = '--')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Number of B cells')
    plt.ylabel('Mean time / simulation [s]')
    plt.ylim(1e-3,1.5e3)
    plt.xlim(xmin = 2)
    plt.xlim(xmax = 2.5*1e5)
    #plt.xlim(8e-1,1e5)
    #plt.xlim(5e0,3e4)
    plt.legend(loc = 'lower right', prop={"size":11})
    plt.show()
        
        
    
    
def find_rank(sorted_dict, value):
    #Rutrn the index value of the last item with value > val in the sorted dict
    #sorted dict is a container from the library: from sortedcontainers import SortedDict
    index = sorted_dict.bisect_left(value)
    return index
    



def affinity_function(a0=1):
    new_val = a0 + (random.random())-0.45
    return new_val



def simulate_system(N = 100, Tend = 20, method = 'REGIR', plot = True, constant_T = True):
    
    #initialize populations   
    sorted_Bcells = SortedDict()
    bound_Bcells = list()
    unbound_Bcells = list()
    Population = dict()
    NB = N
    NBT = int(N/10)
    if constant_T:
        NBT = 10
    else:
        NBT = int(N/10)
    
    Population['B'] = NB
    Population['BT'] = NBT
    
    #print(Population)
    
    affinity_values = dict()
    aff_unbound = SortedDict()
    for i in range(NB + NBT+1):
        affinity_values[i] = affinity_function()
    for i in range(NB):
        aff_unbound[-affinity_values[i]] = i
    
    mean_affinity1 = np.ones(timepoints+1)
    mean_affinity2 = np.ones(timepoints+1)
    
    for i in range(NB):
        unbound_Bcells.append(i)
    for i in range(int(NBT)):
        bound_Bcells.append(NB+1)
    
    
    def perform_reaction(mu, reactantIDs):
        if mu == 0: #affinity change
            reactantID = reactantIDs['B']
            a0 = affinity_values[reactantID]
            new_affinity = affinity_function(a0)
            affinity_values[reactantID] = new_affinity
            
            if 'sorted' in method:
                aff_unbound.pop(-a0)
                aff_unbound[-new_affinity] = reactantID
            
        elif mu == 1: #Bcell switch
            reactantID1 = reactantIDs['B']
            reactantID2 = reactantIDs['BT']
            bound_Bcells.remove(reactantID2)
            bound_Bcells.append(reactantID1)
            unbound_Bcells.remove(reactantID1)
            unbound_Bcells.append(reactantID2)
            
            if 'sorted' in method:
                aff_unbound.pop(-affinity_values[reactantID1])
                aff_unbound[-affinity_values[reactantID2]] = reactantID2
            
            
        
        
    def simulate_REGIR(plot = True):
        
        def compute_propensities():
            a1 = NB*rB
            a2 = NB*NBT*rBT
            return np.array([a1,a2])
        
        
        
        t_start = datetime.now()        

        t = 0
        ti = 0
        while t < Tend: #Monte Carlo step
            
            if t >= ti*Tend/timepoints:#record populations
                ti = int(t/Tend*(timepoints))+1
                mean_affinity1[ti] = np.mean([affinity_values[i] for i in unbound_Bcells])
                mean_affinity2[ti] = np.mean([affinity_values[i] for i in bound_Bcells])
                
        
            a_array = compute_propensities()
            a0 = np.sum(a_array)
            
            if a0 == 0: #if propensities are zero, quickly end the simulation
                t += Tend/timepoints/2
    
            else: 
                #2 ----- Generate random time step (exponential distribution)
                r1 = random.random()
                tau = 1/a0*log(1/r1)
                t += tau
                
                #3 ----- Chose the reaction mu that will occurs (depends on propensities)
                r2 = random.random()
                mu = 0
                p_sum = 0.0
                while p_sum < r2*a0:
                    p_sum += a_array[mu]
                    mu += 1
                mu = mu - 1 
                    
                
                #4 ----- Perform the reaction
                if mu == 0:
                    reactantID = unbound_Bcells[random.randint(0,NB-1)]
                    reactantIDs = {'B': reactantID}
                    perform_reaction(mu, reactantIDs)   
                else:
                    reactantID1 = unbound_Bcells[random.randint(0,NB-1)]
                    reactantID2 = bound_Bcells[random.randint(0,NBT-1)]
                    reactantIDs = {'B': reactantID1, 'BT':reactantID2}
                    
                    if (affinity_values[reactantID1] > affinity_values[reactantID2]):
                        perform_reaction(mu, reactantIDs)   
                
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        final_time = t_elapsed.total_seconds()
                
                
        if plot:
            time_array = np.linspace(0,Tend, timepoints+1)
            plt.plot(time_array, mean_affinity1)
            plt.plot(time_array, mean_affinity2)
            plt.show()
                
        
        return final_time
    
    
    
    
    def simulate_Gillespie_sorted_array(plot = True):
        
        #Todo, sorted dict of affinity values in unbound dict. Not sure how to get it though
        
       
        
        
        def compute_propensities_and_get_IDs():
            a1 = NB*rB
            indiv_prop = []
            allowed_reacIDs = []
            for reactantID in bound_Bcells:
                aff = affinity_values[reactantID]
                rank = find_rank(aff_unbound, -aff)
                nPairs = rank
                allowed_reac = np.array([ aff_unbound.peekitem(i)[1] for i in range(rank)])
                ak = nPairs*rBT
                indiv_prop.append(ak)
                allowed_reacIDs.append(allowed_reac)
            a_all = [a1] + indiv_prop
            allow_reac = [unbound_Bcells] + allowed_reacIDs
            
            
            
            
            return np.array(a_all), allow_reac
        
        
        t_start = datetime.now()        

        t = 0
        ti = 0
        while t < Tend: #Monte Carlo step
            
            if t >= ti*Tend/timepoints:#record populations
                ti = int(t/Tend*(timepoints))+1
                mean_affinity1[ti] = np.mean([affinity_values[i] for i in unbound_Bcells])
                mean_affinity2[ti] = np.mean([affinity_values[i] for i in bound_Bcells])
                
        
            a_array, allowed_reacIDs = compute_propensities_and_get_IDs()
            a0 = np.sum(a_array)
            
            if a0 == 0: #if propensities are zero, quickly end the simulation
                t += Tend/timepoints/2
    
            else: 
                #2 ----- Generate random time step (exponential distribution)
                r1 = random.random()
                tau = 1/a0*log(1/r1)
                t += tau
                
                #3 ----- Chose the reaction mu that will occurs (depends on propensities)
                r2 = random.random()
                mu = 0
                p_sum = 0.0
                while p_sum < r2*a0:
                    p_sum += a_array[mu]
                    mu += 1
                mu = mu - 1 
                    
                
                #4 ----- Perform the reaction
                if mu == 0:
                    reactantID = unbound_Bcells[random.randint(0,NB-1)]
                    reactantIDs = {'B': reactantID}
                    perform_reaction(mu, reactantIDs)   
                else:
                    reacIDs = allowed_reacIDs[mu]
                    Ncandidates = len(reacIDs)
                    reactantID1 = reacIDs[random.randint(0,Ncandidates-1)]
                    reactantID2 = bound_Bcells[mu-1]
                    
                    reactantIDs = {'B': reactantID1, 'BT':reactantID2}
                    perform_reaction(mu, reactantIDs)   
                
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        final_time = t_elapsed.total_seconds()
                
                
        if plot:
            plt.plot(mean_affinity1)
            plt.plot(mean_affinity2)
            plt.show()
                
        
        return final_time
            
            
            
                
    
    
    
    
    def simulate_Gillespie(plot = True):
        
        
        def compute_propensities_and_get_IDs():
            a1 = NB*rB
            aff_unbound =  np.array([affinity_values[i] for i in unbound_Bcells])
            
            indiv_prop = []
            allowed_reacIDs_index = []
            for reactantID in bound_Bcells:
                aff = affinity_values[reactantID]
                nwhere = np.where(aff < aff_unbound)[0]
                nPairs = len(nwhere)
                ak = nPairs*rBT
                indiv_prop.append(ak)
                #allowed_reacIDs.append([unbound_Bcells[i] for i in nwhere])
                allowed_reacIDs_index.append(nwhere)
            a_all = [a1] + indiv_prop
            allow_reac_index = allowed_reacIDs_index
            
            
            return np.array(a_all), allow_reac_index
        
        
        t_start = datetime.now()        

        t = 0
        ti = 0
        while t < Tend: #Monte Carlo step
            
            if t >= ti*Tend/timepoints:#record populations
                ti = int(t/Tend*(timepoints))+1
                mean_affinity1[ti] = np.mean([affinity_values[i] for i in unbound_Bcells])
                mean_affinity2[ti] = np.mean([affinity_values[i] for i in bound_Bcells])
                
        
            a_array, allowed_reacIDs_index = compute_propensities_and_get_IDs()
            a0 = np.sum(a_array)
            
            if a0 == 0: #if propensities are zero, quickly end the simulation
                t += Tend/timepoints/2
    
            else: 
                #2 ----- Generate random time step (exponential distribution)
                r1 = random.random()
                tau = 1/a0*log(1/r1)
                t += tau
                
                #3 ----- Chose the reaction mu that will occurs (depends on propensities)
                r2 = random.random()
                mu = 0
                p_sum = 0.0
                while p_sum < r2*a0:
                    p_sum += a_array[mu]
                    mu += 1
                mu = mu - 1 
                    
                
                #4 ----- Perform the reaction
                if mu == 0:
                    reactantID = unbound_Bcells[random.randint(0,NB-1)]
                    reactantIDs = {'B': reactantID}
                    perform_reaction(mu, reactantIDs)   
                else:                   
                    reacIDs_index = allowed_reacIDs_index[mu-1]
                    Ncandidates = len(reacIDs_index)
                    reactantID1 = unbound_Bcells[reacIDs_index[random.randint(0,Ncandidates-1)]]
                    reactantID2 = bound_Bcells[mu-1]
                    
                    reactantIDs = {'B': reactantID1, 'BT':reactantID2}
                    perform_reaction(mu, reactantIDs)   
                
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        final_time = t_elapsed.total_seconds()
                
                
        if plot:
            plt.plot(mean_affinity1)
            plt.plot(mean_affinity2)
            plt.show()
                
        
        return final_time
    
    
    def simulate_DelaySSA(plot = True):
        
        
        
        delayed_reactions = SortedDict()
        delayed_reactions[1000000000000000000000000000] = [-1,-1]
        id_delay = dict()
        
        id_delay[0] = dict()
        id_delay[1] = dict()
        
        
        def compute_propensities_and_get_IDs():
            a1 = NB*rB
            aff_unbound =  np.array([affinity_values[i] for i in unbound_Bcells])
            
            indiv_prop = []
            allowed_reacIDs_index = []
            for reactantID in bound_Bcells:
                aff = affinity_values[reactantID]
                nwhere = np.where(aff < aff_unbound)[0]
                nPairs = len(nwhere)
                ak = nPairs*rBT
                indiv_prop.append(ak)
                #allowed_reacIDs.append([unbound_Bcells[i] for i in nwhere])
                allowed_reacIDs_index.append(nwhere)
            a_all = [a1] + indiv_prop
            allow_reac_index = allowed_reacIDs_index
            
            
            return np.array(a_all), allow_reac_index
        
        
        t_start = datetime.now()   
        
        
        t = 0
        
        a_array, allowed_reacIDs_index = compute_propensities_and_get_IDs()
        #print(allowed_reacIDs_index)
        for product_id in unbound_Bcells:
            #print(product_id)
            if product_id in allowed_reacIDs_index[0]:
                t_delay = generate_delay_from_distribution(rBT, 1, 'weibull')
                delayed_reactions[t + t_delay] = [1, product_id]
                id_delay[1][product_id] = t + t_delay
                
        for product_id in unbound_Bcells:
            t_delay = generate_delay_from_distribution(rB, 1, 'weibull')
            delayed_reactions[t + t_delay] = [0, product_id]
            id_delay[0][product_id] = t + t_delay


        
        ti = 0
        while t < Tend:
            
            a0 = np.sum(a_array)
            r1 = random.random()
            tau = 1/a0*log(1/r1)
            
            if t >= ti*Tend/timepoints:#record populations
                ti = int(t/Tend*(timepoints))+1
                mean_affinity1[ti] = np.mean([affinity_values[i] for i in unbound_Bcells])
                mean_affinity2[ti] = np.mean([affinity_values[i] for i in bound_Bcells])
                
                
            #2 ----- Generate random time step (exponential distribution)           
            t_delay_min, (mu, reactantID) = next(iter(delayed_reactions.items()))
            #print(t)
            t = t + tau
            #t = t_delay_min + tau/2
            
            
            #Here need to update the DelaySSA queu
            if mu == 0: #B cell affinity change
                reactantIDs = {'B': reactantID}
                perform_reaction(mu, reactantIDs) 
                #affinity has changed after this reaction was performed
                
                a_array, allowed_reacIDs_index = compute_propensities_and_get_IDs()
                
                #Remove the current delay and all other processes involved
                id_list = list(id_delay[1].keys())
                for reactant_id in id_list: #need to remove all other B cells that are no longer allowed to react
                    if reactant_id not in allowed_reacIDs_index[0]:
                        time_delay = id_delay[1][reactant_id]
                        delayed_reactions.pop(time_delay)
                        id_delay[1].pop(reactant_id)
                
                #Add time for all the new processes involved
                for product_id in unbound_Bcells: #the new Bcell has a new affinity, need to update the propensities for all bcells as they now have a new competitor
                    if product_id in allowed_reacIDs_index[0]:
                        if product_id not in id_delay[1]:
                            t_delay = generate_delay_from_distribution(rBT, 1, 'weibull')
                            delayed_reactions[t + t_delay] = [0, product_id]
                            id_delay[1][product_id] = t + t_delay
                
                
            
            else: #B cell switch
                
                a_array, allowed_reacIDs_index = compute_propensities_and_get_IDs()
                
                #reacIDs_index = allowed_reacIDs_index[mu-1]
                #Ncandidates = len(reacIDs_index)
                reactantID1 = reactantID
                reactantID2 = bound_Bcells[mu-1]
                
                reactantIDs = {'B': reactantID1, 'BT':reactantID2}
                perform_reaction(mu, reactantIDs)   
                
                
                #Remove the current delay and all other processes involved
                id_list = list(id_delay[1].keys())
                for reactant_id in id_list: #need to remove all other B cells that are no longer allowed to react
                    if reactant_id not in allowed_reacIDs_index[0]:
                        time_delay = id_delay[1][reactant_id]
                        delayed_reactions.pop(time_delay)
                        id_delay[1].pop(reactant_id)
                
                #Add time for all the new processes involved
                for product_id in unbound_Bcells: #the new Bcell has a new affinity, need to update the propensities for all bcells as they now have a new competitor
                    if product_id in allowed_reacIDs_index[0]:
                        if product_id not in id_delay[1]:
                            t_delay = generate_delay_from_distribution(rBT, 1, 'weibull')
                            delayed_reactions[t + t_delay] = [1, product_id]
                            id_delay[1][product_id] = t + t_delay
                            
                            
                time_delay = id_delay[0][reactantID1] #reactant1 is now bound
                delayed_reactions.pop(time_delay)
                id_delay[0].pop(reactantID1)
                
                t_delay = generate_delay_from_distribution(rB, 1, 'weibull') #reactant 2 is now unbound
                delayed_reactions[t + t_delay] = [1, reactantID2]
                id_delay[1][reactantID2] = t + t_delay
                            
                            
                #update the 
                
                
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        final_time = t_elapsed.total_seconds()
                
                
        if plot:
            plt.plot(mean_affinity1)
            plt.plot(mean_affinity2)
            plt.show()
                
        
        return final_time
    
    
    
    if method == 'REGIR':
        return simulate_REGIR(plot = plot)
    elif method == 'Gillespie':
        return simulate_Gillespie(plot = plot)
    elif method == 'DelaySSA':
        return simulate_DelaySSA(plot = plot)
    else:
        return simulate_Gillespie_sorted_array(plot = plot)
        
    
    
    
def fit_log_line(x0,y0):
    
    y0 = np.array(y0)
    x0 = np.array(x0)
    
    x1 = x0[~np.isnan(y0)]
    y1 = y0[~np.isnan(y0)]
    
    x = np.log(x1)
    y = np.log(y1)
    m, b = np. polyfit(x, y, 1)
    print(m)
    
    fitted_y = np.exp(m*np.log(x0) + b)
    return fitted_y



def generate_delay_from_distribution(rate, shape_param, distribution, N_ = None):
    
    if N_ is None:
        N = 1
    else:
        N = N_
    
    if distribution.lower() in ['exponential', 'exp']:
        delay = stat.expon.rvs(loc = 0, scale = 1/rate, size=N)
            
    elif distribution.lower() in ['weibull','weib']:
        alpha = shape_param
        if shape_param == 1:
            delay = stat.expon.rvs(loc = 0, scale = 1/rate, size=N)
        else:
            beta = (alpha) * (rate * gamma((alpha + 1)/(alpha)))**(alpha)
            delay = stat.weibull_min.rvs(alpha, loc = 0, scale = (alpha/beta)**(1/alpha), size=N)  #alpha/beta or beta/alpha
            
    elif distribution.lower() in ['gaussian', 'normal', 'norm']:
        mu = 1/rate
        sigma = mu*shape_param
        delay = stat.norm.rvs(loc = mu, scale = sigma, size=N)
        
    elif distribution.lower() in ['gamma','gam']:
        alpha = shape_param
        beta = alpha*rate
        delay = stat.gamma.rvs(alpha, loc = 0, scale =  1/beta, size=N)
        
        
    elif distribution.lower() in ['cauchy', 'cau']:
        gam = shape_param
        mu = 1/rate
        delay = stat.cauchy.rvs(loc = mu, scale = gam, size=N) #check gam or 1/gam
        
        
    elif distribution.lower() in ['pareto', 'par']:
        alpha = shape_param
        mu = 1/rate* 2**(-1/alpha)
        delay = stat.pareto.rvs(alpha, scale = mu, size=N) #check gam or 1/gam
    
    if N_ is None:
        return delay[0]
    else:
        return delay  
    
    
    
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})
    main()
        