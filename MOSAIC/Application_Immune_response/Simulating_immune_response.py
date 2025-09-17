




import numpy as np
import pandas as pd
import random
import sys
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stat
from math import gamma
import pickle
import seaborn as sns

import lib_REGIR_with_rate_function as gil

plt.rcParams.update({'font.size': 19})



"""
    # ------ intercellular GC dynamic related parameters ------------------------    
    r_activation        = 4.1         #0 -> CB GC seeding rate
    r_division          = 0.134       #CB -> 2CB Division rate
    r_migration         = 0.4         #CB -> CC migration rate
    r_apoptosis         = 0.084       #CC -> 0 apoptosis rate
    r_recirculate       = 3.75        #CCsel -> CB recirculation rate
    r_unbinding         = 2.000       #CCTC -> CC + TC       
    r_FDCencounter      = 1           #CC antigen uptake (should be 10)
    r_TCencounter       = 10          #CC + TC = [CCTC] Tcell encounter rate
    rhoTC               = 1/42        #TC:CC ratio
    p                   = 0.7         #CCsel recirculation probability
    r_exit = r_recirculate/p - r_recirculate  
"""






Ninit = 1000
Nsim = 1


rdiv = 1
rbind = 0.1
rdeath = 0.1
rinteract = 1


N_init = {'B':100, 'T':10, 'Bdiv':0, '[BT]':0}

class param:
    Tend = 100
    unit = 'h'
    N_simulations = 1          #The simulation results should be averaged over many trials
    timepoints = 100            #Number of timepoints to record (make surethat this number isnt too big)
    
    
    

def affinity_function(a0=0.5, Nstep = 10):
    new_val = a0 + (random.random() - a0)/Nstep
    return new_val


def main():    
    
    
    a0_values = np.linspace(0, 1, 50)
    prob_increase = 1 - a0_values

    plt.figure(figsize=(8, 5))
    plt.plot(a0_values, prob_increase, linestyle='--', color = 'black', lw = 2)
    plt.xlabel("B cell initial affinity")
    plt.ylabel("Probability of affinity increase")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    
    sys.exit()

    
    
    
    def perform_reaction_division(G_simul, forced_ID = None): #Bdiv -> B + B
        
        if forced_ID is None:
            index, reactant = G_simul.get_random_element(G_simul.reactant_list['Bdiv'])
            reactantID = reactant.id
        else:
            index = G_simul.find_in_reactant_list(G_simul.reactant_list['Bdiv'], forced_ID) #Really need to optimize this function
            reactantID = forced_ID
                    
        #update populations
        product1 = gil.Reactant()
        product2 = gil.Reactant()
        G_simul.reactant_list['Bdiv'].pop(index)
        G_simul.reactant_population['Bdiv'] -= 1
        G_simul.reactant_list['B'].append(product1)
        G_simul.reactant_list['B'].append(product2)
        G_simul.reactant_population['B'] += 2
        
        #Draw new affinity values
        a0 = G_simul.Bcell_affinity[reactantID]
        new_affinity1 = affinity_function(a0)
        new_affinity2 = affinity_function(a0)
        G_simul.Bcell_affinity[product1.id] = new_affinity1
        G_simul.Bcell_affinity[product2.id] = new_affinity2
        
        cloneID = G_simul.Ancestor[reactantID]
        G_simul.Ancestor[product1.id] = cloneID
        G_simul.Ancestor[product2.id] = cloneID
        
        product_dict = dict()
        reactant_dict = dict()
        reactant_dict[reactantID] = 'Bdiv'
        product_dict[product1.id] = 'B'
        product_dict[product2.id] = 'B'
    
        return reactant_dict, product_dict
        
        
    def perform_reaction_TBexchange(G_simul, forced_ID = None): #[B1T] + B2 → [B2T] + B1
        indexB, reactantB = G_simul.get_random_element(G_simul.reactant_list['B'])
        indexBT, reactantBT = G_simul.get_random_element(G_simul.reactant_list['[BT]'])
        affinity1 = G_simul.Bcell_affinity[reactantB.id]
        affinity2 = G_simul.Bcell_affinity[reactantBT.id]
        
        
        product_dict = dict()
        reactant_dict = dict()
        if affinity1 > affinity2: #only perform the reaction  if affinity1 is higher than affinity2
            productB = gil.Reactant(ID = reactantBT.id)
            productBT = gil.Reactant(ID = reactantB.id)
            G_simul.reactant_list['[BT]'].pop(indexBT)
            G_simul.reactant_list['B'].pop(indexB)
            G_simul.reactant_list['B'].append(productB)
            G_simul.reactant_list['[BT]'].append(productBT)
            
            reactant_dict[reactantB.id] = 'B'
            reactant_dict[reactantBT.id] = '[BT]'
            product_dict[productB.id] = 'B'
            product_dict[productBT.id] = '[BT]'
            
        return reactant_dict, product_dict
            
    def perform_reaction_TBbinding(G_simul, forced_ID = None): #B + T -> [BT]
        indexB, reactantB = G_simul.get_random_element(G_simul.reactant_list['B'])
        indexT, reactantT = G_simul.get_random_element(G_simul.reactant_list['T'])
        
        productBT = gil.Reactant(ID = reactantB.id)
        G_simul.reactant_list['B'].pop(indexB)
        G_simul.reactant_list['T'].pop(indexT)
        G_simul.reactant_list['[BT]'].append(productBT)
        
        G_simul.reactant_population['B'] -= 1
        G_simul.reactant_population['T'] -= 1
        G_simul.reactant_population['[BT]'] += 1
        
        product_dict = dict()
        reactant_dict = dict()
        reactant_dict[reactantB.id] = 'B'
        reactant_dict[reactantT.id] = 'T'
        product_dict[productBT.id] = '[BT]'
    
        return reactant_dict, product_dict
            
            
    def perform_reaction_TBunbinding(G_simul, forced_ID = None): #[BT] -> Bdiv + T
        
        if forced_ID is None:
            index, reactant = G_simul.get_random_element(G_simul.reactant_list['[BT]'])
            reactantID = reactant.id
        else:
            index = G_simul.find_in_reactant_list(G_simul.reactant_list['[BT]'], forced_ID)
            reactantID = forced_ID
            
        productB = gil.Reactant(ID = reactantID)
        productT = gil.Reactant()
        
        G_simul.reactant_list['[BT]'].pop(index)
        G_simul.reactant_list['T'].append(productT)
        G_simul.reactant_list['Bdiv'].append(productB)
        
        G_simul.reactant_population['[BT]'] -= 1
        G_simul.reactant_population['T'] += 1
        G_simul.reactant_population['Bdiv'] += 1
        
        product_dict = dict()
        reactant_dict = dict()
        reactant_dict[reactantID] = '[BT]'
        product_dict[productB.id] = 'Bdiv'
        product_dict[productT.id] = 'T'
    
        return reactant_dict, product_dict
                              

    

    all_max_pop = []
    for ni in range(Nsim):    
        print('Simulation %s/%s' % (ni+1, Nsim))
        reaction_channel_list = []
        
        
        channel = gil.Reaction_channel(param,
                                       rate = rdeath, 
                                       shape_param = 0.1, 
                                       distribution = 'logNormal', 
                                       name = 'B -> 0', 
                                       precompute_delay = True, 
                                       reactants = ['B'], 
                                       products = []
                                       )
        reaction_channel_list.append(channel)
        
        
        channel = gil.Reaction_channel(param,
                                       rate = rdiv, 
                                       shape_param = 0.1, 
                                       distribution = 'logNormal', 
                                       name = 'Bdiv -> B + B', 
                                       precompute_delay = True, 
                                       perform_reaction_custom = perform_reaction_division,
                                       reactants = ['Bdiv'], 
                                       products = ['B', 'B']
                                       )
        reaction_channel_list.append(channel)
        
        
        
        channel = gil.Reaction_channel(param,
                                       rate = rbind, 
                                       distribution = 'exponential', 
                                       name = 'B + T -> [BT]', 
                                       perform_reaction_custom = perform_reaction_TBbinding,
                                       reactants = ['B', 'T'], 
                                       products = ['[BT]']
                                       )
        reaction_channel_list.append(channel)
        
        
    
        channel = gil.Reaction_channel(param,
                                       rate = rinteract,
                                       shape_param = 4,
                                       distribution = 'Gamma', 
                                       name = '[BT] -> Bdiv + T', 
                                       perform_reaction_custom = perform_reaction_TBunbinding,
                                       precompute_delay = True, 
                                       reactants = ['[BT]'], 
                                       products = ['Bdiv', 'T']
                                       )
        reaction_channel_list.append(channel)
        
        
        
        channel = gil.Reaction_channel(param,
                                       rate = rbind,
                                       shape_param = 4,
                                       distribution = 'exponential', 
                                       perform_reaction_custom = perform_reaction_TBexchange,
                                       name = '[B1T] + B2 → [B2T] + B1', 
                                       reactants = ['B', '[BT]'], 
                                       products = ['B', '[BT]']
                                       )
        reaction_channel_list.append(channel)
        
        
        
        
        
        #initialise the Gillespie simulation
        G_simul = gil.Gillespie_simulation(N_init,param, reaction_channel_list, print_warnings = True)
        
        #Here you can create and initialise any map that you will use in your function below
        G_simul.Bcell_affinity = dict()
        G_simul.Ancestor = dict()
        G_simul.ancestor_dict = dict()
        
        #run multiple Gillespie simulationand average them
        G_simul.run_simulations(param.Tend, delay_method = 'Standard')
        
        #G_simul.plot_inter_event_time_distribution(plot_fitted = False)
        #G_simul.plot_populations(reactant_list = ['B'])
        
        
        
        
        #B cell affinity evolving through time:
        mean_affinity = np.array([np.mean(G_simul.ancestor_dict[ti][1]) for ti in range(1,param.timepoints)])*1.3
        std_affinity = np.array([np.std(G_simul.ancestor_dict[ti][1]) for ti in range(1,param.timepoints)])*1.3
        time = np.arange(1,param.timepoints)
        plt.plot(time, mean_affinity)
        plt.fill_between(time, mean_affinity-std_affinity, mean_affinity+std_affinity, alpha = 0.3)
        plt.xlim(0,50)
        plt.xlabel('Days after immunization')
        plt.ylabel('B-cell affinity')
        plt.show()
        
        
        #B cell clones population
        a_list = np.sort(list(set(G_simul.ancestor_dict[1][0])))
        for a in a_list:
            pop = []
            norm_pop = []
            affinity = []
            ti = 1
            while ti <= param.timepoints:
                nwhere = np.where(a == G_simul.ancestor_dict[ti][0])[0]
                if len(nwhere) > 0:
                    pop.append(len(nwhere))
                    norm_pop.append(len(nwhere)/len(G_simul.ancestor_dict[ti][0]))
                    affinity.append(np.array(G_simul.ancestor_dict[ti][1])[nwhere])
                ti += 1
                
            x_array = np.arange(param.timepoints)/2+6
            x_array = x_array[:len(norm_pop)]
            plt.plot(x_array, np.array(norm_pop), label = 'Clone %s' % (a+1))
            
        plt.xlabel('Days after immunization')
        plt.ylabel('Clone dominance')
        plt.legend(fontsize = 10)
        plt.show()
        
        
        #B cell clones affinity
        a_list = np.sort(list(set(G_simul.ancestor_dict[1][0])))
        for a in a_list:
            pop = []
            affinity = []
            ti = 1
            while ti <= param.timepoints:
                nwhere = np.where(a == G_simul.ancestor_dict[ti][0])[0]
                if len(nwhere) > 0:
                    pop.append(len(nwhere))
                    affinity.append(np.mean(np.array(G_simul.ancestor_dict[ti][1])[nwhere]))
                ti += 1
                
            x_array = np.arange(param.timepoints)/2+6
            x_array = x_array[:len(affinity)]
            plt.plot(x_array, np.array(affinity), label = 'Clone %s' % (a+1))
            
        plt.xlabel('Days after immunization')
        plt.ylabel('Clone Affinity')
        plt.legend(fontsize = 10)
        plt.show()
    
                
        #Dominance score
        max_pop = []
        ti = 1
        while ti <= param.timepoints:
            norm_pop = []
            for a in a_list:
                nwhere = np.where(a == G_simul.ancestor_dict[ti][0])[0]
                norm_pop.append(len(nwhere)/len(G_simul.ancestor_dict[ti][0]))
            max_pop.append(np.max(norm_pop))
            ti += 1
        max_pop = np.array(max_pop)
        
        
        all_max_pop.append(max_pop)
        
    
       
    filename = 'GC_diversity.xlsx'
    df = pd.read_excel(filename, sheet_name=['AID-Confetti','AID-Confetti B1-8'])
    dates = np.array([3,5,7,11,15,19,23])+5
    NDS_exp = [[] for i in range(len(dates))]
    
    days = np.array(df['AID-Confetti'].values[:,19][2:])+5
    NDS_ = np.array(df['AID-Confetti'].values[:,22][2:])
    DZdensity = np.array(df['AID-Confetti'].values[:,20][2:])
    nkeep = np.where(DZdensity >= 0.2)
    NDS_ = NDS_[nkeep]
    days = days[nkeep]
    
    for i in range(len(dates)):
        for j in np.where(days == dates[i]):
            NDS_exp[i].append(NDS_[j])
            
    
    for max_pop in all_max_pop:
        plt.plot(np.arange(param.timepoints)/2+6, max_pop, color = sns.color_palette()[0], alpha = 0.5, lw = 0.5)
    all_max_pop = np.array(all_max_pop)
    plt.plot(np.arange(param.timepoints)/2+6, all_max_pop.mean(axis=0), lw=2.5, color='blue', label = 'Av. simulation')
    plt.xlabel('Days after immunization')
    plt.ylabel('Dominance')   
    
    av = []
    for i in range(len(NDS_exp)):
        for j in range(len(NDS_exp[i][0])):
            plt.scatter(dates[i],NDS_exp[i][0][j], color = "black",s=12)
        avi = np.median(NDS_exp[i])
        plt.plot([dates[i]-0.6,dates[i]+0.6],[avi,avi], color = 'black', lw = 4)
        av.append(avi)
    plt.plot([50,53],[avi,avi], color = 'black', lw = 5, label = "[Tas & al., 2016]") #just for the label
    plt.legend(fontsize = 13)
    plt.ylim(0,1.1)
    plt.xlim(5,35)
    plt.show()
    
    
    
def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, 4)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)    
    
    
    
    
    
if __name__ == '__main__':
    main()