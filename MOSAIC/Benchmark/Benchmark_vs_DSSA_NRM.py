import numpy as np
import random
import scipy
import sys
import random as rd
import matplotlib.pyplot as plt
from datetime import datetime

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import MOSAIC as gil

from scipy import stats
plt.rcParams.update({'font.size': 18})

"""
    This implementation is using sorted containers, the equivalent of std::map in c++
    https://pypi.org/project/sortedcontainers/
    https://grantjenks.com/docs/sortedcontainers/sorteddict.html
    Elements are always sorted by keys, inserting or deleting an element is O(log(N))
    
    Setting a new (key,value) is O(log(N)), but in practice it takes 5 microseconds for N=10 and 10 microseconds for N=1e6
    This is a small difference and the update of this step is kinda negligible in the context of Gillespi implementation
"""
from sortedcontainers import SortedDict


k1 = 0.005  # Example rate constant for A -> B
k2 = 0.003  # Example rate constant for B -> A
alpha_1 = 6
alpha_2 = 6

Tmax = 100




def main():
    
    #-> plot time per reactions instead of total time
    
    # start with {A->B ; B->A}:
    # Both are with Gamma distribution
    reaction1 = ['A','B','Gamma']
    reaction2 = ['B','A','Gamma']
    reaction_list = [reaction1, reaction2]
    
    N_list = np.logspace(1,4.5, num = 10).astype(int)
    #N_list = [1000]
    
    #time_Next = []
    time_DSSA = []
    time_mosaic = []
    time_gil = []
    for N in N_list:
        print('Benchmaarking N_init = %s' % N)
        tgil, Nreactions_gil = simulate_gil(N, reaction_list)
        tmosaic, Nreactions_reg = simulate_MOSAIC(N, reaction_list)
        #tnext, Nreactions_next = simulate_Next_reaction(N, reaction_list)
        tdssa, Nreactions_dssa = simulate_DSSA(N, reaction_list)
        
        #time_Next.append(tnext/Nreactions_next)
        time_DSSA.append(tdssa/Nreactions_dssa)
        time_mosaic.append(tmosaic/Nreactions_reg)
        time_gil.append(tgil/Nreactions_gil)
        
        
        
    #time_Next = np.array(time_Next)
    time_DSSA = np.array(time_DSSA)
    time_mosaic = np.array(time_mosaic)
    time_gil = np.array(time_gil)

    
    plt.figure(figsize = (8,4))
    #plt.scatter(N_list, time_Next/10, s=60, color = 'green', label = r'Next reaction (gamma, $\alpha=6$)')
    plt.scatter(N_list, time_DSSA/10, s=60, color = 'red', label = r'DSSA (gamma, $\alpha=6$)')
    plt.scatter(N_list, time_mosaic/10, s=60, color = 'blue', label = r'MOSAIC (gamma, $\alpha=6$)')
    plt.scatter(N_list, time_gil/10, s=60, color = 'black', label = r'SG (Poisson)')
    
    
    #fitted_Next = fit_log_line(N_list,time_Next)
    fitted_DSSA = fit_log_line(N_list,time_DSSA)
    fitted_mosaic = fit_log_line(N_list,time_mosaic)
    fitted_gil = fit_log_line(N_list,time_gil)
    
    plt.plot(N_list, fitted_mosaic/10, lw=2, color = 'blue')
    #plt.plot(N_list, fitted_Next/10, lw=2, color = 'green')
    plt.plot(N_list, fitted_DSSA/10, lw=2, color = 'red')
    plt.plot(N_list, fitted_gil/10, lw=2, color = 'black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of reactants')
    plt.ylabel('Av. time / simulation [s]')
    #plt.ylim(1e-3,1e2)
    #plt.xlim(5e0,1e5)
    #plt.legend(loc = 'lower right', prop={"size":15})
    plt.show()
    
    
    
def simulate_DSSA(N_init, reaction_list): 
    
    NA = N_init
    NB = 0
    
    class param:
        Tend = 100
        unit = 'h'
        rd.seed(101)                #length of the simulation, in hours (41 days)
        N_simulations = 1         #The simulation results should be averaged over many trials
        timepoints = 100            #Number of timepoints to record (make surethat this number isnt too big)
    
    class reactant:
        def __init__(self, type_ = '', id_ = 0):
            self.time_delay_over = 0
            self.time_since_reacted = 0
            self.id = id_
            self.type = type_
            
    
    
    def generate_delay_from_distribution(rate, alpha):
        beta = alpha*rate
        delay = stats.gamma.rvs(alpha, loc = 0, scale =  1/beta, size=1)[0]
        return delay      
    
    #usefull simulation parameters
    all_inter_event_times_1 = []
    all_inter_event_times_2 = []
    popA = []
    popB = []
    time = []
    
    
    t_start = datetime.now()
    
    #initialize the simulation
    A_list = dict()
    all_delayed_reaction_1 = SortedDict()
    all_delayed_reaction_1[2*param.Tend] = 0
    
    all_delayed_reaction_2 = SortedDict()
    all_delayed_reaction_2[2*param.Tend] = 0
    

    for ni in range(NA):
        ri = reactant(type_ = 'A', id_ = ni)
        A_list[ni] = ri
        delay_i  = generate_delay_from_distribution(k1,alpha_1)
        all_delayed_reaction_1[delay_i] = ni
        
    
    def perform_reaction_1(j, NA, NB): #A -> B
        
        NA = NA - 1
        NB = NB + 1
        all_inter_event_times_1.append(t - A_list[j].time_since_reacted)
        #A_list.pop(j)
        delay  = generate_delay_from_distribution(k2,alpha_2)
        A_list[j].time_since_reacted = t
        A_list[j].time_delay_over = t + delay
        all_delayed_reaction_2[t+delay] = j  # THis step requires reordering of the array and thus should be O(logN)
        #keep in mind that this wont work if t+d is already in the std:map. However thats extremly unlikely and considered negligible
        popA.append(NA)
        popB.append(NB)
        time.append(t)
        
    def perform_reaction_2(j, NA, NB):  #B -> A
        
        NB -= 1
        NA += 1
        all_inter_event_times_2.append(t - A_list[j].time_since_reacted)
        #B_list.pop(j)
        delay  = generate_delay_from_distribution(k1,alpha_1)
        A_list[j].time_since_reacted = t
        A_list[j].time_delay_over = t + delay
        all_delayed_reaction_1[t+delay] = j  # THis step requires reordering of the array and thus should be O(logN)
        #keep in mind that this wont work if t+d is already in the std:map. However thats extremly unlikely and considered negligible
        popA.append(NA)
        popB.append(NB)
        time.append(t)
        
    #run the simulation
    t = 0
    rate_0 = 0.0000001
    Nreactions = 0
    while t < param.Tend:    
        more_delayed_recations = True
        while more_delayed_recations and t < param.Tend:
            
            a0 = NA*rate_0 #This will lead to a giant time step
            if a0 > 0:
                u = np.random.rand()
                Deltat = 1/a0 * np.log(1/u)
                
                t_delay_min1, mu1 = next(iter(all_delayed_reaction_1.items()))
                t_delay_min2, mu2 = next(iter(all_delayed_reaction_2.items()))
                
                if t_delay_min1 < t_delay_min2:
                    t_delay_min = t_delay_min1
                    k = mu1
                    reaction = 1
                else:
                   t_delay_min = t_delay_min2
                   reaction = 2       
                   k = mu2
                
                if t_delay_min < t + Deltat:
                    more_delayed_recations = True
                    t = t_delay_min
                    if reaction == 1:
                        perform_reaction_1(k, NA, NB)
                        all_delayed_reaction_1.pop(t_delay_min) #or popitem(0) if you know it's the first one
                        Nreactions += 1
                    
                    else:
                        perform_reaction_2(k, NA, NB)
                        all_delayed_reaction_2.pop(t_delay_min) #or popitem(0) if you know it's the first one
                        Nreactions += 1      
                        
                        
                else:
                    more_delayed_recations = False
                    t += Deltat
                    
                    popA.append(NA)
                    popB.append(NB)
                    time.append(t)
                
    t_end = datetime.now()
    t_elapsed = t_end - t_start
    total_time = t_elapsed.total_seconds()
    
    
    # Plotting
    plt.plot(time, popA, label='A')
    plt.plot(time, popB, label='B')
    plt.xlabel('Time')
    plt.ylabel('Number of Molecules')
    plt.title('Gillespie Simulation of A <-> B Reaction')
    plt.legend()
    plt.show()
    
    Nreactions /= param.N_simulations
    total_time /= param.N_simulations
    
    plt.hist(all_inter_event_times_1, bins = 30)
    plt.show()
    
    plt.hist(all_inter_event_times_2, bins = 30)
    plt.show()
    
    print('DSSA:', len(all_inter_event_times_1) + len(all_inter_event_times_2))
    
    return total_time, Nreactions




def simulate_Next_reaction(N_init, reaction_list): 
    
    
    
    """ Next reaction algorithm benchmarking
    [ref] Anderson, David F. "A modified next reaction method for simulating chemical systems with time
     dependent propensities and delays." The Journal of chemical physics 127.21 (2007): 214107.
    """
    
    
    
    NA = N_init
    NB = 0
    
    class param:
        Tend = 100
        unit = 'h'
        rd.seed(101)                #length of the simulation, in hours (41 days)
        N_simulations = 1         #The simulation results should be averaged over many trials
        timepoints = 100            #Number of timepoints to record (make surethat this number isnt too big)
    
    class reactant:
        def __init__(self, type_ = '', id_ = 0):
            self.time_delay_over = 0
            self.time_since_reacted = 0
            self.id = id_
            self.type = type_
    
    
    #usefull simulation parameters
    all_inter_event_times_1 = []
    all_inter_event_times_2 = []
    popA = []
    popB = []
    time = []
    
    
    t_start = datetime.now()
    
    
    
    #initialize the simulation
    A_list = dict()
    all_delayed_reaction_1 = SortedDict()
    all_delayed_reaction_1[2*param.Tend] = 0
    
    all_delayed_reaction_2 = SortedDict()
    all_delayed_reaction_2[2*param.Tend] = 0
    
    T_array = np.zeros(NA)
    a_array = np.ones(NA)*k1
    P_array = np.log(1/np.random.rand(NA))
    Delta_array = (P_array - T_array)/a_array
       
    for ni in range(NA):
        ri = reactant(type_ = 'A', id_ = ni)
        A_list[ni] = ri
        delay_i  = generate_delay_from_distribution(k1,alpha_1)
        all_delayed_reaction_1[delay_i] = ni
    
    def perform_reaction_1(j, NA, NB): #A -> B
        
        NA = NA - 1
        NB = NB + 1
        all_inter_event_times_1.append(t - A_list[j].time_since_reacted)
        #A_list.pop(j)
        delay  = generate_delay_from_distribution(k2,alpha_2)
        A_list[j].time_since_reacted = t
        A_list[j].time_delay_over = t + delay
        all_delayed_reaction_2[t+delay] = j  # THis step requires reordering of the array and thus should be O(logN)
        #keep in mind that this wont work if t+d is already in the std:map. However thats extremly unlikely and considered negligible
        popA.append(NA)
        popB.append(NB)
        time.append(t)
        
    def perform_reaction_2(j, NA, NB):  #B -> A
        
        NB -= 1
        NA += 1
        all_inter_event_times_2.append(t - A_list[j].time_since_reacted)
        #B_list.pop(j)
        delay  = generate_delay_from_distribution(k1,alpha_1)
        A_list[j].time_since_reacted = t
        A_list[j].time_delay_over = t + delay
        all_delayed_reaction_1[t+delay] = j  # THis step requires reordering of the array and thus should be O(logN)
        #keep in mind that this wont work if t+d is already in the std:map. However thats extremly unlikely and considered negligible
        popA.append(NA)
        popB.append(NB)
        time.append(t)
        
    
    #run the simulation
    t = 0    
    Nreactions = 0
    while t < param.Tend:   
        Delta_array = (P_array - T_array)/a_array
        t_delay_min1, mu1 = next(iter(all_delayed_reaction_1.items()))
        t_delay_min2, mu2 = next(iter(all_delayed_reaction_2.items()))
        
        if t_delay_min1 < t_delay_min2:
            t_delay_min = t_delay_min1
            k = mu1
            reaction = 1
        else:
           t_delay_min = t_delay_min2
           reaction = 2       
           k = mu2
        
        mu = np.argmin(Delta_array)
        Delta_min = Delta_array[mu]
        
        if t_delay_min - t <= Delta_min: #If there is a delayed reaction, chose it
            mu = k
            Delta_min = t_delay_min - t
        else:
            pass
        t += Delta_min
        T_array = T_array + a_array*Delta_min
        P_array[mu] = P_array[mu] + 1/np.log(1/np.random.rand())
        
        if reaction == 1:
            perform_reaction_1(mu, NA, NB)
            all_delayed_reaction_1.pop(t_delay_min) #or popitem(0) if you know it's the first one
            Nreactions += 1
        
        else:
            perform_reaction_2(mu, NA, NB)
            all_delayed_reaction_2.pop(t_delay_min) #or popitem(0) if you know it's the first one
            Nreactions += 1           
        
    
    
    
    
    t_end = datetime.now()
    t_elapsed = t_end - t_start
    total_time = t_elapsed.total_seconds()
    
    
    # Plotting
    plt.plot(time, popA, label='A')
    plt.plot(time, popB, label='B')
    plt.xlabel('Time')
    plt.ylabel('Number of Molecules')
    plt.title('Gillespie Simulation of A <-> B Reaction')
    plt.legend()
    plt.show()
    
    Nreactions /= param.N_simulations
    total_time /= param.N_simulations
    
    plt.hist(all_inter_event_times_1, bins = 30)
    plt.show()
    
    plt.hist(all_inter_event_times_2, bins = 30)
    plt.show()
    
    print('Next:', len(all_inter_event_times_1) + len(all_inter_event_times_2))
    
    return total_time, Nreactions
    
def simulate_MOSAIC(NA, reaction_list):    
    
    
    class param:
        Tend = Tmax
        unit = 'h'
        rd.seed(100)                #length of the simulation, in hours (41 days)
        if NA < 1000:
            N_simulations = 5         #The simulation results should be averaged over many trials
        else:
            N_simulations = 1
        timepoints = 100            #Number of timepoints to record (make surethat this number isnt too big)
    
    #initialise reactants
    N_init = dict()
    N_init['A'] = NA
    N_init['B'] = 0
        
    #initialise reaction channels
    react1 = gil.Reaction_channel(param,rate=k1, shape_param=alpha_1, distribution = 'Gamma', name='A -> B')
    react1.reactants = ['A']
    react1.products = ['B']
        
    react2 = gil.Reaction_channel(param,rate=k2, shape_param=alpha_2, distribution = 'Gamma', name='B -> A')
    react2.reactants = ['B']
    react2.products = ['A']

    reaction_channel_list = [react1,react2]
    
    G_simul = gil.Gillespie_simulation(N_init,param)
    G_simul.reaction_channel_list = reaction_channel_list
    print(G_simul)
    
    t_start = datetime.now()
    
    #run multiple Gillespie simulationand average them
    population = G_simul.run_simulations(param.Tend)
    
    t_end = datetime.now()
    t_elapsed = t_end - t_start
    total_time = t_elapsed.total_seconds()
    
    
    #G_simul.plot_inter_event_time_distribution(color_list = ['orange','royalblue'])
    G_simul.plot_populations(color_list = ['orange','royalblue', 'gray'], figsize = (5,4))
    
    
    Nreactions = len(reaction_channel_list[0].wait_times) + len(reaction_channel_list[1].wait_times)
    
    Nreactions /= param.N_simulations
    total_time /= param.N_simulations
    
    return total_time, Nreactions
    

def simulate_gil(N_init, reaction_list): 
    
    
    t_start = datetime.now()
    
    # Define rate constants
    k1 = 0.005  # Example rate constant for A -> B
    k2 = 0.003  # Example rate constant for B -> A
    
    # Initial number of molecules
    NA = N_init  # Number of A molecules
    NB = 0    # Number of B molecules
    
    
    # Simulation
    Nreactions = 0
    
    if N_init <500:
        N_simulations = 2000
    elif N_init <10000:
        N_simulations = 25
    else:
        N_simulations = 1
    
    for i in range(N_simulations):
        times = [0]  # List to store time points
        A_counts = [NA]  # List to store the count of A over time
        B_counts = [NB]  # List to store the count of B over time
        while times[-1] < 100:  # Run simulation for a certain time period, e.g., 100 units
            # Calculate propensities
            a1 = k1 * NA
            a2 = k2 * NB
            a0 = a1 + a2
        
            # Time to next reaction
            tau = -np.log(np.random.random()) / a0
        
            # Select which reaction will occur (A->B or B->A)
            if np.random.random() < a1 / a0:
                NA -= 1
                NB += 1
            else:
                NA += 1
                NB -= 1
        
            # Update time and record state
            times.append(times[-1] + tau)
            A_counts.append(NA)
            B_counts.append(NB)
            Nreactions += 1
            
        t_end = datetime.now()
        t_elapsed = t_end - t_start
        
       
    
    total_time = t_elapsed.total_seconds()
    
    # Plotting
    plt.plot(times, A_counts, label='A')
    plt.plot(times, B_counts, label='B')
    plt.xlabel('Time')
    plt.ylabel('Number of Molecules')
    plt.title('Gillespie Simulation of A <-> B Reaction')
    plt.legend()
    plt.show()
    
    Nreactions /= N_simulations
    total_time /= N_simulations
    

    
    
    return total_time, Nreactions
    
    
    
    
    
    
        
    # Actually for Gon Goff it makes more sense to precompute the delay directly
    # Might be useful only when the number of reactants is HUGE
    # -> In one case we precompute and store the time delay of ALL reactants
    # -> In the other case we store the time it was created and never compute the time delay explicitly
    
    #Do delay reactions of O(N) or O(NlogN) ?
    
    #For small number of reactants (>50), precomputing the delay as in other methods is much faster
    
    #Regarding accuracy, it would take more than 1 trillion simulation to observe a significant difference between 
    #Theoretical and Simulated
    
    
    # while the idea of pre-calculating reaction times for each reactant at creation is intriguing, 
    # it lacks the dynamic adaptability, stochastic accuracy, and computational efficiency in 
    # handling complex reaction networks that the Gillespie algorithm provides.
    
    #If I have like 1 million reactants, I dont need to keep track of any time to next event,
    #With delay I need to COMPUTE the next witing time after each reaction occured (But its very fast though)
    # With MOSAIC its not needed thanks to the rejection.
    
    #Delay recation method can be implemented very easiy, I need to understand its limits
    #they key is that delayed reaction are always stored, is it always exact?
    #Need to read again where Anderson or Cai was mentioning that its not always exact
    #Consuming vs non consuming delayed reaction, is it always exact? I guess yes
    
    #Discussing the propensity update problem
    # Change the state of the system only upon delay completion (CD)
    # Change the state of the system at both initiation and completion (ICD) 
    # A -> B + A (what happens to A during the delay?)
    # B -> C
    
    # Whats surprising to me is that here we are simply precomputing all delay and there is no point to do Any Gillespie anymore, 
    # A -> 2A + B (Gamma distribution, easy with Mosaic)
    # A -> 2A + B (With delay? We can just assume No gillespie, we initialize the random delay for all A as soon as they exist)
    
    #System 1: Simple Epidemic model SIER
    # https://docs.juliahub.com/DelaySSAToolkit/kLahb/0.1.0/tutorials/tutorials/
    # FIXED DELAY not interresting
    #The parameters are b = 10-4/h; r = 10-2/h; tau = 20h
    
    #S + I -> E (b)
    #E -> I (delay tau)
    #I -> R (r)
    
    
    
    
    
    #System 2: Cell population dynamics
    # https://docs.juliahub.com/DelaySSAToolkit/kLahb/0.1.0/tutorials/heterogeneous_delay/
    # FIXED DELAY not interresting
    # For each individual, the production rate of a protein X is drawn from a Gamma distribution, i.e. A~Gamma(8,0.23),
    # the reaction is completed after a delayed time, tau~Gamma(7,1). The protein degradation rate B~Gamma(9,625).
    # plotting 40 simulated trajectories subsampled every minute

    #0 -> X (A with delay tau)
    #X -> 0 (B)
    
    
    
    #System 3: Protein transcription
    # https://docs.juliahub.com/DelaySSAToolkit/kLahb/0.1.0/tutorials/delay_oscillator/
    # Auto-regulatory negative feedback loop whereby a protein X is transcribed by a gene, transformed after a delay time s into a mature protein Y, which binds the promoter and represses the production of X.
    # Note7: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-22919-1/MediaObjects/41467_2021_22919_MOESM1_ESM.pdf
    # 0 -> X (J1(Y))
    # X -> Y (delay tau)
    # Y -> 0 (J2(Y))
    
    
    
    #System 4: Gene regulation with delayed auto-inhibition
    # mRNA transcription initiated at a Poissonian process ofrate g(nP) depending on 
    # the instantaneous number of proteins nP present at generation time, 
    # but it is completed only after a delay time tau
    # 0 -> M (poisson process with delay tau)
    # M -> M + P
    # M -> 0
    # P -> 0
    
    # -> constant or stochastic delay: compare with other algorithms
    # -> Implement the hard case, show how it affects the results and its same speed
    
    
    
    
    #Methods to compare with:
    # Delay rejection method -> https://docs.juliahub.com/DelaySSAToolkit/kLahb/0.1.0/algorithms/delayrejection/
    #   [1] Dmitri A. Bratsun, Dmitri N. Volfson, Jeff Hasty, and Lev S. Tsimring "Non-Markovian processes in gene regulation (Keynote Address)", Proc. SPIE 5845, Noise in Complex Systems and Stochastic Dynamics III, (23 May 2005). https://doi.org/10.1117/12.609707
    #   [2] Manuel Barrio, Kevin Burrage, André Leier, Tianhai Tian. "Oscillatory Regulation of Hes1: Discrete Stochastic Delay Modelling and Simulation", PLoS Computational Biology, 10.1371(2006). https://doi.org/10.1371/journal.pcbi.0020117
    
    # Delay direct method -> https://docs.juliahub.com/DelaySSAToolkit/kLahb/0.1.0/algorithms/delaydirect/
    #   [1] Xiaodong Cai, "Exact stochastic simulation of coupled chemical reactions with delays", The Journal of Chemical Physics 126, 124108(2007). https://doi/10.1063/1.2710253
    
    # Delay modified next reaction method -> https://docs.juliahub.com/DelaySSAToolkit/kLahb/0.1.0/algorithms/delaymnrm/
    #   [1] David F. Anderson, "A modified Next Reaction Method for simulating chemical systems with time dependent propensities and delays", The Journal of Chemical Physics 128, 109903(2008). https://doi/10.1063/1.2799998
    
    # Delay direct method with composition and rejection. -> 
    # Same as delay rejection method, but with a more optimised data structure
    
    # We note that the delay rejection and delay direct methods can offer better performance for small reaction networks, 
    # while the other two are preferable for systems with significant number of reactions.
    


def fit_log_line(x0,y0):
    
    x1 = x0[~np.isnan(y0)]
    y1 = y0[~np.isnan(y0)]
    
    x = np.log(x1)
    y = np.log(y1)
    m, b = np. polyfit(x, y, 1)
    print(m)
    
    fitted_y = np.exp(m*np.log(x0) + b)
    return fitted_y


def generate_delay_from_distribution(rate, alpha):
    beta = alpha*rate
    
    delay = stats.gamma.rvs(alpha, loc = 0, scale =  1/beta, size=1)[0]
    return delay   



if __name__ == '__main__':
    main()