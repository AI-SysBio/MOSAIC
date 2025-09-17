
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import random
import pickle
import sys
import networkx as nx
import scipy.stats as stat
from math import log, gamma, exp, factorial, pi, sqrt, erf, atan
from scipy.special import gammainc, gamma
from sortedcontainers import SortedDict
from scipy.stats import linregress


save_results = True
#Normal Regir is 5min per simulation

#parameters for now
#N = 274


def getID(i,j):
    if i > j: #make sure we always have i < j
        k = i
        i = j
        j = k
    ID = '%s_%s' % (i,j)
    return ID


def run_REGIR_simulation(N, Tend = 2000, case = 'Basic', use_IP = False):
    
    
    #Tend = 7249

    #N1 + N2 -> interaction parameters
    distribution = 'Pareto'
    rAA = 0.1
    alphaAA = 0.9

    #interaction -> 0 parameters
    distribution_A = 'Pareto'
    rA = 0.4
    alphaA = 1.4

    if case == 'first_order':
        alphaAA = 0.75
        
    if case == 'second_order':
        alphaAA = 0.8
        if use_IP:
            alphaAA = 0.7
    
        
    
    """
    This model simulate a temporal network with a Gillespie approach and considers two type of interactions:
        - A + A -> interaction
        - interaction -> 0
    Note that the same agent can be interacting with an unlimited number of agents
    """
    
    if use_IP:
        p_init = stat.gamma.rvs(a = 4, size = N)
        p_init = p_init/np.sum(p_init)
        #plt.hist(p_init, bins = 30, alpha=0.5)
        #plt.show()

    
    #Storing simulation
    first_order_set = set()
    second_order_set = set()
    Node_friend_list = dict()
    for ni in range(N):
        Node_friend_list[ni] = set()
    next_interaction_type_dict = dict()   
    for ni in range(N):
        next_interaction_type_dict[ni] = []
    All_interactions = [] #storing all the interactions
    inter_event_times = [] #just for tracking inter-event times
    interaction_times = []
    Aggregated_Network = np.zeros((N,N))
    
    
    
    #Simulation temporary variables
    Dt_reactant_time = dict()
    for i in range(N):
        Dt_reactant_time[i] = 0
    t = 0
    Interaction_end = SortedDict()
    Interaction_end[1000000000000000000000000000] = [-1,-1]
    Currently_interacting = set()
    next_t = 0
    while t < Tend:
        
        if t > next_t:
            #print('Time %s/%s' % (int(t), Tend))
            next_t += 250
        
        #1 - Set lambda_max, it's constant in the case of Pareto
        mu = 1/rAA* 2**(-1/alphaAA)
        mu = 1
        rmax = alphaAA/mu
        rmax = rmax/(N-1)
        
        #2 - Draw the time until next event
        u = random.random()
        rsum = rmax*N*(N-1)/2
        Dt = - log(u)/rsum
        
        
        t_delay_min, (i, j) = next(iter(Interaction_end.items()))
        if t_delay_min < t + Dt:
            t = t_delay_min
            All_interactions.append([t, i, j, "end interaction"])
            Interaction_end.popitem(index = 0) #The selected reaction is always on top of the sorted_dict
            interact_id = getID(i,j)
            Currently_interacting.remove(interact_id)
            
        else:
            t += Dt
            
            #3 - Draw two nodes with preference toward sommunities
            
            #select the first node uniformly
            if use_IP:
                i = np.random.choice(np.arange(N), p = p_init)
            else:
                i = np.random.randint(0,N)
            
            #select the second node according to sicial bonds
            pi = np.zeros(N)
            for j in range(N):
                interact_id = getID(i,j)
                pi[j] = 0.1 #default weight for unrelated nodes
                if i == j or interact_id in Currently_interacting:
                    pi[j] = 0 #noe already interacting cant interact
                elif interact_id in first_order_set:
                    if case == 'first_order' or case == 'second_order':
                        pi[j] = 1
                elif interact_id in second_order_set:
                    if case == 'second_order':
                        pi[j] = 0.5
                else:
                    pass #weight is 0.05 as the default
                    
            weights = pi/np.sum(pi)
            j = np.random.choice(np.arange(N), p = weights)
            if i > j: #make sure we always have i < j
                k = i
                i = j
                j = k
                       
                
            #4 -  Compute their interaction rate
            ti = t - Dt_reactant_time[i]
            tj = t - Dt_reactant_time[j]
            rate_i = rate_function(ti, r0 = rAA, shape_param = alphaAA)
            rate_j = rate_function(tj, r0 = rAA, shape_param = alphaAA)
            all_rates = [rate_function(t - Dt_reactant_time[k], r0 = rAA, shape_param = alphaAA) for k in range(N)]
            tot_rate = np.sum(all_rates) - (rate_i + rate_j)/2
            if tot_rate > 0:
                rate_ij = (rate_i * rate_j)/tot_rate
            else:
                rate_ij = 0
            
            #4 - Accept or reject the reaction
            if rate_ij >=rmax*random.random():
                inter_event_times.append(ti)
                inter_event_times.append(tj)
                Dt_reactant_time[i] = t
                Dt_reactant_time[j] = t
                
                interact_id = getID(i,j)
                
                if interact_id not in Currently_interacting:
                    
                    Dt = generate_delay_from_distribution(distribution_A, rA, alphaA)
                    Interaction_end[t+Dt] = [i,j]
                    interaction_times.append(Dt)
                    Currently_interacting.add(interact_id)
                    
                    #Store some information
                    All_interactions.append([t, i, j, "start interaction"])
                    Aggregated_Network[i,j] += Dt
                    Aggregated_Network[j,i] += Dt
                    if interact_id in first_order_set:
                        next_interaction_type_dict[i].append(1) 
                        next_interaction_type_dict[j].append(1) 
                    else:
                        next_interaction_type_dict[i].append(0) 
                        next_interaction_type_dict[j].append(0)    
                                        
                    #update first order inderactions
                    if case != 'Basic':
                        Node_friend_list[i].add(j)
                        Node_friend_list[j].add(i)
                        first_order_set.add(interact_id)
                     
                    #update second order inderactions
                    if case == 'second_order':
                        second_order_set.add(interact_id)
                        for ki in Node_friend_list[j]:
                            interact_id = getID(ki,j)
                            second_order_set.add(interact_id)
                        for kj in Node_friend_list[i]:
                            interact_id = getID(kj,j)
                            second_order_set.add(interact_id)
            
            
    return
    #Comparing to the real network:
    Real_data = load_dict('conf17.pkl')
    #key_to_plot = ['Interacting_time', 'Non_interacting_time']
    
    #Make the plot in log-log space
    
    def get_log_pdf(data, forced_bins = None):
        data = np.array(data)
        if forced_bins is None:
            bins = np.logspace(0, np.log10(data.max()), 20)  # Log-spaced bins
        else:
            bins = forced_bins
        hist, bin_edges = np.histogram(data, bins=bins, density=True)  # Normalized histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
        nonzero = hist > 0
        bin_centers = bin_centers[nonzero]
        pdf = hist[nonzero]
        
        return bin_centers, pdf, bins
      
    #Now that we have the power value, get the xmin
    plt.figure(figsize = (9,5))
    
    x,y,bins = get_log_pdf(Real_data['Interacting_time'])
    plt.scatter(x, y, label="Interaction (conf)", alpha=1, color = 'green')
    x,y,_ = get_log_pdf(interaction_times, forced_bins = bins)
    plt.scatter(x, y, label="Interaction (REGIR)", alpha=1, color = 'darkgreen', marker = 'x')
    
    
    x,y,bins = get_log_pdf(Real_data['Non_interacting_time'])
    plt.scatter(x, y, label="Inter-event time (conf)", alpha=1, color = 'red')
    x,y,_ = get_log_pdf(inter_event_times, forced_bins = bins)
    plt.scatter(x, y, label="Inter-event time (REGIR)", alpha=1, color = 'darkred', marker = 'x')
    
    
    plt.xlabel("Time")
    plt.ylabel("PDF")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize = 14)
    plt.show()

    
    
    #Show distribution plot
    t = np.linspace(0, 100, 1000)  
    #pdf_true, fitted_params = fit_pdf_function(interaction_times, t)
    pdf_true = get_pdf_function(t, r0 = rA, shape_param = alphaA)
    plt.plot(t, pdf_true, 'black',lw=2, linestyle = '--') 
    bins = np.linspace(0,100,40)
    plt.hist(interaction_times, density = True, bins = bins, alpha = 0.3, color = 'green', label = 'Simulated Network')
    plt.hist(Real_data['Interacting_time'], label = 'Conference Network', alpha = 0.2, bins = bins, color = 'black', density = True)
    plt.xlabel('Interaction time')
    plt.ylabel('Density')
    plt.yscale('log')
    plt.xlim(-5,100)
    plt.show() 
    
    t = np.linspace(0, 3000, 1000)  
    #pdf_true, fitted_params = fit_pdf_function(inter_event_times, t)
    pdf_true = get_pdf_function(t, r0 = rAA, shape_param = alphaAA)
    plt.plot(t, pdf_true, 'black',lw=2, linestyle = '--') 
    bins = np.linspace(0,3000,40)
    plt.hist(inter_event_times, density = True, bins = bins, alpha = 0.5)
    plt.hist(Real_data['Non_interacting_time'], label = 'Conference Network', alpha = 0.2, bins = bins, color = 'black', density = True)
    plt.xlabel('Time between interactions of individual nodes')
    plt.ylabel('Density')
    if distribution == 'Pareto':
        plt.yscale('log')
    plt.show()


    number_of_interactions = np.sum(Aggregated_Network, axis = 1)
    counts, bins, patches = plt.hist(Real_data['Interactions'], bins = 30, alpha = 0.5, density = True, color = 'gray')
    plt.hist(number_of_interactions, bins = bins, alpha = 0.5, density = True)
    plt.xlabel('Number of interactions')
    plt.show()
    print(' %.2f vs %.2f' % (np.mean(number_of_interactions), np.mean(Real_data['Interactions'])))
    
    prop_of_friends = np.sum(Aggregated_Network > 0, axis = 1)/N
    conf_prop_of_friends = np.array(Real_data['Friends'])/274
    bins = np.linspace(0,1,30)
    plt.hist(conf_prop_of_friends, bins = bins, alpha = 0.5, density = True, color = 'gray')
    plt.hist(prop_of_friends, bins = bins, color = 'brown', density = True, alpha = 0.5)
    plt.xlabel('Proportion of node interacted')
    plt.show()
    print(' %.2f vs %.2f' % (np.mean(prop_of_friends), np.mean(conf_prop_of_friends)))
    
    next_interaction_ratio = []
    for ai, agent in enumerate(np.arange(N)):
        f = np.count_nonzero(next_interaction_type_dict[agent])
        tot = len(next_interaction_type_dict[agent])
        if tot>100:
            next_interaction_ratio.append(f/tot) 
    bins = np.linspace(0,1,30)
    plt.hist(Real_data['Proportion'], bins = bins, alpha = 0.5, density = True, color = 'gray')
    plt.hist(next_interaction_ratio, bins = bins, color = 'blue', density = True, alpha = 0.5)
    plt.xlabel('Proportion of friend interactions') 
    plt.show()
    #print( 'Mean proportion accross agents is %.2f' % np.mean(next_interaction_ratio))
    print(' %.2f vs %.2f' % (np.mean(next_interaction_ratio), np.mean(Real_data['Proportion'])))
    
    
    if save_results:
        print('Writting network interaction file')
        if use_IP:
            write_interaction_file(All_interactions, 'REGIR_networks/Sim_conf_%s_IP.csv' % case)
        else:
            write_interaction_file(All_interactions, 'REGIR_networks/Sim_conf_%s.csv' % case)
        
        all_results = dict()
        all_results['Interacting_time'] = np.array(interaction_times)
        all_results['Non_interacting_time'] = np.array(inter_event_times)
        #all_results['Network'] = logNetwork[nsorted,:][:,nsorted]
        all_results['Interactions'] = number_of_interactions
        all_results['Friends'] = prop_of_friends*N
        all_results['Proportion'] = next_interaction_ratio
        if use_IP:
            save_dict(all_results, 'REGIR_networks/Sim_conf_%s_IP.pkl' % case)
        else:
            save_dict(all_results, 'REGIR_networks/Sim_conf_%s.pkl' % case)
    
    sys.exit()
    
    
    #Make the graph
    nAgents = N
    logNetwork = np.log2(Aggregated_Network+1)
    na = nAgents
    G = nx.Graph()
    for node in np.arange(N):
        G.add_node(int(node))
        
    edge_width_dict = dict()
    k = 3
    max_width = np.max(logNetwork)**k
    for ai in range(na):
        for aj in range(ai+1,na):
            #edge = (ai, aj, {'weight': logNetwork[ai,aj]})
            if logNetwork[ai,aj] > 4:
                #edge = (ai, aj, logNetwork[ai,aj])
                edge = (ai, aj)
                G.add_edge(*edge, weight = logNetwork[ai,aj])
                edge_width_dict[edge] = logNetwork[ai,aj]**k/max_width*2   
    
    
    
    dist_matrix = 1-logNetwork/np.max(logNetwork)
    for i in range(logNetwork.shape[0]):
        dist_matrix[i,i] = 0
    communities = nx.community.louvain_communities(G, weight='weight')
    clusters_ = np.zeros(nAgents)
    for ci,com in enumerate(communities):
        for node in com:
            clusters_[int(node)] = ci
    
    
    
    
    
    
    
    cluster_list = np.sort(list(set(clusters_)))
    nclust = len(cluster_list)
    
    #Sort the clusters by similarities
    cluster_dist_matrix = np.zeros((nclust,nclust))
    clust_size = np.zeros(nclust)
    for ci, clusti in enumerate(cluster_list):
        nwherei = np.where(clusti == clusters_)[0]
        clust_size[ci] = len(nwherei)
        for cj, clustj in enumerate(cluster_list):
            if cj >= ci+1:
                nwherej = np.where(clustj == clusters_)[0]
                cluster_dist_matrix[ci,cj] = np.mean(dist_matrix[:,nwherei][nwherej,:])
                cluster_dist_matrix[cj,ci] = cluster_dist_matrix[ci,cj]
                
    
    #mean_clust_dist = np.mean(cluster_dist_matrix, axis=0)
    #c0 = np.argmax(mean_clust_dist)
    c0 = np.argmax(clust_size)
    c_list = [c0]
    ci = c0
    candidate_c = list(np.arange(len(cluster_list)))
    for i in range(nclust-1):
        candidate_c.remove(ci)
        cnext = candidate_c[np.argmin(cluster_dist_matrix[ci,candidate_c])]
        c_list.append(cnext)
        ci = cnext

    c_dict = {c:ci for ci,c in enumerate(c_list)}
    clusters = np.array([c_dict[c] for c in clusters_])
    nsorted = np.argsort(clusters)
    
    plt.figure(figsize=(9,7))                
    #plt.title('Aggregated network heatmap (Louvain clusters)')
    plt.imshow(logNetwork[nsorted,:][:,nsorted], origin='lower', cmap = 'inferno')
    plt.colorbar()
    plt.show()
        
    
    
    

    
    
    pdf_true, fitted_params = fit_pdf_function(inter_event_times, t)
    shape_param_fit = fitted_params[1]
    r_fit = fitted_params[0]*2**(-1/shape_param_fit)
    print()
    print('Param intrisic distrbution:')
    print('  r0 = %.2f, shape_param = %.2f' % (rAA, alphaAA))
    print()
    print('Param fitted distrbution for N = %s:' % N)
    print('  r0 = %.2f, shape_param = %.2f' % (r_fit, shape_param_fit))
    


def rate_function(t, r0 = 1, shape_param = 1, return_sdf_and_pdf = False, distribution = 'Pareto'): #r0,alpha

    if distribution == 'Exponential':
        rate = r0
        sdf = exp(-r0*t)
        pdf = r0*sdf


    elif distribution == 'Weibull':
        alpha = shape_param 
        beta = (alpha) * (r0 * gamma((alpha + 1)/(alpha)))**(alpha)
        rate = (t**(alpha-1))*beta
        sdf = exp(-beta*t**alpha / alpha)
        pdf = rate*sdf
        
    elif distribution == 'Pareto':
        alpha = shape_param 
        mu = 1/r0 * 2**(-1/alpha)
        mu = 1
        
        #In this work, we need mu == 1, so we gonna shift pareto to match that
        if t < mu:
            rate = 0
            sdf = 1
            pdf = 0
        else:
            rate = alpha/t
            pdf = alpha * mu**alpha / t**(alpha + 1)
            sdf = (mu/t)**alpha
            
    elif distribution == 'Normal':
        sigma = shape_param
        sigma = 1/r0*sigma
        mu = 1/r0
        if t > mu + 7*sigma: #necessary as the precision of cdf calculation is not precise enough
            rate = (t-mu)/sigma**2
            pdf = 1/(sigma*sqrt(2*pi)) * exp(-(1/2) * (t-mu)**2/(sigma**2))
            cdf = 1/2*(1+erf((t-mu)/(sigma*sqrt(2))))
            sdf = 1-cdf
        else:
            pdf = 1/(sigma*sqrt(2*pi)) * exp(-(1/2) * (t-mu)**2/(sigma**2))
            cdf = 1/2*(1+erf((t-mu)/(sigma*sqrt(2))))
            rate = pdf/(1-cdf)
            sdf = 1-cdf
            
    elif distribution == 'Gamma':
        alpha = shape_param 
        if alpha < 1 and t == 0:
            rate = 0
            pdf = 0
            sdf = 0
        else:
            #only works for alpha >= 1
            beta = alpha*r0
            pdf = (beta**alpha)*(t**(alpha-1))*exp(-beta*t)/gamma(alpha)
            cdf = gammainc(alpha,beta*t)
            rate = pdf/(1-cdf)
            sdf = 1-cdf
    

    elif distribution == 'Cauchy':
        sigma = shape_param
        gam = 1/r0*sigma
        mu = 1/r0
        pdf = 1 / (pi*gam*(1 + ((t-mu)/gam)**2))
        cdf = (1/pi) * atan( (t-mu)/gam ) + 1/2
        rate = pdf/(1-cdf)
        sdf = 1-cdf


    if return_sdf_and_pdf:
        return rate, pdf, sdf
    else:
        return rate
    
    
def fit_pdf_function(wait_times, t, distribution = 'Pareto'): 
    
    if distribution == 'Exponential':
        P = stat.expon.fit(wait_times,floc=0)
        pdf = stat.expon.pdf(t, *P)
        params = [1/P[1],1]
        
    elif distribution == 'Weibull':
        P = stat.weibull_min.fit(wait_times,3, floc=0)
        pdf = stat.weibull_min.pdf(t, *P)
        params = [1/P[2], P[0]]
        
    elif distribution == 'Pareto':
        P = stat.pareto.fit(wait_times, floc=0, fscale = 1)
        pdf = stat.pareto.pdf(t, *P)
        params = [1/P[2], P[0]]
        
        """
        #alternative method to fit the Pareto distribution with bounds
        bounds = [(0.5, 4), (-10, 10), (0.1, 10)]  # Shape (b), Location (loc), Scale (scale)
        from scipy.stats import pareto
        from scipy import stats
        result = stats.fit(pareto, wait_times, bounds=bounds)
        fitted_params = result.params
        P = [fitted_params.b, fitted_params.loc, fitted_params.scale]
        pdf = stat.pareto.pdf(t, *P)
        params = [1/P[2], P[0]]
        """

    elif distribution == 'Normal':
        P = stat.norm.fit(wait_times)
        pdf = stat.norm.pdf(t, *P)
        r0 = 1/P[0]
        sigma = P[1]*r0
        params = [r0, sigma]
            
    elif distribution == 'Gamma':
        P = stat.gamma.fit(wait_times,floc=0)
        pdf = stat.gamma.pdf(t, *P)
        params = [1/P[2], P[0]]
    
    elif distribution == 'Cauchy':
        P = stat.cauchy.fit(wait_times)
        pdf = stat.cauchy.pdf(t, *P)
        r0 = 1/P[0]
        sigma = P[1]*r0
        params = [r0, sigma]
        
    elif distribution == 'Lognormal':
        P = stat.lognorm.fit(wait_times,floc=0)
        pdf = stat.lognorm.pdf(t, *P)
        r0 = 1/P[0]
        sigma = P[1]*r0
        params = [r0, sigma]
        
    return pdf, params




def generate_delay_from_distribution(distribution, rate, shape_param, N_ = None):
    
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
        
    elif distribution.lower() in ['lognormal','lognorm']:
        mu0 = 1/rate
        sigma0 = mu0*shape_param
        mu = log(mu0**2 / sqrt(mu0**2 + sigma0**2))
        sigma = sqrt(log(1+ sigma0**2/mu0**2))
        delay = stat.lognorm.rvs(sigma, loc = 0, scale = mu0, size=N)
        
    elif distribution.lower() in ['cauchy', 'cau']:
        gam = shape_param
        mu = 1/rate
        delay = stat.cauchy.rvs(loc = mu, scale = gam, size=N) #check gam or 1/gam
        
        
    elif distribution.lower() in ['pareto', 'par']:
        alpha = shape_param
        mu = 1/rate* 2**(-1/alpha)
        mu = 1
        delay = stat.pareto.rvs(alpha, scale = mu, size=N) #check gam or 1/gam
    
    if N_ is None:
        return delay[0]
    else:
        return delay  
    
    
def get_pdf_function(t_array, r0 = 1, shape_param = 1): #r0,alpha

    pdf_array = np.zeros(len(t_array))
    for ti,t in enumerate(t_array):
        _, pdf, _ = rate_function(t, r0 = r0, shape_param = shape_param, return_sdf_and_pdf = True)
        pdf_array[ti] = pdf
        
    return pdf_array


def get_random_element(a_huge_key_list, weights = None):
    L = len(a_huge_key_list)
    if weights is None:
        i = np.random.randint(0, L)
    else:
        weights = weights/np.sum(weights)
        i = np.random.choice(np.arange(L), p = weights)
                             
    return i, a_huge_key_list[i] 






def write_interaction_file(interaction_list, fname, resolution = 1):
    
    #interaction is [timestamp, node1, node2, start/end]
    All_interactions = np.array(interaction_list)
    interaction_csv = []
    current_interacting = set()
    for i in range(len(All_interactions)):
        itype = All_interactions[i,3]
        timestamp = float(All_interactions[i,0])
        n1,n2 = All_interactions[i,1:3].astype(int)
        if 'start' in itype:
            current_interacting.add((n1,n2))
        if 'end' in itype:
            current_interacting.remove((n1,n2))
        for int_nodes in current_interacting:
            interaction_csv.append([timestamp/resolution, int_nodes[0], int_nodes[1]])
            
    #Now, regroup times together with a specific resolution
    df = pd.DataFrame(interaction_csv, columns=["timestamp", "node1", "node2"])
    df["timestamp"] = df["timestamp"].round().astype(int)
    df_aggregated = (
        df.groupby("timestamp")[["node1", "node2"]]
        .apply(lambda x: np.unique(x.values, axis=0))
        .reset_index())
    
    
    #filling missing tinestamps
    # Create a new DataFrame with no skipped timestamps
    all_timestamps = np.arange(df_aggregated["timestamp"].min(), df_aggregated["timestamp"].max() + 1)
    
    # Iterate over all timestamps to add rows for missing timestamps
    prev_interactions = None
    output_lines = []
    for timestamp in all_timestamps:
        if timestamp in df_aggregated["timestamp"].values:
            interactions = df_aggregated[df_aggregated["timestamp"] == timestamp].values[0][1]
            prev_interactions = interactions  # Update to the current interactions
        else:
            interactions = prev_interactions  # Use previous interactions for missing timestamps
    
        # Append the timestamp and its interactions
        for interaction in interactions:
            output_lines.append(f"{timestamp} {int(interaction[0])} {int(interaction[1])}")
    
    file_path = fname
    with open(file_path, 'w') as f:
        f.write("\n".join(output_lines))
        
        
def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, 4)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)    
    
    
    
def select_xmin(data):
    from scipy.stats import ks_2samp
    """
    Systematically find the best xmin that minimizes the KS statistic.
    """
    xmins = np.arange(1,10)/2
    best_xmin = xmins[0]
    best_alpha = None
    best_ks = float('inf')
    
    for xmin in xmins:
        filtered_data = data[data >= xmin]
        alpha = 1 + len(filtered_data) / np.sum(np.log(filtered_data / xmin))
        cdf_empirical = np.arange(1, len(filtered_data) + 1) / len(filtered_data)
        cdf_theoretical = 1 - (xmin / filtered_data)**(alpha - 1)
        ks_stat, _ = ks_2samp(cdf_empirical, cdf_theoretical)
        if ks_stat < best_ks:
            best_ks = ks_stat
            best_xmin = xmin
            best_alpha = alpha
    
    return best_xmin, best_alpha


def fit_powerlaw(data):
    
    xmin, alpha_fit = select_xmin(data)
        
    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 20)  # Log-spaced bins
    hist, bin_edges = np.histogram(data, bins=bins, density=True)  # Normalized histogram
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
    nonzero = hist > 0
    bin_centers = bin_centers[nonzero]
    pdf = hist[nonzero]
    log_bin_centers = np.log(bin_centers)
    log_pdf = np.log(pdf)
    slope, intercept, r_value, p_value, std_err = linregress(log_bin_centers, log_pdf)
    alpha_fit_ = -slope  # Slope is -alpha
    fitted_line = slope * log_bin_centers + intercept
      
    #Now that we have the power value, get the xmin
    plt.title('power law MLE alpha = %.2f, mu = %s' % (alpha_fit, xmin), fontsize = 15)
    plt.plot(np.exp(log_bin_centers), np.exp(fitted_line), color="black", ls = '--', lw = 1)
    plt.scatter(np.exp(log_bin_centers), np.exp(log_pdf), label="Empirical PDF", alpha=1, color = 'black')
    plt.xlabel("Interaction Time")
    plt.ylabel("PDF")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    
    
    #xmin = 1
    alpha_fit = 1.05

    from scipy.stats import pareto
    t = np.linspace(-1, 3000, 1000)  
    bins = np.linspace(0,3000,50)
    pdf_pareto = pareto.pdf(t, b=alpha_fit, scale=xmin, loc = -xmin)
    plt.plot(t, pdf_pareto, label=f"Pareto Fit (alpha={alpha_fit:.2f})", color="red")
    plt.hist(data, bins = bins, density = True, alpha = 0.5)
    plt.yscale("log")
    plt.show()
    
    
def quick_test(distribution = 'Pareto'):
    
    #Quick check:
    Real_data = load_dict('conf17.pkl')
    Real_data['Interacting_time']
    Real_data['Non_interacting_time']
    
    
    data = np.array(Real_data['Interacting_time'])
    #fit_powerlaw(Real_data['Interacting_time'])
    #fit_powerlaw(Real_data['Non_interacting_time'])
    
    #AA params:
    # alpha_fit = 2.22, xmin = 4.5
    
    #A params:
    # alpha_fit = 1.05, xmin = 4.5
    
    #sys.exit()
    
        
    t = np.linspace(0, 100, 1000)  
    pdf_true, fitted_params = fit_pdf_function(Real_data['Interacting_time'], t)
    shape_param_fit = fitted_params[1]
    r_fit = fitted_params[0]*2**(-1/shape_param_fit)
    print(r_fit, shape_param_fit)
    
    plt.plot(t, pdf_true, 'black',lw=2, linestyle = '--') 
    bins = np.linspace(0,100,40)
    plt.hist(Real_data['Interacting_time'], label = 'Conference Network', alpha = 0.2, bins = bins, color = 'black', density = True)
    plt.xlabel('Time between interactions of individual nodes')
    plt.ylabel('Density')
    if distribution == 'Pareto':
        plt.yscale('log')
    plt.show()
    
    t = np.linspace(0, 3000, 1000)  
    pdf_true, fitted_params = fit_pdf_function(Real_data['Non_interacting_time'], t)
    shape_param_fit = fitted_params[1]
    r_fit = fitted_params[0]*2**(-1/shape_param_fit)
    print(r_fit, shape_param_fit)
    
    plt.plot(t, pdf_true, 'black',lw=2, linestyle = '--') 
    bins = np.linspace(0,3000,40)
    plt.hist(Real_data['Non_interacting_time'], label = 'Conference Network', alpha = 0.2, bins = bins, color = 'black', density = True)
    plt.xlabel('Time between interactions of individual nodes')
    plt.ylabel('Density')
    if distribution == 'Pareto':
        plt.yscale('log')
    plt.show()
        
    
    sys.exit()
        

    
if __name__ == '__main__':
    run_REGIR_simulation(15)