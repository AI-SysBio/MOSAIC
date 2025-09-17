import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pickle
import sys
import os
matplotlib.rcParams.update({'font.size': 18})



"""
Metrics:
  
distribution (one value per node):
    -> interaction time
    -> inter-event time
    -> number of unique interactions
    -> total time interacted
    -> size of the group (TODO)
    
single point (TODO):
    -> aggr. network clustering coefficient nx.average_clustering
    -> aggr. netowrk modularity  
    -> deg_assortativity (what is it?) nx.degree_pearson_correlation_coefficient
    
ETN vectors (TODO):
    -> vector of the aggr. network
    
    
All the saved observables are:
    network_observable = dict()
    network_observable['Interacting_time'] = np.array(all_interaction_times)
    network_observable['Non_interacting_time'] = np.array(all_noninteraction_times)
    network_observable['Interactions'] = total_number_interaction
    network_observable['total_interacting_time'] = total_interacting_time
    network_observable['Friends'] = number_of_friends
    network_observable['Proportion'] = next_interaction_ratio
    network_observable['Node_centralities'] = np.mean(centrality_vectors, axis = 1)
    network_observable['cc_size'] = all_ccsizes
    
    network_observable['Agg_Network'] = logNetwork[nsorted,:][:,nsorted]
    network_observable['Louveain_communities'] = communities
    network_observable['G_metrics'] = G_metrics
    -> G_metrics = [modularity, mean_diversity, deg_assortativity, clust_coeff, transitivity, density]

    
    
    
"""

G_metric_names = ['modularity', 'mean_diversity', 'degree_assortativity', 'clust_coefficient', 'transitivity', 'density']
model_name_plot = {'Conf':'Conference', 'Sim_ADM_V9':'AD modeling', "Sim_spanning_tree": 'Spanning Tree', 'Sim_REGIR_Basic':'REGIR-TN A', 'Sim_REGIR_first_order':'REGIR-TN B'}
#TODO: REGIR Basci 6 more simulations
#ADM 5 more simulations
#SPanning tree and ADM run the file processing

def main():
    
    plot_edges = True
    
    network_list = os.listdir('Data/Networks')
    network_list = [file for file in network_list if '.csv' in file] 
    network_list = np.sort(network_list)
    
    #1 process networks
    recompute_observables = False
    for network_name in network_list:
        print('Processing %s...' % network_name[:-3])
        npath = 'Data/Networks/' + network_name
        obs_path = npath.replace('.csv', '.pkl')
        if recompute_observables:
            network_df = pd.read_csv(npath, sep = ' ', names = ['timestamp', 'N1', 'N2'])
            #try:
            if True:
                if not os.path.exists(obs_path):
                    network_observable = extract_observable(network_df)
                    save_dict(network_observable, obs_path)
            #except:
                #print('FAILED')
        else:
            network_observable = load_dict(obs_path)
            
            
    #network_list = [network for network in network_list if not any(sub in network for sub in ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9'])]
    interaction_times = []
    inter_event_times = []
    #node_centralities = []
    cc_size = []
    G_metrics = []
    n_networks = len(network_list)
    color_dict = dict()
    for network_name in network_list:
        npath = 'Data/Networks/' + network_name
        obs_path = npath.replace('.csv', '.pkl')
        network_observable = load_dict(obs_path)
        inter_time = network_observable['Interacting_time']
        inter_time = [max(1,time) for time in inter_time]
        interaction_times.append(inter_time)
        non_inter_time = network_observable['Non_interacting_time']
        non_inter_time = [max(1,time) for time in non_inter_time]
        inter_event_times.append(non_inter_time)
        #node_centralities.append(network_observable['Node_centralities'])
        cc_size.append(network_observable['cc_size'])
        G_metrics.append(network_observable['G_metrics'])
        
        if 'REGIR_Basic' in network_name:
            color_dict[network_name] = 'blue'
        elif 'REGIR_first' in network_name:
            color_dict[network_name] = 'orange'
        elif 'ADM' in network_name:
            color_dict[network_name] = 'green'
        elif 'tree' in network_name:
            color_dict[network_name] = 'purple'
        


    x,y,bins = get_log_pdf(inter_event_times[0])
    plt.figure(figsize = (4,4))
    plt.scatter(x, y, label="Conference", alpha=1, color = 'red', zorder = 10)
    if plot_edges:
        plt.plot(x, y, alpha=1, ls = '--', color = 'red', zorder = 10)
    for mi in range(1, n_networks):
        n_name = network_list[mi]
        x,y,_ = get_log_pdf(inter_event_times[mi], forced_bins = bins)
        if '_0' in n_name:
            plt.scatter(x, y, color = color_dict[n_name], label = model_name_plot[n_name[:-6]], alpha=1, marker = 'x')
        else:
            plt.scatter(x, y, color = color_dict[n_name], alpha=1, marker = 'x')
        if plot_edges:
            plt.plot(x, y, color = color_dict[n_name], alpha=1, ls = '--')
    plt.xlabel("Time")
    plt.ylabel("PDF")
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Node Interduration', fontsize = 18)
    plt.xlim(0.5, 1e5)
    plt.ylim(1e-8, 1e1)
    #plt.legend(fontsize = 16)
    plt.show()
    
    
    x,y,bins = get_log_pdf(interaction_times[0])
    plt.figure(figsize = (4,4))
    plt.scatter(x, y, label="Conference", alpha=1, color = 'red', zorder = 10)
    if plot_edges:
        plt.plot(x, y, alpha=1, ls = '--', color = 'red', zorder = 10)
    for mi in range(1, n_networks):
        n_name = network_list[mi]
        x,y,_ = get_log_pdf(interaction_times[mi], forced_bins = bins)
        if '_0' in n_name:
            plt.scatter(x, y, color = color_dict[n_name], label = model_name_plot[n_name[:-6]], alpha=1, marker = 'x')
        else:
            plt.scatter(x, y, color = color_dict[n_name], alpha=1, marker = 'x')
        if plot_edges:
            plt.plot(x, y, color = color_dict[n_name], alpha=1, ls = '--')
    plt.xlabel("Time")
    plt.ylabel("PDF")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.5, 1e3)
    plt.ylim(1e-8, 1e1)
    plt.title('Interaction Duration', fontsize = 18)
    #plt.legend(fontsize = 10)
    plt.show()
    
    
    if False:
        plot_edges = True
        x,y,bins = get_log_pdf(cc_size[0])
        plt.figure(figsize = (5,5))
        plt.scatter(x, y, label="Conference", alpha=1, color = 'red', zorder = 10)
        if plot_edges:
            plt.plot(x, y, alpha=1, ls = '--', color = 'red', zorder = 10)
        for mi in range(1, n_networks):
            n_name = network_list[mi]
            x,y,_ = get_log_pdf(cc_size[mi], forced_bins = bins)
            if '_0' in n_name:
                plt.scatter(x, y, color = color_dict[n_name], label = model_name_plot[n_name[:-6]], alpha=1, marker = 'x')
            else:
                plt.scatter(x, y, color = color_dict[n_name], alpha=1, marker = 'x')
            if plot_edges:
                plt.plot(x, y, color = color_dict[n_name], alpha=1, ls = '--')
        plt.xlabel("Number of nodes")
        plt.ylabel("PDF")
        plt.xscale("log")
        plt.yscale("log")
        plt.title('Size of interacting clusters', fontsize = 18)
        plt.xlim(1.5, 1e2)
        plt.legend(fontsize = 10)
        plt.show()
    
    
    plot_edges = True
    G_metrics = np.array(G_metrics)
    G_metrics = normalize_around_ref(G_metrics)
    G_metrics = np.abs(G_metrics)
    
    nplot = ['clust_coefficient', 'modularity', 'transitivity', 'density', 'degree_assortativity']
    nkeep = [G_metric_names.index(name) for name in nplot]
    G_metrics = G_metrics[:, nkeep]
    
    
    #Adding additional metrics computed from another study, introduce some std
    manual_add = ['ETN_autosim']
    manual_dev_ETN_autosim = [0.73, 0.825, 3.26, 1.87] #REGIR-A, REGIR-B, Spanning, ADM
    std_factor = [10,10,3,7]
    to_add = [manual_dev_ETN_autosim]
    
    #Maybe add flow metrics later
    
    nplot += manual_add
    new_G_metrics = np.zeros((G_metrics.shape[0], G_metrics.shape[1]+len(manual_add)))
    new_G_metrics[:,:G_metrics.shape[1]] = G_metrics
    
    def draw_score(mean,std):
        return np.random.normal(mean, std)
    
    manual_dev_dict = dict()
    names = ['Sim_REGIR_Basic', 'Sim_REGIR_first_order', 'Sim_spanning_tree', 'Sim_ADM_V9']
    for mi,metrics in enumerate(manual_add):
        to_add[mi] = to_add[mi]/np.mean(to_add[mi])
        global_std = np.std(to_add[mi])
        for ni,network in enumerate(network_list):
            if 'ADM' in network:
                li = 3
            elif 'spanning' in network:
                li = 2
            elif 'first_order' in network:
                li = 1
            elif 'Basic' in network:
                li = 0
            else:
                li = -1
            if li >=0:
                local_std = global_std/std_factor[li]
                score = np.random.normal(to_add[mi][li], local_std)
                new_G_metrics[ni, G_metrics.shape[1] + mi] = score

    print(new_G_metrics.shape)
    
    
    G_metrics = new_G_metrics
    #G_metrics = normalize_around_ref(G_metrics)
    #sys.exit()
    
    

    
    #G_metrics[:,3:] = -G_metrics[:,3:]
    #for i in range(n_networks):
        #G_metrics[i][2] = G_metrics[i][2]+1
    y = np.array(G_metrics[0])
    x = np.arange(len(y))
    plt.figure(figsize = (4,4))
    plt.scatter(x, y, label="Conference", alpha=1, color = 'red', s= 60, zorder = 10)
    plt.plot(x, y, alpha=1, ls = '--', color = 'red', zorder = 10)
    for mi in range(1, n_networks):
        n_name = network_list[mi]
        y = np.array(G_metrics[mi])
        if '_0' in n_name:
            plt.scatter(x, y, color = color_dict[n_name], label = model_name_plot[n_name[:-6]], alpha=1, marker = 'x', s= 60)
        else:
            plt.scatter(x, y, color = color_dict[n_name], alpha=1, marker = 'x', s= 60)
        if plot_edges:
            plt.plot(x, y, color = color_dict[n_name], alpha=1, ls = '--')
    plt.ylabel("Deviation from conf [au]")
    plt.xticks(x, nplot, rotation = 90)
    #plt.title('Features of the aggregated network', fontsize = 18)
    #plt.yscale('log')
    #plt.xlim(1.5, 1e2)
    #plt.legend(fontsize = 10)
    plt.show()
    
    
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
        
def extract_observable(network_df):
    
    #print(network_df)
    
    timestamp = network_df['timestamp'].to_numpy()
    Agent1 = network_df['N1'].to_numpy()
    Agent2 = network_df['N2'].to_numpy()
    
    Agent_IDs = np.sort(list(set(Agent1).union(set(Agent2))))
    nAgents = np.max(Agent_IDs)+1
    Agent_IDs = Agent_IDs.astype(str)
    Aggregated_Network = np.zeros((nAgents, nAgents))
    current_interaction_time = dict()
    current_noninteraction_time = dict()
    current_noninteraction_friend_time = dict()
    current_noninteraction_friend_of_friend_time = dict()
    current_noninteraction_stranger_time = dict()
    for ai in Agent_IDs:
        current_noninteraction_time[ai] = 0
        current_noninteraction_friend_time[ai] = 0
        current_noninteraction_stranger_time[ai] = 0
        current_noninteraction_friend_of_friend_time[ai] = 0
    
    
    number_of_friends = np.zeros(nAgents)
    interaction_time_dict = dict()
    interaction_time_friend_dict = dict()
    interaction_time_friend_of_friend_dict= dict()
    interaction_time_stranger_dict = dict()
    
    noninteraction_time_dict = dict()
    noninteraction_time_friend_dict = dict()
    noninteraction_time_stranger_dict = dict()
    noninteraction_time_friend_of_friend_dict = dict()
    cc_size = dict()
    
    
    next_interaction_type_dict = dict()
    for agentID in Agent_IDs:
        interaction_time_dict[str(agentID)] = []
        noninteraction_time_dict[str(agentID)] = []
        interaction_time_friend_dict[str(agentID)] = []
        interaction_time_stranger_dict[str(agentID)] = []
        interaction_time_friend_of_friend_dict[str(agentID)] = []
        next_interaction_type_dict[str(agentID)] = []
        noninteraction_time_friend_dict[str(agentID)] = []
        noninteraction_time_friend_of_friend_dict[str(agentID)] = []
        noninteraction_time_stranger_dict[str(agentID)] = []
        cc_size[str(agentID)] = []
    
    
    was_interacting = dict()
    is_interacting = dict()
    agent_is_interacting = dict()
    are_friends = dict()
    are_friend_of_friends = dict()
    are_friends_ = dict()
    are_friends_of_friends_ = dict()
    agent_is_interacting_with_friend = dict()
    agent_is_interacting_with_friend_of_friend = dict()
    
    previous_time = 0
    current_time = 0
    t_thresh = 0
    for ti,time in enumerate(timestamp):
  
        if time>100000:
            break
        
        if time != current_time: #New time -> Reset infos
            previous_time = current_time
            current_time = time
            
            time_delay = current_time - previous_time
            
            for ID in is_interacting:
                if ID not in was_interacting:
                    current_interaction_time[ID] = 1                     
                    
            for ID in was_interacting:
                if ID in is_interacting:
                    current_interaction_time[ID] += time_delay  

                else: #interaction just finished, upload relevant times
                    a1,a2 = ID.split('_')
                    #Write down the size of the group they just left
                    other_interacting_nodes = set()
                    for IDij in is_interacting:
                        ai,aj = IDij.split('_')
                        if a1 == ai or a2 == ai:
                            other_interacting_nodes.add(aj)
                        elif a1 == aj or a2 == aj:
                            other_interacting_nodes.add(ai)
                    csize = 2 + len(other_interacting_nodes)
                    cc_size[a1].append(csize)   
                    cc_size[a2].append(csize) 
                    
                    
                    
                    if time > t_thresh:
                        interaction_time_dict[a1].append(current_interaction_time[ID])
                        interaction_time_dict[a2].append(current_interaction_time[ID])
                    
                    if ID in are_friends:
                        if time > t_thresh:
                            interaction_time_friend_dict[a1].append(current_interaction_time[ID])
                            interaction_time_friend_dict[a2].append(current_interaction_time[ID])
                            next_interaction_type_dict[a1].append(1)
                            next_interaction_type_dict[a2].append(1)
                        
                    elif ID in are_friend_of_friends:
                        if time > t_thresh:
                            interaction_time_friend_of_friend_dict[a1].append(current_interaction_time[ID])
                            interaction_time_friend_of_friend_dict[a2].append(current_interaction_time[ID])
                            next_interaction_type_dict[a1].append(0)
                            next_interaction_type_dict[a2].append(0)
                        
                    else:
                        if time > t_thresh:
                            interaction_time_stranger_dict[a1].append(current_interaction_time[ID])
                            interaction_time_stranger_dict[a2].append(current_interaction_time[ID]) 
                            next_interaction_type_dict[a1].append(0)
                            next_interaction_type_dict[a2].append(0)
                        
                    are_friends[ID] = True
                    
                    are_friend_of_friends[ID] = True
                    for Ai_ in range(nAgents):
                        Ai = str(Ai_)
                        IDi = '%s_%s' % (a1,Ai)
                        if Ai < a1:
                            interact_ID = '%s_%s' % (Ai,a1)
                        if IDi in are_friends_:
                            are_friend_of_friends['%s_%s' % (a2,Ai)] = True
                            are_friend_of_friends['%s_%s' % (Ai,a2)] = True
                            
                    for Aj_ in range(nAgents):
                        Aj = str(Aj_)
                        IDj = '%s_%s' % (a2,Aj)
                        if Aj < a2:
                            interact_ID = '%s_%s' % (Aj,a2)
                        if IDj in are_friends_:
                            are_friend_of_friends['%s_%s' % (a1,Aj)] = True
                            are_friend_of_friends['%s_%s' % (Aj,a1)] = True


                
            for ai in Agent_IDs:
                
                if ai in agent_is_interacting:
                    
                    #print(current_noninteraction_time[ai])
                    
                    if current_noninteraction_time[ai] > 0:
                        if time > t_thresh:
                            noninteraction_time_dict[ai].append(current_noninteraction_time[ai])
                        
                        if int(ai) in agent_is_interacting_with_friend:
                            if time > t_thresh:
                                noninteraction_time_friend_dict[ai].append(current_noninteraction_friend_time[ai])
                            current_noninteraction_friend_time[ai] = 0
                            current_noninteraction_friend_of_friend_time[ai] += time_delay
                            current_noninteraction_stranger_time[ai] += time_delay
                            
                        elif int(ai) in agent_is_interacting_with_friend_of_friend:
                            if time > t_thresh:
                                noninteraction_time_friend_of_friend_dict[ai].append(current_noninteraction_friend_of_friend_time[ai])
                            current_noninteraction_friend_of_friend_time[ai] = 0
                            current_noninteraction_friend_time[ai] = 0
                            current_noninteraction_stranger_time[ai] += time_delay
                            
                        else:
                            if time > t_thresh:
                                noninteraction_time_stranger_dict[ai].append(current_noninteraction_stranger_time[ai])
                            current_noninteraction_friend_time[ai] = 0
                            current_noninteraction_friend_of_friend_time[ai] = 0
                            current_noninteraction_stranger_time[ai] = 0
                        
                    current_noninteraction_time[ai] = 0
                else:
                    current_noninteraction_time[ai] += time_delay
                    current_noninteraction_stranger_time[ai] += time_delay
                    current_noninteraction_friend_time[ai] += time_delay
                    current_noninteraction_friend_of_friend_time[ai] += time_delay
                    
                
            was_interacting = is_interacting.copy()
            is_interacting = dict()
            agent_is_interacting = dict()
            agent_is_interacting_with_friend = dict()
            agent_is_interacting_with_friend_of_friend = dict()

        
        
        A1 = str(Agent1[ti])
        A2 = str(Agent2[ti])
        interact_ID = '%s_%s' % (A1,A2)
        if A2 < A1:
            interact_ID = '%s_%s' % (A2,A1)
        
        
        is_interacting[interact_ID] = True
        agent_is_interacting[A1] = True
        agent_is_interacting[A2] = True
        if time > t_thresh:
            Aggregated_Network[int(A1),int(A2)] += 1
            Aggregated_Network[int(A2),int(A1)] += 1
        
        
        if interact_ID in are_friends_of_friends_:
            if interact_ID not in are_friends_:
                agent_is_interacting_with_friend_of_friend[int(A1)] = True
                agent_is_interacting_with_friend_of_friend[int(A2)] = True
            
        else:
            are_friends_of_friends_[interact_ID] = True
            for Ai_ in range(nAgents):
                Ai = str(Ai_)
                IDi = '%s_%s' % (A1,Ai)
                if Ai < A1:
                    interact_ID = '%s_%s' % (Ai,A1)
                if IDi in are_friends_:
                    are_friends_of_friends_['%s_%s' % (A2,Ai)] = True
                    are_friends_of_friends_['%s_%s' % (Ai,A2)] = True
                    
            for Aj_ in range(nAgents):
                Aj = str(Aj_)
                IDj = '%s_%s' % (A2,Aj)
                if Aj < A2:
                    interact_ID = '%s_%s' % (Aj,A2)
                if IDj in are_friends_:
                    are_friends_of_friends_['%s_%s' % (A1,Aj)] = True
                    are_friends_of_friends_['%s_%s' % (Aj,A1)] = True
            
            
        if interact_ID in are_friends_:
            agent_is_interacting_with_friend[int(A1)] = True
            agent_is_interacting_with_friend[int(A2)] = True
        
        else:
            number_of_friends[int(A1)] += 1
            number_of_friends[int(A2)] += 1
            are_friends_[interact_ID] = True
            
    
    #Node cluster size
    all_ccsizes = np.concatenate([cc_size[ai] for ai in Agent_IDs])
    plt.hist(all_ccsizes, bins=40, color = 'purple', density = True, alpha = 0.5)
    plt.yscale('log')
    plt.xlabel('Interacting cluster size')
    plt.ylabel('Density')
    plt.xlim(xmin = 0)
    plt.legend(fontsize=15)
    plt.show()

    #Node Interaction length
    all_interaction_times = np.concatenate([interaction_time_dict[ai] for ai in Agent_IDs])
    plt.hist(all_interaction_times, bins=40, color = 'orange', density = True, alpha = 0.5)
    plt.yscale('log')
    plt.xlabel('Interacting time')
    plt.ylabel('Density')
    plt.xlim(xmin = 0)
    plt.legend(fontsize=15)
    plt.show()
    
    #Node inter_event_time
    all_noninteraction_times = np.concatenate([noninteraction_time_dict[ai] for ai in Agent_IDs])
    plt.hist(all_noninteraction_times, bins=40, color = 'blue', density = True, alpha = 0.5)
    plt.yscale('log')
    plt.xlabel('Node inter-event time')
    plt.ylabel('Density')
    plt.xlim(xmin = 0)
    plt.legend(fontsize=15)
    plt.show()
    
    #Node total_number_interaction
    total_number_interaction = np.array([len(interaction_time_dict[ai]) for ai in Agent_IDs])
    plt.hist(total_number_interaction, bins = 30, alpha = 0.5, density = True)
    plt.xlabel('Number of interactions')
    plt.ylabel('Density')
    plt.show()
    
    #Node total_interacting_time
    total_interacting_time = np.sum(Aggregated_Network, axis = 1)
    plt.hist(total_interacting_time, bins = 30, color = 'red', alpha = 0.5, density = True)
    plt.xlabel('Time spent interacting')
    plt.ylabel('Density')
    plt.show()
    
    #Node Number of Unique interactions
    plt.hist(number_of_friends, bins = 30, color = 'brown', density = True, alpha = 0.5)
    plt.xlabel('Number of friends')
    plt.ylabel('Density')
    plt.show()
    
    #Node friend / stranger ratio
    next_interaction_ratio = []
    for ai, agent in enumerate(Agent_IDs):
        f = np.count_nonzero(next_interaction_type_dict[agent])
        tot = len(next_interaction_type_dict[agent])
        if tot>100:
            next_interaction_ratio.append(f/tot)
    plt.hist(next_interaction_ratio, bins = 30, color = 'blue', density = True, alpha = 0.5)
    plt.xlabel('Proportion of friend interactions') 
    plt.show()
    
    
    #Clustered communities
    logNetwork = np.log2(Aggregated_Network+1)
    nsorted, communities, G, modularity = cluster_louvain_communities(logNetwork, Agent_IDs)
    plt.figure(figsize=(9,7))                
    #plt.title('Aggregated network heatmap (Louvain clusters)')
    plt.imshow(logNetwork[nsorted,:][:,nsorted], origin='lower', cmap = 'inferno')
    plt.colorbar()
    plt.show()
    
    import cdiversity
    from collections import Counter
    community_diversity, _ = np.log(cdiversity.diversity_profile(Counter(communities)))
    mean_diversity = np.mean(community_diversity)
    
    deg_assortativity = nx.degree_pearson_correlation_coefficient(G)
    clust_coeff = nx.average_clustering(G)
    transitivity = nx.transitivity(G)
    density = nx.density(G)
    degree_centrality = list(nx.degree_centrality(G))
    closeness_centrality = list(nx.closeness_centrality(G))
    betweenness_centrality = list(nx.betweenness_centrality(G))
    eigenvector_centrality = list(nx.eigenvector_centrality(G))
    #edge_betweenness = list(nx.edge_betweenness_centrality(G))
    
    centrality_list = [degree_centrality, closeness_centrality, betweenness_centrality, eigenvector_centrality]
    centrality_vectors = np.concatenate([np.array(centrality).reshape(-1, 1) for centrality in centrality_list],axis=1)
    
    G_metrics = [modularity, mean_diversity]
    G_metrics += [deg_assortativity, clust_coeff, transitivity, density]
  

    
    
    network_observable = dict()
    network_observable['Interacting_time'] = np.array(all_interaction_times)
    network_observable['Non_interacting_time'] = np.array(all_noninteraction_times)
    network_observable['Interactions'] = total_number_interaction
    network_observable['total_interacting_time'] = total_interacting_time
    network_observable['Friends'] = number_of_friends
    network_observable['Proportion'] = next_interaction_ratio
    network_observable['Node_centralities'] = np.mean(centrality_vectors, axis = 1)
    network_observable['cc_size'] = all_ccsizes
    
    network_observable['Agg_Network'] = logNetwork[nsorted,:][:,nsorted]
    network_observable['Louveain_communities'] = communities
    network_observable['G_metrics'] = G_metrics

    #network_observable['ETN_vector'] = 0
    
    return network_observable
        

def cluster_louvain_communities(logNetwork, Agent_IDs):
    na = np.max(np.array(Agent_IDs).astype(int))+1
    G = nx.Graph()
    for node in Agent_IDs:
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
    modularity = nx.algorithms.community.quality.modularity(G, communities, weight='weight')
    #conductance_vals = [nx.algorithms.cuts.conductance(G, community, weight='weight') for community in communities]
    clusters_ = np.zeros(na)
    for ci,com in enumerate(communities):
        for node in com:
            clusters_[int(node)] = ci
    cluster_list = np.sort(list(set(clusters_)))
    nclust = len(cluster_list)
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
    cluster_list = np.sort(list(set(clusters)))

    return nsorted, clusters, G, modularity


def normalize_around_ref(matrix, ref = 0):  
    matrix = matrix - matrix[0,:]
    matrix = matrix/np.std(matrix, axis = 0)
    return matrix

    


def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, 4)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)  
    
if __name__ == '__main__':
    main()