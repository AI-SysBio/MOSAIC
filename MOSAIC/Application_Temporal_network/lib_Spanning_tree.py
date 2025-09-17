import networkx as nx 
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import scipy
import os
import itertools
from concurrent.futures import ProcessPoolExecutor





def run_Spanning_tree_simulation(N, Tend = 2000):
    
    m = int(N/2)
    #tol_time = 7249 #number of time steps
    
    tol_time = Tend
    
    recompute_topology = True
    recompute_spanning_tree = True
    
    if recompute_topology:
        #generate_static_topology_fully_connected(N, k)
        generate_static_topology_Barabasi(N, m)
    if recompute_spanning_tree:
        net_trajectory_node, net_trajectory_edge, all_edge_array = compute_spanning_tree(N)
    else:
        net_trajectory_edge = np.load('Temp/net_trajectory_edge' + '_' + str(N) + '.npy')
        
    plot_inter_event_times_from_interactions(net_trajectory_edge, all_edge_array, N,  tol_time)
    plot_interaction_durations(net_trajectory_edge)
    
    return
    generate_interaction_files_by_network(net_trajectory_edge, all_edge_array, 'Spanning_Trees')
    
    
def generate_interaction_files_by_network(net_trajectory_edge, all_edge_array, output_dir):
    """
    Generate separate files for each network containing interactions (timestep, node1, node2).

    Parameters:
    - net_trajectory_edge: 3D numpy array (n_net, num_of_edge, tol_time + 1).
    - all_edge_array: 2D numpy array (num_of_edge, 2) mapping edges to node pairs.
    - output_dir: str, directory to save the output files.
    - tol_time: int, total simulation time.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    n_net = net_trajectory_edge.shape[0]
    tol_time = net_trajectory_edge.shape[2]
    
    for net in range(n_net):  # Loop over networks
        output_file = os.path.join(output_dir, f"interaction_network_{net}.txt")
        with open(output_file, 'w') as file:
            for timestep in range(tol_time):  # Loop over timesteps
                # Get active edges for this timestep
                active_edges = np.where(net_trajectory_edge[net, :, timestep] == 1)[0]
                for edge_idx in active_edges:
                    # Map edge index to node pair
                    node1, node2 = all_edge_array[edge_idx]
                    # Write to file
                    file.write(f"{timestep} {node1} {node2}\n")
        print(f"File for network {net} written to: {output_file}")
        
        
def plot_inter_event_times_from_interactions(net_trajectory_edge, all_edge_array, N, tol_time):
    """
    Calculate and plot the inter-event time distribution for node interactions.
    
    Parameters:
    - net_trajectory_edge: 3D array (n_net, num_of_edge, tol_time+1) with edge activation states.
    - all_edge_array: 2D array mapping edge indices to node pairs.
    - N: Total number of nodes.
    - tol_time: Total simulation time.
    """
    inter_event_times = []
    # Loop through all networks
    for net in range(net_trajectory_edge.shape[0]):
        # Initialize a dictionary to store active times for each node
        node_active_times = {node: [] for node in range(N)}
        
        # Loop through timesteps
        for t in range(tol_time + 1):
            # Identify active edges at this timestep
            active_edges = np.where(net_trajectory_edge[net, :, t] == 1)[0]
            
            # Add this timestep to the active times of nodes involved in these edges
            for edge_idx in active_edges:
                node1, node2 = all_edge_array[edge_idx]
                node_active_times[node1].append(t)
                node_active_times[node2].append(t)
        
        # Calculate inter-event times for each node
        for node, times in node_active_times.items():
            if len(times) > 1:
                # Sort times to ensure order
                sorted_times = sorted(times)
                # Calculate inter-event times
                inter_event_times.extend(np.diff(sorted_times))
    
    # Plot the histogram of inter-event times
    plt.hist(inter_event_times, bins=50, alpha=0.5, density=True)
    plt.xlabel("Inter-event time (steps)")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.show()

        


def plot_inter_event_times(trajectory_node, tol_time):
    """
    Calculate and plot the inter-event time distribution for node interactions.
    
    Parameters:
    - trajectory_node: 3D array (n_net, N, tol_time+1) with node activation states.
    - tol_time: Total simulation time.
    """
    inter_event_times = []

    # Loop through all networks and nodes
    for network in trajectory_node:
        for node_trajectory in network:
            # Identify time steps where the node is active
            active_times = np.where(node_trajectory == 1)[0]
            
            # Calculate inter-event times
            if len(active_times) > 1:
                inter_event_times.extend(np.diff(active_times))
    
    # Plot the histogram of inter-event times
    plt.hist(inter_event_times, bins=50, alpha=0.5, density=True)
    plt.xlabel("Inter-event time (steps)")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.show()
    
    
def plot_interaction_durations(trajectory_edge):
    """
    Calculate and plot the distribution of interaction durations for edges.

    Parameters:
    - trajectory_edge: 3D array (n_net, num_of_edge, tol_time+1) with edge activation states.
    """
    interaction_durations = []

    # Loop through all networks and edges
    for network in trajectory_edge:
        for edge_trajectory in network:
            # Find active periods (consecutive "1"s)
            active_times = np.where(edge_trajectory == 1)[0]
            if len(active_times) > 1:
                # Calculate consecutive differences to find bursts of activity
                gaps = np.diff(active_times)
                durations = [len(list(group)) for k, group in itertools.groupby(gaps, lambda x: x == 1) if k]
                interaction_durations.extend(durations)
    
    # Plot the distribution of interaction durations
    plt.hist(interaction_durations, bins=50, alpha=0.5, density=True)
    plt.xlabel("Interaction duration (steps)")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.show()
    
    
    
def generate_static_topology_fully_connected(N):
    # Generate a fully connected graph (complete graph)
    graph = nx.complete_graph(N)
    root = np.random.choice(range(N))
    generation_node_record, direct_neibour_record, \
                direct_number_record, tree_graph = f_graph_tree_strc(graph, root)
    
    static_adj_mat_origin = graph_to_matrix(graph)
    static_adj_mat_tree = graph_to_matrix(tree_graph)
    
    str1 = 'Temp/static_adj_mat_origin' + '_' + str(N) + '.npy'
    str2 = 'Temp/static_adj_mat_tree' + '_' + str(N) + '.npy'
    
    np.save(str1, static_adj_mat_origin)
    np.save(str2, static_adj_mat_tree)
    
    print(nx.algorithms.tree.recognition.is_tree(tree_graph))
    print(nx.average_clustering(graph))
    
    
    
def generate_static_topology_Barabasi(N, m):
    
    graph = nx.random_graphs.barabasi_albert_graph(N, m)
    root = np.random.choice(range(N))
    generation_node_record, direct_neibour_record, \
                direct_number_record, tree_graph = f_graph_tree_strc(graph, root, N)
    
    static_adj_mat_origin = graph_to_matrix(graph, N)
    static_adj_mat_tree = graph_to_matrix(tree_graph, N)
    
    str1 = 'Temp/static_adj_mat_origin' + '_' + str(N) + '.npy'
    str2 = 'Temp/static_adj_mat_tree' + '_' + str(N) + '.npy'
    
    np.save(str1, static_adj_mat_origin)
    np.save(str2, static_adj_mat_tree)
    
    #print(nx.algorithms.tree.recognition.is_tree(tree_graph))
    #print(nx.average_clustering(graph))


def compute_spanning_tree(N):
    
    
    n_net = 1 #  simulating multiple network realizations
    num_of_edge = N - 1 #a spanning tree always has N - 1 edges for N nodes.
    alpha1 = 1.735 #Shape the temporal dynamics of node activities
    alpha2 = 1.01 #Shape the temporal dynamics of node activities
    tol_time = 7249
    
    
    root = 0
    str1 = 'Temp/static_adj_mat_origin' + '_' + str(N) + '.npy'
    str2 = 'Temp/static_adj_mat_tree' + '_' + str(N) + '.npy'
    matrix = np.load(str2)
    graph = matrix_to_graph(matrix)
    generation_node_record, direct_neibour_record, direct_number_record = \
                                            f_generate_generation(graph, root, N)
    generation_num = len(generation_node_record)
    generation_node_record, generation_num_record = \
                     f_node_generation(generation_node_record, generation_num, N)         
    all_edge_list = graph.edges()
    num_of_edge = len(all_edge_list)
    all_edge_array = np.zeros((num_of_edge, 2), dtype=np.int16)
    tik = 0
    for x, y in all_edge_list:
        if x > y:
            x, y = y, x
        all_edge_array[tik, 0] = x
        all_edge_array[tik, 1] = y
        tik += 1
    alpha = alpha1
    const1 = scipy.special.zetac(alpha) + 1
    cutoff = tol_time + 10
    prob_vec = f_generate_prob_vec(cutoff, alpha, const1)
    cond_prob_vec_node = f_generate_cond_dist(prob_vec, cutoff)
    
    alpha = alpha2
    const2 = scipy.special.zetac(alpha) + 1
    prob_vec = f_generate_prob_vec(cutoff, alpha, const2)
    cond_prob_vec_edge = f_generate_cond_dist(prob_vec, cutoff)
    pool = ProcessPoolExecutor()
    cond_prob_vec_node = [cond_prob_vec_node] * n_net
    cond_prob_vec_edge = [cond_prob_vec_edge] * n_net
    generation_node_record = [generation_node_record] * n_net
    generation_num_record = [generation_num_record] * n_net
    direct_neibour_record = [direct_neibour_record] * n_net
    direct_number_record = [direct_number_record] * n_net
    all_edge_array = [all_edge_array] * n_net
    num_of_edge = [num_of_edge] * n_net
    generation_num = [generation_num] * n_net
    Ns = [N] * n_net
    tol_times = [tol_time] * n_net
    result_list = list(pool.map(f_single_turn, cond_prob_vec_node, cond_prob_vec_edge, 
                  generation_node_record, generation_num_record,
                  direct_neibour_record, direct_number_record,
                  all_edge_array, num_of_edge, generation_num, Ns, tol_times))
    net_trajectory_node = np.zeros((n_net, N, tol_time+1), dtype=np.int8)
    for i in range(n_net):
        net_trajectory_node[i, :, :] = result_list[i][0]

    matrix = np.load(str1)
    graph = matrix_to_graph(matrix)
    all_edge_list = graph.edges()
    num_of_edge_origin = len(all_edge_list)
    all_edge_array = np.zeros((num_of_edge_origin, 2), dtype=np.int16)
    tik = 0
    for x, y in all_edge_list:
        if x > y:
            x, y = y, x
        all_edge_array[tik, 0] = x
        all_edge_array[tik, 1] = y
        tik += 1
    net_trajectory_edge = f_orgin_net_trajectory(net_trajectory_node, 
                                                  all_edge_array,
                                                  num_of_edge_origin, n_net, tol_time)
    
    str1 = 'Temp/net_trajectory_node' + '_' + str(N) + '.npy'
    str2 = 'Temp/net_trajectory_edge' + '_' + str(N) + '.npy'
    np.save(str1, net_trajectory_node)
    np.save(str2, net_trajectory_edge)
    
    
    return net_trajectory_node, net_trajectory_edge, all_edge_array

def matrix_to_graph(matrix):
    num, num = matrix.shape
    graph = nx.empty_graph()
    for x in range(num):
        for y in range(x, num):
            if matrix[x, y] > 0:
                graph.add_edge(x, y)
    return graph

def f_node_neibour(graph, N):  
    number_record = np.zeros(N, dtype=np.int16)
    neibour_record = np.zeros((N, N), dtype=np.int16)-1
    for col_label, row_label in graph.edges():
        neibour_record[col_label, number_record[col_label]] = row_label
        neibour_record[row_label, number_record[row_label]] = col_label
        number_record[col_label] += 1
        number_record[row_label] += 1
    return neibour_record, number_record

def f_generate_generation(graph, root, N):
    neibour_record, number_record = f_node_neibour(graph, N)
    max_val = max(number_record)
    active_node_list = []
    active_node_list.append(root)
    generation_node_record = []
    generation_node_record.append([root])
    direct_neibour_record = np.zeros((N, max_val), dtype=np.int32) - 1
    direct_number_record = np.zeros(N, dtype=np.int32)
    tik = 0
    while len(active_node_list) < N:
        node_list = []
        for vertex in generation_node_record[tik]:
            neibour = neibour_record[vertex, :number_record[vertex]]
            for id_x in neibour:
                if (id_x in active_node_list) == False:
                    node_list.append(id_x)
                    active_node_list.append(id_x)
                    direct_neibour_record[vertex, direct_number_record[vertex]] = id_x
                    direct_number_record[vertex] += 1
        generation_node_record.append(node_list)
        tik += 1
    return generation_node_record, direct_neibour_record, \
           direct_number_record

def f_node_generation(generation_node_record, generation_num, N):
    generation_num_record = np.zeros(generation_num, dtype=np.int16)
    generation_node_record_array = np.zeros((generation_num, N), dtype=np.int16) - 1
    for g in range(generation_num):
        seq1 = generation_node_record[g]
        length = len(seq1)
        generation_num_record[g] = length
        generation_node_record_array[g, :length] = seq1 
    return generation_node_record_array, generation_num_record

@njit
def f_generate_prob_vec(cutoff, alpha, const):
    prob_vec = np.zeros(cutoff)
    for i in range(cutoff):
        prob_vec[i] = 1 / ((i + 1) ** alpha)
    return prob_vec / const

@njit
def f_generate_cond_dist(prob_vec, length):
    cond_prob_vec = np.zeros(length)
    for i in range(length):
        cond_prob_vec[i] = prob_vec[i] / (1 - np.sum(prob_vec[ : i]))
    return cond_prob_vec 

@njit
def f_cal_prob(trajectory, cond_prob_vec, time):
    m = 0
    for i in range(time):
        if trajectory[i] == 1:
            m = i
    index = time - m - 1
    val = cond_prob_vec[index]
    return val 

@njit
def f_cal_dist(val_x, val_y, val_z):
    prob_array = np.zeros(4)
    prob_array[0] = val_z
    prob_array[1] = val_x - val_z
    prob_array[2] = val_y - val_z
    prob_array[3] = 1 + val_z - val_x - val_y
    return prob_array

@njit
def f_find_edge_index(all_edge_array, num_of_edge, seq):
    for i in range(num_of_edge):
        temp = all_edge_array[i, :]
        if np.prod(temp == seq):
            return i
    return -1

@njit
def f_single_turn(cond_prob_vec_node, cond_prob_vec_edge, 
                  generation_node_record, generation_num_record,
                  direct_neibour_record, direct_number_record,
                  all_edge_array, num_of_edge, generation_num, N, tol_time):
    trajectory_node = np.zeros((N, tol_time+1), dtype=np.int8)
    for i in range(N):
        trajectory_node[i, 0] = 1
        
    trajectory_edge = np.zeros((num_of_edge, tol_time+1), dtype=np.int8)
    for i in range(num_of_edge):
        trajectory_edge[i, 0] = 1
        
    for time in range(1, tol_time+1):
        res_array = np.zeros(N, dtype=np.int32) - 1
        for g in range(generation_num-1):
            root_array = generation_node_record[g, :generation_num_record[g]]
            for root in root_array:
                val_root = f_cal_prob(trajectory_node[root, :], cond_prob_vec_node, time)
                if g == 0:
                    rand = np.random.random()
                    if rand < val_root:
                        res_root = 1
                        res_array[root] = 1
                    else:
                        res_root = 0
                        res_array[root] = 0
                else:
                    res_root = res_array[root]
                neibour = direct_neibour_record[root, : direct_number_record[root]]
                for vertex in neibour:
                    if vertex > root:
                        x, y = root, vertex
                    else:
                        x, y = vertex, root
                    seq = np.array([x, y])
                    edge_index = f_find_edge_index(all_edge_array, num_of_edge, seq)
                    val_leaf = f_cal_prob(trajectory_node[vertex, :], cond_prob_vec_node, time)
                    val_edge = f_cal_prob(trajectory_edge[edge_index, :], cond_prob_vec_edge, time)
                    prob_array = f_cal_dist(val_root, val_leaf, val_edge)
                    if np.sum(prob_array < 0) > 0:
                        print(time)
                        print(prob_array)
                        print('Distribution incompatibility!')
                        break
                    if res_root == 1:
                        rand = np.random.random()
                        val = prob_array[0] / val_root
                        if rand < val:
                            res_array[vertex] = 1
                            trajectory_edge[edge_index, time] = 1
                        else:
                            res_array[vertex] = 0
                            trajectory_edge[edge_index, time] = 0
                    else:
                        rand = np.random.random()
                        val = prob_array[2] / (1 - val_root)
                        if rand < val:
                            res_array[vertex] = 1
                            trajectory_edge[edge_index, time] = 0
                        else:
                            res_array[vertex] = 0
                            trajectory_edge[edge_index, time] = 0
        for j in range(N):
            trajectory_node[j, time] = res_array[j]
    return trajectory_node, trajectory_edge

        



@njit
def f_orgin_net_trajectory(net_trajectory_node, all_edge_array, num_of_edge, n_net, tol_time):
    net_trajectory_edge = np.zeros((n_net, num_of_edge, tol_time+1), dtype=np.int8)
    for i in range(n_net):
        for time in range(tol_time+1):
            tik = 0
            for j in range(num_of_edge):
                id_x = all_edge_array[j, 0]
                id_y = all_edge_array[j, 1]
                if time == 0:
                    net_trajectory_edge[i, tik, time] = 1
                else:
                    net_trajectory_edge[i, tik, time] = \
                                 net_trajectory_node[i, id_x, time] *  \
                                 net_trajectory_node[i, id_y, time]
                tik += 1
    return net_trajectory_edge     

    


def f_graph_tree_strc(graph, root, N):
    neibour_record, number_record = f_node_neibour(graph, N)
    max_val = max(number_record)
    active_node_list = []
    active_node_list.append(root)
    generation_node_record = []
    generation_edge_record = []
    generation_node_record.append([root])
    direct_neibour_record = np.zeros((N, max_val), dtype=np.int8) - 1
    direct_number_record = np.zeros(N, dtype=np.int8)
    tik = 0
    while len(active_node_list) < N:
        node_list = []
        edge_list = []
        seq = generation_node_record[tik]
        for vertex in seq:
            neibour = neibour_record[vertex, :number_record[vertex]]
            for id_x in neibour:
                if (id_x in active_node_list) == False:
                    node_list.append(id_x)
                    edge_list.append((vertex, id_x))
                    active_node_list.append(id_x)
                    direct_neibour_record[vertex, direct_number_record[vertex]] = id_x
                    direct_number_record[vertex] += 1
        generation_node_record.append(node_list)
        generation_edge_record.extend(edge_list)
        tik += 1
    tree_graph = nx.empty_graph()
    tree_graph.add_edges_from(generation_edge_record)
    return generation_node_record, direct_neibour_record, \
           direct_number_record, tree_graph
           
def graph_to_matrix(graph, N):
    static_adj_mat = np.zeros((N, N), dtype=np.int16)
    for x, y in graph.edges():
        static_adj_mat[x, y] = 1
        static_adj_mat[y, x] = 1
    return static_adj_mat












if __name__=="__main__":
    run_Spanning_tree_simulation(10)
    
    
    
    




