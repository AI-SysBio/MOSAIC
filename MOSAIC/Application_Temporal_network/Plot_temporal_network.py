import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import MDS
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
from collections import Counter
import sys



color_list = ["deepskyblue", "#ff7f0e", "#6b5b95", "#88b04b", "#f7cac9", "#1f77b4", "#92a8d1", "#955251", "#b565a7", 
              "#009b77", "#dd4124", "#45b8ac", "#e94b3c", "#5b5ea6", "#9b2335", "#d65076", "#2ca02c", 
              "#e15d44", "#7fcdcd", "#bc243c", "#98b4d4", "#ff6f61", "#deeaee", "#c94c4c"]

color_list = np.tile(color_list, 1000)





def find_best_match(component, list_of_components):
    
    n_commons = []
    for key,comp in list_of_components.items():
        comm = len(set(component).intersection(set(comp)))
        n_commons.append(comm)

    imax = np.argmax(n_commons)
    best_match = list(list_of_components.keys())[imax]
    
    return best_match


def main():
    
    
    network_df = pd.read_csv('Data/Networks/Conf17.csv', sep = ' ', names = ['timestamp', 'N1', 'N2'])
    timestamp = network_df['timestamp'].to_numpy()
    N1 = network_df['N1'].to_numpy()
    N2 = network_df['N2'].to_numpy()
    
    Tend = np.max(timestamp)  # Max time for plotting
    N = max(np.max(N1),np.max(N2))+1
    grouped = network_df.groupby('timestamp')
    
    #G = nx.Graph()
    #for ni in range(N):
        #G.add_node(ni)
    
    #pos = nx.spring_layout(G, seed=42)  # Use a layout algorithm with a fixed seed for reproducibility
    grouped = network_df.groupby('timestamp')
    for t, group in grouped:
        if t > Tend:
            break
        
        if t >= 5000:
            plt.clf()
            plt.figure(figsize=(8, 8))
        
            # Add edges for the current time step
              # Use spring layout (or try Kamada-Kawai/spectral layout)
            #G.remove_edges_from(list(G.edges))  # Remove all edges
            G = nx.Graph()
            G.add_edges_from(zip(group['N1'], group['N2']))  # Add edges for the current time step
            pos = nx.spring_layout(G, seed=42)
            # Generate a new layout for this timestep
            
        
            # Draw the graph with the new layout
            nx.draw(
                G, pos, with_labels=False, node_color='lightblue', node_size=200, edge_color='black', font_weight='bold'
            )
        
            # Add a title showing the current timestamp
            plt.title(f"Graph at Timestamp {t}")
            plt.pause(0.5)
    
    plt.show()
    
    sys.exit()
        
    
    

    Tend = np.max(timestamp)
    Tend = 200
    existing_components = dict()
    node_component = np.zeros(N).astype(int)
    for i in range(N):
        node_component[i] = i
    for i in range(N):
        existing_components[i] = [i]
        
        
    
    node_component_list = []
    previosuly_connected_nodes = []
    for t, group in grouped:
        if t > Tend:
            break
        G = nx.Graph()
        G.add_edges_from(zip(group['N1'], group['N2']))
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        next_max = np.max(node_component) + 1
        connected_nodes = []
        for ci,comp in enumerate(components):
            
            best_comp_id = find_best_match(comp, existing_components)
            for i in comp:
                node_component[i] = best_comp_id
                connected_nodes.append(i)
        for i in range(N):
            if i not in connected_nodes and i in previosuly_connected_nodes:
                node_component[i] = next_max #Make sure its a component ID that was never used
        existing_components = dict()
        for i in range(np.max(node_component)+1):
            existing_components[i] = []
        for i in range(N):
            existing_components[node_component[i]].append(i)
        previosuly_connected_nodes = connected_nodes
        node_component_list.append(np.copy(node_component))

    
    #rename the components by aggregated time, and set to -1 the ones that are in isolation
    comp_count = dict()
    for ti in range(1,Tend+1):
        counter = Counter(node_component_list[ti])
        for comp in counter:
            if counter[comp] > 1: #Dont count non-interacting clusters
                if comp in comp_count:
                    comp_count[comp] += counter[comp]
                else:
                    comp_count[comp] = counter[comp]
                
    sorted_count = dict(sorted(comp_count.items(), key=lambda item: item[1], reverse=True))
    comp_rank = {key: rank + 1 for rank, (key, value) in enumerate(sorted_count.items())}
    new_node_component_list = []
    for node_components in node_component_list:
        unique_values, counts = np.unique(node_components, return_counts=True)
        unique_indices = [i for i, value in enumerate(node_components) if counts[np.where(unique_values == value)[0][0]] == 1]
        new_node_components = np.zeros(N).astype(int)
        for ni in range(N):
            if ni in unique_indices:
                new_node_components[ni] = -1
            else:
                comp = node_components[ni]
                new_node_components[ni] = comp_rank[comp]
        new_node_component_list.append(new_node_components)
        
        
    def rank_by_frequency(integer_list):
        frequency_counts = Counter(integer_list)
        sorted_by_frequency = [item[0] for item in frequency_counts.most_common()]
        integer_mapping = {original: rank + 1 for rank, original in enumerate(sorted_by_frequency)}
        renamed_list = np.array([integer_mapping[num] for num in integer_list])
        return renamed_list
    
    
    ranked_node_component_list = []
    size_of_components_list = []
    for ti in range(Tend):
        node_pos = rank_by_frequency(new_node_component_list[ti])
        node_pos[np.array(new_node_component_list[ti]) == -1]  = 0
        ranked_node_component_list.append(node_pos)
        size_of_components_list.append(Counter(node_pos))
        
        
        
    comp_rank_dict = dict()
    component_list = set()
    for ti in range(Tend):
        counter = Counter(new_node_component_list[ti])
        sorted_count = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        comp_rank_dict[ti] = dict()
        
        rank = 0
        for comp,count in sorted_count.items():
            component_list.add(comp)
            if rank >= 1:
                comp_rank_dict[ti][comp] = rank
            rank += 1
        


    t = np.arange(Tend)
    size_below = dict()
    comp_size_dict = dict()
    #comp_size_rank = dict()
    for ti in range(Tend):
        counter = Counter(ranked_node_component_list[ti])
        sorted_count = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))
        size_below[ti] = dict()
        comp_size_dict[ti] = dict()
        
        cur_val = 0
        rank = 0
        for comp,count in sorted_count.items():
            if rank >= 1:
                comp_size_dict[ti][rank] = count
                size_below[ti][rank] = cur_val
                cur_val += count
            rank += 1


    #print(size_below)
    plt.figure(figsize=(13, 4))
    #print(component_list)
    from scipy.interpolate import UnivariateSpline
    t_array = np.arange(0, Tend)  # Original indices
    t_inter = np.linspace(0, Tend, 1000)
    for ci, comp in enumerate(component_list):
        comp_size = []
        mean_pos = []
        for ti in range(Tend):
            if comp in comp_rank_dict[ti]:
                rank = comp_rank_dict[ti][comp]
                comp_size.append(comp_size_dict[ti][rank])
                mean_pos.append(size_below[ti][rank])
            else:
                comp_size.append(0)
                mean_pos.append(0)
                
        comp_size = np.array(comp_size)
        mean_pos = np.array(mean_pos)
        ymin = mean_pos 
        ymax = mean_pos + comp_size -0.8
        
        fit_1D = False
        if fit_1D:
            ymin_it = UnivariateSpline(t_array, ymin, s=1)(t_inter)
            ymax_it = UnivariateSpline(t_array, ymax, s=1)(t_inter)
            ymax_it[ymax_it < 0.5] = np.nan
            ymax_it[ymax_it < 0.5] = np.nan
            plt.fill_between(t_inter[:-100], ymin_it[:-100], ymax_it[:-100], alpha = 1) #, color = color_list[ci]
        else:
            plt.fill_between(t_array, ymin, ymax, alpha = 1) #, color = color_list[comp]
        
        
    plt.xlabel('Time')
    #plt.yticks([],[])
    #plt.fill_between([0, Tend], [-0.25,-0.25], [0.25,0.25], color = 'black', alpha = 1)
    
    sys.exit()
        
        
        

        
    
    
        
        
    from scipy.interpolate import UnivariateSpline
    node_component_array = np.array(ranked_node_component_list).T  # Shape: (num_vectors, vector_length)
    original_x = np.linspace(0, Tend-1, node_component_array.shape[1])  # Original indices
    interpolated_x = np.linspace(0, Tend-1, 500)  # Desired finer grid
    
    smoothed_array = np.array([
        UnivariateSpline(original_x, row, s=10)(interpolated_x) for row in node_component_array
    ])
    
    plt.figure(figsize=(13, 4))
    for i, original in enumerate(node_component_array):
        plt.plot(interpolated_x, smoothed_array[i], color = 'black', lw = 1)
    plt.xlabel('Time')
    plt.fill_between([0, Tend], [-0.25,-0.25], [0.25,0.25], color = 'black', alpha = 1)
    
    sys.exit()
        
    
    
         
    plt.figure(figsize=(13, 4))
    for ti in range(1,Tend):
        node_pos = ranked_node_component_list[ti]
        prev_node_pos = ranked_node_component_list[ti-1]
        for ni in range(N):
            plt.plot([ti-1,ti], [prev_node_pos[ni], node_pos[ni]], color = 'black', lw = 1)
        
    plt.xlabel('Time')
    #plt.ylabel('Interacting components')
    #plt.yticks([],[])
    plt.fill_between([0, Tend], [0,0], [0.5,0.5], color = 'black', alpha = 1)
    plt.show()
    
    sys.exit()
        
    prev_node_pos = np.arange(N)
    prev_node_pos[prev_node_pos>10] = -2
    for ti in range(1,Tend):
        node_pos = node_component_list[ti]
        node_pos[node_pos>10] = -2
        for ni in range(N):
            plt.plot([ti-1,ti], [prev_node_pos[ni]*3, node_pos[ni]*3], color = 'black', lw = 0.5)
            
        prev_node_pos = node_pos
        
    plt.xlabel('Time')
    #plt.ylabel('Interacting components')
    plt.yticks([],[])
    plt.show()
    
    
    print_cluster_activity = True
    if print_cluster_activity:
        Tend = np.max(timestamp)  # Max time for plotting
        #simple population plot
        plt.figure(figsize=(13, 4))
        for t, group in grouped:
            if t > Tend:
                break
            G = nx.Graph()
            G.add_edges_from(zip(group['N1'], group['N2']))
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            comp_sizes = [len(comp) for comp in components]
    
            cdist = 0
            old_cdist = 0
            for ci,comp in enumerate(components):
                cdist = cdist + comp_sizes[ci]
                plt.plot([t,t], [old_cdist, cdist], lw = 2, color = color_list[ci])
                old_cdist = cdist
        plt.xlabel('Time')
        plt.ylabel('Interacting components')
        plt.show()
    
    sys.exit()
    

    #Now plot the final result 
    from matplotlib.colors import ListedColormap
    node_component_array = np.array(new_node_component_list).T
    num_classes = np.max(node_component_array) + 1  # Number of unique integers
    colors = ["#000000"] + list(plt.cm.tab20.colors[:num_classes - 1])   # Start with black for index 0, followed by distinct colors
    custom_cmap = ListedColormap(colors[:num_classes])
    plt.imshow(node_component_array, cmap = custom_cmap)
    plt.xlabel('Time')
    plt.ylim(0,50)
    plt.xlim(0,200)
    plt.show()
    
    
    sys.exit()
    
    
    plt.figure(figsize=(13, 4))
    prev_node_pos = np.arange(N)
    prev_node_pos[prev_node_pos>10] = -2
    for ti in range(1,Tend):
        node_pos = node_component_list[ti]
        node_pos[node_pos>10] = -2
        for ni in range(N):
            plt.plot([ti-1,ti], [prev_node_pos[ni]*3, node_pos[ni]*3], color = 'black', lw = 0.5)
            
        prev_node_pos = node_pos
        
    plt.xlabel('Time')
    #plt.ylabel('Interacting components')
    plt.yticks([],[])
    plt.show()
    
    
    #TODO: plot the node_component smartly
    #  - For start, leave a blank between each node (do for example 3 times more space)
    #  - Then set the position of the biggest cluster at the borrom, size size is proportional to its size
    
    
    sys.exit()    
    

        
    
    
    
    
    
    
    sys.exit()
    
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    for t, group in grouped:
        if t > Tend:
            break
        for _, row in group.iterrows():
            plt.plot([t, t], [row['N1'], row['N2']], color='black', alpha=0.5)
    
    # Add labels and improve visuals
    plt.xlabel('Time')
    plt.ylabel('Interactions')
    plt.title('Dynamic Network Visualization')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
    sys.exit()
    
    ni = len(network_df)
    Tend = 200
    
    t = 0
    i = 0
    plt.figure(figsize = (12,4))
    while t < Tend and i < ni:   
        nwhere = np.where(t == timestamp)[0]
        for j in nwhere:
            plt.plot([t,t], [N1[j], N2[j]], color = 'black')
            i = j
        t += 1  
    plt.xlabel('Time')
    plt.ylabel('Interactions')
    plt.show()
        
    
    
    
    
    sys.exit()
    
    print(network_df)
    
    
    """
    The x-axis shows the spatial layout of nodes (based on their proximity).
    The y-axis represents time steps.
    Edges between nodes indicate proximity relationships during specific time steps.
    """
    
    
    
    
    # Example: Simulated dynamic network data
    time_steps = [
        {"time": 1, "edges": [("A", "B"), ("B", "C")]},
        {"time": 2, "edges": [("A", "B"), ("C", "D")]},
        {"time": 3, "edges": [("B", "C"), ("C", "D"), ("A", "D")]},
    ]
    
    default_dist = 6  # Example proximity threshold (not used in static demo)
    
    # Create a list of graphs for each time step
    graphs = []
    for step in time_steps:
        G = nx.Graph()
        G.add_edges_from(step["edges"])
        graphs.append((step["time"], G))
    
    # Dimensionality reduction for node layout (Sammon-like projection)
    layout = MDS(n_components=2, random_state=42)
    
    # Plot proximity timeline
    fig, ax = plt.subplots(figsize=(12, 6))
    
    node_positions = {}  # Store node positions for consistency
    for time, G in graphs:
        nodelist = list(G.nodes)  # Dynamic nodelist for each graph
        if len(nodelist) > 1:  # Avoid MDS errors with single-node graphs
            adj_matrix = nx.adjacency_matrix(G, nodelist=nodelist).toarray()
            new_positions = layout.fit_transform(adj_matrix)  # MDS layout based on proximity
            for i, node in enumerate(nodelist):
                node_positions[node] = new_positions[i]
    
        # Plot edges for the current graph
        for edge in G.edges:
            x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
            y_coords = [time, time]
            ax.plot(x_coords, y_coords, 'k-', alpha=0.5)  # Edge lines
    
        # Plot nodes for the current graph
        for node in G.nodes:
            ax.scatter(node_positions[node][0], time, s=100, label=node)
    
    # Customizing plot
    ax.set_xlabel("Node Layout (Proximity in Reduced Space)")
    ax.set_ylabel("Time Steps")
    ax.set_title("Proximity Timeline")
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()

    
    
if __name__ == '__main__':
    main()