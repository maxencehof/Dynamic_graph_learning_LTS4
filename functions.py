import numpy as np
import networkx as nx

from helpers import *

def from_number_nodes_to_all_L_multiple(N, n_edges, step_prior = 0.01, step_prior_Frob=None, threshold = 0.1, seed_int=42):

    G, _ = create_graph_clustered_and_full_multiple(N, n_edges, seed_int=seed_int)
    _, G1 = create_graph_clustered_and_full_multiple(N, n_edges, seed_int=89*seed_int)
    prior = nx.laplacian_matrix(G).todense()
    L = nx.laplacian_matrix(G1).todense()
    pinv_signal = np.linalg.pinv(L)
    
    n_signals = 100
    epsilon = 1.5/np.sqrt(n_signals)
    # Sample signals
    np.random.seed(42)
    graph_signals = epsilon*np.random.multivariate_normal(mean=np.zeros_like(pinv_signal[0]), cov=pinv_signal, size=n_signals)

    L_wp, L_p, L_pF, prob_wp, prob_p, prob_pF = give_all_Laplacian(graph_signals, prior=prior, step_prior=step_prior, step_prior_Frob=step_prior_Frob, threshold=threshold, retall=True)

    return prior, L, L_wp, L_p, L_pF, graph_signals, prob_wp, prob_p, prob_pF 


def create_graph_clustered_and_full_multiple(N, n_edges, seed_int=42, weighted=False):
    # Create graphs for each small cluster
    graphs = []
    rng = np.random.default_rng(seed_int)
    k=0
    for n in N:
        connected=False
        k=k+1
        while(connected==False):
            tmp_graph = nx.erdos_renyi_graph(n, 0.5, seed=rng)
            rng = np.random.default_rng(seed_int*k)
            k=k+1
            connected = nx.is_connected(tmp_graph)
        graphs.append(tmp_graph)
        

    # Create a new graph
    G = nx.Graph()
    #plot_graph(graphs[1])
    # Add nodes from all small clusters
    offset = 0
    for graph in graphs:
        G.add_nodes_from((n + offset, d) for n, d in graph.nodes(data=True))
        offset += len(graph)

    # Add edges from all small clusters with random weights
    seed_offset = 5*seed_int
    for i, graph in enumerate(graphs):
        np.random.seed(seed_offset + i)
        offset = sum(N[:i])  # Offset for node IDs
        if weighted:
            G.add_weighted_edges_from((u + offset, v + offset, np.random.uniform(0.1, 1.0)) for u, v in graph.edges())
        else:
            G.add_weighted_edges_from((u + offset, v + offset, 1) for u, v in graph.edges())
    
    # Connect consecutive small clusters with an edge
    for i in range(len(graphs)-1):
        np.random.seed(seed_offset+10*i)
        node_a = list(graphs[i].nodes())[-1]
        #node_a = np.random.choice(list(graphs[i].nodes()))
        np.random.seed(seed_offset+20*i)
        node_b = list(graphs[i + 1].nodes())[0]
        #node_b = np.random.choice(list(graphs[i + 1].nodes()))
        offset_a = sum(N[:i])  # Offset for node IDs in the first cluster
        offset_b = sum(N[:i+1])  # Offset for node IDs in the second cluster
        np.random.seed(77*i)  # You can choose any seed here
        if weighted:
            weight = round(np.random.uniform(0.3, 1.0), 2)
        else:
            weight = 1
        G.add_edge(node_a + offset_a, node_b + offset_b, weight=weight)
    if(len(graphs)>2):
        node_a = list(graphs[-1].nodes())[-1]
        #node_a = np.random.choice(list(graphs[i].nodes()))
        node_b = list(graphs[0].nodes())[0]
        offset_a = sum(N[:-1])  # Offset for node IDs in the second cluster
        np.random.seed(374)  # You can choose any seed here
        if weighted:
            weight = round(np.random.uniform(0.3, 1.0), 2)
        else:
            weight = 1
        G.add_edge(node_a+offset_a, node_b, weight=weight)
    # Connect small clusters with the specified number of edges
    G_full = G.copy()
    k = 0
    seed_offset = seed_offset+1
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            k = 0
            while k < n_edges:
                np.random.seed(seed_offset)
                seed_offset = seed_offset+1
                node_a = np.random.choice(list(graphs[i].nodes()))
                np.random.seed(seed_offset)
                seed_offset = seed_offset+1
                node_b = np.random.choice(list(graphs[j].nodes()))
                if not G_full.has_edge(node_a + sum(N[:i]), node_b + sum(N[:j])):
                    np.random.seed(seed_offset)
                    seed_offset = seed_offset+1
                    if weighted:
                        weight = round(np.random.uniform(0.1, 1.0), 2)
                    else:
                        weight = 1
                    G_full.add_edge(node_a + sum(N[:i]), node_b + sum(N[:j]), weight=weight)
                    k += 1
    return G, G_full

def spectral_gap(G):
    # Compute Laplacian matrix
    L = nx.laplacian_matrix(G).todense()

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)

    # Sort eigenvalues in ascending order
    sorted_eigenvalues = np.sort(eigenvalues)

    # Compute spectral gap
    spectral_gap = sorted_eigenvalues[1] - sorted_eigenvalues[0]
    return spectral_gap

def spectral_gap_L(L):
    # Compute Laplacian matrix

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)

    # Sort eigenvalues in ascending order
    sorted_eigenvalues = np.sort(eigenvalues)

    # Compute spectral gap
    spectral_gap = (sorted_eigenvalues[1] - sorted_eigenvalues[0])
    return spectral_gap


def plot_graph(G, pos=None, weight=True, show=True, ax=None):
    # Plot the graph
    plt.figure()
    if pos is None:
        pos = nx.spring_layout(G)  # Layout algorithm

    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Use provided ax or the current axes
    ax = ax or plt.gca()

    if show:
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=200, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=weights, ax=ax)

    if weight:
        labels = {(u, v): f"{w:.2f}" for (u, v, w) in G.edges(data='weight')}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)

    # If show is True and ax is not provided, then show the plot
    if show and ax is None:
        plt.show()

    return pos



def give_all_Laplacian(graph_signals, prior, step_prior=0.1, step_prior_Frob=None, threshold=1e-1, retall=False):
    if step_prior_Frob is None:
        step_prior_Frob = step_prior
    if retall:
        W2_kal, problem_kal = log_degree_barrier((graph_signals).T, alpha=1, beta=1, retall=True)
        W2_kal_prior, problem_kal_prior = log_degree_barrier((graph_signals).T, alpha=1, beta=1, prior=prior, step_prior=step_prior, retall=True)
        W2_kal_prior_Frob, problem_kal_prior_Frob = log_degree_barrier((graph_signals).T, alpha=1, beta=1, prior_Frob=prior, step_prior=step_prior_Frob, retall=True)
    else:
        W2_kal = log_degree_barrier((graph_signals).T, alpha=1, beta=1)
        W2_kal_prior = log_degree_barrier((graph_signals).T, alpha=1, beta=1, prior=prior, step_prior=step_prior)
        W2_kal_prior_Frob = log_degree_barrier((graph_signals).T, alpha=1, beta=1, prior_Frob=prior, step_prior=step_prior_Frob)


    W2_kal_thresholded = W2_kal.copy()
    W2_kal_thresholded[abs(W2_kal_thresholded) < threshold] = 0
    L2_kal_thresholded = W_to_L(W2_kal_thresholded)
    
    W2_kal_prior_thresholded = W2_kal_prior.copy()
    W2_kal_prior_thresholded[abs(W2_kal_prior_thresholded) < threshold] = 0
    L2_kal_prior_thresholded = W_to_L(W2_kal_prior_thresholded)

    W2_kal_prior_thresholded_Frob = W2_kal_prior_Frob.copy()
    W2_kal_prior_thresholded_Frob[abs(W2_kal_prior_thresholded_Frob) < threshold] = 0
    L2_kal_prior_thresholded_Frob = W_to_L(W2_kal_prior_thresholded_Frob)
    if retall:
        return L2_kal_thresholded, L2_kal_prior_thresholded, L2_kal_prior_thresholded_Frob, problem_kal, problem_kal_prior, problem_kal_prior_Frob
    return L2_kal_thresholded, L2_kal_prior_thresholded, L2_kal_prior_thresholded_Frob




def change_graph(G, n, seed_int=42, weighted=False):
    if n > len(G.edges):
        raise ValueError("n cannot be greater than the number of edges in the graph")
    
    rng = np.random.default_rng(seed_int)
    
    # Step 1: Select n random edges to remove
    edges = list(G.edges)
    edges_to_remove = rng.choice(edges, n, replace=False)
    
    # Step 2: Remove these edges from the graph
    G.remove_edges_from(edges_to_remove)
    
    nodes = list(G.nodes)
    new_edges = set()
    
    # Step 3: Add n new edges between random pairs of nodes that are not currently connected
    k = 1
    while len(new_edges) < n:
        rng_step = np.random.default_rng(seed_int + k)
        u, v = rng_step.choice(nodes, 2, replace=False)
        if not G.has_edge(u, v) and (u, v) not in new_edges and (v, u) not in new_edges:
            rng_step_weight = np.random.default_rng(seed_int + k + 1)
            if weighted:
                weight = round(rng_step_weight.uniform(0.3, 1.0), 2)
            else:
                weight = 1
            new_edges.add((u, v))
            G.add_edge(u, v, weight=weight)
        k += 1
    
    return G


def learn_graph_prior(graph_signals, prior, threshold=0.05, step_prior=0.05, verbosity='NONE'):
    return signal_to_laplacian_prior(graph_signals, prior=prior, step_prior=step_prior, threshold=threshold, verbosity=verbosity)


def learn_graph(graph_signals, threshold=0.05,verbosity='NONE'):
    return signal_to_laplacian(graph_signals, threshold=threshold, verbosity=verbosity)


def learn_graph_prior_Frob(graph_signals, prior_Frob, threshold=0.05, step_prior=0.05, verbosity='NONE'):
    return signal_to_laplacian_prior_Frob(graph_signals, prior_Frob=prior_Frob, step_prior=step_prior, threshold=threshold, verbosity=verbosity)




