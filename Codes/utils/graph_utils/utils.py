import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import sys
sys.path.append("./utils/graph_utils/")
from graph import Graph


def vect_norm(x):
    return np.sqrt(x.T@x)

def kernel(G:Graph, entropy_reg = 0.1, deg = 2):
    n = len(G.nodes)
    dist = np.zeros((n,n))
    for (i,source_node) in enumerate(G.nodes):
        for (j,target_node) in enumerate(G.nodes):
            dist[i,j] = nx.shortest_path_length(G,source = source_node, target = target_node)
    if deg == 1:
        return np.exp(-dist/entropy_reg), dist
    else:
        return np.exp(-dist**2/entropy_reg), dist**2