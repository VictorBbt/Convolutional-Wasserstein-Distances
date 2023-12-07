import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix, csc_matrix, find, coo_matrix
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh
from scipy.io import  savemat
import networkx as nx
import pylab as pyl
import time



class Graph(nx.Graph):
    def __init__(self, R, N, graph = None):
        """
        Initialize the mesh from a path
        """
        if graph == None:
            self.G = nx.full_rary_tree(R, N)
        else:
            self.G = graph
        self.nodes = self.G.nodes
        self.edges = self.G.edges
        self.A = self.compute_adj_matrix()
        self.D = None
        self.L = None

# INit from edges list or adjacency dict
    def clear(self):
        self.G.clear()
        self.nodes = None
        self.edges = None

    def get_neighbors(self, node_ind):
        return self.G.adj[node_ind]
    
    def get_degree(self, node_ind):
        return self.G.degree[node_ind]
    
    def get_nodes(self) -> list:
        return list(self.nodes) 

    def get_edges(self) -> list:
        return list(self.edges)
    
    def add_edge(self, node1:int, node2:int):
        self.G.add_edge(node1, node2)
        # Update fields
        self.edges = self.G.edges
        self.nodes = self.G.nodes

    def add_node(self, node:int):
       self.G.add_edge(node)
       # Update fields
       self.edges = self.G.edges
       self.nodes = self.G.nodes

    def compute_adj_matrix(self):
        self.A =  nx.adjacency_matrix(self.G, nodelist=None, dtype=None, weight=None) 
        
    def compute_incidence_matrix(self):
        self.D = nx.incidence_matrix(self.G, nodelist=None, edgelist=None, oriented=False, weight=None)

    def compute_laplacian_matrix(self):
        self.L = nx.normalized_laplacian_matrix(self.G, nodelist=None, weight='weight')
        
    def compute_heat_kernel(self,t):
        if self.L == None:
            self.compute_laplacian_matrix()
        return scipy.linalg.expm(-t*self.L.todense())

        
      
    