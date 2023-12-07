import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix, csc_matrix, find, coo_matrix
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh
from scipy.io import  savemat



import sys
sys.path.append("./..")
import mesh_utils as mesh # Follow the above procedure
import graph_utils as graph
from IPython.display import display, HTML



def diffuse_mesh(f, mesh, t, steps = -1):
    """
    Diffuse a function f on a mesh for time t
    
    Input
    --------------
    f       : (n,) - function values
    mesh    : MyMesh - mesh on which to diffuse
    t       : float - time for which to diffuse
    
    Output
    --------------
    f_diffuse : (n,) values of f after diffusion
    """
    mesh.compute_laplacian()
    
    if steps<=0:
        f_diffuse = scipy.sparse.linalg.spsolve(mesh.A + t*mesh.W, mesh.A@f)
        
    else:
        h = t/steps
        f_diffuse = scipy.sparse.linalg.spsolve(mesh.A + h*mesh.W, mesh.A@f)
        
        for i in range(steps):
            f_diffuse = scipy.sparse.linalg.spsolve(mesh.A + h*mesh.W, mesh.A@f_diffuse)
   
    return f_diffuse


def diffuse_graph(f, graph:graph.Graph, t, steps = -1):
    """
    Diffuse a function f on a mesh for time t
    
    Input
    --------------
    f       : (n,) - function values
    graph    : Graph - graph on which to diffuse
    t       : float - time for which to diffuse
    
    Output
    --------------
    f_diffuse : (n,) values of f after diffusion
    """
    if graph.A == None:
        graph.compute_adj_matrix()

    graph.compute_laplacian_matrix()
    
    if steps<=0:
        # f_diffuse = scipy.sparse.linalg.spsolve(graph.A + t*graph.L, graph.A@f)
        f_diffuse = scipy.sparse.linalg.spsolve(np.eye(graph.A.shape[0]) - t*graph.L, graph.A@f)
        
    else:
        h = t/steps
        f_diffuse = scipy.sparse.linalg.spsolve(graph.A + h*graph.L, graph.A@f)
        
        for i in range(steps):
            f_diffuse = scipy.sparse.linalg.spsolve(graph.A + h*graph.L, graph.A@f_diffuse)
   
    return f_diffuse