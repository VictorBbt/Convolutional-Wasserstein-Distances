import networkx as nx
import numpy as np


import sys
sys.path.append("./..")
import graph_utils as graph # Follow the above procedure
import diffusion_utils as diff
import mesh_utils as mesh
from IPython.display import display, HTML


ENTROPY_REG = 0.1




def convolutional_wasserstein(mu_0, mu_1, a, Kernel,  entropy_reg = ENTROPY_REG, n_iter = 2000):
    N = len(mu_0)
    w = np.ones(N)
    Err_mu_0 = []
    Err_mu_1 = []
    
   
    
    for i in range(n_iter):
        
        # sinkhorn step 1
        v = mu_0/ Kernel(a*w)  
         
        # error computation
        Err_mu_1.append(graph.vect_norm(w * (Kernel(a*v)) - mu_1)/graph.vect_norm(mu_1))
        
        # sinkhorn step 2
        w = mu_1 / Kernel(a*v)
        Err_mu_0.append(graph.vect_norm(v * (Kernel(a*w)) - mu_0)/graph.vect_norm(mu_0))

    wass = entropy_reg * a.T @ (mu_0*np.log(np.maximum(1e-19*np.ones(N),v)) + mu_1*np.log(np.maximum(1e-19*np.ones(N),w)) )
    
    # if second order wass distance it returns the square of the true distance
    return wass, Err_mu_0, Err_mu_1
   
    
def wasserstein_barycenter(mu, a,  Kernel, n_iter = 2000, lambd = None):
    
    R = mu.shape[1]
    N = mu.shape[0]
    
    if lambd == None:
        lambd = np.ones(R)/R
    w = np.ones((N,R))
    v = np.copy(w)
    error = np.zeros(n_iter)
    

    for i in range(n_iter):
        
        # First step of the Bergman projection (onto C1)
        target_mu = np.log(np.ones(N))
        for k in range(R):
            # To monitor $\sum_k \norm{PI_k * a - mu_k} $
            error[i] += graph.vect_norm(w[:,k] * Kernel(a*v[:,k]) - mu[:,k])/graph.vect_norm(mu[:,k])
            w[:,k] = mu[:,k]/Kernel(a*v[:,k]) 
        
        
            target_mu +=  lambd[k] * np.log(np.maximum(1e-19*np.ones(len(w[:,k])), v[:,k]*Kernel(a*w[:,k])))
        target_mu = np.exp(target_mu)
                
        # Second projection (onto C2)
        for k in range(R):
            v[:,k] =  target_mu / Kernel(a*w[:,k])
            
    return target_mu, error
    