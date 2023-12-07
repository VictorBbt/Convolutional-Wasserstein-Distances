import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap

def plot(G,mu):
    pos = nx.spring_layout(G,seed=1)
    YorRd = mpl.colormaps['YlOrRd'].resampled(1000)(np.linspace(0,1,1000))
    YorRd[:10,:] = np.array([0., 0., 0., 1.])
    newcmp = ListedColormap(YorRd)
    N = len(mu) 
    nx.draw_networkx(G, pos = pos, node_color=mu, cmap = newcmp, vmin = 0, vmax = 1, with_labels = False)
    plt.tight_layout()
    plt.axis("off")
    plt.show()