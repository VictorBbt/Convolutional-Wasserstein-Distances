import matplotlib.pyplot as plt
import networkx as nx

def plot(G,mu):
    pos = nx.spring_layout(G,seed=1)
    N = len(mu) 
    nx.draw_networkx(G, pos = pos, node_color=mu, cmap = 'Greys', vmin = 0, vmax = 1, with_labels = False)
    plt.tight_layout()
    plt.axis("off")
    plt.show()