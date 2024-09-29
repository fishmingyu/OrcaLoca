import matplotlib.pyplot as plt
import networkx as nx
import PIL

# gnp_random_graph(n, p, seed=None, directed=False)

# Returns a G_{n,p} random graph, also known as an Erdős-Rényi graph or a binomial graph.
# The G_{n,p} model chooses each of the possible edges with probability p.

# Parameters
# n (int) – The number of nodes.
# p (float) – Probability for edge creation.

# Returns
# G – A random graph, also known as an Erdős-Rényi graph or a binomial graph.

G = nx.gnp_random_graph(6, 0.4, directed=True)
DAG = nx.DiGraph([(u, v, {"weight": 1}) for (u, v) in G.edges() if u < v])

# Draw the graph
nx.draw(DAG, with_labels=True)

# save the graph
plt.savefig("random_graph.png")
