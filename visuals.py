import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# Load the Facebook data
facebook = pd.read_csv(
    r"C:\Users\venka\Downloads\gplus_combined.txt.gz",
    compression="gzip",
    sep=" ",
    names=["start_node", "end_node"],
)

# Create the graph from the edge list
G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")

# Detect communities using the greedy modularity algorithm
communities = greedy_modularity_communities(G)

# Create a mapping of node to community
community_map = {}
for community_id, community in enumerate(communities):
    for node in community:
        community_map[node] = community_id

# Assign a unique color to each community
num_communities = len(communities)
community_colors = {i: plt.cm.tab20(i % 20) for i in range(num_communities)}
node_colors = [community_colors[community_map[node]] for node in G.nodes()]

# Calculate the spring layout positions
pos = nx.spring_layout(G, iterations=15, seed=1721)

# Plot the graph with the spring layout and community colors
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")

nx.draw_networkx(
    G,
    pos=pos,
    ax=ax,
    node_size=10,
    node_color=node_colors,
    with_labels=False,
    width=0.15,
)

plt.title(f"Graph Visualization with {num_communities} Communities", fontsize=16)
plt.show()
