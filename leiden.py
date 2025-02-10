import networkx as nx
import matplotlib.pyplot as plt
import string
import numpy as np
import igraph as ig
import leidenalg

def generate_colors(n):
    """Generate n distinct colors"""
    return plt.cm.Set3(np.linspace(0, 1, n)) 

def write_communities_to_file(communities, nodes, filename="abc.txt"):
    """Write communities and their member nodes to a file"""
    community_groups = {}
    for node_idx, comm_id in communities.items():
        if comm_id not in community_groups:
            community_groups[comm_id] = []
        community_groups[comm_id].append(list(nodes)[node_idx])
    
    with open(filename, 'w') as f:
        for comm_id, members in sorted(community_groups.items()):
            f.write(f"Community {comm_id}:\n")
            for member in sorted(members):
                f.write(f"  - {member}\n")
            f.write("\n")

def create_country_graph(filename):

    with open(filename, 'r') as f:
        countries = [line.strip() for line in f]
    
    G = nx.DiGraph()
    G.add_nodes_from(countries)

    letters = list(string.ascii_lowercase)
    colors = generate_colors(len(letters))
    letter_colors = dict(zip(letters, colors))
    
    edges = []
    edge_colors = []
    for country1 in countries:
        for country2 in countries:
            if country1 != country2:
                connecting_letter = country1[-1].lower()
                if connecting_letter == country2[0].lower():
                    edges.append((country1, country2))
                    edge_colors.append(letter_colors[connecting_letter])
    
    G.add_edges_from(edges)
    
    G_undirected = G.to_undirected()
    adj_matrix = nx.adjacency_matrix(G_undirected)
    g_igraph = ig.Graph.Adjacency(adj_matrix.toarray().tolist())
    
    partitions = leidenalg.find_partition(g_igraph, 
                                        leidenalg.RBConfigurationVertexPartition,
                                        resolution_parameter=1.0)
    
    communities = {i: membership for i, membership in enumerate(partitions.membership)}
    write_communities_to_file(communities, G.nodes())
    

    plt.figure(figsize=(24, 24), facecolor='white')
    
    pos = nx.kamada_kawai_layout(G)
    
    num_communities = len(set(communities.values()))
    community_colors = plt.cm.Pastel1(np.linspace(0, 1, num_communities))
    node_colors = [community_colors[communities[i]] for i in range(len(G.nodes()))]

    node_sizes = [4000 * (1 + G.degree(node)) / G.number_of_nodes() 
                 for node in G.nodes()]
    
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          arrows=True,
                          arrowsize=20,
                          alpha=0.6,
                          width=2,
                          connectionstyle="arc3,rad=0.2")

    nodes = nx.draw_networkx_nodes(G, pos,
                                 node_color=node_colors,
                                 node_size=node_sizes,
                                 alpha=0.9,
                                 edgecolors='white',
                                 linewidths=2)
    

    labels = nx.draw_networkx_labels(G, pos,
                                   font_size=10,
                                   font_weight='bold',
                                   bbox=dict(facecolor='white',
                                           edgecolor='none',
                                           alpha=0.7,
                                           pad=0.5))
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=color,
                                 label=f'Community {i}',
                                 markersize=15,
                                 markeredgecolor='white',
                                 markeredgewidth=2)
                      for i, color in enumerate(community_colors)]
    
    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1.1, 0.5),
              title='Communities',
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=12)
    
    plt.title("Country Name Communities", 
              pad=20,
              fontsize=20,
              fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_country_graph("countries.txt")