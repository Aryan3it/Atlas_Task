# louvian.py
import networkx as nx
import matplotlib.pyplot as plt
import string
import numpy as np
import community.community_louvain as community

def generate_colors(n):
    """Generate n distinct colors"""
    return plt.cm.rainbow(np.linspace(0, 1, n))

# Add to main function after creating visualization:
    # write_color_documentation(communities, letter_colors)
def create_country_graph(filename):
    """Create and visualize a directed graph from country names"""
    # Read country names
    with open(filename, 'r') as f:
        countries = [line.strip() for line in f]
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (countries)
    G.add_nodes_from(countries)
    
    # Create color mapping for letters
    letters = list(string.ascii_lowercase)
    colors = generate_colors(len(letters))
    letter_colors = dict(zip(letters, colors))
    
    # Create edges with colors based on connecting letter
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
    
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    
    # Detect communities using Louvain method
    communities = community.best_partition(G_undirected)
    
    # Generate colors for communities with enhanced palette
    num_communities = len(set(communities.values()))
    community_colors = plt.cm.Pastel1(np.linspace(0, 1, num_communities))
    node_colors = [community_colors[communities[node]] for node in G.nodes()]
    
    # Enhanced visualization
    plt.figure(figsize=(20, 20))
    pos = nx.kamada_kawai_layout(G)
    
    # Draw nodes with enhanced styling
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=3000,
                          edgecolors='white',
                          linewidths=2)
    
    # Draw edges with curved arrows
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          arrows=True,
                          arrowsize=20,
                          alpha=0.6,
                          connectionstyle="arc3,rad=0.2")
    
    # Add labels with white background
    nx.draw_networkx_labels(G, pos,
                           font_size=10,
                           font_weight='bold',
                           bbox=dict(facecolor='white',
                                   edgecolor='none',
                                   alpha=0.7,
                                   pad=0.5))
    
    # Add legend for communities
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
              shadow=True)
    
    plt.title("Country Name Communities", 
              pad=20,
              fontsize=20,
              fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # write_color_documentation(communities, letter_colors)

if __name__ == "__main__":
    create_country_graph("countries.txt")