import networkx as nx
import matplotlib.pyplot as plt
import string
import numpy as np
from collections import defaultdict

def analyze_country_connections(filename):
    # Read countries and create graph
    with open(filename, 'r') as f:
        countries = [line.strip() for line in f]
    
    G = nx.DiGraph()
    G.add_nodes_from(countries)
    
    # Create edges
    letter_connections = defaultdict(int)
    for country1 in countries:
        for country2 in countries:
            if country1 != country2:
                if country1[-1].lower() == country2[0].lower():
                    G.add_edge(country1, country2)
                    letter_connections[country1[-1].lower()] += 1
    
    return G, letter_connections

def analyze_graph_statistics(G):
    stats = {
        'Total Countries': G.number_of_nodes(),
        'Total Connections': G.number_of_edges(),
        'Average Connections': G.number_of_edges() / G.number_of_nodes(),
        'Most Connected': max(dict(G.degree()).items(), key=lambda x: x[1])[0],
        'Isolated Countries': list(nx.isolates(G))
    }
    return stats

def analyze_country_ratios(G):
    country_ratios = {}
    for country in G.nodes():
        in_deg = G.in_degree(country)
        out_deg = G.out_degree(country)
        total_deg = in_deg + out_deg
        if total_deg > 0:
            ratio = in_deg / total_deg
            country_ratios[country] = {
                'ratio': ratio,
                'in_degree': in_deg,
                'out_degree': out_deg,
                'total': total_deg
            }
        else:
            country_ratios[country] = {
                'ratio': 0,
                'in_degree': 0,
                'out_degree': 0,
                'total': 0
            }
    return country_ratios

def visualize_beautiful_graph(G, country_ratios):
    # Create figure with a specific size and gridspec for colorbar
    fig = plt.figure(figsize=(24, 24), facecolor='white')
    gs = fig.add_gridspec(1, 20)  # 20 columns grid
    ax_main = fig.add_subplot(gs[:, :-1])  # Main plot uses all but last column
    ax_cbar = fig.add_subplot(gs[:, -1])  # Colorbar uses last column
    
    # Use kamada_kawai_layout for better distribution
    pos = nx.kamada_kawai_layout(G)
    
    # Node sizes and colors based on ratio
    node_sizes = []
    node_colors = []
    for node in G.nodes():
        ratio = country_ratios[node]['ratio']
        total_connections = country_ratios[node]['total']
        node_sizes.append(4000 * (1 + total_connections) / G.number_of_nodes())
        node_colors.append(plt.cm.RdYlBu(ratio))
    
    # Draw edges with curved arrows
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          alpha=0.4,
                          width=1.5,
                          connectionstyle="arc3,rad=0.2",
                          ax=ax_main)
    
    # Draw nodes with white borders
    nodes = nx.draw_networkx_nodes(G, pos,
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 alpha=0.9,
                                 edgecolors='white',
                                 linewidths=2,
                                 ax=ax_main)
    
    # Add labels with white background
    labels = nx.draw_networkx_labels(G, pos,
                                   font_size=10,
                                   font_weight='bold',
                                   bbox=dict(facecolor='white',
                                           edgecolor='none',
                                           alpha=0.7,
                                           pad=0.5),
                                   ax=ax_main)
    
    # Add colorbar legend
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Incoming/Outgoing Ratio', size=12)
    
    ax_main.set_title("Country Connections Network",
                     pad=20,
                     fontsize=20,
                     fontweight='bold')
    
    ax_main.axis('off')
    plt.tight_layout()

def main(filename):
    G, letter_connections = analyze_country_connections(filename)
    
    # Display statistics
    stats = analyze_graph_statistics(G)
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nMost Common Connecting Letters:")
    sorted_letters = sorted(letter_connections.items(), key=lambda x: x[1], reverse=True)
    for letter, count in sorted_letters[:5]:
        print(f"'{letter}': {count} connections")
    
    # Display country ratios
    print("\nCountry Connection Ratios:")
    country_ratios = analyze_country_ratios(G)
    sorted_countries = sorted(country_ratios.items(), 
                            key=lambda x: x[1]['ratio'], 
                            reverse=True)
    
    print("\n{:<20} {:<10} {:<10} {:<10} {:<10}".format(
        "Country", "Ratio", "In", "Out", "Total"))
    print("-" * 60)
    
    for country, data in sorted_countries:
        print("{:<20} {:<10.2f} {:<10d} {:<10d} {:<10d}".format(
            country[:19], 
            data['ratio'],
            data['in_degree'],
            data['out_degree'],
            data['total']
        ))
    
    # Visualize the graph
    visualize_beautiful_graph(G, country_ratios)
    plt.show()

if __name__ == "__main__":
    main("countries.txt")