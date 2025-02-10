import networkx as nx
import matplotlib.pyplot as plt
import string
import colorsys

def generate_colors(n):
    return [colorsys.hsv_to_rgb(i/n, 0.7, 0.95) for i in range(n)]

def create_country_graph(filename):
    # Read countries from file
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
    
    # Draw graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color='lightgray',
                          node_size=2000)
    
    # Draw edges with colors
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          arrows=True,
                          arrowsize=10)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos,
                           font_size=8,
                           font_weight='bold')
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], color=color, label=letter.upper())
                      for letter, color in letter_colors.items()
                      if any(country[-1].lower() == letter for country in countries)]
    
    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              title='Connecting Letters')
    
    plt.title("Country Name Connection Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G

# Example usage
graph = create_country_graph('countries.txt')