import networkx as nx
import matplotlib.pyplot as plt
import string
import colorsys

def generate_colors(n):
    return [colorsys.hsv_to_rgb(i/n, 0.7, 0.95) for i in range(n)]

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
    
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    nx.draw_networkx_nodes(G, pos,
                          node_color='lightgray',
                          node_size=2000)
    
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          arrows=True,
                          arrowsize=10)
    
    nx.draw_networkx_labels(G, pos,
                           font_size=8,
                           font_weight='bold')
    
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

graph = create_country_graph('countries.txt')