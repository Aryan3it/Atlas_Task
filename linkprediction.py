import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from node2vec import Node2Vec
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import time

class LinkPredictionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

def create_features(countries):
    """Create letter-based node features"""
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    features = []
    
    for country in countries:
        first_letter = np.zeros(26)
        last_letter = np.zeros(26)
        
        first_idx = alphabet.index(country[0].lower())
        last_idx = alphabet.index(country[-1].lower())
        
        first_letter[first_idx] = 1
        last_letter[last_idx] = 1
        
        length_feature = np.array([len(country) / 20.0])
        
        feature = np.concatenate([first_letter, last_letter, length_feature])
        features.append(feature)
    
    return np.array(features)

def evaluate_node2vec(model, G, test_edges, test_non_edges):
    """Enhanced Node2Vec evaluation with weighted scoring"""
    embeddings = {}
    for node in G.nodes():
        embeddings[node] = model.wv[str(node)]
    
    scores = []
    labels = []
    
    for edge in test_edges:
        cosine_sim = np.dot(embeddings[edge[0]], embeddings[edge[1]]) / \
                    (np.linalg.norm(embeddings[edge[0]]) * np.linalg.norm(embeddings[edge[1]]))
        euclidean_dist = np.linalg.norm(embeddings[edge[0]] - embeddings[edge[1]])
        
        # Weighted score
        score = 0.7 * cosine_sim + 0.3 * (1 / (1 + euclidean_dist))
        scores.append(score)
        labels.append(1)
    
    for edge in test_non_edges:
        cosine_sim = np.dot(embeddings[edge[0]], embeddings[edge[1]]) / \
                    (np.linalg.norm(embeddings[edge[0]]) * np.linalg.norm(embeddings[edge[1]]))
        euclidean_dist = np.linalg.norm(embeddings[edge[0]] - embeddings[edge[1]])
        
        score = 0.7 * cosine_sim + 0.3 * (1 / (1 + euclidean_dist))
        scores.append(score)
        labels.append(0)
    
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap

def train_node2vec(G, dimensions=128, walk_length=40, num_walks=400):
    """Train Node2Vec model with optimized parameters for directed letter graphs"""
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    
    node2vec = Node2Vec(
        G, 
        dimensions=dimensions,
        walk_length=walk_length,  
        num_walks=num_walks,      
        workers=4,
        p=0.25,  
        q=2.0    
    )
    
    
    model = node2vec.fit(
        window=15,       
        min_count=1,
        batch_words=4,
        sg=1,          
        epochs=20,      
        alpha=0.025     
    )
    
    return model

def main(filename):

    print("\nLoading and preprocessing data...")
    with open(filename, 'r') as f:
        countries = [line.strip() for line in f]
    
    country_to_idx = {country: idx for idx, country in enumerate(countries)}
    
    edges = []
    for c1 in countries:
        for c2 in countries:
            if c1 != c2 and c1[-1].lower() == c2[0].lower():
                edges.append((country_to_idx[c1], country_to_idx[c2]))
    
    features = create_features(countries)
    
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edges).t()
    
    data = Data(x=x, edge_index=edge_index)
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    
    train_data, val_data, test_data = transform(data)
    
    test_pos_edges = test_data.edge_label_index[:, test_data.edge_label == 1].t().numpy()
    test_neg_edges = test_data.edge_label_index[:, test_data.edge_label == 0].t().numpy()
    print("\n=== Training Node2Vec ===")
    n2v_start_time = time.time()
    G = nx.DiGraph()
    G.add_nodes_from(range(len(countries)))
    G.add_edges_from(edges)
    n2v_model = train_node2vec(G)
    n2v_train_time = time.time() - n2v_start_time
    
    n2v_auc, n2v_ap = evaluate_node2vec(n2v_model, G, test_pos_edges, test_neg_edges)
    
    print("\n=== Training GNN ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinkPredictionGNN(in_channels=53, hidden_channels=64).to(device)  
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        out = model.decode(z, train_data.edge_label_index)
        loss = criterion(out, train_data.edge_label)
        loss.backward()
        optimizer.step()
        return float(loss)
    
    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index)
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    
    gnn_start_time = time.time()
    best_val_auc = 0
    for epoch in range(100):
        loss = train()
        if (epoch + 1) % 10 == 0:
            val_auc = test(val_data)
            test_auc = test(test_data)
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, '
                  f'Val: {val_auc:.4f}, Test: {test_auc:.4f}')
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
    
    gnn_train_time = time.time() - gnn_start_time
    final_test_auc = test(test_data)
    
    print("\n=== Final Results ===")
    print("\nModel Performance:")
    print("-" * 60)
    print(f"{'Metric':<20} {'Node2Vec':<15} {'GNN':<15}")
    print("-" * 60)
    print(f"{'Test AUC':<20} {n2v_auc:.4f}{'':<15} {final_test_auc:.4f}")
    print(f"{'Training Time (s)':<20} {n2v_train_time:.2f}{'':<15} {gnn_train_time:.2f}")
    print("-" * 60)
    
    print("\nModel Characteristics:")
    print("Node2Vec:")
    print("- Uses pure graph structure")
    print("- Fast training")
    print("- Memory efficient for sparse graphs")
    
    print("\nGNN:")
    print("- Uses both graph structure and node features")
    print("- Can capture complex patterns")
    print("- More suitable for inductive learning")
    
    print("\nRecommendation:")
    if n2v_auc > final_test_auc:
        print("Node2Vec performs better for this specific task.")
    else:
        print("GNN performs better for this specific task.")

if __name__ == "__main__":
    main("countries.txt")