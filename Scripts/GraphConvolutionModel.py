import numpy as np
import torch
import torch.nn as nn
import networkx as nx

device = "cpu"

def make_adjacency(graph):
    n_nodes = len(graph)
    adj = np.zeros((n_nodes, n_nodes))
    for (i, j) in graph.edges:
        adj[i, j] = 1
    # for i in range(n_nodes):
    #    adj[i, [key for key in x[i].keys()]] = 1
    return adj

class FeedForward(nn.Module):
    def __init__(self, n_in, n_out):
        super(FeedForward, self).__init__()
        self.ffd = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.ffd(x)


class GraphConvolutionModel(nn.Module):
    def __init__(self, A, n_features, n_embd=10):
        super(GraphConvolutionModel, self).__init__()
        self.n_data = np.shape(A)[0]
        self.A = torch.tensor(A) + torch.diag(torch.ones(self.n_data))
        self.D_sqr_inv = torch.diag(1/torch.sqrt(torch.sum(self.A, dim=1)))
        self.A_hat = self.D_sqr_inv @ self.A @ self.D_sqr_inv
        self.n_features = n_features
        self.n_embd = n_embd
        self.ln1 = nn.LayerNorm(n_features)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffd1 = FeedForward(n_features, n_embd)
        self.ffd2 = FeedForward(n_embd, n_features)

    def forward(self, node_feats):
        x = self.ffd(self.ln1(node_feats))
        x = self.A_hat @ x  # message passing
        # skip connection: x = x + node_feats
        return self.ffd2(self.ln2(x))

graph = nx.karate_club_graph()
labels = list(set([d['club'] for d in graph._node.values()]))

labels_to_int = {label: i for i, label in enumerate(labels)}

print(np.shape(graph.nodes))
print(graph.nodes)

graph_features = torch.tensor([labels_to_int[d['club']] for d in graph._node.values()]) # simply label of which group member belongs to
graph_features = torch.unsqueeze(graph_features, dim=1)
adj = make_adjacency(graph)
model = GraphConvolutionModel(adj, n_features=1, n_embd=3)

x_update = model(graph_features)
print(x_update)