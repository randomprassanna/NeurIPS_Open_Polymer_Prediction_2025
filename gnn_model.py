import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm

class PolymerGNN(nn.Module):
    def __init__(self, num_node_features=7, num_edge_features=3, global_features_dim=10, 
                 hidden_dim=128, num_layers=4, num_targets=5, dropout=0.2):
        super(PolymerGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node embedding
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        
        # Edge embedding
        self.edge_embedding = nn.Linear(num_edge_features, hidden_dim)
        
        # GNN layers (using GAT for attention mechanism)
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, 
                                              edge_dim=hidden_dim, dropout=dropout))
            else:
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, 
                                              edge_dim=hidden_dim, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Graph-level representation
        self.graph_conv = nn.Linear(hidden_dim * 3, hidden_dim)  # 3 pooling methods
        
        # Global features processing
        self.global_fc = nn.Sequential(
            nn.Linear(global_features_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        global_features = data.global_features
        if len(global_features.shape) == 1:
            batch_size = torch.max(batch).item() + 1
            global_features = global_features.view(batch_size, -1)
        
        # Embed nodes and edges
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply GNN layers
        for i in range(self.num_layers):
            x_residual = x
            x = self.gnn_layers[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            
            # Residual connection
            if x.size() == x_residual.size():
                x = x + x_residual
        
        # Graph-level pooling
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_add = global_add_pool(x, batch)
        
        # Combine different pooling methods
        graph_repr = torch.cat([graph_mean, graph_max, graph_add], dim=1)
        graph_repr = self.graph_conv(graph_repr)
        graph_repr = F.relu(graph_repr)
        # Process global molecular descriptors
        global_repr = self.global_fc(global_features)
        
        # Combine graph and global representations
        combined = torch.cat([graph_repr, global_repr], dim=1)
        
        # Final prediction
        output = self.predictor(combined)
        
        return output

class SimpleGCN(nn.Module):
    """Simpler GCN model for faster training"""
    def __init__(self, num_node_features=7, global_features_dim=10, 
                 hidden_dim=64, num_targets=5, dropout=0.3):
        super(SimpleGCN, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.global_fc = nn.Sequential(
            nn.Linear(global_features_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets)
        )
        
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_features = data.global_features
        if len(global_features.shape) == 1:
            batch_size = torch.max(batch).item() + 1
            global_features = global_features.view(batch_size, -1)
        
        # GCN layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        # Graph pooling
        x = global_mean_pool(x, batch)
        
        # Global features
        global_repr = self.global_fc(global_features)
        
        # Combine and predict
        combined = torch.cat([x, global_repr], dim=1)
        output = self.predictor(combined)
        
        return output