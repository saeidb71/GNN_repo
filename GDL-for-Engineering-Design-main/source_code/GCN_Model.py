from torch_geometric.nn import GraphConv,GCNConv,GATConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp , global_add_pool as gadd

class GCN_Model(torch.nn.Module):
    def __init__(self, num_layers, num_features, embedding_size, num_output):
        
        super(GCN_Model, self).__init__()
        torch.manual_seed(12345)
        self.embedding_size=embedding_size

        # Initialize a list to hold the GATConv layers
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(num_features, embedding_size))

        # Add additional GATConv layers based on the specified num_layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(embedding_size , embedding_size))

        # Output layer
        self.out = Linear(embedding_size * 3, num_output)


    def forward(self, x, edge_index, batch):

        hidden = x
        for conv_layer in self.conv_layers:
            hidden = F.relu(conv_layer(hidden, edge_index))

        # Concatenate the pooling results
        hidden = torch.cat([gmp(hidden, batch),
                            gap(hidden, batch),
                            gadd(hidden, batch)], dim=1)

        hidden = F.dropout(hidden, p=0.5, training=self.training)
        #hidden = F.dropout(hidden, p=0.5, training=self.training)
        out = self.out(hidden)  # For regression
        #out=torch.exp(-out)

        return out