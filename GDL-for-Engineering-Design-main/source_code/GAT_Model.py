from torch_geometric.nn import GraphConv,GCNConv,GATConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp , global_add_pool as gadd

class GAT_Model(torch.nn.Module):
    def __init__(self, num_layers, num_features, embedding_size, num_output, num_heads):
        
        super(GAT_Model, self).__init__()
        torch.manual_seed(12356)
        self.embedding_size=embedding_size
        self.num_heads=num_heads

        # Initialize a list to hold the GATConv layers
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GATConv(num_features, embedding_size, heads=num_heads))
        

        # Add additional GATConv layers based on the specified num_layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(embedding_size * num_heads , embedding_size, heads=num_heads))

        # Output layer
        self.out = Linear(embedding_size * 3, num_output) #was 3


    def forward(self, x, edge_index, batch):

        hidden = x
        for conv_layer in self.conv_layers:
            hidden = F.relu(conv_layer(hidden, edge_index))

        # Average over all heads
        hidden = torch.mean(hidden.view(-1, self.num_heads, self.embedding_size), dim=1)

        # Concatenate the pooling results
        hidden = torch.cat([gmp(hidden, batch),
                            gap(hidden, batch),
                            gadd(hidden, batch)], dim=1)
    

        hidden = F.dropout(hidden, p=0.5, training=self.training)
        out = self.out(hidden)  # For regression
        #out=torch.exp(-out)

        return out
