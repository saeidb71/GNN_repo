

import torch
"""> Make sure you clicked "RESTART RUNTIME" above (if torch version was different)!"""
#@title
# Install rdkit
import sys
import os
import requests
import subprocess
import shutil
from logging import getLogger, StreamHandler, INFO
import pickle as pkl
import numpy as np
from  torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, summary, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp , global_add_pool as gadd
from torch_geometric.nn import SAGPooling, ASAPooling
import mlflow
import random
logger = getLogger(__name__)
logger.addHandler(StreamHandler())
logger.setLevel(INFO)
pytorch_version = f"torch-{torch.__version__}.html"
import rdkit
from torch_geometric.datasets import MoleculeNet
from sklearn.manifold import TSNE
from scipy.stats import norm
import argparse
import scipy.io
import pandas as pd

class GAT_Model(torch.nn.Module):
    def __init__(self, num_layers, num_heads, num_features, embedding_size, num_output):
        # Init parent
        super(GAT_Model, self).__init__()
        torch.manual_seed(41) 
        self.num_heads=num_heads
        self.embedding_size=embedding_size

        # Initialize a list to hold the GATConv layers
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GATConv(num_features, embedding_size, heads=num_heads))

        # Add additional GATConv layers based on the specified num_layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(embedding_size * num_heads, embedding_size, heads=num_heads))

        # Output layer
        self.out = Linear(embedding_size * 3, num_output)

    def forward(self, x, edge_index, batch_index):
        # Apply the GATConv layers
        hidden = x
        for conv_layer in self.conv_layers:
            hidden = F.tanh(conv_layer(hidden, edge_index))

        # Average over all heads
        hidden = torch.mean(hidden.view(-1, self.num_heads, self.embedding_size), dim=1)

        # Concatenate the pooling results
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index),
                            gadd(hidden, batch_index)], dim=1)

        hidden = F.dropout(hidden, p=0.5, training=self.training)
        out = self.out(hidden)  # For regression
        #out=torch.exp(-out)

        return out, hidden