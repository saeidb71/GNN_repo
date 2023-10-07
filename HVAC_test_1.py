# Enforce pytorch version 1.6.0
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

mlflow.pytorch.autolog()

#torch.set_num_threads(60)

def map_scalar_to_range(arr, scalar):
    # Find the minimum and maximum values in the array
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Map the scalar to the range [0, 1]
    if arr_min == arr_max:
        # Handle the case where arr_min and arr_max are the same to avoid division by zero
        return 0.5  # Return 0.5 as a neutral value
    else:
        mapped_value = (scalar - arr_min) / (arr_max - arr_min)
        return mapped_value


def map_to_classes(arr, num_classes=10):
    # Check if the input array is empty
    if len(arr) == 0:
        return np.array([])  # Return an empty array if input is empty
    
    # Calculate the minimum and maximum values in the input array
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Map the input array values to classes starting from 0 to (num_classes - 1)
    # using linear interpolation
    mapped_values = (arr - min_val) * (num_classes - 1) / (max_val - min_val)
    
    # Round the mapped values to the nearest integer and convert to int
    mapped_values = np.round(mapped_values).astype(int)
    
    # Clip values to ensure they are within the desired range [0, num_classes - 1]
    mapped_values = np.clip(mapped_values, 0, num_classes - 1)
    
    return mapped_values



def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

#-----------------------------------------Load Data----------------------------------------
#with open('rawTrainData_single_3', 'rb') as file:
with open('rawTrainData_single_3_GNN', 'rb') as file:
        distrbPop_3 = pkl.load(file)
        tEndPopBest_3 = pkl.load(file)
        Label_3 = pkl.load(file)
        tEndPopBest_conifgs_123459_3 = pkl.load(file)
        Label_conifgs_123459_3 = pkl.load(file)
        tEndPopBest_conifgs_678101112_3 = pkl.load(file)
        Label_conifgs_678101112_3 = pkl.load(file)
        t_allPop_3 = pkl.load(file)

with open('rawTrainData_single_4_randint', 'rb') as file:
        distrbPop_4 = pkl.load(file)
        tEndPopBest = pkl.load(file)
        Label_common_4 = pkl.load(file)
        unique_classes_4 = pkl.load(file)
        count_dict_4 = pkl.load(file)
        result_dict_4 = pkl.load(file)
        t_allPop_4 = pkl.load(file)

with open('rawTrainData_single_5_GNN', 'rb') as file:
        distrbPop_5 = pkl.load(file)
        tEndPopBest_5 = pkl.load(file)
        Label_5 = pkl.load(file)
        t_allPop_5 = pkl.load(file)

with open('Pop_Multi_3_GNN_all_data', 'rb') as file:
        Data_dict = pkl.load(file)
distrb_arrays_multi_3 = Data_dict['distrb_arrays']
Edges_list_multi_3= Data_dict['Edges_list_all']
t_allPop_multi_3 = Data_dict['time_End']
Edges_list_multi_3_long64=[]
for numpy_array in Edges_list_multi_3:
    torch_tensor = torch.tensor(numpy_array, dtype=torch.long)
    Edges_list_multi_3_long64.append(torch_tensor)

with open('Pop_Multi_4_GNN_all_data', 'rb') as file:
        Data_dict = pkl.load(file)
distrb_arrays_multi_4 = Data_dict['distrb_arrays']
Edges_list_multi_4= Data_dict['Edges_list_all']
t_allPop_multi_4 = Data_dict['time_End']
Edges_list_multi_4_long64=[]
for numpy_array in Edges_list_multi_4:
    torch_tensor = torch.tensor(numpy_array, dtype=torch.long)
    Edges_list_multi_4_long64.append(torch_tensor)

classes_pop_3=np.zeros(t_allPop_3.shape,dtype=int)
for i in np.arange(len(distrbPop_3)):
     classes_pop_3[:,i]=map_to_classes(t_allPop_3[:,i])

classes_pop_4=np.zeros(t_allPop_4.shape,dtype=int)
for i in np.arange(len(distrbPop_4)):
     classes_pop_4[:,i]=map_to_classes(t_allPop_4[:,i])

classes_pop_5=np.zeros(t_allPop_5.shape,dtype=int)
for i in np.arange(len(distrbPop_5)):
     classes_pop_5[:,i]=map_to_classes(t_allPop_5[:,i])

with open('Edge_GNN', 'rb') as file:
    Edge_GNN_dict = pkl.load(file)

num_comp_3=len(Edge_GNN_dict[3])
num_comp_4=len(Edge_GNN_dict[4])
num_comp_5=len(Edge_GNN_dict[5])
num_comp_6=len(Edge_GNN_dict[6])
num_comp_7=len(Edge_GNN_dict[7])

num_comp_3_multi=len(t_allPop_multi_3)
num_comp_4_multi=len(t_allPop_multi_4)

#-----------------------------------------Build Graph---------------------------------------
# Initialize an empty dictionary for the edge list
edge_list_dict_3={}
for g in np.arange(num_comp_3):
     g_i=Edge_GNN_dict[3][g]
     source_nodes=[]
     target_nodes=[]
     for j in np.arange(len(g_i)):
          edge_series=g_i[j]
          # Iterate through the list of nodes and create edges
          for i in range(len(edge_series) - 2):
            source_nodes.append(edge_series[i])
            target_nodes.append(edge_series[i + 1])
     edge_list_dict_3[g] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 0
                         source_nodes+target_nodes, # Source Nodes
                         target_nodes+source_nodes  # Target Nodes
                        ], dtype=torch.long)
     
edge_list_dict_4={}
for g in np.arange(num_comp_4):
     g_i=Edge_GNN_dict[4][g]
     source_nodes=[]
     target_nodes=[]
     for j in np.arange(len(g_i)):
          edge_series=g_i[j]
          # Iterate through the list of nodes and create edges
          for i in range(len(edge_series) - 2):
            source_nodes.append(edge_series[i])
            target_nodes.append(edge_series[i + 1])
     edge_list_dict_4[g] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 0
                         source_nodes+target_nodes, # Source Nodes
                         target_nodes+source_nodes  # Target Nodes
                        ], dtype=torch.long)
     
edge_list_dict_5={}
for g in np.arange(num_comp_5):
     g_i=Edge_GNN_dict[5][g]
     source_nodes=[]
     target_nodes=[]
     for j in np.arange(len(g_i)):
          edge_series=g_i[j]
          # Iterate through the list of nodes and create edges
          for i in range(len(edge_series) - 2):
            source_nodes.append(edge_series[i])
            target_nodes.append(edge_series[i + 1])
     edge_list_dict_5[g] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 0
                         source_nodes+target_nodes, # Source Nodes
                         target_nodes+source_nodes  # Target Nodes
                        ], dtype=torch.long)

"""edge_list_dict = {}
edge_list_dict[0] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 0
                         [0, 0, 0, 1, 2, 3], # Source Nodes
                         [1, 2, 3, 0, 0, 0]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[1] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 1
                         [0, 0, 3, 2, 3, 1], # Source Nodes
                         [2, 3, 1, 0, 0, 3]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[2] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 2
                         [0, 0, 3, 1, 3, 2], # Source Nodes
                         [1, 3, 2, 0, 0, 3]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[3] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 3
                         [0, 0, 1, 2, 1, 3], # Source Nodes
                         [2, 1, 3, 0, 0, 1]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[4] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 4
                         [0, 0, 2, 1, 2, 3], # Source Nodes
                         [1, 2, 3, 0, 0, 2]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[5] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 5
                         [0, 0, 2, 3, 2, 1], # Source Nodes
                         [3, 2, 1, 0, 0, 2]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[6] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 6
                         [0, 3, 2, 3, 2, 1], # Source Nodes
                         [3, 2, 1, 0, 3, 2]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[7] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 7
                         [0, 2, 1, 2, 1, 3], # Source Nodes
                         [2, 1, 3, 0, 2, 1]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[8] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 8
                         [0, 2, 3, 2, 3, 1], # Source Nodes
                         [2, 3, 1, 0, 2, 3]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[9] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 9
                         [0, 0, 1, 3, 1, 2], # Source Nodes
                         [3, 1, 2, 0, 0, 1]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[10] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 10
                         [0, 3, 1, 3, 1, 2], # Source Nodes
                         [3, 1, 2, 0, 3, 1]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[11] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 11
                         [0, 1, 3, 1, 3, 2], # Source Nodes
                         [1, 3, 2, 0, 1, 3]  # Target Nodes
                        ], dtype=torch.long)
edge_list_dict[12] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 12
                         [0, 1, 2, 1, 2, 3], # Source Nodes
                         [1, 2, 3, 0, 1, 2]  # Target Nodes
                        ], dtype=torch.long)"""

# 6 Features for each node (4x6 - Number of nodes x NUmber of features)
node_features_list_dict_3={}
for i in np.arange(len(distrbPop_3)):
        node_features_list_dict_3[i] = torch.tensor([
                            [0.0, 0.0, 0.0, 1.0], # Features of Node 0
                            [0.0, distrbPop_3[i][0]/np.sum(distrbPop_3[i]),distrbPop_3[i][0]/1000.0, 0.0], # Features of Node 1
                            [0.0, distrbPop_3[i][1]/np.sum(distrbPop_3[i]),distrbPop_3[i][1]/1000.0, 0.0], # Features of Node 2
                            [0.0, distrbPop_3[i][2]/np.sum(distrbPop_3[i]),distrbPop_3[i][2]/1000.0, 0.0], # Features of Node 3
                            ],dtype=torch.float32)#torch.long)
        
node_features_list_dict_4={}
for i in np.arange(len(distrbPop_4)):
        node_features_list_dict_4[i] = torch.tensor([
                            [0.0, 0.0, 0.0, 1.0], # Features of Node 0
                            [0.0, distrbPop_4[i][0]/np.sum(distrbPop_4[i]) ,distrbPop_4[i][0]/1000.0, 0.0], # Features of Node 1
                            [0.0, distrbPop_4[i][1]/np.sum(distrbPop_4[i]) ,distrbPop_4[i][1]/1000.0, 0.0], # Features of Node 2
                            [0.0, distrbPop_4[i][2]/np.sum(distrbPop_4[i]) ,distrbPop_4[i][2]/1000.0, 0.0], # Features of Node 3
                            [0.0, distrbPop_4[i][3]/np.sum(distrbPop_4[i]) ,distrbPop_4[i][3]/1000.0, 0.0], # Features of Node 4
                            ],dtype=torch.float32)#torch.long)
        
node_features_list_dict_5={}
for i in np.arange(len(distrbPop_5)):
        node_features_list_dict_5[i] = torch.tensor([
                            [0.0, 0.0, 0.0, 1.0], # Features of Node 0
                            [0.0, distrbPop_5[i][0]/np.sum(distrbPop_5[i]) ,distrbPop_5[i][0]/1000.0, 0.0], # Features of Node 1
                            [0.0, distrbPop_5[i][1]/np.sum(distrbPop_5[i]) ,distrbPop_5[i][1]/1000.0, 0.0], # Features of Node 2
                            [0.0, distrbPop_5[i][2]/np.sum(distrbPop_5[i]) ,distrbPop_5[i][2]/1000.0, 0.0], # Features of Node 3
                            [0.0, distrbPop_5[i][3]/np.sum(distrbPop_5[i]) ,distrbPop_5[i][3]/1000.0, 0.0], # Features of Node 4
                            [0.0, distrbPop_5[i][4]/np.sum(distrbPop_5[i]) ,distrbPop_5[i][4]/1000.0, 0.0], # Features of Node 5
                            ],dtype=torch.float32)#torch.long)
        
node_features_list_dict_3_multy={}
for i in np.arange(len(distrb_arrays_multi_3)):
        node_features_list_dict_3_multy[i] = torch.tensor([
                            [1.0, distrb_arrays_multi_3[i][0]/np.sum(distrb_arrays_multi_3[i]),distrb_arrays_multi_3[i][0]/1000.0, 0.0], # Features of Node -2+2--> junction
                            [0.0, 0.0, 0.0, 1.0], # Features of Node -1+2 ---> tank
                            [0.0, distrb_arrays_multi_3[i][1]/np.sum(distrb_arrays_multi_3[i]),distrb_arrays_multi_3[i][1]/1000.0, 0.0], # Features of Node 0+2
                            [0.0, distrb_arrays_multi_3[i][2]/np.sum(distrb_arrays_multi_3[i]),distrb_arrays_multi_3[i][2]/1000.0, 0.0], # Features of Node 1+2
                            ],dtype=torch.float32)#torch.long)
        
node_features_list_dict_4_multy={}
for i in np.arange(len(distrb_arrays_multi_4)):
        node_features_list_dict_4_multy[i] = torch.tensor([
                            [1.0, distrb_arrays_multi_4[i][0]/np.sum(distrb_arrays_multi_4[i]),distrb_arrays_multi_4[i][0]/1000.0, 0.0], # Features of Node -2+2--> junction
                            [0.0, 0.0, 0.0, 1.0], # Features of Node -1+2 ---> tank
                            [0.0, distrb_arrays_multi_4[i][1]/np.sum(distrb_arrays_multi_4[i]),distrb_arrays_multi_4[i][1]/1000.0, 0.0], # Features of Node 0+2
                            [0.0, distrb_arrays_multi_4[i][2]/np.sum(distrb_arrays_multi_4[i]),distrb_arrays_multi_4[i][2]/1000.0, 0.0], # Features of Node 1+2
                            [0.0, distrb_arrays_multi_4[i][3]/np.sum(distrb_arrays_multi_4[i]),distrb_arrays_multi_4[i][3]/1000.0, 0.0], # Features of Node 2+2
                            ],dtype=torch.float32)#torch.long)
        
"""# 1 Weight for each edge 
edge_weight_list_dict={}
edge_weight = torch.tensor([
                            [35.], # Weight for nodes (0,1)
                            [48.], # Weight for nodes (0,2)
                            [12.], # Weight for nodes (0,3)
                            [10.], # Weight for nodes (1,0)
                            [70.], # Weight for nodes (2,0)
                            [5.], # Weight for nodes (2,3)
                            [15.], # Weight for nodes (3,2)
                            [8.], # Weight for nodes (3,0)   
                            ],dtype=torch.long)"""

# Define the specific size of the list
size_all_data = len(distrbPop_3)* num_comp_3 + len(distrbPop_4)*num_comp_4 + len(distrbPop_5)*num_comp_5 + num_comp_3_multi + num_comp_4_multi
# Initialize an empty list with specific size filled with initial values (e.g., zeros)
Data_list = [0] * size_all_data

# Make a data object to store graph informaiton 
indx=0
for i in np.arange(len(distrbPop_3)):
        for j in np.arange(num_comp_3):
                Data_list[indx] = Data(x=node_features_list_dict_3[i], edge_index=edge_list_dict_3[j],y=t_allPop_3[j][i])#,y=t_allPop_3[j][i])#,  ,y=classes_pop_3[j][i]) edge_attr=edge_weight)
                #torch.save(Data_list[indx], os.path.join(os.getcwd()+'/Pop3_Dataset/') + f'3_data_{indx}.pt')
                indx=indx+1

for i in np.arange(len(distrbPop_4)):
        for j in np.arange(num_comp_4):
                Data_list[indx] = Data(x=node_features_list_dict_4[i], edge_index=edge_list_dict_4[j],y=t_allPop_4[j][i])#,y=t_allPop_4[j][i])#, ,y=classes_pop_4[j][i] edge_attr=edge_weight)
                #torch.save(Data_list[indx], os.path.join(os.getcwd()+'/Pop3_Dataset/') + f'4_data_{indx}.pt')
                indx=indx+1

for i in np.arange(len(distrbPop_5)):
        for j in np.arange(num_comp_5):
                Data_list[indx] = Data(x=node_features_list_dict_5[i], edge_index=edge_list_dict_5[j],y=t_allPop_5[j][i])#,y=t_allPop_4[j][i])#, ,y=classes_pop_4[j][i] edge_attr=edge_weight)
                #torch.save(Data_list[indx], os.path.join(os.getcwd()+'/Pop3_Dataset/') + f'5_data_{indx}.pt')
                indx=indx+1

for i in np.arange(len(distrb_arrays_multi_3)):       
        Data_list[indx] = Data(x=node_features_list_dict_3_multy[i], edge_index=Edges_list_multi_3_long64[i],y=t_allPop_multi_3[i])#,y=t_allPop_4[j][i])#, ,y=classes_pop_4[j][i] edge_attr=edge_weight)
        #torch.save(Data_list[indx], os.path.join(os.getcwd()+'/Pop3_Dataset/') + f'3_data_multy{indx}.pt')
        indx=indx+1

for i in np.arange(len(distrb_arrays_multi_4)):       
        Data_list[indx] = Data(x=node_features_list_dict_4_multy[i], edge_index=Edges_list_multi_4_long64[i],y=t_allPop_multi_4[i])#,y=t_allPop_4[j][i])#, ,y=classes_pop_4[j][i] edge_attr=edge_weight)
        #torch.save(Data_list[indx], os.path.join(os.getcwd()+'/Pop3_Dataset/') + f'4_data_multy{indx}.pt')
        indx=indx+1

#list of graphs in nx format
graphs_list_nx=[]
for i in np.arange(len(Data_list)):
     graphs_list_nx.append(to_networkx(Data_list[i], to_undirected=True))


#-----------------------------------------Batch Loader---------------------------------------

NUM_GRAPHS_PER_BATCH = 50#64

#--!!!!!!!!!!!!!!!!!modified!!!!!!!!!!!!!!!!!!!
#Data_list=Data_list[0:35299] #only singel split cases
#Data_list=Data_list[35299:] #only multi split cases

data_size = len(Data_list)
random.seed(42)
# Shuffle the list in place using the seeded random generator
Data_list_shuffled= random.sample(Data_list, len(Data_list))
loader = DataLoader(Data_list_shuffled[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
loader_test = DataLoader(Data_list_shuffled[int(data_size * 0.8):],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

"""for batch in loader:
       print(batch.x.float())
       print(batch.edge_index)
       print(batch.batch)
       print(batch.y)"""
"""for i in np.arange(13):
        g =to_networkx(Data_list[i], to_undirected=True)
        #nx.draw(g)
        nx.draw_networkx(g)
        plt.show()  # Add this line to display the plot
        k=1
        plt.clf()"""

#-----------------------------------------GCN Model---------------------------------------
embedding_size = 64#90#64#32
num_features= Data_list[0].x.shape[1]
num_output=1#10 # 1:regression 1:clasification: cross entropy
class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        #self.conv4 = GCNConv(embedding_size, embedding_size)
        #self.SAGPooling = SAGPooling(embedding_size, ratio=0.5)
        #self.ASAPooling = ASAPooling(embedding_size, ratio=0.5)
        #self.TopKPooling = TopKPooling(embedding_size, ratio=0.5)

        # Output layer
        self.out = Linear(embedding_size*3, num_output)
        #self.out = Linear(embedding_size*6, num_output)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        #hidden = F.relu(hidden)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        #hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        #hidden = F.relu(hidden)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        #hidden = F.relu(hidden)
        hidden = F.tanh(hidden)

        #-------------levrel 4-----------------------
        #hidden = self.conv4(hidden, edge_index)
        #hidden = F.tanh(hidden)

        """hidden, edge_index, _, batch_index, _, _ =self.SAGPooling(hidden,edge_index,batch=batch_index)
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)"""
        
        # Apply SAGPooling
        #x_sag, edge_index_sag, _, batch_index_sag, _, _ = self.SAGPooling(hidden, edge_index, batch=batch_index)

        # Apply TopKPooling
        #x_topk, edge_index_topk, _, batch_index_topk, _, _ = self.TopKPooling(hidden, edge_index, batch=batch_index)

        # Apply ASAPooling
        #x_asa, edge_index_asa, _, batch_index_asa, _, _ = self.ASAPooling(hidden, edge_index, batch=batch_index)

        # Concatenate the output of both pooling layers
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index),
                            gadd(hidden, batch_index)], dim=1)
        #x = torch.cat([gmp(x_sag, batch_index_sag), gap(x_sag, batch_index_sag), gadd(x_sag, batch_index_sag),
        #               gmp(x_topk, batch_index_topk), gap(x_topk, batch_index_topk), gadd(x_topk, batch_index_topk)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)   #:for regression
        #out = self.out(x)

        #hidden = hidden.relu()
        #out = F.dropout(out, p=0.5, training=self.training)

        # Output layer : regression
        #out = F.softmax(self.out(hidden), dim=1)

        return out, hidden
   
#-----------------------------------------GATConv Model---------------------------------------
embedding_size = 32#32#32-->saved GAT
num_features= Data_list[0].x.shape[1]
num_output=1#10 # 1:regression 1:clasification: cross entropy
numHeads=4
class GAT(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GAT, self).__init__()
        torch.manual_seed(41) #41

        # GCN layers
        self.initial_conv = GATConv(num_features, embedding_size,heads=numHeads)
        self.conv1 = GATConv(embedding_size * numHeads, embedding_size, heads=numHeads)
        self.conv2 = GATConv(embedding_size * numHeads, embedding_size, heads=numHeads)
        self.conv3 = GATConv(embedding_size * numHeads, embedding_size, heads=numHeads)
        #self.conv4 = GCNConv(embedding_size, embedding_size)
        #self.SAGPooling = SAGPooling(embedding_size, ratio=0.5)
        #self.ASAPooling = ASAPooling(embedding_size, ratio=0.5)
        #self.TopKPooling = TopKPooling(embedding_size, ratio=0.5)

        # Output layer
        self.out = Linear(embedding_size*3, num_output)
        #self.out = Linear(embedding_size*6, num_output)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        #hidden = F.relu(hidden)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        #hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        #hidden = F.relu(hidden)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        #hidden = F.relu(hidden)
        hidden = F.tanh(hidden)

        #avergae over all 4 heads
        hidden= torch.mean(hidden.view(-1, numHeads, embedding_size), dim=1) #4: num heads

        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index),
                            gadd(hidden, batch_index)], dim=1)
        out = self.out(hidden)   #:for regression

        return out, hidden

#-----------------model----------------------------
#model = GCN()
model = GAT()
# Specify the file path where you saved the model.
model_path ='trained_model_1_saved_GAT.pth' # 'trained_model_1.pth'
# Load the saved state dictionary into the model.
model.load_state_dict(torch.load(model_path))
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

for batch in loader:
       print(batch.x.float())
       print(batch.edge_index)
       print(batch.batch)
       print(batch.y)
       print(summary(model, batch.x.float(), batch.edge_index,batch.batch))
       break

#-----------------------------------------Test GNN Model---------------------------------------
with torch.no_grad():
        out, h = model(batch.x, batch.edge_index, batch.batch)
print(f'Embedding shape: {list(h.shape)}')
#visualize_embedding(h, color=batch.y)

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(out.detach().numpy().flatten(),batch.y, '-o',
            linewidth=2, markersize=8, markerfacecolor='none', markeredgewidth=2)
ax.set_xlabel('$\mathrm{estimate-y}$', fontsize=15)
ax.set_ylabel('$\mathrm{real-y}$', fontsize=15)
# ax.legend(prop=fontP)
#ax.set_ylim([timeEndVecSorted[-1, 1]-0.7,
#                timeEndVecSorted[0, 1]+0.7])  # Time [s]
#ax.set_title(f'numHX: {numHX} , distrb: {distrb}', fontsize=15)
#fig.savefig(os.getcwd()+'/Results/' +f'All_Flow.png')
#ax.set_yticks(np.linspace(
#        timeEndVecSorted[-1, 1], timeEndVecSorted[0, 1], 5))
#fig.savefig(plot_Address_Directory+f'/bruteForce.png')
plt.close(fig)"""

#-----------------------------------------Learning---------------------------------------

# Root mean squared error
loss_fn = torch.nn.MSELoss() # torch.nn.CrossEntropyLoss() # torch.nn.MSELoss() $ classifican/ regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def train():
    # Enumerate over the data
    for batch in loader:
      # Use GPU
      batch.to(device)
      # Reset gradients
      optimizer.zero_grad()
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
      # Calculating the loss and gradients
      loss = loss_fn(pred.flatten(), batch.y.float()) #for regression
      #loss = loss_fn(pred, torch.tensor(batch.y)) #for classification
      loss.backward()
      # Update using the gradients
      optimizer.step()
    return loss, embedding

"""print("Starting training...")

with mlflow.start_run():
    mlflow.log_param("embedding_size", embedding_size)
    mlflow.log_param("num_features", num_features)
    losses = []
    for epoch in range(10000):
        loss, h = train()
        losses.append(loss)
        if epoch % 100 == 0:
            model_path = 'trained_model_1.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch} | Train Loss {loss}")
            mlflow.log_metric("train_loss", loss.item())

            test_loss_avg=[]
            with torch.no_grad():
                 for batch_test in loader_test:
                    batch_test.to(device)
                    pred, embed = model(batch_test.x.float(), batch_test.edge_index, batch_test.batch)
                    loss_test = loss_fn(pred.flatten(), batch_test.y.float()) 
                    test_loss_avg.append(loss_test)
            avg_totall_loss_test=sum(test_loss_avg) / len(test_loss_avg)
            print(f"Epoch {epoch} | Test Loss avg {avg_totall_loss_test}")
            mlflow.log_metric("test_loss", avg_totall_loss_test)

    mlflow.pytorch.log_model(model, "models")

model_path = 'trained_model_1.pth'
torch.save(model.state_dict(), model_path)"""


#-----------------------------------------Test Learned Model---------------------------------------
# Analyze the results for one batch
#test_batch = next(iter(loader_test))
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
with torch.no_grad():
    for batch_test in loader_test:
        batch_test.to(device)
        pred, embed = model(batch_test.x.float(), batch_test.edge_index, batch_test.batch)
        ax.plot(pred.detach().numpy().flatten(),batch_test.y, '-o',
            linewidth=2, markersize=8, markerfacecolor='none', markeredgewidth=2)
    ax.set_xlabel('$\mathrm{estimate-y: test}$', fontsize=15)
    ax.set_ylabel('$\mathrm{real-y: test}$', fontsize=15)
fig.savefig('test_data_learned_model.png')
mlflow.log_artifact("test_data_learned_model.png")


# embedding
"""model.eval()
all_node_embeddings = []
with torch.no_grad():
    for i in np.arange(len(Data_list)):
        pred,embed=model(Data_list[i].x.float(), Data_list[i].edge_index, Data_list[i].batch)
        all_node_embeddings.append(embed)
all_node_embeddings = torch.cat(all_node_embeddings, dim=0).numpy()
#tsne = TSNE(n_components=2, random_state=42)
#graph_embedding_2d = tsne.fit_transform(all_node_embeddings)
#tsne_3d = TSNE(n_components=3, random_state=42)
#graph_embedding_3d = tsne_3d.fit_transform(all_node_embeddings)

classess_graphs=np.zeros(len(Data_list),dtype=int)
for i in np.arange(len(Data_list)):
     if i<len(distrbPop_3)* num_comp_3:
          classess_graphs[i]=0
     elif i<len(distrbPop_3)* num_comp_3 + len(distrbPop_4)*num_comp_4:
          classess_graphs[i]=1
     elif i<len(distrbPop_3)* num_comp_3 + len(distrbPop_4)*num_comp_4 + len(distrbPop_5)*num_comp_5:
          classess_graphs[i]=2
     elif i<len(distrbPop_3)* num_comp_3 + len(distrbPop_4)*num_comp_4 + len(distrbPop_5)*num_comp_5 + num_comp_3_multi:
          classess_graphs[i]=3
     else:
          classess_graphs[i]=4

# Define a color map for the classes with five distinct colors
colors = ['r', 'g', 'b', 'c', 'm']  # You can customize these colors
# Map class labels to colors
class_colors = [colors[i] for i in classess_graphs]
# Create a figure and axes
fig, ax = plt.subplots()
# Create a scatter plot on the axes
scatter = ax.scatter(graph_embedding_2d[:, 0], graph_embedding_2d[:, 1], c=class_colors)#, label=classess_graphs)
# Create a custom legend for the defined five classes
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Class {i}') for i, color in enumerate(colors)]
# Customize the plot (e.g., add labels, title, legend, etc.)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Node Embeddings Visualization')
#ax.legend()
ax.legend(handles=custom_legend, title='Legend', loc='upper right')  # Include the custom legend
plt.savefig('graph_embedding.png')
mlflow.log_artifact("graph_embedding.png")"""


t_pred_3=np.zeros((num_comp_3,len(distrbPop_3)))
pred_best_class_3=np.zeros(len(distrbPop_3),dtype=int)
group_num=0
with torch.no_grad():
     for i in np.arange(len(distrbPop_3)):
          t_true=t_allPop_3[:,i]
          for j in np.arange(num_comp_3):
               indcx=group_num*num_comp_3+j
               t_p,hid=model(Data_list[indcx].x.float(), Data_list[indcx].edge_index, Data_list[indcx].batch)
               t_pred_3[j,i]=t_p.numpy().flatten()[0]
          pred_best_class_3[i]=np.argmax(t_pred_3[:,i])
          group_num=group_num+1

pred_3_best=np.zeros(len(distrbPop_3))
num_clasess_btter_than_best_pred_3_single=np.zeros(len(distrbPop_3))
pred_3_best_rel_val=np.zeros(len(distrbPop_3))
for g in np.arange(len(distrbPop_3)):
    y_val_all=t_allPop_3[:,g]
    y_val_pred_best=t_allPop_3[pred_best_class_3[g],g]
    # Count the elements greater than 0.6
    num_clasess_btter_than_best_pred_3_single[g] = float(np.sum(y_val_all >= y_val_pred_best))
    pred_3_best[g]=y_val_pred_best
    pred_3_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(distrbPop_3)):
     y_val_all=t_allPop_3[:,g]
     x_val=g*np.ones(len(y_val_all))
     ax.plot(x_val,y_val_all,'o')
ax.plot(np.arange(len(distrbPop_3)),pred_3_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
fig.savefig('all_pop_3_labels_pred.png')
mlflow.log_artifact("all_pop_3_labels_pred.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pred_3_best_rel_val,'o-')     
fig.savefig('all_pop_3_labels_pred_2.png')
mlflow.log_artifact("all_pop_3_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_3_single,'o-')    
ax.set_title(f'num Graphs: {y_val_all.shape[0]}') 
fig.savefig('num_clasess_btter_than_best_pred_3_single.png')
mlflow.log_artifact("num_clasess_btter_than_best_pred_3_single.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(distrbPop_3)):
     ax.plot(t_pred_3[:,i],t_allPop_3[:,i], '-o',
            linewidth=2, markersize=8, markerfacecolor='none', markeredgewidth=2)
ax.set_xlabel('$\mathrm{estimate-y:3}$', fontsize=15)
ax.set_ylabel('$\mathrm{real-y:3}$', fontsize=15)
fig.savefig('end_result_3_all.png')
mlflow.log_artifact("end_result_3_all.png")

last3_indx=indcx
t_pred_4=np.zeros((num_comp_4,len(distrbPop_4)))
group_num=0
pred_best_class_4=np.zeros(len(distrbPop_4),dtype=int)
with torch.no_grad():
     for i in np.arange(len(distrbPop_4)):
          t_true=t_allPop_4[:,i]
          for j in np.arange(num_comp_4):
               indcx=group_num*num_comp_4+j+(last3_indx+1)
               t_p,hid=model(Data_list[indcx].x.float(), Data_list[indcx].edge_index, Data_list[indcx].batch)
               t_pred_4[j,i]=t_p.numpy().flatten()[0]
          group_num=group_num+1
          pred_best_class_4[i]=np.argmax(t_pred_4[:,i])

pred_4_best=np.zeros(len(distrbPop_4))
pred_4_best_rel_val=np.zeros(len(distrbPop_4))
num_clasess_btter_than_best_pred_4_single=np.zeros(len(distrbPop_4))
for g in np.arange(len(distrbPop_4)):
    y_val_all=t_allPop_4[:,g]
    y_val_pred_best=t_allPop_4[pred_best_class_4[g],g]
    pred_4_best[g]=y_val_pred_best
    num_clasess_btter_than_best_pred_4_single[g] = float(np.sum(y_val_all >= y_val_pred_best))
    pred_4_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(distrbPop_4)):
     y_val_all=t_allPop_4[:,g]
     x_val=g*np.ones(len(y_val_all))
     ax.plot(x_val,y_val_all,'o')
ax.plot(np.arange(len(distrbPop_4)),pred_4_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
fig.savefig('all_pop_4_labels_pred.png')
mlflow.log_artifact("all_pop_4_labels_pred.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pred_4_best_rel_val,'o-')     
fig.savefig('all_pop_4_labels_pred_2.png')
mlflow.log_artifact("all_pop_4_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_4_single,'o-')    
ax.set_title(f'num Graphs: {y_val_all.shape[0]}') 
fig.savefig('num_clasess_btter_than_best_pred_4_single.png')
mlflow.log_artifact("num_clasess_btter_than_best_pred_4_single.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(distrbPop_4)):
     ax.plot(t_pred_4[:,i],t_allPop_4[:,i], '-o',
            linewidth=2, markersize=8, markerfacecolor='none', markeredgewidth=2)
ax.set_xlabel('$\mathrm{estimate-y:4}$', fontsize=15)
ax.set_ylabel('$\mathrm{real-y:4}$', fontsize=15)
fig.savefig('end_result_4_all.png')
mlflow.log_artifact("end_result_4_all.png")

last4_indx=indcx
t_pred_5=np.zeros((num_comp_5,len(distrbPop_5)))
group_num=0
pred_best_class_5=np.zeros(len(distrbPop_5),dtype=int)
with torch.no_grad():
     for i in np.arange(len(distrbPop_5)):
          t_true=t_allPop_5[:,i]
          for j in np.arange(num_comp_5):
               indcx=group_num*num_comp_5+j+(last4_indx+1)
               t_p,hid=model(Data_list[indcx].x.float(), Data_list[indcx].edge_index, Data_list[indcx].batch)
               t_pred_5[j,i]=t_p.numpy().flatten()[0]
          group_num=group_num+1
          pred_best_class_5[i]=np.argmax(t_pred_5[:,i])

pred_5_best=np.zeros(len(distrbPop_5))
pred_5_best_rel_val=np.zeros(len(distrbPop_5))
num_clasess_btter_than_best_pred_5_single=np.zeros(len(distrbPop_5))
for g in np.arange(len(distrbPop_5)):
    y_val_all=t_allPop_5[:,g]
    y_val_pred_best=t_allPop_5[pred_best_class_5[g],g]
    pred_5_best[g]=y_val_pred_best
    num_clasess_btter_than_best_pred_5_single[g] = float(np.sum(y_val_all >= y_val_pred_best))
    pred_5_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(distrbPop_5)):
     y_val_all=t_allPop_5[:,g]
     x_val=g*np.ones(len(y_val_all))
     ax.plot(x_val,y_val_all,'o')
ax.plot(np.arange(len(distrbPop_5)),pred_5_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
fig.savefig('all_pop_5_labels_pred.png')
mlflow.log_artifact("all_pop_5_labels_pred.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pred_5_best_rel_val,'o-')     
fig.savefig('all_pop_5_labels_pred_2.png')
mlflow.log_artifact("all_pop_5_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_5_single,'o-')   
ax.set_title(f'num Graphs: {y_val_all.shape[0]}') 
fig.savefig('num_clasess_btter_than_best_pred_5_single.png')
mlflow.log_artifact("num_clasess_btter_than_best_pred_5_single.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(distrbPop_5)):
     ax.plot(t_pred_5[:,i],t_allPop_5[:,i], '-o',
            linewidth=2, markersize=8, markerfacecolor='none', markeredgewidth=2)
ax.set_xlabel('$\mathrm{estimate-y:5}$', fontsize=15)
ax.set_ylabel('$\mathrm{real-y:5}$', fontsize=15)
fig.savefig('end_result_5_all.png')
mlflow.log_artifact("end_result_5_all.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(np.argmax(t_pred_3,0),'o-',label='pre: 3')
ax.plot(np.argmax(t_allPop_3,0),'--o',label='true: 3')
ax.legend()
fig.savefig('end_result_3_label.png')
mlflow.log_artifact("end_result_3_label.png")
print(np.sum(np.argmax(t_allPop_3,0)==np.argmax(t_pred_3,0))/len(distrbPop_3))
mlflow.log_param("label_accuracy_3", np.sum(np.argmax(t_allPop_3,0)==np.argmax(t_pred_3,0))/len(distrbPop_3))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(np.argmax(t_pred_4,0),'o-',label='pre: 4')
ax.plot(np.argmax(t_allPop_4,0),'--o',label='true: 4')
ax.legend()
fig.savefig('end_result_4_label.png')
print(np.sum(np.argmax(t_allPop_4,0)==np.argmax(t_pred_4,0))/len(distrbPop_4))
mlflow.log_artifact("end_result_4_label.png")
mlflow.log_param("label_accuracy_4", np.sum(np.argmax(t_allPop_4,0)==np.argmax(t_pred_4,0))/len(distrbPop_4))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(np.argmax(t_pred_5,0),'o-',label='pre: 5')
ax.plot(np.argmax(t_allPop_5,0),'--o',label='true: 5')
ax.legend()
fig.savefig('end_result_5_label.png')
print(np.sum(np.argmax(t_allPop_5,0)==np.argmax(t_pred_5,0))/len(distrbPop_5))
mlflow.log_artifact("end_result_5_label.png")
mlflow.log_param("label_accuracy_5", np.sum(np.argmax(t_allPop_5,0)==np.argmax(t_pred_5,0))/len(distrbPop_5))


#-------multi 3 multi, fora each disturbance (out of 30) we have 9 diffrent calssess--> you cna see branches usjg edges--------
unique_sets_multi_3 = {}
for idx, arr in enumerate(distrb_arrays_multi_3):
    # Sort the array to ignore permutations
    arr_sorted = sorted(arr)
    key = tuple(arr_sorted)

    if key in unique_sets_multi_3:
        unique_sets_multi_3[key].append(idx)
    else:
        unique_sets_multi_3[key] = [idx]
# Find sets of values with more than one occurrence (same content)
same_content_sets = {key: indices for key, indices in unique_sets_multi_3.items() if len(indices) > 1}
for key, indices in same_content_sets.items():
    print(f"For the same content {key}, indices are: {indices}")
# Convert unique_sets to a list of lists
unique_dist_group_multi_3 = list(unique_sets_multi_3.values())
print(unique_dist_group_multi_3)

last5_indx=indcx
t_pred_3_multi=np.zeros(num_comp_3_multi)
with torch.no_grad():
     for i in np.arange(len(distrb_arrays_multi_3)):
            indcx=i+(last5_indx+1)
            t_p,hid=model(Data_list[indcx].x.float(), Data_list[indcx].edge_index, Data_list[indcx].batch)
            t_pred_3_multi[i]=t_p.numpy().flatten()[0]

t_pred_3_multi_gropuped=np.zeros((len(unique_dist_group_multi_3[0]) , len(unique_dist_group_multi_3)))
t_real_3_multi_gropuped=np.zeros((len(unique_dist_group_multi_3[0]) , len(unique_dist_group_multi_3)))
for i in np.arange(len(unique_dist_group_multi_3)):
     t_pred_3_multi_gropuped[:,i]=t_pred_3_multi[unique_dist_group_multi_3[i]]
     t_real_3_multi_gropuped[:,i]=t_allPop_multi_3[unique_dist_group_multi_3[i]]

t_pred_3_multi_gropuped_best_index=t_pred_3_multi_gropuped.argmax(axis=0)
t_real_3_multi_gropuped_best=t_real_3_multi_gropuped.max(axis=0)
t_pred_3_multi_gropuped_best=np.zeros(len(unique_dist_group_multi_3))
t_pred_3_multi_gropuped_best_rel_val=np.zeros(len(unique_dist_group_multi_3))
for i in np.arange(len(unique_dist_group_multi_3)):
     t_pred_3_multi_gropuped_best[i]=t_real_3_multi_gropuped[t_pred_3_multi_gropuped_best_index[i],i]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
num_clasess_btter_than_best_pred_3_multi=np.zeros(len(unique_dist_group_multi_3))
for g in np.arange(len(unique_dist_group_multi_3)):
     y_val_all=t_real_3_multi_gropuped[:,g]
     x_val=g*np.ones(len(y_val_all))
     y_val_pred_best=t_pred_3_multi_gropuped_best[g]
     num_clasess_btter_than_best_pred_3_multi[g] = float(np.sum(y_val_all >= y_val_pred_best))
     t_pred_3_multi_gropuped_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)
     ax.plot(x_val,y_val_all,'o')
ax.plot(np.arange(len(unique_dist_group_multi_3)),t_pred_3_multi_gropuped_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
fig.savefig('all_pop_3_multi_labels_pred.png')
mlflow.log_artifact("all_pop_3_multi_labels_pred.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_pred_3_multi_gropuped_best_rel_val,'o-')     
fig.savefig('all_pop_3_multi_labels_pred_2.png')
mlflow.log_artifact("all_pop_3_multi_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_3_multi,'o-')   
ax.set_title(f'num Graphs: {y_val_all.shape[0]}') 
fig.savefig('num_clasess_btter_than_best_pred_3_multi.png')
mlflow.log_artifact("num_clasess_btter_than_best_pred_3_multi.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(unique_dist_group_multi_3)):
     ax.plot(t_pred_3_multi_gropuped[:,i],t_real_3_multi_gropuped[:,i], '-o',
            linewidth=2, markersize=8, markerfacecolor='none', markeredgewidth=2)
ax.set_xlabel('$\mathrm{estimate-y:3 mutli}$', fontsize=15)
ax.set_ylabel('$\mathrm{real-y:3 multi}$', fontsize=15)
fig.savefig('end_result_3_multi_all.png')
mlflow.log_artifact("end_result_3_multi_all.png")

print(np.sum(np.argmax(t_real_3_multi_gropuped,0)==np.argmax(t_pred_3_multi_gropuped,0))/len(unique_dist_group_multi_3))
mlflow.log_param("label_accuracy_3_multi", np.sum(np.argmax(t_real_3_multi_gropuped,0)==np.argmax(t_pred_3_multi_gropuped,0))/len(unique_dist_group_multi_3))


#-------multi 4 multi, fora each disturbance (out of 20) we have 88 diffrent calssess--> you can see branches usjg edges--------
unique_sets_multi_4 = {}
for idx, arr in enumerate(distrb_arrays_multi_4):
    # Sort the array to ignore permutations
    arr_sorted = sorted(arr)
    key = tuple(arr_sorted)

    if key in unique_sets_multi_4:
        unique_sets_multi_4[key].append(idx)
    else:
        unique_sets_multi_4[key] = [idx]
# Find sets of values with more than one occurrence (same content)
same_content_sets_multi_4 = {key: indices for key, indices in unique_sets_multi_4.items() if len(indices) > 1}
for key, indices in same_content_sets_multi_4.items():
    print(f"For the same content {key}, indices are: {indices}")
# Convert unique_sets to a list of lists
unique_dist_group_multi_4 = list(unique_sets_multi_4.values())
print(unique_dist_group_multi_4)

last3_indx_multi=indcx
t_pred_4_multi=np.zeros(num_comp_4_multi)
with torch.no_grad():
     for i in np.arange(len(distrb_arrays_multi_4)):
            indcx=i+(last3_indx_multi+1)
            t_p,hid=model(Data_list[indcx].x.float(), Data_list[indcx].edge_index, Data_list[indcx].batch)
            t_pred_4_multi[i]=t_p.numpy().flatten()[0]

t_pred_4_multi_gropuped=np.zeros((len(unique_dist_group_multi_4[0]) , len(unique_dist_group_multi_4)))
t_real_4_multi_gropuped=np.zeros((len(unique_dist_group_multi_4[0]) , len(unique_dist_group_multi_4)))
for i in np.arange(len(unique_dist_group_multi_4)):
     t_pred_4_multi_gropuped[:,i]=t_pred_4_multi[unique_dist_group_multi_4[i]]
     t_real_4_multi_gropuped[:,i]=t_allPop_multi_4[unique_dist_group_multi_4[i]]

t_pred_4_multi_gropuped_best_index=t_pred_4_multi_gropuped.argmax(axis=0)
t_real_4_multi_gropuped_best=t_real_4_multi_gropuped.max(axis=0)
t_pred_4_multi_gropuped_best=np.zeros(len(unique_dist_group_multi_4))
t_pred_4_multi_gropuped_best_rel_val=np.zeros(len(unique_dist_group_multi_4))
for i in np.arange(len(unique_dist_group_multi_4)):
     t_pred_4_multi_gropuped_best[i]=t_real_4_multi_gropuped[t_pred_4_multi_gropuped_best_index[i],i]

num_clasess_btter_than_best_pred_4_multi=np.zeros(len(unique_dist_group_multi_4))
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(unique_dist_group_multi_4)):
     y_val_all=t_real_4_multi_gropuped[:,g]
     x_val=g*np.ones(len(y_val_all))
     y_val_pred_best=t_pred_4_multi_gropuped_best[g]
     num_clasess_btter_than_best_pred_4_multi[g] = float(np.sum(y_val_all >= y_val_pred_best))
     t_pred_4_multi_gropuped_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)
     ax.plot(x_val,y_val_all,'o')
ax.plot(np.arange(len(unique_dist_group_multi_4)),t_pred_4_multi_gropuped_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
fig.savefig('all_pop_4_multi_labels_pred.png')
mlflow.log_artifact("all_pop_4_multi_labels_pred.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_pred_4_multi_gropuped_best_rel_val,'o-')     
fig.savefig('all_pop_4_multi_labels_pred_2.png')
mlflow.log_artifact("all_pop_4_multi_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_4_multi,'o-')    
ax.set_title(f'num Graphs: {y_val_all.shape[0]}') 
fig.savefig('num_clasess_btter_than_best_pred_4_multi.png')
mlflow.log_artifact("num_clasess_btter_than_best_pred_4_multi.png")


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(unique_dist_group_multi_4)):
     ax.plot(t_pred_4_multi_gropuped[:,i],t_real_4_multi_gropuped[:,i], '-o',
            linewidth=2, markersize=8, markerfacecolor='none', markeredgewidth=2)
ax.set_xlabel('$\mathrm{estimate-y:4 multi}$', fontsize=15)
ax.set_ylabel('$\mathrm{real-y:4 multi}$', fontsize=15)
fig.savefig('end_result_4_multi_all.png')
mlflow.log_artifact("end_result_4_multi_all.png")

print(np.sum(np.argmax(t_real_4_multi_gropuped,0)==np.argmax(t_pred_4_multi_gropuped,0))/len(unique_dist_group_multi_4))
mlflow.log_param("label_accuracy_4_multi", np.sum(np.argmax(t_real_4_multi_gropuped,0)==np.argmax(t_pred_4_multi_gropuped,0))/len(unique_dist_group_multi_4))

print(f"pred_3_best_rel_val_mean: {np.mean(pred_3_best_rel_val)}")
print(f"pred_4_best_rel_val_mean: {np.mean(pred_4_best_rel_val)}")
print(f"pred_5_best_rel_val_mean: {np.mean(pred_5_best_rel_val)}")
print(f"t_pred_3_multi_gropuped_best_rel_val_mean: {np.mean(t_pred_3_multi_gropuped_best_rel_val)}")
print(f"t_pred_4_multi_gropuped_best_rel_val_mean: {np.mean(t_pred_4_multi_gropuped_best_rel_val)}")

mlflow.log_param("pred_3_best_rel_val_mean", np.mean(pred_3_best_rel_val))
mlflow.log_param("pred_4_best_rel_val_mean", np.mean(pred_4_best_rel_val))
mlflow.log_param("pred_5_best_rel_val_mean", np.mean(pred_5_best_rel_val))
mlflow.log_param("t_pred_3_multi_gropuped_best_rel_val_mean", np.mean(t_pred_3_multi_gropuped_best_rel_val))
mlflow.log_param("t_pred_4_multi_gropuped_best_rel_val_mean", np.mean(t_pred_4_multi_gropuped_best_rel_val))

