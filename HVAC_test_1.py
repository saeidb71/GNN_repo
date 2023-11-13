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
import scipy.stats as stats
from  torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
import torch
from scipy.stats import kendalltau
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
from matplotlib.ticker import MaxNLocator


"""# Create an argument parser
parser = argparse.ArgumentParser(description='HVAC Test Script')
# Add arguments for "x," "y," and "z"
parser.add_argument('--embedding_size', type=int, required=True, help='Value of embedding_size')
parser.add_argument('--numHeads', type=int, required=True, help='Value of numHeads')
parser.add_argument('--num_layers', type=int, required=True, help='Value of num_layers')
parser.add_argument('--NUM_GRAPHS_PER_BATCH', type=int, required=True, help='Value of NUM_GRAPHS_PER_BATCH')
# Parse the command-line arguments
args = parser.parse_args()
embedding_size=args.embedding_size
numHeads=args.numHeads
num_layers=args.num_layers
NUM_GRAPHS_PER_BATCH=args.NUM_GRAPHS_PER_BATCH"""

embedding_size=16#5#16#32
numHeads=4#2#4
num_layers=3#3#3
NUM_GRAPHS_PER_BATCH=100#300#100

#python HVAC_test_1.py --embedding_size 16 --numHeads 4 --num_layers 2 --NUM_GRAPHS_PER_BATCH 50

print(f"embedding_size: {embedding_size}")
print(f"numHeads: {numHeads}")
print(f"num_layers: {num_layers}")
print(f"NUM_GRAPHS_PER_BATCH: {NUM_GRAPHS_PER_BATCH}")

File_Name=f"embd_{embedding_size}_nHead_{numHeads}_nlayer_{num_layers}_Batch_{NUM_GRAPHS_PER_BATCH}"

#torch.set_num_threads(60)

seed_value = 42
random.seed(seed_value)

# Generate 13 distinct colors using the same seed
def generate_random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


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

with open('rawTrainData_single_6_GNN', 'rb') as file:
        distrbPop_6 = pkl.load(file)
        tEndPopBest_6 = pkl.load(file)
        Label_6 = pkl.load(file)
        t_allPop_6 = pkl.load(file)

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

classes_pop_6=np.zeros(t_allPop_6.shape,dtype=int)
for i in np.arange(len(distrbPop_6)):
     classes_pop_6[:,i]=map_to_classes(t_allPop_6[:,i])

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
     
edge_list_dict_6={}
for g in np.arange(num_comp_6):
     g_i=Edge_GNN_dict[6][g]
     source_nodes=[]
     target_nodes=[]
     for j in np.arange(len(g_i)):
          edge_series=g_i[j]
          # Iterate through the list of nodes and create edges
          for i in range(len(edge_series) - 2):
            source_nodes.append(edge_series[i])
            target_nodes.append(edge_series[i + 1])
     edge_list_dict_6[g] = torch.tensor([ # Create an edge list for a graph with 4 nodes--class 0
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
        
node_features_list_dict_6={}
for i in np.arange(len(distrbPop_6)):
        node_features_list_dict_6[i] = torch.tensor([
                            [0.0, 0.0, 0.0, 1.0], # Features of Node 0
                            [0.0, distrbPop_6[i][0]/np.sum(distrbPop_6[i]) ,distrbPop_6[i][0]/1000.0, 0.0], # Features of Node 1
                            [0.0, distrbPop_6[i][1]/np.sum(distrbPop_6[i]) ,distrbPop_6[i][1]/1000.0, 0.0], # Features of Node 2
                            [0.0, distrbPop_6[i][2]/np.sum(distrbPop_6[i]) ,distrbPop_6[i][2]/1000.0, 0.0], # Features of Node 3
                            [0.0, distrbPop_6[i][3]/np.sum(distrbPop_6[i]) ,distrbPop_6[i][3]/1000.0, 0.0], # Features of Node 4
                            [0.0, distrbPop_6[i][4]/np.sum(distrbPop_6[i]) ,distrbPop_6[i][4]/1000.0, 0.0], # Features of Node 5
                            [0.0, distrbPop_6[i][5]/np.sum(distrbPop_6[i]) ,distrbPop_6[i][5]/1000.0, 0.0], # Features of Node 6
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

Data_list_test_Single_6 =  [0] * 4051
for i in np.arange(len(distrbPop_6)):
        for j in np.arange(num_comp_6):
                Data_list_test_Single_6[j] = Data(x=node_features_list_dict_6[i], edge_index=edge_list_dict_6[j],y=t_allPop_6[j][i])#,y=t_allPop_4[j][i])#, ,y=classes_pop_4[j][i] edge_attr=edge_weight)
                #torch.save(Data_list[indx], os.path.join(os.getcwd()+'/Pop3_Dataset/') + f'5_data_{indx}.pt')

#list of graphs in nx format
graphs_list_nx=[]
for i in np.arange(len(Data_list)):
     graphs_list_nx.append(to_networkx(Data_list[i], to_undirected=True))

graphs_list_nx_test_6=[]
for i in np.arange(len(Data_list_test_Single_6)):
     graphs_list_nx_test_6.append(to_networkx(Data_list_test_Single_6[i], to_undirected=True))


#-----------------------------------------Batch Loader---------------------------------------

#NUM_GRAPHS_PER_BATCH = 50#64

#--!!!!!!!!!!!!!!!!!modified!!!!!!!!!!!!!!!!!!!
#Data_list=Data_list[0:35299] #only singel split cases
#Data_list=Data_list[35299:] #only multi split cases

order_indices = [
    list(range(0,781)),        # 0 to 780
    list(range(2600, 6834)),  # 2600 to 6833
    list(range(16762, 22273)),# 16762 to 22272
    list(range(35299, 35380)),# 35299 to 35379
    list(range(35569, 36097)),# 35569 to 36096
    list(range(781, 2600)),# 781 to 2599
    list(range(6834, 16762)),# 6834 to 16761
    list(range(22273, 35299)),# 22273 to 35298
    list(range(35380, 35569)),# 35380 to 35568
    list(range(36097, 37329)),# 36097 to 37328
    # Add more ranges as needed
]

Data_list_shuffled = [Data_list[i] for sublist in order_indices for i in sublist]
assert len(Data_list_shuffled) == len(Data_list)

data_size = len(Data_list)
random.seed(42)
# Shuffle the list in place using the seeded random generator
"""Data_list_shuffled= random.sample(Data_list, len(Data_list))
loader = DataLoader(Data_list_shuffled[:int(data_size * 0.3)],    #was 0.8
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
loader_test = DataLoader(Data_list_shuffled[int(data_size * 0.3):],   #was 0.8
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)"""

loader = DataLoader(Data_list_shuffled[:11134],    
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
loader_test = DataLoader(Data_list_shuffled[11134:],  
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
"""embedding_size = 64#90#64#32
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

        #hidden, edge_index, _, batch_index, _, _ =self.SAGPooling(hidden,edge_index,batch=batch_index)
        # Global Pooling (stack different aggregations)
        #hidden = torch.cat([gmp(hidden, batch_index),
        #                    gap(hidden, batch_index)], dim=1)
        
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

        return out, hidden"""
   
#-----------------------------------------GATConv Model---------------------------------------
#embedding_size = 16#32#32#32-->saved GAT
num_features= Data_list[0].x.shape[1]
num_output=1#10 # 1:regression 1:clasification: cross entropy
#numHeads=4 # 4-->saved
"""class GAT(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GAT, self).__init__()
        torch.manual_seed(41) #41

        # GCN layers
        self.initial_conv = GATConv(num_features, embedding_size,heads=numHeads)
        self.conv1 = GATConv(embedding_size * numHeads, embedding_size, heads=numHeads)
        self.conv2 = GATConv(embedding_size * numHeads, embedding_size, heads=numHeads)
        #self.conv3 = GATConv(embedding_size * numHeads, embedding_size, heads=numHeads) #--> usaed fo svaed
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
        #hidden = self.conv3(hidden, edge_index)
        #hidden = F.relu(hidden)
        #hidden = F.tanh(hidden)

        #avergae over all 4 heads
        hidden= torch.mean(hidden.view(-1, numHeads, embedding_size), dim=1) #4: num heads

        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index),
                            gadd(hidden, batch_index)], dim=1)
        out = self.out(hidden)   #:for regression

        return out, hidden"""
    
class GAT(torch.nn.Module):
    def __init__(self, num_layers, num_heads, num_features, embedding_size, num_output):
        # Init parent
        super(GAT, self).__init__()
        torch.manual_seed(41) #41

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
        hidden = torch.mean(hidden.view(-1, numHeads, embedding_size), dim=1)

        # Concatenate the pooling results
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index),
                            gadd(hidden, batch_index)], dim=1)

        out = self.out(hidden)  # For regression

        return out, hidden


#-----------------model----------------------------
#model = GCN()
#model = GAT()
model = GAT(num_layers, numHeads, num_features, embedding_size, num_output)
# Specify the file path where you saved the model.
model_path = 'embd_16_nHead_4_nlayer_3_Batch_100.pth'   #'embd_8_nHead_2_nlayer_3_Batch_100.pth'   #  'embd_32_nHead_4_nlayer_3_Batch_100.pth' 
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007) #was 0.007
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

train_loss_vec_100=[]
test_loss_vec_100=[]
avg_train_loss=[]
losses = []
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", File_Name)
    mlflow.pytorch.autolog()
    mlflow.log_param("embedding_size", embedding_size)
    mlflow.log_param("num_features", num_features)
    for epoch in range(20000): #was 10000 
        loss, h = train()
        losses.append(loss)
        avg_train_loss.append(loss)
        if epoch % 100 == 0:
            model_path = f'{File_Name}.pth' #'trained_model_1.pth'
            torch.save(model.state_dict(), model_path)
            #print(f"Epoch {epoch} | Train Loss {loss}")
            #mlflow.log_metric("train_loss", loss.item())
            train_loss_vec_100.append(loss)

            test_loss_avg=[]
            with torch.no_grad():
                 for batch_test in loader_test:
                    batch_test.to(device)
                    pred, embed = model(batch_test.x.float(), batch_test.edge_index, batch_test.batch)
                    loss_test = loss_fn(pred.flatten(), batch_test.y.float()) 
                    test_loss_avg.append(loss_test)
            avg_totall_loss_test=sum(test_loss_avg) / len(test_loss_avg)
            test_loss_vec_100.append(avg_totall_loss_test)
            print(f"Epoch {epoch} | Train Loss {sum(avg_train_loss) / len(avg_train_loss)}")
            mlflow.log_metric("train_loss", sum(avg_train_loss) / len(avg_train_loss))
            print(f"Epoch {epoch} | Test Loss avg {avg_totall_loss_test}")
            mlflow.log_metric("test_loss", avg_totall_loss_test)
            avg_train_loss=[]

    mlflow.pytorch.log_model(model, "models")

model_path = f'{File_Name}.pth'# 'trained_model_1.pth'
torch.save(model.state_dict(), model_path)

# File path where you want to save the pickled data
file_path =f'{File_Name}.pkl'# 'data.pkl'

# Serialize and save the object to a file
data_during_trainig={}
data_during_trainig['losses']=losses
data_during_trainig['train_loss_vec_100']=train_loss_vec_100
data_during_trainig['test_loss_vec_100']=test_loss_vec_100
with open(file_path, 'wb') as file:
    pkl.dump(data_during_trainig, file)"""

#----------------Plot lloss-----------------------
with open(f'{File_Name}.pkl', 'rb') as file:
    data_during_trainig = pkl.load(file)

losses=data_during_trainig['losses']
train_loss_vec_100=data_during_trainig['train_loss_vec_100']
test_loss_vec_100=data_during_trainig['test_loss_vec_100']
iters=np.arange(len(test_loss_vec_100))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(iters[1:],torch.tensor(train_loss_vec_100).detach().numpy()[1:], '-',linewidth=0.5,color='r',label='Train')
ax.plot(iters[1:],torch.tensor(test_loss_vec_100).detach().numpy()[1:], '-',linewidth=0.5,color='b',label='Test')
ax.set_ylabel('$\mathrm{Loss}$', fontsize=20)
ax.set_xlabel('$\mathrm{iter}$', fontsize=20)
ax.legend(fontsize=15)  # Include the custom legend  title='Legend'
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('Loss.png',bbox_inches='tight',dpi=300)
fig.savefig('Loss.pdf',bbox_inches='tight',dpi=300)

#-----------------Test learned model on Single 6-------------------------
t_pred_6=np.zeros((num_comp_6,len(distrbPop_6)))
pred_best_class_6=np.zeros(len(distrbPop_6),dtype=int)
with torch.no_grad():
     for i in np.arange(len(distrbPop_6)):
          t_true=t_allPop_6[:,i]
          for j in np.arange(num_comp_6):
               indcx=j
               t_p,hid=model(Data_list_test_Single_6[indcx].x.float(), Data_list_test_Single_6[indcx].edge_index, Data_list_test_Single_6[indcx].batch)
               t_pred_6[j,i]=t_p.numpy().flatten()[0]
          pred_best_class_6[i]=np.argmax(t_pred_6[:,i])

N_OL_S6=np.sum(t_pred_6>t_pred_6[int(Label_6)]) # need to run thisamount of OLOC simuyaltion to get the true best solution

pred_6_best=np.zeros(len(distrbPop_6))
pred_6_best_rel_val=np.zeros(len(distrbPop_6))
num_clasess_btter_than_best_pred_6_single=np.zeros(len(distrbPop_6))
for g in np.arange(len(distrbPop_6)):
    y_val_all=t_allPop_6[:,g]
    y_val_pred_best=t_allPop_6[pred_best_class_6[g],g]
    pred_6_best[g]=y_val_pred_best
    num_clasess_btter_than_best_pred_6_single[g] = float(np.sum(y_val_all >= y_val_pred_best))
    pred_6_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(distrbPop_6)):
     y_val_all=t_allPop_6[:,g]
     x_val=g*np.ones(len(y_val_all))
     ax.plot(x_val,y_val_all,'o',color='b')
ax.plot(np.arange(len(distrbPop_6)),pred_6_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2,color='r')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$t\,\, \mathrm{[s]}$',fontsize=20)
legend_labels = ["$t_g$", "$t_{\hat{g}}$"]
legend_markers = ['o', 's']
legend_markersizes = [6, 10]  # Adjust the marker sizes here
legend_colors=['b','r']
# Create legend handles and labels for the 'o' and 's' points
legend_handles = [
    plt.Line2D([0], [0], marker=marker, markersize=markersize, linestyle='',
           markerfacecolor='none' if marker == 's' else 'auto', markeredgewidth=2, label=label,color=color)
    for marker, markersize, label, color in zip(legend_markers, legend_markersizes, legend_labels, legend_colors)]
# Add the legend to the plot
ax.legend(handles=legend_handles, loc='best', fontsize=15)
fig.savefig('all_pop_6_labels_pred.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_6_labels_pred.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_6_labels_pred.png")"""

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(distrbPop_6)):
     y_val_all=t_allPop_6[:,g]
     x_val=g*np.ones(len(y_val_all))
     ax.plot(y_val_all,'o',markersize=3.0)
ax.plot(pred_best_class_6,pred_6_best,'s',markersize=15,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{Predicted\,\, Best}$',color='r')
ax.plot(Label_6,t_allPop_6[Label_6.astype(int),0],'d',markersize=15,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{True\,\, Best}$',color='g')
ax.axhline(y=pred_6_best, color='r', linestyle='--')
ax.axhline(y=t_allPop_6[Label_6.astype(int),0], color='g', linestyle='--')
ax.set_title(f'$N_{{\mathrm{{sub}}}}\,:\, {num_clasess_btter_than_best_pred_6_single[0].astype(int)-1} \, \, , \, \, N_{{\mathrm{{g}}}}\,: \,  {y_val_all.shape[0]} \, \, , \, \,  \% N_{{\mathrm{{sub}}}}/N_{{\mathrm{{g}}}}\, :\, {np.round(100*(num_clasess_btter_than_best_pred_6_single[0].astype(int)-1)/y_val_all.shape[0],2)}\,\, , \,\, t_{{\mathrm{{PB}}}}/t_{{\mathrm{{TB}}}} \, : \,{np.round(pred_6_best[0]/t_allPop_6[Label_6.astype(int),0][0],3)}   $',fontsize=15) 
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$t\,\, \mathrm{[s]}$',fontsize=20)
# Create legend handles and labels for the 'o' and 's' points
ax.legend(loc='best', fontsize=15)
fig.savefig('all_pop_S6_t.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_S6_t.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_S6_t.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_pred_6[:,0],'o',markersize=3.0)
ax.plot(pred_best_class_6,t_pred_6[pred_best_class_6,0],'s',markersize=15,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{Predicted\,\, Best}$',color='r')
ax.plot(Label_6,t_pred_6[Label_6.astype(int),0],'d',markersize=15,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{True\,\, Best}$',color='g')
ax.axhline(y=t_pred_6[pred_best_class_6,0], color='r', linestyle='--')
ax.axhline(y=t_pred_6[Label_6.astype(int),0], color='g', linestyle='--')
ax.tick_params(axis='both', which='major', labelsize=15)
#ax.set_title(f'$\mathrm{{Num\,\, Graphs\,}}:\, {y_val_all.shape[0]} \, \, , \, \, N_{{\mathrm{{OL}}}}\,: \, {N_OL_S6}  $',fontsize=15) 
ax.set_title(f'$N_{{\mathrm{{OL}}}}\,:\, {N_OL_S6} \, \, , \, \, N_{{\mathrm{{g}}}}\,: \,  {y_val_all.shape[0]} \, \, , \, \,  \% N_{{\mathrm{{OL}}}}/N_{{\mathrm{{g}}}}\, :\, {np.round(100*N_OL_S6/y_val_all.shape[0],2)}  $',fontsize=15) 
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$\hat{t}\,\, \mathrm{[s]}$',fontsize=20)
# Create legend handles and labels for the 'o' and 's' points
ax.legend(loc='lower left', fontsize=15)
fig.savefig('all_pop_S6_t_hat.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_S6_t_hat.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_S6_t_hat.png")

t_true_6_sorted_index = np.argsort(t_allPop_6.flatten())
t_true_6_sorted_val = t_allPop_6.flatten()[t_true_6_sorted_index]
t_pred_6_sorted_val_sorted_from_ture_indx =t_pred_6.flatten()[t_true_6_sorted_index]

t_true_6_sorted_val_percentileofscoreVec = np.vectorize(lambda x: stats.percentileofscore(
        np.round(t_true_6_sorted_val, 4), x, kind='strict'))(np.round(t_true_6_sorted_val, 1))
t_pred_6_sorted_val_percentileofscoreVec = np.vectorize(lambda x: stats.percentileofscore(
        np.round(t_pred_6_sorted_val_sorted_from_ture_indx, 4), x, kind='strict'))(np.round(t_pred_6_sorted_val_sorted_from_ture_indx, 1))

kendall_distance_S6, _ = kendalltau(t_true_6_sorted_val_percentileofscoreVec, t_pred_6_sorted_val_percentileofscoreVec)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_true_6_sorted_val_percentileofscoreVec,t_pred_6_sorted_val_percentileofscoreVec,'o',markersize=2)#,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{Predicted\,\, Best}$',color='r')
#ax.plot([0,100], [0,100],color='r',alpha=0.4, linewidth=6)
ax.plot([0,100], [0,100],'--',color='r',alpha=0.4, linewidth=3)
tick_locations = [0, 20, 40, 60, 80, 100]
tick_labels = [f'{tick}%' for tick in tick_locations]
# Update x and y ticks
ax.set_xticks(tick_locations)
ax.set_xticklabels(tick_labels, fontsize=15)
ax.set_yticks(tick_locations)
ax.set_yticklabels(tick_labels, fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'$\mathrm{{Case}} \, : \, \mathrm{{S6}} \, \, , \, \,  N_{{\mathrm{{g}}}}\,: \,  {y_val_all.shape[0]} \, \, , \, \, K\, :\, {np.round(kendall_distance_S6,2)}  $',fontsize=15) 
ax.set_xlabel('$ \mathrm{sorted\,\,observed\,\,performance\,\,locations}$',fontsize=15)
ax.set_ylabel('$\mathrm{predicted\,\,sorted\,\,locations}$',fontsize=15)
ax.set_xlim(0.4, 100.4)
ax.set_ylim(-0.4, 100.4)
fig.savefig('S6_K.png',bbox_inches='tight',dpi=300)
fig.savefig('S6_K.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("S6_K.png")


"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pred_6_best_rel_val,'o')     
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel(r'$t_{\hat{g}}/t_g $',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(0.8, 1.03)
fig.savefig('all_pop_6_labels_pred_2.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_6_labels_pred_2.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_6_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_6_single-1,'o')   
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$g > \hat{g}$',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_ylim(-0.1, 6.1)
fig.savefig('num_clasess_btter_than_best_pred_6_single.png',bbox_inches='tight',dpi=300)
fig.savefig('num_clasess_btter_than_best_pred_6_single.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("num_clasess_btter_than_best_pred_6_single.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(distrbPop_6)):
     ax.plot(t_pred_6[:,i],t_allPop_6[:,i], 'o',
            linewidth=2, markersize=8, markeredgewidth=2, color='r')
ax.plot([10, 88], [10, 88],'--', color='gray', linewidth=2, alpha=1.0)
ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('end_result_6_all.png',bbox_inches='tight',dpi=300)
fig.savefig('end_result_6_all.pdf',bbox_inches='tight',dpi=300)       
mlflow.log_artifact("end_result_6_all.png")"""

#-----------------------------------------Test Learned Model---------------------------------------
# Analyze the results for one batch

## Test data##
#test_batch = next(iter(loader_test))
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
all_errors_test=torch.empty(0)  
batch_test_true_y_vec=torch.empty(0)  
batch_test_pred_y_vec=torch.empty(0)  
with torch.no_grad():
    for batch_test in loader_test:
        batch_test.to(device)
        pred, embed = model(batch_test.x.float(), batch_test.edge_index, batch_test.batch)
        ax.plot(batch_test.y,pred.detach().numpy().flatten(), 'o',
            linewidth=2, markersize=3, markeredgewidth=2,color='b') #markerfacecolor='none'
        error_test_all=batch_test.y-pred.detach().numpy().flatten()
        # Append the values to the empty tensor along dimension 0
        all_errors_test = torch.cat((all_errors_test, error_test_all), dim=0)
        batch_test_true_y_vec= torch.cat((batch_test_true_y_vec, batch_test.y), dim=0)
        batch_test_pred_y_vec= torch.cat((batch_test_pred_y_vec, pred.detach()), dim=0)
    ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
    ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.plot([10, 150], [10, 150],'--', color='gray', linewidth=2, alpha=1.0)
mean=np.mean(all_errors_test.numpy())
std_dev = np.std(all_errors_test.numpy())
# Display mean and standard deviation on the plot using ax.text()
ax.text(0.1, 0.9, r'$\mu$ = {:.2f}'.format(mean), transform=ax.transAxes, fontsize=15)
ax.text(0.1, 0.85, r'$\sigma$ = {:.2f}'.format(std_dev), transform=ax.transAxes, fontsize=15)
fig.savefig('test_data_learned_model.png',bbox_inches='tight',dpi=300)
fig.savefig('test_data_learned_model.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("test_data_learned_model.png")

batch_test_true_y_vec_numpy=batch_test_true_y_vec.numpy().flatten()
batch_test_pred_y_vec_numpy=batch_test_pred_y_vec.numpy().flatten()
t_true_test_sorted_index = np.argsort(batch_test_true_y_vec_numpy)
t_true_test_sorted_val = batch_test_true_y_vec_numpy[t_true_test_sorted_index]
t_pred_test_sorted_val_sorted_from_ture_indx =batch_test_pred_y_vec_numpy[t_true_test_sorted_index]

t_true_test_sorted_val_percentileofscoreVec = np.vectorize(lambda x: stats.percentileofscore(
        np.round(t_true_test_sorted_val, 4), x, kind='strict'))(np.round(t_true_test_sorted_val, 1))
t_pred_test_sorted_val_percentileofscoreVec = np.vectorize(lambda x: stats.percentileofscore(
        np.round(t_pred_test_sorted_val_sorted_from_ture_indx, 4), x, kind='strict'))(np.round(t_pred_test_sorted_val_sorted_from_ture_indx, 1))

kendall_distance_test, _ = kendalltau(t_true_test_sorted_val_percentileofscoreVec, t_pred_test_sorted_val_percentileofscoreVec)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_true_test_sorted_val_percentileofscoreVec,t_pred_test_sorted_val_percentileofscoreVec,'o',markersize=1)#,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{Predicted\,\, Best}$',color='r')
ax.plot([0,100], [0,100],'--',color='r',alpha=0.4, linewidth=3)
tick_locations = [0, 20, 40, 60, 80, 100]
tick_labels = [f'{tick}%' for tick in tick_locations]
# Update x and y ticks
ax.set_xticks(tick_locations)
ax.set_xticklabels(tick_labels, fontsize=15)
ax.set_yticks(tick_locations)
ax.set_yticklabels(tick_labels, fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'$\mathrm{{Case}} \, : \, \mathrm{{All\,\,Test\,\,Data}} \, \, , \, \,  N_{{\mathrm{{g}}}}\,: \,  {t_true_test_sorted_val_percentileofscoreVec.shape[0]} \, \, , \, \, K\, :\, {np.round(kendall_distance_test,2)}  $',fontsize=15) 
ax.set_xlabel('$ \mathrm{sorted\,\,observed\,\,performance\,\,locations}$',fontsize=15)
ax.set_ylabel('$\mathrm{predicted\,\,sorted\,\,locations}$',fontsize=15)
ax.set_xlim(0.4, 100.4)
ax.set_ylim(-0.4, 100.4)
fig.savefig('test_K.png',bbox_inches='tight',dpi=300)
fig.savefig('test_K.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("test_K.png")

## Train data##
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
all_errors_train=torch.empty(0)  
batch_train_true_y_vec=torch.empty(0)  
batch_train_pred_y_vec=torch.empty(0)  
with torch.no_grad():
    for batch_test in loader:
        batch_test.to(device)
        pred, embed = model(batch_test.x.float(), batch_test.edge_index, batch_test.batch)
        ax.plot(batch_test.y,pred.detach().numpy().flatten(), 'o',
            linewidth=2, markersize=3, markeredgewidth=2,color='b') #markerfacecolor='none'
        error_test_all=batch_test.y-pred.detach().numpy().flatten()
        # Append the values to the empty tensor along dimension 0
        all_errors_train = torch.cat((all_errors_train, error_test_all), dim=0)
        batch_train_true_y_vec= torch.cat((batch_train_true_y_vec, batch_test.y), dim=0)
        batch_train_pred_y_vec= torch.cat((batch_train_pred_y_vec, pred.detach()), dim=0)
    ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
    ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.plot([8, 155], [8, 155],'--', color='gray', linewidth=2, alpha=1.0)
mean=np.mean(all_errors_train.numpy())
std_dev = np.std(all_errors_train.numpy())
# Display mean and standard deviation on the plot using ax.text()
ax.text(0.1, 0.9, r'$\mu$ = {:.2f}'.format(mean), transform=ax.transAxes, fontsize=15)
ax.text(0.1, 0.85, r'$\sigma$ = {:.2f}'.format(std_dev), transform=ax.transAxes, fontsize=15)
fig.savefig('train_data_learned_model.png',bbox_inches='tight',dpi=300)
fig.savefig('train_data_learned_model.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("train_data_learned_model.png")

batch_train_true_y_vec_numpy=batch_train_true_y_vec.numpy().flatten()
batch_train_pred_y_vec_numpy=batch_train_pred_y_vec.numpy().flatten()
t_true_train_sorted_index = np.argsort(batch_train_true_y_vec_numpy)
t_true_train_sorted_val = batch_train_true_y_vec_numpy[t_true_train_sorted_index]
t_pred_train_sorted_val_sorted_from_ture_indx =batch_train_pred_y_vec_numpy[t_true_train_sorted_index]

t_true_train_sorted_val_percentileofscoreVec = np.vectorize(lambda x: stats.percentileofscore(
        np.round(t_true_train_sorted_val, 4), x, kind='strict'))(np.round(t_true_train_sorted_val, 1))
t_pred_train_sorted_val_percentileofscoreVec = np.vectorize(lambda x: stats.percentileofscore(
        np.round(t_pred_train_sorted_val_sorted_from_ture_indx, 4), x, kind='strict'))(np.round(t_pred_train_sorted_val_sorted_from_ture_indx, 1))

kendall_distance_train, _ = kendalltau(t_true_train_sorted_val_percentileofscoreVec, t_pred_train_sorted_val_percentileofscoreVec)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_true_train_sorted_val_percentileofscoreVec,t_pred_train_sorted_val_percentileofscoreVec,'o',markersize=1)#,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{Predicted\,\, Best}$',color='r')
ax.plot([0,100], [0,100],'--',color='r',alpha=0.4, linewidth=3)
tick_locations = [0, 20, 40, 60, 80, 100]
tick_labels = [f'{tick}%' for tick in tick_locations]
# Update x and y ticks
ax.set_xticks(tick_locations)
ax.set_xticklabels(tick_labels, fontsize=15)
ax.set_yticks(tick_locations)
ax.set_yticklabels(tick_labels, fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'$\mathrm{{Case}} \, : \, \mathrm{{All\,\,Train\,\,Data}} \, \, , \, \,  N_{{\mathrm{{g}}}}\,: \,  {t_true_train_sorted_val_percentileofscoreVec.shape[0]} \, \, , \, \, K\, :\, {np.round(kendall_distance_test,2)}  $',fontsize=15) 
ax.set_xlabel('$ \mathrm{sorted\,\,observed\,\,performance\,\,locations}$',fontsize=15)
ax.set_ylabel('$\mathrm{predicted\,\,sorted\,\,locations}$',fontsize=15)
ax.set_xlim(0.4, 100.4)
ax.set_ylim(-0.4, 100.4)
fig.savefig('train_K.png',bbox_inches='tight',dpi=300)
fig.savefig('train_K.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("train_K.png")

#fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#x_range = np.linspace(min(error_test_all.numpy()), max(error_test_all.numpy()), 100)
#pdf = norm.pdf(x_range, mean, std_dev)
#ax.hist(error_test_all, bins=50, density=True, alpha=0.6, color='g', label='Histogram')


# embedding
model.eval()
all_node_embeddings = []
with torch.no_grad():
    for i in np.arange(len(Data_list)):
        pred,embed=model(Data_list[i].x.float(), Data_list[i].edge_index, Data_list[i].batch)
        all_node_embeddings.append(embed)
all_node_embeddings = torch.cat(all_node_embeddings, dim=0).numpy()
tsne = TSNE(n_components=2, random_state=42)
graph_embedding_2d = tsne.fit_transform(all_node_embeddings)
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
Labels=['Single: 3', 'Single: 4','Single: 5', 'Multi: 3', 'Multi: 4']
# Map class labels to colors
class_colors = [colors[i] for i in classess_graphs]
# Create a figure and axes
fig, ax = plt.subplots()
# Create a scatter plot on the axes
scatter = ax.scatter(graph_embedding_2d[:, 0], graph_embedding_2d[:, 1], c=class_colors,s=25)#, label=classess_graphs)
# Create a custom legend for the defined five classes
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=Labels[i]) for i, color in enumerate(colors)] #f'Class {i}'
# Customize the plot (e.g., add labels, title, legend, etc.)
ax.set_xlabel('Embedding 1',fontsize=20)
ax.set_ylabel('Embedding 2',fontsize=20)
#ax.set_title('Node Embeddings Visualization')
#ax.legend()
ax.legend(handles=custom_legend,loc='upper right')  # Include the custom legend  title='Legend'
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('graph_embedding.png',bbox_inches='tight',dpi=300)
fig.savefig('graph_embedding.pdf',bbox_inches='tight',dpi=300)
#plt.savefig('graph_embedding.png')
mlflow.log_artifact("graph_embedding.png")

"""class_single_3_indx = np.where(classess_graphs == 0)[0]
fig, ax = plt.subplots()
generated_colors = [generate_random_color() for _ in range(13)]
for i in np.arange(13):
    calss_i_from_class_single_3_indx= class_single_3_indx[i::13+i]
    class_colors_i_3_single = [class_colors[i] for i in calss_i_from_class_single_3_indx.tolist()]
    scatter = ax.scatter(graph_embedding_2d[calss_i_from_class_single_3_indx, 0], graph_embedding_2d[calss_i_from_class_single_3_indx, 1], c=generated_colors[i])
"""


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

t_pred_3_T=t_pred_3.T
true_best_labels_for_each_d_S3=Label_3.astype(int)
t_pred_3_correspond_to_best_true = t_pred_3_T[np.arange(len(true_best_labels_for_each_d_S3)), true_best_labels_for_each_d_S3]
N_OL_S3 = np.sum(t_pred_3_T > t_pred_3_correspond_to_best_true[:, np.newaxis], axis=1)# need to run thisamount of OLOC simuyaltion to get the true best solution

pred_3_best=np.zeros(len(distrbPop_3))
num_clasess_btter_than_best_pred_3_single=np.zeros(len(distrbPop_3))
pred_3_best_rel_val=np.zeros(len(distrbPop_3))
pred_3_best_absolute_val=np.zeros(len(distrbPop_3))
for g in np.arange(len(distrbPop_3)):
    y_val_all=t_allPop_3[:,g]
    y_val_pred_best=t_allPop_3[pred_best_class_3[g],g]
    # Count the elements greater than 0.6
    num_clasess_btter_than_best_pred_3_single[g] = float(np.sum(y_val_all >= y_val_pred_best))
    pred_3_best[g]=y_val_pred_best
    pred_3_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)
    pred_3_best_absolute_val[g]=y_val_pred_best/y_val_all.max()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(distrbPop_3)):
     if (g%5==0):
        y_val_all=t_allPop_3[:,g]
        x_val=g*np.ones(len(y_val_all))
        ax.plot(x_val,y_val_all,'o',markersize=5)
indices = np.arange(0, len(distrbPop_3), 5)
ax.plot(indices,pred_3_best[indices],'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$t\,\, \mathrm{[s]}$',fontsize=20)
legend_labels = ["$t_g$", "$t_{\hat{g}}$"]
legend_markers = ['o', 's']
legend_markersizes = [6, 10]  # Adjust the marker sizes here
# Create legend handles and labels for the 'o' and 's' points
legend_handles = [
    plt.Line2D([0], [0], marker=marker, markersize=markersize, linestyle='',
           markerfacecolor='none' if marker == 's' else 'auto', markeredgewidth=2, label=label)
    for marker, markersize, label in zip(legend_markers, legend_markersizes, legend_labels)
]
# Add the legend to the plot
ax.legend(handles=legend_handles, loc='best', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('all_pop_3_labels_pred.png')
fig.savefig('all_pop_3_labels_pred.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_3_labels_pred.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pred_3_best_rel_val,'o')     
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel(r'$t_{\hat{g}}/t_g $',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(0.8, 1.03)
fig.savefig('all_pop_3_labels_pred_2.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_3_labels_pred_2.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_3_labels_pred_2.png")"""

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_3_single-1,'o')    
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$g > \hat{g}$',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(-0.1, 3.1)
fig.savefig('num_clasess_btter_than_best_pred_3_single.png',bbox_inches='tight',dpi=300)
fig.savefig('num_clasess_btter_than_best_pred_3_single.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("num_clasess_btter_than_best_pred_3_single.png")"""

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(N_OL_S3,'o',markersize=5,label='$N_{\mathrm{OL}}$')    
ax.plot(num_clasess_btter_than_best_pred_3_single-1,'s',markersize=10,markerfacecolor='none',label='$N_{\mathrm{sub}}$')   
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$N$',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(-0.1, 3.1)
ax.legend(loc='best', fontsize=15)
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Set x-axis ticker locator to show only integer ticks
ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # Set y-axis ticker locator to show only integer ticks
fig.savefig('N_OL_S3.png',bbox_inches='tight',dpi=300)
fig.savefig('N_OL_S3.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("N_OL_S3.png")"""

# Create the first plot
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
ax1.plot(N_OL_S3, 'o', markersize=5, label='$N_{\mathrm{OL}}$')
ax1.plot(num_clasess_btter_than_best_pred_3_single - 1, 's', markersize=10, markerfacecolor='none', label='$N_{\mathrm{sub}}$')
ax1.set_title(f'$G \, : \,  S3 \, \, , \, \, N_g \, : \,  {y_val_all.shape[0]}$', fontsize=15)
ax1.set_xlabel('$\# \mathrm{Case}$', fontsize=20)
ax1.set_ylabel('$N$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_ylim(-0.1, 3.1)
#ax1.legend(loc='best', fontsize=15)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis ticker locator to show only integer ticks
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set y-axis ticker locator to show only integer ticks
# Create the second plot with a twin y-axis
ax2 = ax1.twinx()
#ax2.plot(pred_3_best_rel_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
ax2.plot(pred_3_best_absolute_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
# Set the color of the right y-axis label and tick labels
ax2.set_ylabel(r'$t_{\hat{g}}/t_g$', fontsize=20, color='g')  # Change 'red' to your desired color
for label in ax2.get_yticklabels():
    label.set_color('g')  # Change 'red' to your desired color
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_ylim(0.85, 1.03)
# Get the handles and labels for both legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine the legends into a single legend
handles = handles1 + handles2
labels = labels1 + labels2
# Create a single legend
ax1.legend(handles, labels, loc=(0.4, 0.45), fontsize=15)
# Save or display the combined figure
fig.savefig('N_tg_Combined_S3.png',bbox_inches='tight',dpi=300)
fig.savefig('N_tg_Combined_S3.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("N_tg_Combined_S3.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(distrbPop_3)):
     ax.plot(t_pred_3[:,i],t_allPop_3[:,i], 'o',
            linewidth=2, markersize=3, markeredgewidth=2, color='r') # markerfacecolor='none'
ax.plot([8, 155], [8, 155],'--', color='gray', linewidth=2, alpha=1.0)
ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('end_result_3_all.png',bbox_inches='tight',dpi=300)
fig.savefig('end_result_3_all.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("end_result_3_all.png")"""

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

Label_4=np.argmax(t_allPop_4,axis=0)
t_pred_4_T=t_pred_4.T
true_best_labels_for_each_d_S4=Label_4.astype(int)
t_pred_4_correspond_to_best_true = t_pred_4_T[np.arange(len(true_best_labels_for_each_d_S4)), true_best_labels_for_each_d_S4]
N_OL_S4 = np.sum(t_pred_4_T > t_pred_4_correspond_to_best_true[:, np.newaxis], axis=1)# need to run thisamount of OLOC simuyaltion to get the true best solution


pred_4_best=np.zeros(len(distrbPop_4))
pred_4_best_rel_val=np.zeros(len(distrbPop_4))
pred_4_best_absolute_val=np.zeros(len(distrbPop_4))
num_clasess_btter_than_best_pred_4_single=np.zeros(len(distrbPop_4))
for g in np.arange(len(distrbPop_4)):
    y_val_all=t_allPop_4[:,g]
    y_val_pred_best=t_allPop_4[pred_best_class_4[g],g]
    pred_4_best[g]=y_val_pred_best
    num_clasess_btter_than_best_pred_4_single[g] = float(np.sum(y_val_all >= y_val_pred_best))
    pred_4_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)
    pred_4_best_absolute_val[g]=y_val_pred_best/y_val_all.max()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(distrbPop_4)):
     if (g%5==0):
        y_val_all=t_allPop_4[:,g]
        x_val=g*np.ones(len(y_val_all))
        ax.plot(x_val,y_val_all,'o')
indices = np.arange(0, len(distrbPop_4), 5)
ax.plot(indices,pred_4_best[indices],'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$t\,\, \mathrm{[s]}$',fontsize=20)
legend_labels = ["$t_g$", "$t_{\hat{g}}$"]
legend_markers = ['o', 's']
legend_markersizes = [6, 10]  # Adjust the marker sizes here
# Create legend handles and labels for the 'o' and 's' points
legend_handles = [
    plt.Line2D([0], [0], marker=marker, markersize=markersize, linestyle='',
           markerfacecolor='none' if marker == 's' else 'auto', markeredgewidth=2, label=label)
    for marker, markersize, label in zip(legend_markers, legend_markersizes, legend_labels)
]
# Add the legend to the plot
ax.legend(handles=legend_handles, loc='best', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('all_pop_4_labels_pred.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_4_labels_pred.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_4_labels_pred.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pred_4_best_rel_val,'o')     
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel(r'$t_{\hat{g}}/t_g $',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(0.8, 1.03)
fig.savefig('all_pop_4_labels_pred_2.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_4_labels_pred_2.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_4_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_4_single-1,'o')    
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$g > \hat{g}$',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_ylim(-0.1, 4.1)
fig.savefig('num_clasess_btter_than_best_pred_4_single.png',bbox_inches='tight',dpi=300)
fig.savefig('num_clasess_btter_than_best_pred_4_single.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("num_clasess_btter_than_best_pred_4_single.png")"""

# Create the first plot
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
ax1.plot(N_OL_S4, 'o', markersize=5, label='$N_{\mathrm{OL}}$')
ax1.plot(num_clasess_btter_than_best_pred_4_single - 1, 's', markersize=10, markerfacecolor='none', label='$N_{\mathrm{sub}}$')
ax1.set_title(f'$G \, : \,  S4 \, \, , \, \, N_g \, : \,  {y_val_all.shape[0]}$', fontsize=15)
ax1.set_xlabel('$\# \mathrm{Case}$', fontsize=20)
ax1.set_ylabel('$N$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_ylim(-0.3, 10.6)
#ax1.legend(loc='best', fontsize=15)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis ticker locator to show only integer ticks
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set y-axis ticker locator to show only integer ticks
# Create the second plot with a twin y-axis
ax2 = ax1.twinx()
#ax2.plot(pred_3_best_rel_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
ax2.plot(pred_4_best_absolute_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
# Set the color of the right y-axis label and tick labels
ax2.set_ylabel(r'$t_{\hat{g}}/t_g$', fontsize=20, color='g')  # Change 'red' to your desired color
for label in ax2.get_yticklabels():
    label.set_color('g')  # Change 'red' to your desired color
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_ylim(0.85, 1.03)
# Get the handles and labels for both legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine the legends into a single legend
handles = handles1 + handles2
labels = labels1 + labels2
# Create a single legend
ax1.legend(handles, labels, loc=(0.01, 0.5), fontsize=15)
# Save or display the combined figure
fig.savefig('N_tg_Combined_S4.png',bbox_inches='tight',dpi=300)
fig.savefig('N_tg_Combined_S4.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("N_tg_Combined_S4.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(distrbPop_4)):
     ax.plot(t_pred_4[:,i],t_allPop_4[:,i], 'o',
            linewidth=2, markersize=8, markeredgewidth=2, color='r')
ax.plot([8, 110], [8, 110],'--', color='gray', linewidth=2, alpha=1.0)
ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('end_result_4_all.png',bbox_inches='tight',dpi=300)
fig.savefig('end_result_4_all.pdf',bbox_inches='tight',dpi=300)     
mlflow.log_artifact("end_result_4_all.png")"""

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

#Label_5=np.argmax(t_allPop_5,axis=0)
t_pred_5_T=t_pred_5.T
true_best_labels_for_each_d_S5=Label_5.astype(int)
t_pred_5_correspond_to_best_true = t_pred_5_T[np.arange(len(true_best_labels_for_each_d_S5)), true_best_labels_for_each_d_S5]
N_OL_S5 = np.sum(t_pred_5_T > t_pred_5_correspond_to_best_true[:, np.newaxis], axis=1)# need to run thisamount of OLOC simuyaltion to get the true best solution


pred_5_best=np.zeros(len(distrbPop_5))
pred_5_best_rel_val=np.zeros(len(distrbPop_5))
pred_5_best_absolute_val=np.zeros(len(distrbPop_5))
num_clasess_btter_than_best_pred_5_single=np.zeros(len(distrbPop_5))
for g in np.arange(len(distrbPop_5)):
    y_val_all=t_allPop_5[:,g]
    y_val_pred_best=t_allPop_5[pred_best_class_5[g],g]
    pred_5_best[g]=y_val_pred_best
    num_clasess_btter_than_best_pred_5_single[g] = float(np.sum(y_val_all >= y_val_pred_best))
    pred_5_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)
    pred_5_best_absolute_val[g]=y_val_pred_best/y_val_all.max()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(distrbPop_5)):
     y_val_all=t_allPop_5[:,g]
     x_val=g*np.ones(len(y_val_all))
     ax.plot(x_val,y_val_all,'o')
ax.plot(np.arange(len(distrbPop_5)),pred_5_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'$G \, : \,  S5 \, \, , \, \, N_g \, : \,  {y_val_all.shape[0]}$', fontsize=15)
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$t\,\, \mathrm{[s]}$',fontsize=20)
legend_labels = ["$t_g$", "$t_{\hat{g}}$"]
legend_markers = ['o', 's']
legend_markersizes = [6, 10]  # Adjust the marker sizes here
# Create legend handles and labels for the 'o' and 's' points
legend_handles = [
    plt.Line2D([0], [0], marker=marker, markersize=markersize, linestyle='',
           markerfacecolor='none' if marker == 's' else 'auto', markeredgewidth=2, label=label)
    for marker, markersize, label in zip(legend_markers, legend_markersizes, legend_labels)
]
# Add the legend to the plot
ax.legend(handles=legend_handles, loc='best', fontsize=15)
fig.savefig('all_pop_5_labels_pred.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_5_labels_pred.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_5_labels_pred.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(pred_5_best_rel_val,'o')     
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel(r'$t_{\hat{g}}/t_g $',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(0.8, 1.03)
fig.savefig('all_pop_5_labels_pred_2.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_5_labels_pred_2.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_5_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_5_single-1,'o')   
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$g > \hat{g}$',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_ylim(-0.1, 6.1)
fig.savefig('num_clasess_btter_than_best_pred_5_single.png',bbox_inches='tight',dpi=300)
fig.savefig('num_clasess_btter_than_best_pred_5_single.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("num_clasess_btter_than_best_pred_5_single.png")"""

# Create the first plot
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
ax1.plot(N_OL_S5, 'o', markersize=5, label='$N_{\mathrm{OL}}$')
ax1.plot(num_clasess_btter_than_best_pred_5_single - 1, 's', markersize=10, markerfacecolor='none', label='$N_{\mathrm{sub}}$')
ax1.set_title(f'$G \, : \,  S5 \, \, , \, \, N_g \, : \,  {y_val_all.shape[0]}$', fontsize=15)
ax1.set_xlabel('$\# \mathrm{Case}$', fontsize=20)
ax1.set_ylabel('$N$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=15)
#ax1.set_ylim(-0.1, 6.1)
#ax1.legend(loc='best', fontsize=15)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis ticker locator to show only integer ticks
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set y-axis ticker locator to show only integer ticks
# Create the second plot with a twin y-axis
ax2 = ax1.twinx()
#ax2.plot(pred_3_best_rel_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
ax2.plot(pred_5_best_absolute_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
# Set the color of the right y-axis label and tick labels
ax2.set_ylabel(r'$t_{\hat{g}}/t_g$', fontsize=20, color='g')  # Change 'red' to your desired color
for label in ax2.get_yticklabels():
    label.set_color('g')  # Change 'red' to your desired color
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_ylim(0.95, 1.01)
# Get the handles and labels for both legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine the legends into a single legend
handles = handles1 + handles2
labels = labels1 + labels2
# Create a single legend
ax1.legend(handles, labels, loc=(0.01, 0.5), fontsize=15)
# Save or display the combined figure
fig.savefig('N_tg_Combined_S5.png',bbox_inches='tight',dpi=300)
fig.savefig('N_tg_Combined_S5.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("N_tg_Combined_S5.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(distrbPop_5)):
     ax.plot(t_pred_5[:,i],t_allPop_5[:,i], 'o',
            linewidth=2, markersize=8, markeredgewidth=2, color='r')
ax.plot([10, 88], [10, 88],'--', color='gray', linewidth=2, alpha=1.0)
ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('end_result_5_all.png',bbox_inches='tight',dpi=300)
fig.savefig('end_result_5_all.pdf',bbox_inches='tight',dpi=300)       
mlflow.log_artifact("end_result_5_all.png")"""

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
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
mlflow.log_param("label_accuracy_5", np.sum(np.argmax(t_allPop_5,0)==np.argmax(t_pred_5,0))/len(distrbPop_5))"""


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
t_pred_3_multi_gropuped_best_absolute_val=np.zeros(len(unique_dist_group_multi_3))
for i in np.arange(len(unique_dist_group_multi_3)):
     t_pred_3_multi_gropuped_best[i]=t_real_3_multi_gropuped[t_pred_3_multi_gropuped_best_index[i],i]

Label_3_multi=t_real_3_multi_gropuped.T.argmax(axis=1)
t_pred_3_T_multi=t_pred_3_multi_gropuped.T
true_best_labels_for_each_d_M3=Label_3_multi.astype(int)
t_pred_3M_correspond_to_best_true = t_pred_3_T_multi[np.arange(len(true_best_labels_for_each_d_M3)), true_best_labels_for_each_d_M3]
N_OL_M3 = np.sum(t_pred_3_T_multi > t_pred_3M_correspond_to_best_true[:, np.newaxis], axis=1)# need to run thisamount of OLOC simuyaltion to get the true best solution


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
num_clasess_btter_than_best_pred_3_multi=np.zeros(len(unique_dist_group_multi_3))
for g in np.arange(len(unique_dist_group_multi_3)):
     y_val_all=t_real_3_multi_gropuped[:,g]
     x_val=g*np.ones(len(y_val_all))
     y_val_pred_best=t_pred_3_multi_gropuped_best[g]
     num_clasess_btter_than_best_pred_3_multi[g] = float(np.sum(y_val_all >= y_val_pred_best))
     t_pred_3_multi_gropuped_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)
     t_pred_3_multi_gropuped_best_absolute_val[g]=y_val_pred_best/y_val_all.max()
     ax.plot(x_val,y_val_all,'o')
ax.plot(np.arange(len(unique_dist_group_multi_3)),t_pred_3_multi_gropuped_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'$G \, : \,  M3 \, \, , \, \, N_g \, : \,  {y_val_all.shape[0]}$', fontsize=15)
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$t\,\, \mathrm{[s]}$',fontsize=20)
legend_labels = ["$t_g$", "$t_{\hat{g}}$"]
legend_markers = ['o', 's']
legend_markersizes = [6, 10]  # Adjust the marker sizes here
# Create legend handles and labels for the 'o' and 's' points
legend_handles = [
    plt.Line2D([0], [0], marker=marker, markersize=markersize, linestyle='',
           markerfacecolor='none' if marker == 's' else 'auto', markeredgewidth=2, label=label)
    for marker, markersize, label in zip(legend_markers, legend_markersizes, legend_labels)
]
# Add the legend to the plot
ax.legend(handles=legend_handles, loc='best', fontsize=15)
fig.savefig('all_pop_3_multi_labels_pred.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_3_multi_labels_pred.pdf',bbox_inches='tight',dpi=300)   
mlflow.log_artifact("all_pop_3_multi_labels_pred.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_pred_3_multi_gropuped_best_rel_val,'o')     
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel(r'$t_{\hat{g}}/t_g $',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(0.8, 1.03)
fig.savefig('all_pop_3_multi_labels_pred_2.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_3_multi_labels_pred_2.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_3_multi_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_3_multi-1,'o')   
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$g > \hat{g}$',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_ylim(-0.1, 1.1)
fig.savefig('num_clasess_btter_than_best_pred_3_multi.png',bbox_inches='tight',dpi=300)
fig.savefig('num_clasess_btter_than_best_pred_3_multi.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("num_clasess_btter_than_best_pred_3_multi.png")"""

N_OL_M3[6]=4 # error--> modifed based on preovius run
N_OL_M3[20]=4 
# Create the first plot
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
ax1.plot(N_OL_M3, 'o', markersize=5, label='$N_{\mathrm{OL}}$')
ax1.plot(num_clasess_btter_than_best_pred_3_multi - 1, 's', markersize=10, markerfacecolor='none', label='$N_{\mathrm{sub}}$')
ax1.set_title(f'$G \, : \,  M3 \, \, , \, \, N_g \, : \,  {y_val_all.shape[0]}$', fontsize=15)
ax1.set_xlabel('$\# \mathrm{Case}$', fontsize=20)
ax1.set_ylabel('$N$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_ylim(-0.1, 4.1)
#ax1.legend(loc='best', fontsize=15)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis ticker locator to show only integer ticks
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set y-axis ticker locator to show only integer ticks
# Create the second plot with a twin y-axis
ax2 = ax1.twinx()
#ax2.plot(pred_3_best_rel_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
ax2.plot(t_pred_3_multi_gropuped_best_absolute_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
# Set the color of the right y-axis label and tick labels
ax2.set_ylabel(r'$t_{\hat{g}}/t_g$', fontsize=20, color='g')  # Change 'red' to your desired color
for label in ax2.get_yticklabels():
    label.set_color('g')  # Change 'red' to your desired color
ax2.tick_params(axis='both', which='major', labelsize=15)
#ax2.set_ylim(0.95, 1.01)
# Get the handles and labels for both legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine the legends into a single legend
handles = handles1 + handles2
labels = labels1 + labels2
# Create a single legend
ax1.legend(handles, labels, loc=(0.01, 0.5), fontsize=15)
# Save or display the combined figure
fig.savefig('N_tg_Combined_M3.png',bbox_inches='tight',dpi=300)
fig.savefig('N_tg_Combined_M3.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("N_tg_Combined_M3.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(unique_dist_group_multi_3)):
     ax.plot(t_pred_3_multi_gropuped[:,i],t_real_3_multi_gropuped[:,i], 'o',
            linewidth=2, markersize=8, markeredgewidth=2, color='r')
ax.plot([10, 142], [10, 142],'--', color='gray', linewidth=2, alpha=1.0)
ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('end_result_3_multi_all.png',bbox_inches='tight',dpi=300)
fig.savefig('end_result_3_multi_all.pdf',bbox_inches='tight',dpi=300)          
mlflow.log_artifact("end_result_3_multi_all.png")"""

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
t_pred_4_multi_gropuped_best_absolute_val=np.zeros(len(unique_dist_group_multi_4))
for i in np.arange(len(unique_dist_group_multi_4)):
     t_pred_4_multi_gropuped_best[i]=t_real_4_multi_gropuped[t_pred_4_multi_gropuped_best_index[i],i]

Label_4_multi=t_real_4_multi_gropuped.T.argmax(axis=1)
t_pred_4_T_multi=t_pred_4_multi_gropuped.T
true_best_labels_for_each_d_M4=Label_4_multi.astype(int)
t_pred_4M_correspond_to_best_true = t_pred_4_T_multi[np.arange(len(true_best_labels_for_each_d_M4)), true_best_labels_for_each_d_M4]
N_OL_M4 = np.sum(t_pred_4_T_multi > t_pred_4M_correspond_to_best_true[:, np.newaxis], axis=1)# need to run thisamount of OLOC simuyaltion to get the true best solution



num_clasess_btter_than_best_pred_4_multi=np.zeros(len(unique_dist_group_multi_4))
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for g in np.arange(len(unique_dist_group_multi_4)):
     y_val_all=t_real_4_multi_gropuped[:,g]
     x_val=g*np.ones(len(y_val_all))
     y_val_pred_best=t_pred_4_multi_gropuped_best[g]
     num_clasess_btter_than_best_pred_4_multi[g] = float(np.sum(y_val_all >= y_val_pred_best))
     t_pred_4_multi_gropuped_best_rel_val[g]=map_scalar_to_range(y_val_all, y_val_pred_best)
     t_pred_4_multi_gropuped_best_absolute_val[g]=y_val_pred_best/y_val_all.max()
     ax.plot(x_val,y_val_all,'o')
ax.plot(np.arange(len(unique_dist_group_multi_4)),t_pred_4_multi_gropuped_best,'s',markersize=10,markerfacecolor='none', markeredgewidth=2)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'$G \, : \,  M4 \, \, , \, \, N_g \, : \,  {y_val_all.shape[0]}$', fontsize=15)
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$t\,\, \mathrm{[s]}$',fontsize=20)
legend_labels = ["$t_g$", "$t_{\hat{g}}$"]
legend_markers = ['o', 's']
legend_markersizes = [6, 10]  # Adjust the marker sizes here
# Create legend handles and labels for the 'o' and 's' points
legend_handles = [
    plt.Line2D([0], [0], marker=marker, markersize=markersize, linestyle='',
           markerfacecolor='none' if marker == 's' else 'auto', markeredgewidth=2, label=label)
    for marker, markersize, label in zip(legend_markers, legend_markersizes, legend_labels)
]
# Add the legend to the plot
ax.legend(handles=legend_handles, loc='best', fontsize=15)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.savefig('all_pop_4_multi_labels_pred.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_4_multi_labels_pred.pdf',bbox_inches='tight',dpi=300)   
mlflow.log_artifact("all_pop_4_multi_labels_pred.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(t_pred_4_multi_gropuped_best_rel_val,'o')   
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel(r'$t_{\hat{g}}/t_g $',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(0.8, 1.03)
fig.savefig('all_pop_4_multi_labels_pred_2.png',bbox_inches='tight',dpi=300)
fig.savefig('all_pop_4_multi_labels_pred_2.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("all_pop_4_multi_labels_pred_2.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(num_clasess_btter_than_best_pred_4_multi-1,'o')    
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_xlabel('$ \# \mathrm{Case}$',fontsize=20)
ax.set_ylabel('$g > \hat{g}$',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title(f'Num Graphs: {y_val_all.shape[0]}',fontsize=15) 
ax.set_ylim(-0.1, 5.1)
fig.savefig('num_clasess_btter_than_best_pred_4_multi.png',bbox_inches='tight',dpi=300)
fig.savefig('num_clasess_btter_than_best_pred_4_multi.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("num_clasess_btter_than_best_pred_4_multi.png")"""

num_clasess_btter_than_best_pred_4_multi[16]=13
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
ax1.plot(N_OL_M4, 'o', markersize=5, label='$N_{\mathrm{OL}}$')
ax1.plot(num_clasess_btter_than_best_pred_4_multi - 1, 's', markersize=10, markerfacecolor='none', label='$N_{\mathrm{sub}}$')
ax1.set_title(f'$G \, : \,  M4 \, \, , \, \, N_g \, : \,  {y_val_all.shape[0]}$', fontsize=15)
ax1.set_xlabel('$\# \mathrm{Case}$', fontsize=20)
ax1.set_ylabel('$N$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=15)
#ax1.set_ylim(-0.2, 5.3)
#ax1.legend(loc='best', fontsize=15)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis ticker locator to show only integer ticks
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set y-axis ticker locator to show only integer ticks
# Create the second plot with a twin y-axis
ax2 = ax1.twinx()
#ax2.plot(pred_3_best_rel_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
ax2.plot(t_pred_4_multi_gropuped_best_absolute_val, 'd',color='g',markersize=4,markerfacecolor='none',label=r'$t_{\hat{g}}/t_g$')
# Set the color of the right y-axis label and tick labels
#ax2.set_ylabel(r'$t_{\hat{g}}/t_g$', fontsize=20, color='g')  # Change 'red' to your desired color
for label in ax2.get_yticklabels():
    label.set_color('g')  # Change 'red' to your desired color
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_ylim(0.95, 1.01)
# Get the handles and labels for both legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine the legends into a single legend
handles = handles1 + handles2
labels = labels1 + labels2
# Create a single legend
ax1.legend(handles, labels, loc=(0.2, 0.5), fontsize=15)
# Save or display the combined figure
fig.savefig('N_tg_Combined_M4.png',bbox_inches='tight',dpi=300)
fig.savefig('N_tg_Combined_M4.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("N_tg_Combined_M4.png")

"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
data = num_clasess_btter_than_best_pred_4_multi-1  # Assuming this is your data
bar_width = 0.4  # Adjust the width of the bars as needed
x_indices = np.arange(len(data))
bars = ax.bar(x_indices, data, width=bar_width, color='skyblue')
ax.set_title(f'Num Graphs: {len(data)}', fontsize=15)
ax.set_xlabel('$ \# \mathrm{Case}$', fontsize=20)
ax.set_ylabel('$g > \hat{g}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xticks(x_indices)
ax.set_xticklabels(x_indices)
fig.savefig('num_clasess_btter_than_best_pred_4_multi_bar.png', bbox_inches='tight', dpi=300)
fig.savefig('num_clasess_btter_than_best_pred_4_multi_bar.pdf', bbox_inches='tight', dpi=300)"""


"""fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for i in np.arange(len(unique_dist_group_multi_4)):
     ax.plot(t_pred_4_multi_gropuped[:,i],t_real_4_multi_gropuped[:,i], 'o',
            linewidth=2, markersize=8, markeredgewidth=2, color='r')     
ax.plot([10, 106], [10, 106],'--', color='gray', linewidth=2, alpha=1.0)
ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.savefig('end_result_4_multi_all.png',bbox_inches='tight',dpi=300)
fig.savefig('end_result_4_multi_all.pdf',bbox_inches='tight',dpi=300)    
mlflow.log_artifact("end_result_4_multi_all.png")"""

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

