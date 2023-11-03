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
from scipy.stats import norm
import argparse
import scipy.io
import pandas as pd


# Create an argument parser
"""parser = argparse.ArgumentParser(description='Vehicle Test Script')
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

embedding_size=16#32
numHeads=4
num_layers=3
NUM_GRAPHS_PER_BATCH=500#100

#python Vehicle_GNN_step_1.py --embedding_size 16 --numHeads 4 --num_layers 2 --NUM_GRAPHS_PER_BATCH 500


print(f"embedding_size: {embedding_size}")
print(f"numHeads: {numHeads}")
print(f"num_layers: {num_layers}")
print(f"NUM_GRAPHS_PER_BATCH: {NUM_GRAPHS_PER_BATCH}")

File_Name=f"Vehicle_embd_{embedding_size}_nHead_{numHeads}_nlayer_{num_layers}_Batch_{NUM_GRAPHS_PER_BATCH}"


my_data = pd.read_csv('Vehicle_Data/my_table_GNN.csv')

Graph_GNN_data = scipy.io.loadmat('Vehicle_Data/Graph_GNN_data.mat')['Graph_GNN_data']
#print(len(Graph_GNN_data))

list_of_dicts = [{} for _ in range(4359)]

for g in np.arange(4359):

    node_feature_i=Graph_GNN_data[g][0][0][0][0]
    edge_list_i=Graph_GNN_data[g][0][0][0][1]
    L_array_i=Graph_GNN_data[g][0][0][0][2][0]
    L_i = [item for subarray in L_array_i for item in subarray]
    A_i=Graph_GNN_data[g][0][0][0][3]
    F_i=Graph_GNN_data[g][0][0][0][4][0][0]
    comp=Graph_GNN_data[g][0][0][0][5][0][0][0]
    xp=Graph_GNN_data[g][0][0][0][5][0][0][1]
    # Extract values and create a list of dictionaries
    try:
        dict_comp_i = [{'min': row[0][0][0], 'max': row[0][1][0], 'I': row[0][2], 'name': row[0][3][0]} for row in comp]
        dict_xp_i = [{'blockName': row[0][0][0], 'variableName': row[0][1][0], 'min': row[0][2], 'max': row[0][3][0]} for row in xp]
    except:
        dict_comp_i={}
        dict_xp_i={}
    xop_opt_i=Graph_GNN_data[g][0][0][0][5][0][0][2].flatten()
    exif_flag_i=Graph_GNN_data[g][0][0][0][5][0][0][3].flatten()

    list_of_dicts[g]['node_feature']=torch.from_numpy(node_feature_i).float()
    list_of_dicts[g]['edge_list']=torch.from_numpy(edge_list_i-1).long() 
    list_of_dicts[g]['L']=L_i
    list_of_dicts[g]['A']=A_i
    list_of_dicts[g]['F']= torch.from_numpy(np.array(F_i)).float() 
    list_of_dicts[g]['dict_comp']=dict_comp_i
    list_of_dicts[g]['xop_opt']=xop_opt_i
    list_of_dicts[g]['exif_flag']=exif_flag_i

Data_list = [0] * 4359
for i in np.arange(4359):       
        Data_list[i] = Data(x=list_of_dicts[i]['node_feature'], edge_index=list_of_dicts[i]['edge_list'],y=list_of_dicts[i]['F']) #,y=t_allPop_4[j][i])#, ,y=classes_pop_4[j][i] edge_attr=edge_weight)
        #torch.save(Data_list[indx], os.path.join(os.getcwd()+'/Pop3_Dataset/') + f'4_data_multy{indx}.pt')

#list of graphs in nx format
graphs_list_nx=[]
for i in np.arange(len(Data_list)):
     graphs_list_nx.append(to_networkx(Data_list[i], to_undirected=True))


node_labels_dict = {0: 's', 1: 'u', 2: 'f', 3: 'm', 4: 'p', 5: 'k', 6: 'b', 7:'bk', 8: 'bkk', 9:'bbk' }


g=2000
node_labels={}
nodes_position=[np.where(row == 1)[0] for row in Data_list[g].x]
for i in np.arange(len(nodes_position)):
     node_labels[i]=node_labels_dict[nodes_position[i][0]]
#labels = {node: node_labels[i] for i, node in enumerate(graphs_list_nx[g].nodes())}
nx.draw_networkx(graphs_list_nx[g],labels=node_labels, with_labels=True)
#list_of_dicts[0]['L']
# nx.draw_networkx(g)
plt.show()  # Add this line to display the plot
plt.clf()

#-----------------------------------------Batch Loader---------------------------------------
data_size = len(Data_list)
random.seed(42)
# Shuffle the list in place using the seeded random generator
Data_list_shuffled= random.sample(Data_list, len(Data_list))
loader = DataLoader(Data_list_shuffled[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
loader_test = DataLoader(Data_list_shuffled[int(data_size * 0.8):],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


#-----------------------------------------GATConv Model---------------------------------------
#embedding_size = 16#32#32#32-->saved GAT
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
num_features= Data_list[0].x.shape[1]
num_output=1#10 # 1:regression 1:clasification: cross entropy
    
model = GAT(num_layers, numHeads, num_features, embedding_size, num_output)
# Specify the file path where you saved the model.
model_path = 'Vehicle_embd_16_nHead_4_nlayer_3_Batch_500.pth' # 'Vehicle_embd_32_nHead_4_nlayer_3_Batch_100.pth' #'embd_32_nHead_4_nlayer_3_Batch_100.pth' #'trained_model_1_saved_GAT.pth' # 'trained_model_1.pth'
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
train_loss_vec_100=[]
test_loss_vec_100=[]
losses = []
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", File_Name)
    mlflow.pytorch.autolog()
    mlflow.log_param("embedding_size", embedding_size)
    mlflow.log_param("num_features", num_features)
    for epoch in range(10000):
        loss, h = train()
        losses.append(loss)
        if epoch % 1 == 0:
            model_path = f'{File_Name}.pth' #'trained_model_1.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch} | Train Loss {loss}")
            mlflow.log_metric("train_loss", loss.item())
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
            print(f"Epoch {epoch} | Test Loss avg {avg_totall_loss_test}")
            mlflow.log_metric("test_loss", avg_totall_loss_test)

    mlflow.pytorch.log_model(model, "models")

model_path = f'{File_Name}.pth'# 'trained_model_1.pth'
torch.save(model.state_dict(), model_path)

# File path where you want to save the pickled data
file_path =f'{File_Name}.pkl'# 'data.pkl

# Serialize and save the object to a file
data_during_trainig={}
data_during_trainig['losses']=losses
data_during_trainig['train_loss_vec_100']=train_loss_vec_100
data_during_trainig['test_loss_vec_100']=test_loss_vec_100
with open(file_path, 'wb') as file:
    pkl.dump(data_during_trainig, file)"""


#-----------------------------------------Test Learned Model---------------------------------------
# Analyze the results for one batch

## Test data##
#test_batch = next(iter(loader_test))
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
all_errors_test=torch.empty(0)  
with torch.no_grad():
    for batch_test in loader_test:
        batch_test.to(device)
        pred, embed = model(batch_test.x.float(), batch_test.edge_index, batch_test.batch)
        ax.plot(batch_test.y,pred.detach().numpy().flatten(), 'o',
            linewidth=2, markersize=3, markeredgewidth=2,color='r') #markerfacecolor='none'
        error_test_all=batch_test.y-pred.detach().numpy().flatten()
        # Append the values to the empty tensor along dimension 0
        all_errors_test = torch.cat((all_errors_test, error_test_all), dim=0)
    ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
    ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.plot([10, 150], [10, 150],'--', color='gray', linewidth=2, alpha=1.0)
mean=np.mean(all_errors_test.numpy())
std_dev = np.std(all_errors_test.numpy())
# Display mean and standard deviation on the plot using ax.text()
ax.text(0.1, 0.9, r'$\mu$ = {:.2f}'.format(mean), transform=ax.transAxes, fontsize=15)
ax.text(0.1, 0.85, r'$\sigma$ = {:.2f}'.format(std_dev), transform=ax.transAxes, fontsize=15)
fig.savefig('Vehicle_test_data_learned_model.png',bbox_inches='tight',dpi=300)
fig.savefig('Vehicle_test_data_learned_model.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("Vehicle_test_data_learned_model.png")

## Train data##
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
all_errors_train=torch.empty(0)  
with torch.no_grad():
    for batch_test in loader:
        batch_test.to(device)
        pred, embed = model(batch_test.x.float(), batch_test.edge_index, batch_test.batch)
        ax.plot(batch_test.y,pred.detach().numpy().flatten(), 'o',
            linewidth=2, markersize=3, markeredgewidth=2,color='r') #markerfacecolor='none'
        error_test_all=batch_test.y-pred.detach().numpy().flatten()
        # Append the values to the empty tensor along dimension 0
        all_errors_train = torch.cat((all_errors_train, error_test_all), dim=0)
    ax.set_ylabel('$\hat{t} \,\, \mathrm{[s]}$', fontsize=20)
    ax.set_xlabel('$t\,\, \mathrm{[s]}$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.plot([8, 155], [8, 155],'--', color='gray', linewidth=2, alpha=1.0)
mean=np.mean(all_errors_train.numpy())
std_dev = np.std(all_errors_train.numpy())
# Display mean and standard deviation on the plot using ax.text()
ax.text(0.1, 0.9, r'$\mu$ = {:.2f}'.format(mean), transform=ax.transAxes, fontsize=15)
ax.text(0.1, 0.85, r'$\sigma$ = {:.2f}'.format(std_dev), transform=ax.transAxes, fontsize=15)
fig.savefig('Vehicle_train_data_learned_model.png',bbox_inches='tight',dpi=300)
fig.savefig('Vehicle_train_data_learned_model.pdf',bbox_inches='tight',dpi=300)
mlflow.log_artifact("Vehicle_train_data_learned_model.png")