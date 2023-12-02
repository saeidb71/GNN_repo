from create_dataset import IterationDataset
from GCN_Model import GCN_Model
from GAT_Model import GAT_Model
import torch
from torch_geometric.loader import DataLoader
import torch
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import from_networkx
import os
from tqdm import tqdm
import random
import argparse
from logging import getLogger, StreamHandler, INFO
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
logger = getLogger(__name__)
logger.addHandler(StreamHandler())
logger.setLevel(INFO)
pytorch_version = f"torch-{torch.__version__}.html"

plot_i=0

def train():
    global plot_i 
    model.train()

    total_loss = 0
    for data in training_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        loss = criterion(out, data.y)  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad() 
        total_loss+=loss.detach().item()*len(data)

    #if (plot_i % 10)==0:
    #    plt.plot(data.y,out.detach(),'o')
    #    plt.show()
    #plot_i+=1
    

    return total_loss / len(training_loader.dataset) 

def test(loader,regres_or_classif,known_median):
    model.eval()

    total_loss=0
    with torch.no_grad():
        correct = 0
        for data in loader: 
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss_test = criterion(out, data.y)  
            total_loss+=loss_test.detach().item()*len(data)
            if regres_or_classif==0: #classification
                pred = out.argmax(dim=1) 
                correct += int((pred == data.y).sum()) 
            elif regres_or_classif==1: #regression
                class_label=data.y>known_median 
                predict_label=out.flatten()>known_median
                correct += int((predict_label == class_label).sum())  

    return total_loss/len(loader.dataset) , correct / len(loader.dataset) 

#--------------------------------------------------Run in Terminal or Console--------------------------------------------
# Create an argument parser
parser = argparse.ArgumentParser(description='Vehicle Test Script')
# Add arguments for "x," "y," and "z"
parser.add_argument('--Model_type', type=int, required=True, help='0: GAT, 1: GCN')
parser.add_argument('--regres_or_classif', type=int, required=True, help='Reg=1 or classification=0')
parser.add_argument('--embedding_size', type=int, required=True, help='Value of embedding_size')
parser.add_argument('--numHeads', type=int, required=True, help='Value of numHeads')
parser.add_argument('--num_layers', type=int, required=True, help='Value of num_layers')
parser.add_argument('--NUM_GRAPHS_PER_BATCH', type=int, required=True, help='Value of NUM_GRAPHS_PER_BATCH')
parser.add_argument('--p_known', type=float, required=True, help='Value of p_known')
parser.add_argument('--training_split', type=float, required=True, help='Value of training_split')
parser.add_argument('--epochs', type=int, required=True, help='epochs')
parser.add_argument('--n', type=int, required=True, help='number of iterations')

# Parse the command-line arguments
try:
    args = parser.parse_args()
    Model_type=args.Model_type #1
    regres_or_classif=args.regres_or_classif #0
    embedding_size=args.embedding_size #64
    numHeads=args.numHeads #4
    num_layers=args.num_layers #3
    NUM_GRAPHS_PER_BATCH=args.NUM_GRAPHS_PER_BATCH #32
    p_known=args.p_known #0.2
    training_split=args.training_split #0.8
    epochs=args.epochs #600
    n=args.n # 5
except:
    Model_type=0
    regres_or_classif=1
    embedding_size=64
    numHeads=1
    num_layers=3
    NUM_GRAPHS_PER_BATCH=4 #4
    p_known=0.2
    training_split=0.8 
    epochs=60000#600 
    n=1

File_Name=f"Mdltype_{Model_type}_reg/class_{regres_or_classif}_embd_{embedding_size}_nH_{numHeads}_nL_{num_layers}_btch_{NUM_GRAPHS_PER_BATCH}_pknown_{p_known}_trinsplt_{training_split}_nepcs_{epochs}_nIter_{n}"
#python run.py --Model_type 0 --regres_or_classif 1 --embedding_size 64 --numHeads 1 --num_layers 3 --NUM_GRAPHS_PER_BATCH 4 --p_known 0.2 --training_split 0.8 --epochs 60000 --n 1

print('-----------------------------------Config Start-------------------------------------------')
print(f"Model_type: {Model_type}")
print(f"regres_or_classif: {regres_or_classif}")
print(f"embedding_size: {embedding_size}")
print(f"numHeads: {numHeads}")
print(f"num_layers: {num_layers}")
print(f"NUM_GRAPHS_PER_BATCH: {NUM_GRAPHS_PER_BATCH}")
print(f"p_known: {p_known}")
print(f"training_split: {training_split}")
print(f"epochs: {epochs}")
print(f"n: {n}")
print(f"File_Name : {File_Name}")
print('-----------------------------------Config End-------------------------------------------')

# Define Intermediate variables
num_features= 8#3 # number of node features
if regres_or_classif==0:
    num_output=2 #classification
    criterion = torch.nn.CrossEntropyLoss()
elif regres_or_classif==1:
    num_output=1 #regression
    #criterion = torch.nn.MSELoss() 
    criterion=torch.nn.L1Loss()


raw_data = loadmat('data/analog_circuits/circuit_data.mat', squeeze_me=True) #raw_data['Graphs'][0]['A'] #raw_data['Graphs'][0]['Labels']
data = pd.DataFrame(raw_data['Graphs'])
#column_names = data.columns   #data['A'][0]   #data['Labels'][0]     #data.loc[5]  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

median_performance = []
csv_save_path = 'csv_save_path' #Enter your desired save path for the csv results
data_save_path = 'data_save_path' #Enter the desired save path to store the data

for run in range(0,n):
    seed = 42 #random.randint(10000,99999)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = time.time()
    for iteration in range(0,n):
        if Model_type==1:
            model = GCN_Model(num_layers, num_features, embedding_size, num_output).to(device)
        elif Model_type==0:
            model=GAT_Model(num_layers, num_features, embedding_size, num_output,numHeads).to(device)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #0.001
        #optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
        #optimizer=torch.optim.Adagrad(model.parameters(), lr=0.0001)
        #optimizer=torch.optim.RMSprop(model.parameters(), lr=0.0001)
        #optimizer=torch.optim.Adadelta(model.parameters(), lr=0.0001)

        if iteration == 0:

            all_perm = np.random.permutation(len(data))
            All_index_split = int(len(data)*p_known)

            known_indices = all_perm[:All_index_split]
            unknown_indices = all_perm[All_index_split+1:]
        else:
            n_known_min = 2000
            """n_known_ones = len(known_ones_index)
            known_set_sizes = []
            if n_known_ones < n_known_min :
                n_known_needed = n_known_min - n_known_ones
                predicted_ones_index_needed = predicted_ones_index[:n_known_needed]

                known_indices = np.concatenate((known_ones_index, predicted_ones_index_needed))
                known_set_sizes.append(len(known_indices))
                unknown_indices = predicted_ones_index[n_known_needed:]            
                print('Known Set Size: ', known_set_sizes[-1])
            else:
                known_indices = known_ones_index
                unknown_indices = predicted_ones_index"""
            print('Worked!')

        

        known_graphs = data.loc[known_indices]
        unknown_graphs = data.loc[unknown_indices]

        known_performance = data['Labels'][known_indices]
        # Convert the Pandas Series to a PyTorch tensor with a suitable dtype
        known_performance_tensor = torch.tensor(known_performance.astype(float).values, dtype=torch.float32)
        # Compute the exponential of known_performance
        known_performance = torch.exp(-known_performance_tensor)

        known_median = np.median(known_performance)

        try:
            os.remove(f'{data_save_path}/known_data/processed/data.pt')
        except OSError as e:

            print('Error')
        
        
        try:
            os.remove(f'{data_save_path}/unknown_data/processed/data.pt')
        except OSError as e:
            print('Error')
        
        if regres_or_classif==0: #classification
            known_torch = IterationDataset(root='known_data_classif', data=known_graphs, performance_threshold=known_median, regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None) #known_torch[0]
            unknown_torch = IterationDataset(root='unknown_data_classif', data=unknown_graphs, performance_threshold=known_median,  regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None)  #unknown_torch[0]
        elif regres_or_classif==1: #refression
            known_torch = IterationDataset(root='known_data_reg', data=known_graphs, performance_threshold=known_median, regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None) #known_torch[0]
            unknown_torch = IterationDataset(root='unknown_data_reg', data=unknown_graphs, performance_threshold=known_median,  regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None) 



        training = known_torch[:int(len(known_torch)*training_split)]
        validation = known_torch[int(len(known_torch)*training_split)+1:]

        training_loader = DataLoader(training, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
        validation_loader = DataLoader(validation, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
        unknown_loader = DataLoader(unknown_torch, batch_size=1, shuffle=False)


        #mlflow.create_experiment(File_Name)
        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", File_Name)
            mlflow.log_param("Model_type", Model_type)
            mlflow.log_param("regres_or_classif", regres_or_classif)
            mlflow.log_param("embedding_size", embedding_size)
            mlflow.log_param("numHeads", numHeads)
            mlflow.log_param("num_layers", num_layers)
            mlflow.log_param("p_known", p_known)
            mlflow.log_param("training_split", training_split)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("n", n)

            for epoch in tqdm(range(1, epochs + 1), total=epochs):
                train_loss=train()
                if (epoch-1) % 100 == 0:  
                    train_loss_noDropOut,train_acc = test(training_loader,regres_or_classif,known_median)       
                    val_loss,val_acc = test(validation_loader,regres_or_classif,known_median) 
                    test_loss,test_acc = test(unknown_loader,regres_or_classif,known_median)    

                    print('-----------------------------------Training Start-------------------------------------------')
                    print(f"train_loss : {train_loss_noDropOut} , train_acc : {train_acc}")
                    print(f"val_loss : {val_loss} , val_acc : {val_acc}")
                    print(f"test_loss : {test_loss} , test_acc : {test_acc}")
                    print('-----------------------------------Training End-------------------------------------------')

                    mlflow.log_metric("train_loss", train_loss_noDropOut)
                    mlflow.log_metric("train_acc", train_acc)
                    mlflow.log_metric("val_loss", val_loss)
                    mlflow.log_metric("val_acc", val_acc)
                    mlflow.log_metric("test_loss", test_loss)
                    mlflow.log_metric("test_acc", test_acc)
                    #print(f"train_loss : {train_loss}")
                    #mlflow.log_metric("train_loss", train_loss)
                
            model_path = f'{File_Name}.pth' #'trained_model_1.pth'
            torch.save(model.state_dict(), model_path)
            #mlflow.pytorch.log_model(model, "models")
            mlflow.pytorch.autolog()
        



        

        """predictions = []
        unknown_index = []
        
        for test_graph in tqdm(unknown_loader, total=len(unknown_loader)):
            test_graph = test_graph.to(device)
            out = model(test_graph.x, test_graph.edge_index, test_graph.batch)
            pred = out.argmax(dim=1)
            predictions.append(pred.item())
            unknown_index.append(test_graph.orig_index.item())
            
        
        
        predictions = np.array(predictions)
        predicted_ones_index = unknown_indices[np.where(predictions == 1)[0]]
        predicted_zeros_index = unknown_indices[np.where(predictions == 0)[0]]

        

        known_classifications = []
        for i in range(len(known_torch)):
            known_classifications.append(known_torch[i].y.item())

        known_classifications = np.array(known_classifications)
        known_ones_index = known_indices[np.where(known_classifications == 1)[0]]
        known_zeros_index = known_indices[np.where(known_classifications == 0)[0]]
        
        print(f'Run {run}, Iteration {iteration}')
        print('Known Ones: ', len(known_ones_index))
        print('Known Zeros: ', len(known_zeros_index))
        
        saved_known_ones = pd.DataFrame(known_ones_index)
        saved_known_ones.to_csv(f'{csv_save_path}/run_{run}_iteration{iteration}_known_ones.csv')

        saved_known_zeros = pd.DataFrame(known_zeros_index)
        saved_known_zeros.to_csv(f'{csv_save_path}/run_{run}_iteration{iteration}_known_zeros.csv')

        saved_predicted_ones = pd.DataFrame(predicted_ones_index)
        saved_predicted_ones.to_csv(f'{csv_save_path}/run_{run}_iteration{iteration}_predicted_ones.csv')

        saved_known_zeros = pd.DataFrame(predicted_zeros_index)
        saved_known_zeros.to_csv(f'{csv_save_path}/run_{run}_iteration{iteration}_predicted_zeros.csv')


        median_performance.append(known_median)"""

    end_time = time.time()
