from create_dataset import IterationDataset
from GCN_Model import GCN_Model
from GAT_Model import GAT_Model
import torch
from torch_geometric.loader import DataLoader
import torch
import time
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pandas as pd
import torch
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import from_networkx
import os
from tqdm import tqdm
import random
from scipy.stats import kendalltau
import argparse
from logging import getLogger, StreamHandler, INFO
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
logger = getLogger(__name__)
logger.addHandler(StreamHandler())
logger.setLevel(INFO)
pytorch_version = f"torch-{torch.__version__}.html"
import torch.nn as nn

def custom_loss(outputs, targets, p):
    # Calculate mean squared error

    c=torch.nn.L1Loss()
    L1_loss=c(outputs, targets)
    
    #penalty = torch.where(targets >= p, torch.abs(torch.minimum(outputs - p, torch.zeros_like(outputs))), 
                                    #torch.abs(torch.maximum(outputs - p, torch.zeros_like(outputs))))
    # Combine the MSE loss and the penalty term
    total_loss = L1_loss #+ 2*penalty.mean()

    return total_loss

def train(regres_or_classif):
    model.train()
    total_loss = 0
    for data in training_loader:
        data = data.to(device)
        if regres_or_classif==1:
            out = model(data.x, data.edge_index, data.batch).flatten()  
        else: 
            out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)  
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad() 
        total_loss+=loss.detach().item()*len(data)
    return total_loss / len(training_loader.dataset) 

def test(loader,regres_or_classif,known_median):
    model.eval()
    total_loss=0
    with torch.no_grad():
        correct = 0
        for data in loader: 
            data = data.to(device)
            if regres_or_classif==1:
                out = model(data.x, data.edge_index, data.batch).flatten()
            else:
                out = model(data.x, data.edge_index, data.batch)
            loss_test = criterion(out, data.y)  
            total_loss+=loss_test.detach().item()*len(data)
            if regres_or_classif==0: #classification
                pred = out.argmax(dim=1) 
                correct += int((pred == data.y).sum()) 
            elif regres_or_classif==1: #regression
                class_label=data.y>known_median 
                predict_label=out>known_median
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
    embedding_size=64#64#64
    numHeads=4#4
    num_layers=3
    NUM_GRAPHS_PER_BATCH=4 #4
    p_known=0.003
    training_split=0.8
    epochs=1000#600 
    n=15#5

File_Name=f"Saved_Files/Mdltype_{Model_type}_regclass_{regres_or_classif}_embd_{embedding_size}_nH_{numHeads}_nL_{num_layers}_btch_{NUM_GRAPHS_PER_BATCH}_pknown_{p_known}_trinsplt_{training_split}_nepcs_{epochs}_nIter_{n}"
#python source_code/run.py --Model_type 0 --regres_or_classif 1 --embedding_size 64 --numHeads 4 --num_layers 3 --NUM_GRAPHS_PER_BATCH 4 --p_known 0.2 --training_split 0.8 --epochs 1000 --n 1

Train_or_Check=1; #Train: 1 , Test : 0
# Set up early stopping parameters
early_stopping_counter = 0
best_val_accuracy = 0.0
patience = 12#12#7  # Number of consecutive iterations without improvement to tolerate
break_outer = False
# Define Intermediate variables
num_features= 3#8 # number of node features
if regres_or_classif==0:
    num_output=2 #classification
elif regres_or_classif==1:
    num_output=1 #regression

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

raw_data = loadmat('data/analog_circuits/circuit_data.mat', squeeze_me=True) #raw_data['Graphs'][0]['A'] #raw_data['Graphs'][0]['Labels']
data_all = pd.DataFrame(raw_data['Graphs'])
#column_names = data.columns   #data['A'][0]   #data['Labels'][0]     #data.loc[5]  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sortel_all_index=data_all['Labels'].argsort()
sorted_all_labels=data_all['Labels'][sortel_all_index]

median_performance = []
csv_save_path = 'csv_save_path' #Enter your desired save path for the csv results
data_save_path = 'data_save_path' #Enter the desired save path to store the data

text_file_path = f'{File_Name}.txt'

for run in range(0,n):
    seed = 42 #random.randint(10000,99999)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_time = time.time()
    for iteration in range(0,n): #
        early_stopping_counter = 0
        best_val_accuracy = 0.0
        break_outer=False
        if Model_type==1:
            model = GCN_Model(num_layers, num_features, embedding_size, num_output).to(device)
        elif Model_type==0:
            model=GAT_Model(num_layers, num_features, embedding_size, num_output,numHeads).to(device)
        if Train_or_Check==0:
            model_path = f'{File_Name}.pth' 
            model.load_state_dict(torch.load(model_path))
        if iteration>=1:
            model_path = f'{File_Name}_best_iter{iteration-1}.pth' 
            model.load_state_dict(torch.load(model_path))
        print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #0.001
        #optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
        #optimizer=torch.optim.Adagrad(model.parameters(), lr=0.0001)
        #optimizer=torch.optim.RMSprop(model.parameters(), lr=0.0001)
        #optimizer=torch.optim.Adadelta(model.parameters(), lr=0.0001)

        if iteration == 0:

            all_perm = np.random.permutation(len(data_all))
            All_index_split = int(len(data_all)*p_known)

            """known_indices = all_perm[:All_index_split]
            unknown_indices = all_perm[All_index_split+1:]"""

            known_indices = sortel_all_index[:int(len(data_all)*p_known)]
            unknown_indices = sortel_all_index[int(len(data_all)*p_known):]
        else:
            n_known_min =  100#2500
            n_known_ones = len(known_ones_index)
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
                unknown_indices = predicted_ones_index
            print('Worked!')
        
        known_graphs = data_all.loc[known_indices]
        unknown_graphs = data_all.loc[unknown_indices]

        known_performance = data_all['Labels'][known_indices]
        # Convert the Pandas Series to a PyTorch tensor with a suitable dtype
        known_performance_tensor = torch.tensor(known_performance.astype(float).values, dtype=torch.float32)
        # Compute the exponential of known_performance
        known_performance = torch.exp(-known_performance_tensor)

        known_median = np.median(known_performance)

        if regres_or_classif==0:
            num_output=2 #classification
            criterion = torch.nn.CrossEntropyLoss()
        elif regres_or_classif==1:
            num_output=1 #regression
            #criterion = torch.nn.MSELoss() 
            #criterion=torch.nn.L1Loss()
            criterion = lambda outputs,targets: custom_loss(outputs, targets, known_median)

        try:
            os.remove(f'known_data_reg_p{p_known}/processed/data.pt')
        except OSError as e:

            print('Error')
        
        
        try:
            os.remove(f'unknown_data_reg_p{p_known}/processed/data.pt')
        except OSError as e:
            print('Error')
        
        if regres_or_classif==0: #classification
            known_torch = IterationDataset(root=f'known_data_classif_p{p_known}', data=known_graphs, performance_threshold=known_median, regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None) #known_torch[0]
            unknown_torch = IterationDataset(root=f'unknown_data_classif_p{p_known}', data=unknown_graphs, performance_threshold=known_median,  regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None)  #unknown_torch[0]
        elif regres_or_classif==1: #refression
            known_torch = IterationDataset(root=f'known_data_reg_p{p_known}', data=known_graphs, performance_threshold=known_median, regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None) #known_torch[0]
            unknown_torch = IterationDataset(root=f'unknown_data_reg_p{p_known}', data=unknown_graphs, performance_threshold=known_median,  regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None) 

        training = known_torch[:int(len(known_torch)*training_split)]
        validation = known_torch[int(len(known_torch)*training_split)+1:]

        training_loader = DataLoader(training, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
        validation_loader = DataLoader(validation, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
        unknown_loader = DataLoader(unknown_torch, batch_size=1, shuffle=False)
        unknown_loader2 = DataLoader(unknown_torch, batch_size=100000, shuffle=False)


        #mlflow.create_experiment(File_Name)
        if Train_or_Check==1:
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
                    train_loss=train(regres_or_classif)
                    if (epoch-1) % 10 == 0:  
                        train_loss_noDropOut,train_acc = test(training_loader,regres_or_classif,known_median)       
                        val_loss,val_acc = test(validation_loader,regres_or_classif,known_median) 
                        #test_loss,test_acc = test(unknown_loader2,regres_or_classif,known_median)    

                        print('-----------------------------------Training Start-------------------------------------------')
                        print(f"train_loss : {train_loss_noDropOut} , train_acc : {train_acc}")
                        print(f"val_loss : {val_loss} , val_acc : {val_acc}")
                        #print(f"test_loss : {test_loss} , test_acc : {test_acc}")
                        print('-----------------------------------Training End-------------------------------------------')

                        mlflow.log_metric("train_loss", train_loss_noDropOut)
                        mlflow.log_metric("train_acc", train_acc)
                        mlflow.log_metric("val_loss", val_loss)
                        mlflow.log_metric("val_acc", val_acc)
                       # mlflow.log_metric("test_loss", test_loss)
                       # mlflow.log_metric("test_acc", test_acc)
                        #print(f"train_loss : {train_loss}")
                        #mlflow.log_metric("train_loss", train_loss)

                        if epoch<90: #90
                            early_stopping_counter=0
                        elif val_acc > best_val_accuracy:
                            best_val_accuracy = val_acc
                            early_stopping_counter = 0
                            # Save your model if needed
                            torch.save(model.state_dict(),  f'{File_Name}_best_iter{iteration}.pth' )
                        else:
                            early_stopping_counter += 1

                        if early_stopping_counter >= patience:
                            break_outer = True
                            break
                    if break_outer:
                        break

                
                model_path = f'{File_Name}_iter{iteration}.pth' #'trained_model_1.pth'
                torch.save(model.state_dict(), model_path)
                #mlflow.pytorch.log_model(model, "models")
                mlflow.pytorch.autolog()

        model.load_state_dict(torch.load(f'{File_Name}_best_iter{iteration}.pth'))
        
        model.eval()

        out_known_train=np.zeros(len(training_loader))
        Label_known_train=np.zeros(len(training_loader))
        Loss_known_train=np.zeros(len(training_loader))
        out_Class_known_train=np.zeros(len(training_loader))
        Label_Class_known_train=np.zeros(len(training_loader))
        Correct_known_train=np.zeros(len(training_loader))

        out_known_validation=np.zeros(len(validation_loader))
        Label_known_validation=np.zeros(len(validation_loader))
        Loss_known_validation=np.zeros(len(validation_loader))
        out_Class_known_validation=np.zeros(len(validation_loader))
        Label_Class_known_validation=np.zeros(len(validation_loader))
        Correct_known_validation=np.zeros(len(validation_loader))

        out_unknown=np.zeros(len(unknown_loader))
        Label_unknown=np.zeros(len(unknown_loader))
        Loss_unknown=np.zeros(len(unknown_loader))
        out_Class_unknown=np.zeros(len(unknown_loader))
        Label_Class_unknown=np.zeros(len(unknown_loader))
        Correct_unknown=np.zeros(len(unknown_loader))

        with torch.no_grad():
            for i in np.arange(len(training_loader)):
                data=training[i]
                data = data.to(device)
                if regres_or_classif==1:
                    out = model(data.x, data.edge_index, data.batch).flatten()
                else:
                    out = model(data.x, data.edge_index, data.batch)
                loss= criterion(out, data.y)  
                if regres_or_classif==0: #classification
                    pred = out.argmax(dim=1) 
                    correct = int((pred == data.y).sum()) 
                elif regres_or_classif==1: #regression
                    class_label=data.y>known_median 
                    predict_label=out>known_median
                    correct = (class_label==predict_label)
                if regres_or_classif==1:
                    out_known_train[i]=out
                    Label_known_train[i]=data.y
                    Loss_known_train[i]=loss
                    out_Class_known_train[i]=predict_label
                    Label_Class_known_train[i]=class_label
                    Correct_known_train[i]=correct
                else:
                    out_Class_known_train[i]=pred
                    Label_Class_known_train[i]=data.y
                    Loss_known_train[i]=loss
                    Correct_known_train[i]=correct
        known_train_matrix=np.zeros((2,2))
        if regres_or_classif==1:
            known_train_matrix[0][0]=len(np.intersect1d(np.where(out_known_train > known_median)[0], np.where(Label_known_train < known_median)[0]))
            known_train_matrix[0][1]=len(np.intersect1d(np.where(out_known_train >= known_median)[0], np.where(Label_known_train >= known_median)[0]))
            known_train_matrix[1][0]=len(np.intersect1d(np.where(out_known_train < known_median)[0], np.where(Label_known_train < known_median)[0]))
            known_train_matrix[1][1]=len(np.intersect1d(np.where(out_known_train <= known_median)[0], np.where(Label_known_train >= known_median)[0]))
        else:
            known_train_matrix[0, 0] = np.sum((Label_Class_known_train == 0) & (out_Class_known_train == 0))
            known_train_matrix[0, 1] = np.sum((Label_Class_known_train == 0) & (out_Class_known_train == 1))
            known_train_matrix[1, 0] = np.sum((Label_Class_known_train == 1) & (out_Class_known_train == 0))
            known_train_matrix[1, 1] = np.sum((Label_Class_known_train == 1) & (out_Class_known_train == 1))

        with torch.no_grad():
            for i in np.arange(len(validation_loader)):
                data=validation[i]
                data = data.to(device)
                if regres_or_classif==1:
                    out = model(data.x, data.edge_index, data.batch).flatten()
                else:
                    out = model(data.x, data.edge_index, data.batch)
                loss= criterion(out, data.y)  
                if regres_or_classif==0: #classification
                    pred = out.argmax(dim=1) 
                    correct = int((pred == data.y).sum()) 
                elif regres_or_classif==1: #regression
                    class_label=data.y>known_median 
                    predict_label=out>known_median
                    correct = (class_label==predict_label)
                if regres_or_classif==1:
                    out_known_validation[i]=out
                    Label_known_validation[i]=data.y
                    Loss_known_validation[i]=loss
                    out_Class_known_validation[i]=predict_label
                    Label_Class_known_validation[i]=class_label
                    Correct_known_validation[i]=correct
                else:
                    out_Class_known_validation[i]=pred
                    Label_Class_known_validation[i]=data.y
                    Loss_known_validation[i]=loss
                    Correct_known_validation[i]=correct
        known_validation_matrix=np.zeros((2,2))
        if regres_or_classif==1:
            known_validation_matrix[0][0]=len(np.intersect1d(np.where(out_known_validation > known_median)[0], np.where(Label_known_validation < known_median)[0]))
            known_validation_matrix[0][1]=len(np.intersect1d(np.where(out_known_validation >= known_median)[0], np.where(Label_known_validation >= known_median)[0]))
            known_validation_matrix[1][0]=len(np.intersect1d(np.where(out_known_validation < known_median)[0], np.where(Label_known_validation < known_median)[0]))
            known_validation_matrix[1][1]=len(np.intersect1d(np.where(out_known_validation <= known_median)[0], np.where(Label_known_validation >= known_median)[0]))
        else:
            known_validation_matrix[0, 0] = np.sum((Label_Class_known_validation == 0) & (out_Class_known_validation == 0))
            known_validation_matrix[0, 1] = np.sum((Label_Class_known_validation == 0) & (out_Class_known_validation == 1))
            known_validation_matrix[1, 0] = np.sum((Label_Class_known_validation == 1) & (out_Class_known_validation == 0))
            known_validation_matrix[1, 1] = np.sum((Label_Class_known_validation == 1) & (out_Class_known_validation == 1))

        with torch.no_grad():
            for i in np.arange(len(unknown_loader)):
                data=unknown_torch[i]
                data = data.to(device)
                if regres_or_classif==1:
                    out = model(data.x, data.edge_index, data.batch).flatten()
                else:
                    out = model(data.x, data.edge_index, data.batch)
                loss= criterion(out, data.y)  
                if regres_or_classif==0: #classification
                    pred = out.argmax(dim=1) 
                    correct = int((pred == data.y).sum()) 
                elif regres_or_classif==1: #regression
                    class_label=data.y>known_median 
                    predict_label=out>known_median
                    correct = (class_label==predict_label)
                if regres_or_classif==1:
                    out_unknown[i]=out
                    Label_unknown[i]=data.y
                    Loss_unknown[i]=loss
                    out_Class_unknown[i]=predict_label
                    Label_Class_unknown[i]=class_label
                    Correct_unknown[i]=correct
                else:
                    out_Class_unknown[i]=pred
                    Label_Class_unknown[i]=data.y
                    Loss_unknown[i]=loss
                    Correct_unknown[i]=correct
        unknown_matrix=np.zeros((2,2))
        if regres_or_classif==1:
            unknown_matrix[0][0]=len(np.intersect1d(np.where(out_unknown> known_median)[0], np.where(Label_unknown < known_median)[0]))
            unknown_matrix[0][1]=len(np.intersect1d(np.where(out_unknown >= known_median)[0], np.where(Label_unknown >= known_median)[0]))
            unknown_matrix[1][0]=len(np.intersect1d(np.where(out_unknown < known_median)[0], np.where(Label_unknown < known_median)[0]))
            unknown_matrix[1][1]=len(np.intersect1d(np.where(out_unknown <= known_median)[0], np.where(Label_unknown>= known_median)[0]))
        else:
            unknown_matrix[0, 0] = np.sum((Label_Class_unknown == 0) & (out_Class_unknown == 0))
            unknown_matrix[0, 1] = np.sum((Label_Class_unknown == 0) & (out_Class_unknown == 1))
            unknown_matrix[1, 0] = np.sum((Label_Class_unknown == 1) & (out_Class_unknown == 0))
            unknown_matrix[1, 1] = np.sum((Label_Class_unknown == 1) & (out_Class_unknown == 1))


        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if regres_or_classif==1:
            ax.plot(Label_known_train,out_known_train, 'o',linewidth=2, markersize=1.0, markeredgewidth=2,markerfacecolor='blue')#,color='r')   
            ax.axhline(y=known_median, color='gray', linestyle='--', label='Horizontal Line at y=2')
            ax.axvline(x=known_median, color='gray', linestyle='--', label='Vertical Line at y=2')
            #ax.plot([0, 1], [0, 1],linewidth=2, linestyle='--', color='red',alpha=0.6, label='Line from (0, 0) to (1, 1)')
        ax.set_ylabel('$\mathrm{Predict}$', fontsize=20)
        ax.set_xlabel('$\mathrm{True}$', fontsize=20)
        left_corner_ax = fig.add_axes([0.12, 0.65, 0.2, 0.2])  # Adjust the position and size as needed
        left_corner_ax.imshow(known_train_matrix, interpolation='nearest', cmap='gray', vmin=-np.max(known_train_matrix), vmax=np.max(known_train_matrix))
        for i in range(2):
            for j in range(2):
                left_corner_ax.text(j, i, f'{known_train_matrix[i, j]:.0f}', ha='center', va='center', color='black')
        left_corner_ax.set_xticks([])
        left_corner_ax.set_yticks([])
        left_corner_ax.set_xticklabels([])
        left_corner_ax.set_yticklabels([])
        ax.set_title(f'$ \mathrm{{Known-Train}}  \, \, , \, \, \mathrm{{Loss}} \, \, : \, \, {np.round(Loss_known_train.mean(),2)} \, \, , \,\, \mathrm{{Acc}} \, \, : \, \, {np.round(len(np.where(Correct_known_train==1)[0])/len(training_loader),2)}$')
        fig.savefig('Known_Train.png',bbox_inches='tight',dpi=300)
        fig.savefig('Known_Train.pdf',bbox_inches='tight',dpi=300)


        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if regres_or_classif==1:
            ax.plot(Label_known_validation,out_known_validation, 'o',linewidth=2, markersize=1.0, markeredgewidth=2)#,color='r')   
            ax.axhline(y=known_median, color='gray', linestyle='--', label='Horizontal Line at y=2')
            ax.axvline(x=known_median, color='gray', linestyle='--', label='Vertical Line at y=2')
            #ax.plot([0, 1], [0, 1],linewidth=2, linestyle='--', color='red',alpha=0.6, label='Line from (0, 0) to (1, 1)')
        ax.set_ylabel('$\mathrm{Predict}$', fontsize=20)
        ax.set_xlabel('$\mathrm{True}$', fontsize=20)
        left_corner_ax = fig.add_axes([0.12, 0.65, 0.2, 0.2])  # Adjust the position and size as needed
        left_corner_ax.imshow(known_validation_matrix, interpolation='nearest', cmap='gray', vmin=-np.max(known_validation_matrix), vmax=np.max(known_validation_matrix))
        for i in range(2):
            for j in range(2):
                left_corner_ax.text(j, i, f'{known_validation_matrix[i, j]:.0f}', ha='center', va='center', color='black')
        left_corner_ax.set_xticks([])
        left_corner_ax.set_yticks([])
        left_corner_ax.set_xticklabels([])
        left_corner_ax.set_yticklabels([])
        ax.set_title(f'$ \mathrm{{Known-Validation}}  \, \, , \, \, \mathrm{{Loss}} \, \, : \, \, {np.round(Loss_known_validation.mean(),2)} \, \, , \,\, \mathrm{{Acc}} \, \, : \, \, {np.round(len(np.where(Correct_known_validation==1)[0])/len(validation_loader),2)}$')
        fig.savefig('Known_Valid.png',bbox_inches='tight',dpi=300)
        fig.savefig('Known_Valid.pdf',bbox_inches='tight',dpi=300)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if regres_or_classif==1:
            ax.plot(Label_unknown,out_unknown, 'o',linewidth=2, markersize=1.0, markeredgewidth=2)#,color='r')   
            ax.axhline(y=known_median, color='gray', linestyle='--', label='Horizontal Line at y=2')
            ax.axvline(x=known_median, color='gray', linestyle='--', label='Vertical Line at y=2')
            #ax.plot([0, 1], [0, 1],linewidth=2, linestyle='--', color='red',alpha=0.6, label='Line from (0, 0) to (1, 1)')
        ax.set_ylabel('$\mathrm{Predict}$', fontsize=20)
        ax.set_xlabel('$\mathrm{True}$', fontsize=20)
        left_corner_ax = fig.add_axes([0.12, 0.67, 0.2, 0.2])  # Adjust the position and size as needed
        left_corner_ax.imshow(unknown_matrix, interpolation='nearest', cmap='gray', vmin=-np.max(unknown_matrix), vmax=np.max(unknown_matrix))
        for i in range(2):
            for j in range(2):
                left_corner_ax.text(j, i, f'{unknown_matrix[i, j]:.0f}', ha='center', va='center', color='black')
        left_corner_ax.set_xticks([])
        left_corner_ax.set_yticks([])
        left_corner_ax.set_xticklabels([])
        left_corner_ax.set_yticklabels([])
        ax.set_title(f'$ \mathrm{{Unknown}}  \, \, , \, \, \mathrm{{Loss}} \, \, : \, \, {np.round(Loss_unknown.mean(),2)} \, \, , \,\, \mathrm{{Acc}} \, \, : \, \, {np.round(len(np.where(Correct_unknown==1)[0])/len(unknown_loader),2)}$')
        fig.savefig('Unknown.png',bbox_inches='tight',dpi=300)
        fig.savefig('Unknown.pdf',bbox_inches='tight',dpi=300)

        TP=unknown_matrix[0,1]
        TN=unknown_matrix[1,0]
        FP=unknown_matrix[0,0]
        FN=unknown_matrix[1,1]
        N=TP+TN+FP+FN

        Accuracy= (TP+TN)/N
        Precision=TP/(TP+FP)
        Recall=TP/(TP+FN)
        F1=2*(Precision*Recall)/(Precision+Recall)
        MCC=(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))


        df = pd.DataFrame()
        df[f'{p_known}'] = np.array([Accuracy,Precision,Recall,F1,MCC])
        print(df)

        t_true_sorted_index = np.argsort(Label_unknown)
        t_true_sorted_val = Label_unknown[t_true_sorted_index]
        t_pred_sorted_val_sorted_from_ture_indx =out_unknown[t_true_sorted_index]

        t_true_sorted_val_percentileofscoreVec = np.vectorize(lambda x: stats.percentileofscore(
        t_true_sorted_val, x, kind='strict'))(t_true_sorted_val)
        t_pred_sorted_val_percentileofscoreVec = np.vectorize(lambda x: stats.percentileofscore(
        t_pred_sorted_val_sorted_from_ture_indx, x, kind='strict'))(t_pred_sorted_val_sorted_from_ture_indx)

        kendall_distance_S6, _ = kendalltau(t_true_sorted_val_percentileofscoreVec, t_pred_sorted_val_percentileofscoreVec)

        N_OL=np.sum(t_pred_sorted_val_sorted_from_ture_indx>t_pred_sorted_val_sorted_from_ture_indx[-1]) # need to run thisamount of OLOC simuyaltion to get the true best solution
        N_sub=np.sum(Label_unknown>Label_unknown[out_unknown.argmax()]) 
        f_sub=Label_unknown[out_unknown.argmax()]/Label_unknown.max()*100
        N_OL_percent=N_OL/len(Label_unknown)*100
        N_sub_percent=N_sub/len(Label_unknown)*100

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(t_pred_sorted_val_sorted_from_ture_indx,'o',markersize=0.5,label='$\mathrm{Corresponding \,\,outputs}$')
        ax.plot(t_true_sorted_val,'s',linewidth=3,markersize=1.0,label='$\mathrm{Sorted\,\, targets}$')
        ax.axhline(y=known_median, linewidth=3, color='gray', linestyle='-',label='$\mathrm{Median\,\, known\,\, targets}$')
        ax.axhline(y=np.median(out_unknown), linewidth=3, color='k', linestyle=':',label='$\mathrm{Median\,\, unknown\,\, outputs}$')
        ax.legend(loc='best', fontsize=15)
        ax.set_title(rf'$m_{{\mathrm{{known}}}} \,\, : \, \, {known_median:.3f} \,\, , \,\, m_{{\mathrm{{unknown}}}} \,\, : \,\, {np.median(out_unknown):.3f} $')
        ax.set_xlabel('$ \mathrm{sorted\,\,targets\,\,locations}$',fontsize=15)
        ax.set_ylabel('$\mathrm{Graph\,\,Value}$',fontsize=15)
        fig.savefig('Starget_sorted_plot.png',bbox_inches='tight',dpi=300)
        fig.savefig('Starget_sorted_plot.pdf',bbox_inches='tight',dpi=300)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(t_true_sorted_val_percentileofscoreVec,t_pred_sorted_val_percentileofscoreVec,'o',markersize=0.5)#,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{Predicted\,\, Best}$',color='r')
        #ax.plot(t_true_sorted_val,t_pred_sorted_val_sorted_from_ture_indx,'o',markersize=1.0)
        ax.plot([0,100], [0,100],color='r',alpha=0.4, linewidth=6)
        #ax.plot([0,1], [0,1],'--',color='r',alpha=0.4, linewidth=3)
        #ax.plot([0,100], [0,100],'--',color='r',alpha=0.4, linewidth=3)
        tick_locations = [0, 20, 40, 60, 80, 100]
        tick_labels = [f'{tick}%' for tick in tick_locations]
        # Update x and y ticks
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels, fontsize=15)
        ax.set_yticks(tick_locations)
        ax.set_yticklabels(tick_labels, fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.set_title(f'$\mathrm{{Case}} \, : \, S_6 \, \, , \, \,  N_{{\mathrm{{g}}}}\,: \,  {y_val_all.shape[0]} \, \, , \, \, K\, :\, {np.round(kendall_distance_S6,2)}  $',fontsize=15) 
        ax.set_xlabel('$ \mathrm{sorted\,\,observed\,\,performance\,\,locations}$',fontsize=15)
        ax.set_ylabel('$\mathrm{predicted\,\,sorted\,\,locations}$',fontsize=15)
        ax.set_title(f'$K \,\, : \, \, {np.round(kendall_distance_S6,2)}$',fontsize=15)
        ax.set_xlim(0.4, 100.4)
        ax.set_ylim(-0.4, 100.4)
        fig.savefig('K_metric.png',bbox_inches='tight',dpi=300)
        fig.savefig('SK_metric.pdf',bbox_inches='tight',dpi=300)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(out_unknown,'o',markersize=0.5)
        ax.plot(out_unknown.argmax(),out_unknown.max(),'s',markersize=15,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{Predicted\,\, Best\, \, :} \, \, \hat{J}(G_{\hat{i}^{*}}) $',color='r')
        ax.plot(Label_unknown.argmax(),out_unknown[Label_unknown.argmax()],'d',markersize=15,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{True\,\, Best\,\, :} \, \, \hat{J}(G_{i^{*}})$',color='g')
        ax.axhline(y=out_unknown.max(), color='r', linestyle='--')
        ax.axhline(y=out_unknown[Label_unknown.argmax()], color='g', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=15)
        #ax.set_title(f'$\mathrm{{Num\,\, Graphs\,}}:\, {y_val_all.shape[0]} \, \, , \, \, N_{{\mathrm{{OL}}}}\,: \, {N_OL_S6}  $',fontsize=15) 
        ax.set_title(f'$N_{{\mathrm{{OL}}}}\,:\, {N_OL} \, \, , \, \, N_{{\mathrm{{g}}}}\,: \,  {out_unknown.shape[0]} \, \, , \, \,  \% N_{{\mathrm{{OL}}}}/N_{{\mathrm{{g}}}}\, :\, {np.round(100*N_OL/out_unknown.shape[0],3)}  $',fontsize=15) 
        ax.set_xlabel('$ \mathrm{Case\,\,ID}$',fontsize=20)
        ax.set_ylabel('$\hat{J}\, (G_i)$',fontsize=20)
        # Create legend handles and labels for the 'o' and 's' points
        ax.legend(loc='lower left', fontsize=15)
        fig.savefig('N_OL.png',bbox_inches='tight',dpi=300)
        fig.savefig('N_OL.pdf',bbox_inches='tight',dpi=300)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(Label_unknown,'o',markersize=0.5)
        ax.plot(out_unknown.argmax(),Label_unknown[out_unknown.argmax()],'s',markersize=15,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{Predicted\,\, Best \, \, :} \, \, J(G_{\hat{i}^{*}}) $',color='r')
        ax.plot(Label_unknown.argmax(),Label_unknown.max(),'d',markersize=15,markerfacecolor='none', markeredgewidth=2,label='$\mathrm{True\,\, Best\,\, :} \, \, J(G_{i^{*}})$',color='g')
        ax.axhline(y=Label_unknown[out_unknown.argmax()], color='r', linestyle='--')
        ax.axhline(y=Label_unknown.max(), color='g', linestyle='--')
        ax.set_title(f'$N_{{\mathrm{{sub}}}}\,:\, {N_sub.astype(int)-1} \, \, , \, \, N_{{\mathrm{{g}}}}\,: \,  {Label_unknown.shape[0]} \, \, , \, \,  \% N_{{\mathrm{{sub}}}}/N_{{\mathrm{{g}}}}\, :\, {np.round(100*(N_sub.astype(int)-1)/Label_unknown.shape[0],2)}\,\, , \,\, J_{{\mathrm{{sub}}}} \, : \,{np.round(Label_unknown[out_unknown.argmax()]/Label_unknown.max(),4)}   $',fontsize=15) 
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel('$ \mathrm{Case\,\,ID}$',fontsize=20)
        ax.set_ylabel('$J\, (G_i) \,\,\mathrm{[s]}$',fontsize=20)
        # Create legend handles and labels for the 'o' and 's' points
        ax.legend(loc='best', fontsize=15)
        fig.savefig('N_sub.png',bbox_inches='tight',dpi=300)
        fig.savefig('N_sub.pdf',bbox_inches='tight',dpi=300)


        All_data=pd.DataFrame(raw_data['Graphs'])
        known_performance_all_data = All_data['Labels']
        known_performance_tensor_all_data = torch.tensor(known_performance_all_data.astype(float).values, dtype=torch.float32)
        known_performance_all_data = torch.exp(-known_performance_tensor_all_data)
        known_median_all_data = np.median(known_performance_all_data)
        All_data_torch = IterationDataset(root='All_Data', data=All_data, performance_threshold=known_median_all_data, regres_or_classif=regres_or_classif, transform=None, pre_transform=None, pre_filter=None) 
        
        out_all=np.zeros(len(All_data_torch))
        Label_all=np.zeros(len(All_data_torch))
        Loss_all=np.zeros(len(All_data_torch))
        out_Class_all=np.zeros(len(All_data_torch))
        Label_Class_all=np.zeros(len(All_data_torch))
        Correct_all=np.zeros(len(All_data_torch))

        with torch.no_grad():
            for i in np.arange(len(All_data_torch)):
                data=All_data_torch[i]
                data = data.to(device)
                if regres_or_classif==1:
                    out = model(data.x, data.edge_index, data.batch).flatten()
                else:
                    out = model(data.x, data.edge_index, data.batch)
                loss= criterion(out, data.y)  
                if regres_or_classif==0: #classification
                    pred = out.argmax(dim=1) 
                    correct = int((pred == data.y).sum()) 
                elif regres_or_classif==1: #regression
                    class_label=data.y>known_median_all_data
                    predict_label=out>known_median
                    correct = (class_label==predict_label)
                if regres_or_classif==1:
                    out_all[i]=out
                    Label_all[i]=data.y
                    Loss_all[i]=loss
                    out_Class_all[i]=predict_label
                    Label_Class_all[i]=class_label
                    Correct_all[i]=correct
                else:
                    out_Class_all[i]=pred
                    Label_Class_all[i]=data.y
                    Loss_all[i]=loss
                    Correct_all[i]=correct

        out_all_sorted_index = np.argsort(out_all)[::-1]
        out_all_sorted_val = out_all[out_all_sorted_index]
        Label_all_sorted_index = np.argsort(Label_all)[::-1]
        Label_all_sorted_val = Label_all[Label_all_sorted_index]

        final_set_size=2920
        out_all_sorted_index_final_set=out_all_sorted_index[0:final_set_size]
        Label_all_sorted_index_first_10=Label_all_sorted_index[0:10]
        Label_all_sorted_index_first_100=Label_all_sorted_index[0:100]
        Label_all_sorted_index_first_1000=Label_all_sorted_index[0:1000]

        # Check how many elements from Label_all_sorted_index_first_10 are in out_all_sorted_index_final_set
        count_existing_elements_10 = np.sum(np.isin(Label_all_sorted_index_first_10, out_all_sorted_index_final_set))
        count_existing_elements_100 = np.sum(np.isin(Label_all_sorted_index_first_100, out_all_sorted_index_final_set))
        count_existing_elements_1000 = np.sum(np.isin(Label_all_sorted_index_first_1000, out_all_sorted_index_final_set))
        print(count_existing_elements_10)
        print(count_existing_elements_100)
        print(count_existing_elements_1000)


        #-------------------prepare for next iteraiton--------<this pat shoudl be modified for clasification (not regression0) task>
        known_classifications = []
        for i in range(len(known_torch)):
            if known_torch[i].y.item()>known_median:
                known_classifications.append(1)
            else:
                known_classifications.append(0)
        known_classifications = np.array(known_classifications)
        known_ones_index = known_indices[np.where(known_classifications == 1)[0]]
        known_zero_index = known_indices[np.where(known_classifications == 0)[0]]


        predicted_ones_index = unknown_indices[np.where(out_Class_unknown == 1)[0]]
        predicted_zero_index = unknown_indices[np.where(out_Class_unknown == 0)[0]]

        plt.close('all')

        with open(text_file_path, 'a') as text_file:
            # Write variables to the file
            text_file.write(f"iter {iteration} - END\n")
            text_file.write(f"known_ones_index: {len(known_ones_index)}\n")
            text_file.write(f"predicted_ones_index: {len(predicted_ones_index)}\n")

            text_file.write(f"count_existing_elements_10: {count_existing_elements_10}\n")
            text_file.write(f"count_existing_elements_100: {count_existing_elements_100}\n")
            text_file.write(f"count_existing_elements_1000: {count_existing_elements_1000}\n")

            text_file.write(f"Accuracy: {Accuracy}\n")
            text_file.write(f"Precision: {Precision}\n")
            text_file.write(f"Recall: {Recall}\n")
            text_file.write(f"F1: {F1}\n")
            text_file.write(f"MCC: {MCC}\n")
            
            text_file.write(f"------------------\n")
            text_file.write(f"------------------\n")
            
        

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
