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
import torch.nn as nn

def custom_loss(outputs, targets, p):
    # Calculate mean squared error
    """above_p_indices = targets >= p
    below_p_indices = targets < p
    c=torch.nn.L1Loss()
    L1_Loss_above_p=0.0
    L1_Loss_below_p=0.0

    if any(above_p_indices):
        L1_Loss_above_p=c(outputs[above_p_indices],targets[above_p_indices])
    #if any(below_p_indices):
    #    L1_Loss_below_p=c(outputs[below_p_indices],targets[below_p_indices])

    penalty = torch.where(targets >= p, torch.abs(torch.minimum(outputs - p, torch.zeros_like(outputs))), 
                                    torch.abs(torch.maximum(outputs - p, torch.zeros_like(outputs))))

    weighted_penalty =  torch.where(targets >= p, 0.0*penalty, 0.3*penalty)
    total_loss=5.0*L1_Loss_above_p+L1_Loss_below_p+weighted_penalty.mean()

    if math.isnan(total_loss):
        kk=1"""

    c=torch.nn.L1Loss()
    L1_loss=c(outputs, targets)
    
    penalty = torch.where(targets >= p, torch.abs(torch.minimum(outputs - p, torch.zeros_like(outputs))), 
                                    torch.abs(torch.maximum(outputs - p, torch.zeros_like(outputs))))
    # Combine the MSE loss and the penalty term
    total_loss = L1_loss + 2*penalty.mean()
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
    embedding_size=64#64
    numHeads=4#4
    num_layers=3
    NUM_GRAPHS_PER_BATCH=4 #4
    p_known=0.05
    training_split=0.8 
    epochs=700#600 
    n=1

File_Name=f"Saved_Files/Mdltype_{Model_type}_regclass_{regres_or_classif}_embd_{embedding_size}_nH_{numHeads}_nL_{num_layers}_btch_{NUM_GRAPHS_PER_BATCH}_pknown_{p_known}_trinsplt_{training_split}_nepcs_{epochs}_nIter_{n}"
#python source_code/run.py --Model_type 0 --regres_or_classif 1 --embedding_size 64 --numHeads 4 --num_layers 3 --NUM_GRAPHS_PER_BATCH 4 --p_known 0.2 --training_split 0.8 --epochs 1000 --n 1

Train_or_Check=1; #Train: 1 , Test : 0
# Set up early stopping parameters
early_stopping_counter = 0
best_val_accuracy = 0.0
patience = 12#7  # Number of consecutive iterations without improvement to tolerate
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
        if Train_or_Check==0:
            model_path = f'{File_Name}.pth' 
            model.load_state_dict(torch.load(model_path))
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

        if regres_or_classif==0:
            num_output=2 #classification
            criterion = torch.nn.CrossEntropyLoss()
        elif regres_or_classif==1:
            num_output=1 #regression
            #criterion = torch.nn.MSELoss() 
            #criterion=torch.nn.L1Loss()
            criterion = lambda outputs,targets: custom_loss(outputs, targets, known_median)

        try:
            os.remove(f'{data_save_path}/known_data/processed/data.pt')
        except OSError as e:

            print('Error')
        
        
        try:
            os.remove(f'{data_save_path}/unknown_data/processed/data.pt')
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

                        if epoch<90:
                            early_stopping_counter=0
                        elif val_acc > best_val_accuracy:
                            best_val_accuracy = val_acc
                            early_stopping_counter = 0
                            # Save your model if needed
                            torch.save(model.state_dict(),  f'{File_Name}_best.pth' )
                        else:
                            early_stopping_counter += 1

                        if early_stopping_counter >= patience:
                            break_outer = True
                            break
                    if break_outer:
                        break

                
                model_path = f'{File_Name}.pth' #'trained_model_1.pth'
                torch.save(model.state_dict(), model_path)
                #mlflow.pytorch.log_model(model, "models")
                mlflow.pytorch.autolog()

        model.load_state_dict(torch.load(f'{File_Name}_best.pth'))
        
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
            ax.plot([0, 1], [0, 1],linewidth=2, linestyle='--', color='red',alpha=0.6, label='Line from (0, 0) to (1, 1)')
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
            ax.plot([0, 1], [0, 1],linewidth=2, linestyle='--', color='red',alpha=0.6, label='Line from (0, 0) to (1, 1)')
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
            ax.plot([0, 1], [0, 1],linewidth=2, linestyle='--', color='red',alpha=0.6, label='Line from (0, 0) to (1, 1)')
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
