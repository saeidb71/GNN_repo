from scipy.io import loadmat
import pandas as pd
from torch_geometric.data import InMemoryDataset
import torch
import numpy as np
import torch_geometric
from tqdm import tqdm
from typing import Union
from torch import Tensor
from collections.abc import Sequence
from torch_geometric.utils import from_networkx
import networkx as nx

IndexType = Union[slice, Tensor, np.ndarray, Sequence]

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class IterationDataset(InMemoryDataset):
    def __init__(self, root: str, data, performance_threshold, regres_or_classif, transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.performance_threshold = performance_threshold
        self.regres_or_classif=regres_or_classif # 1: reg, 0:class
        
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return 'Data/circuit_data.mat'
        

    @property
    def processed_file_names(self) -> list[str]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        
        r'''Create Known Graphs'''
        data_list = []
        for index, cir in tqdm(self.data.iterrows(), total=len(self.data)):
            nxg = nx.Graph(self.data['A'][index])
            nxg = self._get_node_features(nxg, self.data['Ln'][index],self.data['np'][index],self.data['nz'][index]) #
            pt_graph = self._get_graph_object(nxg)
            pt_graph.y = self._get_known_graph_label(self.data['Labels'][index], self.performance_threshold)
            pt_graph.performance = torch.tensor(self.data['Labels'][index], dtype=torch.float)
            pt_graph.complexity = self._get_complexity(self.data['types'][index])
            pt_graph.orig_index = torch.tensor(index, dtype=torch.long)
            data_list.append(pt_graph)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        
        data_known, slices_known = self.collate(data_list)


        torch.save((data_known, slices_known), self.processed_paths[0])


    def _get_complexity(self, types):
        comp = len(types)
        return torch.tensor(comp, dtype=torch.float)

    def _get_node_features(self, nx_graph, node_labels,np,nz):
        betweenness = list(nx.betweenness_centrality(nx_graph).values())
        eigenvector = list(nx.eigenvector_centrality(nx_graph, max_iter=600).values())


        degree_centrality = list(nx.degree_centrality(nx_graph).values())
        closeness_centrality = list(nx.closeness_centrality(nx_graph).values())
        pagerank = list(nx.pagerank(nx_graph).values())
        eccentricity = list(nx.eccentricity(nx_graph).values())
        katz_centrality = list(nx.katz_centrality(nx_graph).values())
        harmonic_centrality = list(nx.harmonic_centrality(nx_graph).values())
        average_neighbor_degree = list(nx.average_neighbor_degree(nx_graph).values())
        local_clustering_coefficient = list(nx.square_clustering(nx_graph).values())

        node_label_dict = dict(enumerate(node_labels))

        mapping_dict = {'C': 0, 'G': 1, 'I': 2, 'O': 3, 'P': 4, 'R': 5}
        component_labels = []
        np_list=[]
        nz_list=[]

        for value in node_label_dict.values():
            if value in mapping_dict:
                component_labels.append(mapping_dict[value])
            np_list.append(np)
            nz_list.append(nz)

        """new_node_feature_hot_key = [[1 if i == label else 0 for i in range(6)] for label in component_labels]"""

        all_features = zip(betweenness, 
                           eigenvector, 
                           degree_centrality, 
                           closeness_centrality ,
                           pagerank,
                           eccentricity,
                           katz_centrality,
                           harmonic_centrality,
                           average_neighbor_degree,
                           local_clustering_coefficient,
                           component_labels,
                           np_list,
                           nz_list)
        """all_features = zip(betweenness, 
                           eigenvector, 
                           component_labels,
                           np_list,
                           nz_list)"""
        """all_features = zip(betweenness, eigenvector, 
                           list(np.array(new_node_feature_hot_key)[:,0]),
                           list(np.array(new_node_feature_hot_key)[:,1]),
                           list(np.array(new_node_feature_hot_key)[:,2]),
                           list(np.array(new_node_feature_hot_key)[:,3]),
                           list(np.array(new_node_feature_hot_key)[:,4]),
                           list(np.array(new_node_feature_hot_key)[:,5]))"""
        all_features = dict(enumerate(all_features))

        nx.set_node_attributes(nx_graph, all_features, 'features')

        return nx_graph

    
    def _get_graph_object(self, nx_graph):
        nxg = from_networkx(nx_graph, group_node_attrs=['features'])
        return nxg

    def _get_known_graph_label(self, performance, threshold):
        if self.regres_or_classif==1:
            return torch.tensor(np.exp(-performance), dtype=torch.float)
            #return torch.tensor(performance, dtype=torch.float)
        else:
            if np.exp(-performance) > threshold:
                return torch.tensor(1, dtype=torch.long)
            else:
                return torch.tensor(0, dtype=torch.long)
    
    @property
    def num_node_features(self) -> int:
        return 13#13#5#8

    @property
    def num_classes(self) -> int:
        return 2
