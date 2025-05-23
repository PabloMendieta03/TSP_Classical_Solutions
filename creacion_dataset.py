# Creación de la Base de Datos para tener un gran conjunto de problemas del viajante con los que trabajar 


# ----- Importación de Librerias ----- #
from torch_geometric.data import Data
import itertools
import torch
import numpy as np 

import dataset
import importlib
importlib.reload(dataset)
from dataset import DatasetCreation


# ======================== Base de Datos para entrenar la Red Neuronal ======================== #

TSPs = []

num_datasets = 1000

for num_nodes in [5, 10, 12, 15]:
  node_coords, distance_matrices,solution_paths, solution_adjacencies, distances = DatasetCreation.create_dataset(num_nodes = num_nodes,  num_datasets = num_datasets)

  # Peso para el entrenamiento de la red neuronal, mayor número de nodos mayor peso tiene en el entrenamiento. 
  num_pos = num_nodes
  num_neg = num_nodes**2-num_nodes

  weight_pos_class = (num_neg/num_pos)

  edge_index = torch.tensor(list(itertools.product(np.arange(num_nodes),np.arange(num_nodes))), dtype=torch.long).T.contiguous()
  for i in range(len(node_coords)):
      edge_attr = torch.tensor(((distance_matrices[i])).flatten()).float().unsqueeze(1)

      x = torch.tensor(node_coords[i]).float()
      y = torch.tensor(solution_adjacencies[i].flatten()).float().unsqueeze(1)


      data = Data(x=x, edge_index=edge_index, y= y, edge_attr=edge_attr)
      data.edge_weight = torch.tensor(((distance_matrices[i])).flatten()).float().unsqueeze(1)
      data.true_path = torch.Tensor(solution_paths[i])
      data.true_distance = torch.Tensor([distances[i]]).unsqueeze(1)
      data.num_nodes = num_nodes
      data.pos_class_weight = weight_pos_class
      data.disntace_matrices = distance_matrices
      TSPs.append(data)

# Guardar la Base de Datos 
torch.save(TSPs, "tsps1000.pt")




