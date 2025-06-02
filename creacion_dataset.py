
# ----- GENERACIÓN DE DATA SET PARA LAS REDES NEURONALES ----- #

'''
En este archivo .py, se han creado las diferentes Bases de Datos que se han utilizado para la generación de Problemas del Viajante del Comercio, con su solución y 
con el formato necesario para que las GNNs lo entiendan y puedan utilizarlo.

Se han creado los siguientes datasets: 
 - tsps.pt: 500 TSP, tamaños: 5, 8, 10, 12
 - tsps100.pt: 1000 TSP, tamaño: 5, 10, 12, 15  (~= 5 horas de ejecución)
'''

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

num_datasets = 100

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
torch.save(TSPs, "test100.pt")




