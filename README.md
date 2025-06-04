<h1 align="center">TSP_Solutions</h1>

Este repositorio contiene distintas implementaciones para resolver el problema del Viajante del
Comercio (Traveling Salesman Problem, TSP). Se incluyen algoritmos clásicos como fuerza bruta,
Held‑Karp, nearest neighbor o metaheurísticas (colonia de hormigas y algoritmo genético) y un
modelo basado en Graph Neural Networks (GNN) entrenado con PyTorch Geometric.

## Clonación del repositorio
```bash
git clone https://github.com/PabloMendieta03/TSP_Classical_Solutions.git
```

## Creación del entorno
```bash
 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
```bash
 python -m venv tsp-venv
```
```bash
 .\tsp-venv\Scripts\Activate
```

## Instalación de dependencias
```bash
 pip install -r requirements.txt
```

## Estructura del repositorio
```
TSP_Classical_Solutions/
├── classic_algorithms.py        # Implementaciones clásicas del TSP
├── creacion_dataset.py          # Generación de datasets para la GNN
├── dataset.py                   # Utilidades de creación de datos
├── solutions.py                 # Evaluación y construcción de tours
├── GNNs.ipynb                   # Notebook de entrenamiento de la GNN
├── classical_solution.ipynb     # Ejemplos con los algoritmos clásicos
├── transformer_RL.ipynb         # Experimentos con Transformer + RL
├── TSP_data/                    # Problemas generados para entrenar
├── TSP_problems/                # Instancias clásicas de TSPLIB
├── trained_models/              # Modelos de GNN ya entrenados
└── requirements.txt
```

