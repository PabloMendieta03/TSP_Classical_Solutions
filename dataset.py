# Creación de la base de datos de problemas del TSP 


# ----- Importación de librerías ----- #
import numpy as np 
import math
from typing import Dict, List, Tuple
from python_tsp.exact import solve_tsp_dynamic_programming # Solución exacta 
# from python_tsp.heuristics import solve_tsp_local_search # Solución heurística


class DatasetCreation:
    
    # Función para la creación de la base de datos 
    @staticmethod
    def create_dataset(num_nodes, num_datasets):
        """
        Generación de una base de datos de problemas del Viajante del Comercio.

        Variables de Entrada:
            num_nodos: Número de 'cidudades' en cada problema del TSP.
            num_problemas: Número de problemas a resolver. 
            
            num_nodes (int): The number of nodes (cities) in each dataset.
            num_datasets (int): The number of datasets to generate.
            check_correct (bool, optional): If True, verify the correctness of the TSP solution. Defaults to False.

        Variables de Salida:
            Una tupla: 
                - nodo_coords: Una lista de arrays que representan las coordenadas de los nodos para cada conjunto de datos.
                - distancia_matriz: Una lista de matrices de distancias para cada conjunto de datos.
                - sols: Una lista de matrices de adyacencia que representan las soluciones TSP para cada conjunto de datos.
                - distancias: Una lista de distancias totales para cada conjunto de datos.
                - knn_matrices_adj: Una lista de matrices de adyacencia de los k-vecinos más cercanos para cada conjunto de datos.
        """

        node_coords = []
        distance_matrices = []
        solution_paths = []
        solution_adjacencies = []
        solution_distances = []


        for _ in range(num_datasets):
            # 1) Generar puntos aleatorios en [0,100)×[0,100)
            points = 100.0 * np.random.rand(num_nodes, 2)

            # 2) Calcular matriz de distancias euclídeas
            distance_matrix = np.sqrt(((points[:, np.newaxis] - points) ** 2).sum(axis=2))

            # 3) Resolver el TSP con python-tsp (exacto)
            permutation, solution_value = solve_tsp_dynamic_programming(distance_matrix)
            # Si prefieres heurística, comenta la línea anterior y usa:
            # permutation, solution_value = solve_tsp_local_search(distance_matrix)

            # 4) Convertir la permutación en matriz de adyacencia
            solution_adjacency = DatasetCreation.tsp_solution_to_adjacency(permutation)

            # 5) Almacenar resultados
            node_coords.append(points)
            distance_matrices.append(distance_matrix)
            solution_paths.append(permutation)
            solution_adjacencies.append(solution_adjacency)
            solution_distances.append(solution_value)

        return (
            node_coords,
            distance_matrices,
            solution_paths,
            solution_adjacencies,
            solution_distances
        )
    
    
    def tsp_solution_to_adjacency(permutation, distance=None, distance_matrix=None):
        """
        Convierte una secuencia de permutación en una matriz de adyacencia.

        Variables de Entrada:
            permutation (list o array-like): Secuencia de permutación que representa el orden de los nodos.
            distance (float, opcional): Valor de la distancia esperado. Por defecto None.
            distance_matrix (array-like, opcional): Matriz de distancias entre nodos. Por defecto None.

        Variables de salida :
            numpy.ndarray: Matriz de adyacencia que representa las conexiones entre los nodos.

        Expcepciones:
            AssertionError: Si se proporciona `distance` y `distance_matrix` no es None, y la suma de distancias
                            calculada a partir de la matriz de adyacencia no coincide con el valor dado
                            dentro de una tolerancia.

        Notea:
            - La secuencia de permutación representa la solución (aprox.) del TSP.
            - La matriz de adyacencia resultante refleja las conexiones entre nodos según la permutación dada.
            Cada fila y columna corresponden a un nodo, y el valor en (i, j) es 1 si existe una conexión
            de i a j, y 0 en caso contrario.
            - Si se proporcionan `distance` y `distance_matrix`, la función verifica que la suma de distancias
            calculada coincida con el valor esperado dentro de una tolerancia.
        """

        size = len(permutation)
        to_return = np.zeros((size, size))
        for i in range(size - 1):
            curr = permutation[i]
            next_ = permutation[i + 1]
            to_return[curr, next_] = 1
        to_return[next_, 0] = 1

        if distance is not None:
            if distance_matrix is not None:
                masked_values = to_return * distance_matrix
                # Sum along the appropriate axis
                sum_values = np.sum(masked_values, axis=(0, 1))
                assert np.allclose(sum_values, distance, rtol=0.5, atol=0)

        return to_return


# Estudio de problemas clásicos del TSP 
class FamousTSP:
    """
    Clase para gestionar instancias TSP de TSPLIB y tours óptimos.

    Atributos:
        dimension: int                         Número de nodos.
        coords: Dict[int, Tuple[float, float]] Coordenadas de cada nodo.
        dist_matrix: List[List[int]]          Matriz de distancias redondeadas según TSPLIB.
        tour: List[int]                       Secuencia de nodos del tour.
        cost: int                             Coste total del tour.
    """

    def __init__(self, tsp_file: str, tour_file: str):
        # Leer dimensión y coordenadas
        self.dimension = self._read_dimension(tsp_file)
        self.coords = self._read_tsp(tsp_file)
        if len(self.coords) != self.dimension:
            raise ValueError(f"Número de coordenadas leído ({len(self.coords)}) != dimensión ({self.dimension})")
        # Generar matriz de distancias TSPLIB (distancia euclídea + redondeo)
        self.dist_matrix = self._compute_distance_matrix(self.coords)
        # Leer tour óptimo
        self.tour = self._read_tour(tour_file)
        if len(self.tour) != self.dimension:
            raise ValueError(f"Longitud del tour ({len(self.tour)}) != dimensión ({self.dimension})")
        # Calcular coste total del tour
        self.cost = self._compute_tour_cost(self.tour, self.dist_matrix)

    @staticmethod
    def _read_dimension(filename: str) -> int:
        """
        Lee la línea DIMENSION en el .tsp para obtener el número de nodos.
        """
        with open(filename, 'r') as f:
            for line in f:
                if line.upper().startswith('DIMENSION'):
                    # Formato: DIMENSION : 280
                    parts = line.replace(':', ' ').split()
                    for token in parts:
                        if token.isdigit():
                            return int(token)
        raise ValueError('No se encontró la línea DIMENSION en el archivo .tsp')

    @staticmethod
    def _read_tsp(filename: str) -> Dict[int, Tuple[float, float]]:
        """
        Parsea un archivo .tsp en formato TSPLIB y devuelve coordenadas.
        Sólo procesa la sección NODE_COORD_SECTION.
        """
        coords: Dict[int, Tuple[float, float]] = {}
        with open(filename, 'r') as f:
            in_node_section = False
            for line in f:
                line = line.strip()
                if line.upper() == 'NODE_COORD_SECTION':
                    in_node_section = True
                    continue
                if in_node_section:
                    if not line or line.upper() == 'EOF':
                        break
                    parts = line.split()
                    idx = int(parts[0])
                    coords[idx] = (float(parts[1]), float(parts[2]))
        return coords

    @staticmethod
    def _compute_distance_matrix(coords: Dict[int, Tuple[float, float]]) -> List[List[int]]:
        """
        Crea la matriz de distancias TSPLIB entre nodos.
        Cada distancia se redondea como: int(d + 0.5).
        Se recorre únicamente sobre las claves presentes en coords.
        """
        # Determine máximo índice para dimensionar la matriz
        max_idx = max(coords.keys())
        dist: List[List[int]] = [[0] * (max_idx + 1) for _ in range(max_idx + 1)]
        # Calcular distancias solo entre nodos existentes
        for i, (xi, yi) in coords.items():
            for j, (xj, yj) in coords.items():
                d = math.hypot(xi - xj, yi - yj)
                dist[i][j] = int(d + 0.5)
        return dist

    @staticmethod
    def _read_tour(filename: str) -> List[int]:
        """
        Parsea un archivo .opt.tour TSPLIB y devuelve la secuencia de nodos del tour.
        Lee la sección TOUR_SECTION hasta encontrar -1 o EOF.
        """
        tour: List[int] = []
        with open(filename, 'r') as f:
            in_tour_section = False
            for line in f:
                line = line.strip()
                if line.upper() == 'TOUR_SECTION':
                    in_tour_section = True
                    continue
                if in_tour_section:
                    if not line or line == '-1' or line.upper() == 'EOF':
                        break
                    tour.append(int(line))
        return tour

    @staticmethod
    def _compute_tour_cost(tour: List[int], dist_matrix: List[List[int]]) -> int:
        """
        Suma el coste de un tour cerrado usando la matriz de distancias.
        """
        total = 0
        for k in range(len(tour)):
            i = tour[k]
            j = tour[(k + 1) % len(tour)]
            total += dist_matrix[i][j]
        return total
