# Creación de la base de datos de problemas del TSP 


# Importación de librerías 
import numpy as np 
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



