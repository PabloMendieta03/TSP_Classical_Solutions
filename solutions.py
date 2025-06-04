#functions that are useful for assessing quality of solutions

import torch
import math

class SolutionAnalysys: 

    def tour_from_probs(probs: torch.Tensor, num_nodes: int) -> list:
        """
        Construye un tour Hamiltoniano (cerrado) a partir de:
        • probs: tensor 1D de tamaño E = N*(N-1) con la probabilidad de cada arista i->j,
                en el orden “bloque por bloque”:
                Bloque 0: (0->1, 0->2, …, 0->N-1)  → N-1 elementos
                Bloque 1: (1->0, 1->2, …, 1->N-1)  → N-1 elementos
                …
                Bloque i: (i->0, …, i->i-1, i->i+1, …, i->N-1)  → N-1 elementos
        • num_nodes: número de nodos N

        Empieza en el nodo 0 y, en cada paso, elige entre todas las aristas (current→j)
        donde j no esté visitado, la de mayor probabilidad. Finalmente cierra el ciclo.

        Retorna:
            tour (list[int]): Lista de N+1 nodos (ciclo), p.ej. [0, 3, 5, 2, …, 0].
        """
        N = num_nodes
        visited = {0}
        tour = [0]
        current = 0

        for _ in range(N - 1):
            best_p = -1.0
            best_j = None

            # Cada nodo i tiene sus (N-1) aristas “i→j” en el bloque:
            # bloque_i = probs[i*(N-1) : (i+1)*(N-1)]
            base = current * (N - 1)
            for offset in range(N - 1):
                p = probs[base + offset].item()
                # Convertir offset en el j real (saltando i):
                if offset < current:
                    j = offset
                else:
                    j = offset + 1
                if j not in visited and p > best_p:
                    best_p = p
                    best_j = j

            if best_j is None:
                # Si no encuentra candidato (caso límite), toma el primer no visitado
                rem = set(range(N)) - visited
                best_j = rem.pop()

            tour.append(best_j)
            visited.add(best_j)
            current = best_j

        tour.append(0)
        return tour

    def find_greedy_max_neighbor_traversal(weighted_matrix):
        """
        Find the order of traversal starting from the first node as the root
        by greedily selecting the most probable neighbor that is not in the path yet.

        Parameters:
            weighted_matrix (torch.Tensor): Weighted adjacency matrix representing the graph.

        Returns:
            Tensor: Order of traversal starting from the first node.
        """
        # Start traversal from the first node (index 0)
        path = [0]  # Start with the first node
        num_nodes = weighted_matrix.size(0)
        current_node = 0

        # Traverse through the nodes until reaching the last node
        while len(path) < num_nodes:
            # Find the next node (neighbor) with the maximum weight
            max_weight = -float('inf')
            next_node = -1
            for neighbor, weight in enumerate(weighted_matrix[current_node]):
                if neighbor not in path and weight > max_weight:
                    max_weight = weight
                    next_node = neighbor
            if next_node == -1:
                break  # No valid neighbor found, exit the loop
            else:
                current_node = next_node
                path.append(current_node)

        return torch.Tensor(path)

    def count_disagreements(tensor1, tensor2):
        """
        Count the number of positions where two tensors of the same length disagree.

        Parameters:
            tensor1 (torch.Tensor): First tensor.
            tensor2 (torch.Tensor): Second tensor.

        Returns:
            int: Number of positions where the tensors disagree.
        """

        # Ensure both tensors have the same length
        assert tensor1.size() == tensor2.size(), "Tensors must have the same length"

        # Count number of disagreements
        num_disagreements_one_way = (tensor1[1:] != tensor2[1:]).sum().item()
        num_disagreements_other_way = (tensor1[1:] != torch.flip(tensor2[1:], [0])).sum().item()


        return min(num_disagreements_one_way,num_disagreements_other_way)


    def mean_per_batch_optimality_metrics(heatmap_pred, heatmap_true, num_graphs, num_nodes, edge_weights, true_distance):
        l1_path_distances = 0
        rel_l1_optimality_gaps = 0
        for i in range(num_graphs):
            solution_path = SolutionAnalysys.find_greedy_max_neighbor_traversal(heatmap_pred[i,:,:])
            l1_path_distances += SolutionAnalysys.count_disagreements(solution_path,heatmap_true[i,:])

            pred_distance  = sum([edge_weights[i, int(solution_path[j].item()), int(solution_path[j+1].item()) ] for j in range(num_nodes-1)] + [edge_weights[i, int(solution_path[-1].item()),0 ] ])
            rel_l1_optimality_gaps +=  (pred_distance - true_distance[i])/ true_distance[i]
        return l1_path_distances/num_graphs , rel_l1_optimality_gaps/num_graphs
    

class Opt:
    """
    Clase para “deshacer cruces” en un tour TSP usando heurística 2-opt,
    con límite de pasadas para evitar que el bucle se eternice.
    """

    def __init__(self, coordinates):
        """
        Inicializa la clase con las coordenadas de los nodos.

        Parámetros:
            - coordinates: lista de pares [(x1, y1), (x2, y2), …]
        """
        # Nos aseguramos de tener una lista de tuplas [(x, y), ...]
        self.coordinates = [tuple(p) for p in coordinates]

    @staticmethod
    def _orientation(a, b, c):
        """Devuelve >0 si (a,b,c) son CCW, <0 si CW, 0 si colineales."""
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def _on_segment(a, b, c):
        """True si el punto c está sobre el segmento recta a–b (colinealidad incluida)."""
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    @classmethod
    def _segments_intersect(cls, p1, p2, p3, p4):
        """
        Determina si los segmentos p1–p2 y p3–p4 se intersectan.
        Devuelve True incluso si son colineales y se superponen en algún punto.
        """
        o1 = cls._orientation(p1, p2, p3)
        o2 = cls._orientation(p1, p2, p4)
        o3 = cls._orientation(p3, p4, p1)
        o4 = cls._orientation(p3, p4, p2)

        # Caso general: orientaciones opuestas en ambos pares
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True

        # Casos colineales: un punto de un segmento está sobre el otro
        if o1 == 0 and cls._on_segment(p1, p2, p3): return True
        if o2 == 0 and cls._on_segment(p1, p2, p4): return True
        if o3 == 0 and cls._on_segment(p3, p4, p1): return True
        if o4 == 0 and cls._on_segment(p3, p4, p2): return True

        return False

    def uncross(self, tour, max_passes=160):
        """
        Aplica heurística 2-opt para deshacer cruces en el tour dado, pero
        forzando una salida si supera 'max_passes' pasadas sin converger.

        Parámetros:
            - tour: lista de índices de nodos que define el orden (ej. [0,3,2,1,0]).
                    Se espera que el primer y último índice sean iguales (ciclo cerrado).
            - max_passes: número máximo de iteraciones “externas” de 2-opt.
                          Tras alcanzarlo, devuelve el tour que tenga en ese punto,
                          aunque aún queden cruces.

        Retorna:
            - tour_opt: tour optimizado (o al menos reducido de cruces) en <= max_passes.
        """
        # Verificamos brevemente si el tour está cerrado
        if tour[0] != tour[-1]:
            raise ValueError("El tour debe comenzar y terminar en el mismo nodo.")

        coords = self.coordinates
        n = len(tour)
        tour_opt = tour.copy()
        passes = 0

        # Iteramos hasta max_passes o hasta no haber mejoras
        while passes < max_passes:
            passes += 1
            improved = False

            # Recorremos todas las parejas de aristas no consecutivas
            for i in range(1, n - 2):
                # Sacamos índices de nodos para no repetir búsquedas en la lista
                A_idx = tour_opt[i - 1]
                B_idx = tour_opt[i]
                pA = coords[A_idx]
                pB = coords[B_idx]

                for j in range(i + 1, n - 1):
                    C_idx = tour_opt[j]
                    D_idx = tour_opt[j + 1]
                    pC = coords[C_idx]
                    pD = coords[D_idx]

                    # Si las aristas se cruzan, invertimos la sección i…j
                    if self._segments_intersect(pA, pB, pC, pD):
                        # Intercambiamos (2-opt swap)
                        tour_opt[i : j + 1] = reversed(tour_opt[i : j + 1])
                        improved = True
                        break  # Salimos de 'j'
                if improved:
                    # Rompemos 'i' para volver a comprobar desde el principio
                    break

            if not improved:
                # Si en esta pasada no hubo ningún cruce intercambiado, ya hemos terminado
                break
        return tour_opt
