#functions that are useful for assessing quality of solutions

import torch

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
    
