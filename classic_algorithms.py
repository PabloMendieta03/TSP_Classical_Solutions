
# Creación de los 5 algortimos clásicos para la solución del TSP estudiados en el Proyecto Fin de Grado. 

import itertools
from typing import List, Tuple
import numpy as np
import time

class classic_algorithms: 


    # Algoritmo de fuerza Bruta 
    def brute_force(
        nodes: List[Tuple[float, float]],
        dist_matrix: np.ndarray
    ) -> Tuple[List[int], float, float]:
        """
        Resuelve el TSP por fuerza bruta, aprovechando la simetría de la matriz de distancias.

        Variables de Entrada:
        - nodes: lista de tuplas (x, y), no se usa directamente aquí pero queda
                para futuros cálculos o checks.
        - dist_matrix: matriz de distancias simétrica N×N.

        Variables de Salida:
        - best_route: lista de índices [0, ..., 0] con la mejor ruta.
        - best_cost: coste total mínimo.
        - elapsed_time: tiempo de ejecución en segundos.
        """
        N = len(nodes)
        if dist_matrix.shape != (N, N):
            raise ValueError("La matriz de distancias debe ser de tamaño N×N")

        start_time = time.perf_counter()

        best_cost = float('inf')
        best_route: List[int] = []

        # índice 0 fijado como inicio y fin
        # gracias a la simetría, descartamos la mitad de permutaciones equivalentes
        for perm in itertools.permutations(range(1, N)):
            # descartamos la permutación si su primera posición es mayor que la última,
            # pues invertirla produciría la misma ruta en sentido contrario
            if perm[0] > perm[-1]:
                continue

            route = [0] + list(perm) + [0]
            cost = sum(dist_matrix[route[i], route[i+1]] for i in range(N))
            if cost < best_cost:
                best_cost = cost
                best_route = route

        elapsed_time = time.perf_counter() - start_time
        return best_route, best_cost, elapsed_time

    # Algoritmo Programación Dinámica. Held y Karp 
    def held_karp(
        nodes: List[Tuple[float, float]],
        dist_matrix: np.ndarray
    ) -> Tuple[List[int], float, float]:
        """
        Resuelve el TSP usando el algoritmo de Held-Karp (programación dinámica).

        Variables de Entrada:
        - nodes: lista de tuplas (x, y), no se usa directamente aquí.
        - dist_matrix: matriz de distancias simétrica N×N.

        Variables de Salida:
        - best_route: lista de índices [0, ..., 0] con la mejor ruta.
        - best_cost: coste total mínimo.
        - elapsed_time: tiempo de ejecución en segundos.
        """
        N = len(nodes)
        if dist_matrix.shape != (N, N):
            raise ValueError("La matriz de distancias debe ser de tamaño N×N")

        start = 0
        all_sets = 1 << N  # 2^N posibles subconjuntos

        start_time = time.perf_counter()

        # DP[mask][j] = coste mínimo para recorrer el subconjunto mask,
        # acabando en nodo j.
        DP = [dict() for _ in range(all_sets)]
        parent = [dict() for _ in range(all_sets)]

        # Caso base: desde el inicio al nodo j sin pasar por nadie más
        for j in range(1, N):
            mask = (1 << start) | (1 << j)
            DP[mask][j] = dist_matrix[start, j]
            parent[mask][j] = start

        # Llenar DP para todos los tamaños de subconjunto desde 3 hasta N
        for mask in range(all_sets):
            # ignorar si no incluye el start o tiene menos de 2 bits
            if not (mask & 1) or bin(mask).count("1") < 2:
                continue
            for j in range(1, N):
                if not (mask & (1 << j)):
                    continue
                prev_mask = mask ^ (1 << j)
                # buscamos el mejor k que venga a j
                best_cost = float('inf')
                best_prev = None
                for k in range(1, N):
                    if k == j or not (prev_mask & (1 << k)):
                        continue
                    cost = DP[prev_mask].get(k, float('inf')) + dist_matrix[k, j]
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = k
                # actualizar DP sólo si encontramos un camino válido
                if best_prev is not None:
                    DP[mask][j] = best_cost
                    parent[mask][j] = best_prev

        # Cerramos el ciclo volviendo al inicio
        full_mask = (1 << N) - 1
        best_cost = float('inf')
        best_last = None
        for j in range(1, N):
            cost = DP[full_mask].get(j, float('inf')) + dist_matrix[j, start]
            if cost < best_cost:
                best_cost = cost
                best_last = j

        # Reconstrucción de la ruta
        route = [start]
        mask = full_mask
        last = best_last
        for _ in range(N - 1):
            route.append(last)
            prev = parent[mask][last]
            mask ^= (1 << last)
            last = prev
        route.append(start)
        route.reverse()  # porque reconstruimos de atrás hacia adelante

        elapsed_time = time.perf_counter() - start_time
        return route, best_cost, elapsed_time
        





