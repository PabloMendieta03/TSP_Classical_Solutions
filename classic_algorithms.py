
# Creación de los 5 algortimos clásicos para la solución del TSP estudiados en el Proyecto Fin de Grado. 

import itertools
from typing import List, Tuple
import numpy as np
import time
import math 
import random 

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
        

    # Algortimo del Vecino más Cercano 
    def nearest_neighbor(
        nodes: List[Tuple[float, float]],
        dist_matrix: np.ndarray
    ) -> Tuple[List[int], float, float]:
        """
        Resuelve el TSP usando la heurística del Vecino Más Cercano.

        Variables de Entrada:
        - nodes: lista de tuplas (x, y), no se usa directamente aquí pero queda
                para futuros cálculos o checks.
        - dist_matrix: matriz de distancias simétrica N×N.

        Variables de Salida:
        - best_route: lista de índices [0, ..., 0] con la ruta construida.
        - total_cost: coste total de esa ruta.
        - elapsed_time: tiempo de ejecución en segundos.
        """
        N = len(nodes)
        if dist_matrix.shape != (N, N):
            raise ValueError("La matriz de distancias debe ser de tamaño N×N")

        start_time = time.perf_counter()

        unvisited = set(range(1, N))
        route = [0]  # comenzamos en el nodo 0
        total_cost = 0.0
        current = 0

        # Mientras queden nodos por visitar, elige el vecino más cercano
        while unvisited:
            # busca el nodo no visitado con mínima distancia desde `current`
            next_node = min(unvisited, key=lambda j: dist_matrix[current, j])
            total_cost += dist_matrix[current, next_node]
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        # volver al origen
        total_cost += dist_matrix[current, 0]
        route.append(0)

        elapsed_time = time.perf_counter() - start_time
        return route, total_cost, elapsed_time
    

# Algoritmo de Optimización por Colonización de Hormigas 
class AntColonyOptimizer:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        n_ants: int = 200,
        n_best: int = 15,
        n_iterations: int = 150,
        decay: float = 0.25,
        alpha: float = 1.0,
        beta: float = 2.5,
        candidate_size: int = 30,
        stagnation_limit: int = 20
    ):
        self.dist_matrix = distance_matrix
        self.N = distance_matrix.shape[0]
        self.pheromone = np.ones((self.N, self.N)) / self.N
        self.candidates = [
            list(np.argsort(distance_matrix[i])[1:candidate_size+1])
            for i in range(self.N)
        ]
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.stagnation_limit = stagnation_limit

    def run(self):
        best_cost = float('inf')
        best_route = None
        stagnation = 0
        start_time = time.perf_counter()

        for iteration in range(self.n_iterations):
            all_routes = self._generate_all_routes()
            costs = [self._route_cost(r) for r in all_routes]
            idx_sorted = np.argsort(costs)

            if costs[idx_sorted[0]] < best_cost:
                best_cost = costs[idx_sorted[0]]
                best_route = all_routes[idx_sorted[0]]
                stagnation = 0
            else:
                stagnation += 1

            self._spread_pheromone(
                [all_routes[i] for i in idx_sorted[:self.n_best]],
                [costs[i] for i in idx_sorted[:self.n_best]]
            )
            self.pheromone *= (1 - self.decay)

            if stagnation >= self.stagnation_limit:
                for i in range(len(best_route)-1):
                    u, v = best_route[i], best_route[i+1]
                    self.pheromone[u][v] += 1.0 / best_cost
                    self.pheromone[v][u] += 1.0 / best_cost
                stagnation = 0

            print(f"Iter {iteration+1}/{self.n_iterations}, best_cost: {best_cost}")

        total_time = time.perf_counter() - start_time
        return best_route, best_cost, total_time

    def _generate_route(self, start):
        route = [start]
        visited = set(route)
        while len(route) < self.N:
            current = route[-1]
            cand = [c for c in self.candidates[current] if c not in visited]
            if not cand:
                cand = [j for j in range(self.N) if j not in visited]
            weights = []
            for j in cand:
                pher = self.pheromone[current][j] ** self.alpha
                heuristic = (1.0 / self.dist_matrix[current][j]) ** self.beta
                weights.append(pher * heuristic)
            total = sum(weights)
            if total <= 0 or np.isnan(total):
                probs = [1/len(cand)] * len(cand)
            else:
                probs = [w/total for w in weights]
            next_city = np.random.choice(cand, p=probs)
            route.append(next_city)
            visited.add(next_city)
        route.append(route[0])
        return route

    def _generate_all_routes(self):
        return [self._generate_route(random.randrange(self.N))
                for _ in range(self.n_ants)]

    def _route_cost(self, route):
        return sum(
            self.dist_matrix[route[i]][route[i+1]]
            for i in range(len(route)-1)
        )

    def _spread_pheromone(self, routes, costs):
        for route, cost in zip(routes, costs):
            deposit = 1.0 / cost
            for i in range(len(route)-1):
                u, v = route[i], route[i+1]
                self.pheromone[u][v] += deposit
                self.pheromone[v][u] += deposit



# Resolución por Algoritmos Genéticos 
class GeneticAlgorithm:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        population_size: int = 100,
        elite_size: int = 20,
        mutation_rate: float = 0.01,
        generations: int = 500,
        tournament_size: int = 5
    ):
        """
        Optimized GA for TSP with PMX crossover.

        distance_matrix: symmetric np.array of distances
        population_size: number of individuals
        elite_size: number of best to carry over
        mutation_rate: prob of swap mutation
        generations: number of generations
        tournament_size: number of competitors in tournament selection
        """
        self.dist_matrix = distance_matrix
        self.N = distance_matrix.shape[0]
        self.pop_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.tournament_size = tournament_size

    def _create_route(self):
        route = np.arange(self.N, dtype=int)
        np.random.shuffle(route)
        return np.append(route, route[0])

    def _initial_population(self):
        return [self._create_route() for _ in range(self.pop_size)]

    def _route_distance(self, route):
        idx = np.arange(len(route)-1)
        return np.sum(self.dist_matrix[route[idx], route[idx+1]])

    def _rank_routes(self, population):
        distances = np.array([self._route_distance(r) for r in population])
        sorted_idx = np.argsort(distances)
        return [population[i] for i in sorted_idx], distances[sorted_idx]

    def _tournament_selection(self, population, distances):
        selected = []
        elites = [population[i] for i in sorted(range(len(distances)), key=lambda i: distances[i])[:self.elite_size]]
        selected.extend(elites)
        for _ in range(self.pop_size - self.elite_size):
            contenders = random.sample(range(self.pop_size), self.tournament_size)
            best = min(contenders, key=lambda i: distances[i])
            selected.append(population[best])
        return selected

    def _pmx_crossover(self, p1, p2):
        size = self.N
        cx1, cx2 = sorted(random.sample(range(1, size), 2))
        child = np.full(size, -1, dtype=int)
        child[cx1:cx2] = p1[cx1:cx2]
        for i in range(cx1, cx2):
            if p2[i] not in child:
                val = p2[i]
                pos = i
                while True:
                    val2 = p1[pos]
                    pos = np.where(p2 == val2)[0][0]
                    if child[pos] == -1:
                        child[pos] = val
                        break
        for i in range(size):
            if child[i] == -1:
                child[i] = p2[i]
        return np.append(child, child[0])

    def _breed_population(self, selected):
        children = selected[:self.elite_size]
        for _ in range(self.pop_size - self.elite_size):
            p1, p2 = random.sample(selected, 2)
            child = self._pmx_crossover(p1[:-1], p2[:-1])
            children.append(child)
        return children

    def _mutate(self, route):
        for i in range(1, self.N):
            if random.random() < self.mutation_rate:
                j = random.randint(1, self.N-1)
                route[i], route[j] = route[j], route[i]
        route[-1] = route[0]
        return route

    def _mutate_population(self, population):
        return [self._mutate(r.copy()) for r in population]

    def run(self):
        pop = self._initial_population()
        best_route = None
        best_dist = float('inf')
        start = time.perf_counter()
        for gen in range(1, self.generations + 1):
            ranked, distances = self._rank_routes(pop)
            if distances[0] < best_dist:
                best_dist = distances[0]
                best_route = ranked[0]
            print(f"Gen {gen}/{self.generations}, best dist: {best_dist}")
            selected = self._tournament_selection(pop, distances)
            bred = self._breed_population(selected)
            pop = self._mutate_population(bred)
        elapsed = time.perf_counter() - start
        return best_route.tolist(), best_dist, elapsed




