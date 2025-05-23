
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
        """
        distance_matrix: symmetric matrix of distances
        n_ants: number of ants per iteration
        n_best: number of best ants depositing pheromone
        n_iterations: number of iterations
        decay: pheromone decay factor
        alpha: pheromone importance
        beta: heuristic importance
        candidate_size: number of nearest neighbors per node
        stagnation_limit: iterations without improvement to trigger reset
        """
        self.dist_matrix = distance_matrix
        self.N = distance_matrix.shape[0]
        # Initialize pheromone
        self.pheromone = np.ones((self.N, self.N)) / self.N
        # Candidate list: top-k nearest neighbors per node
        self.candidates = [
            list(np.argsort(distance_matrix[i])[:candidate_size+1])  # includes self
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

            # Update best
            if costs[idx_sorted[0]] < best_cost:
                best_cost = costs[idx_sorted[0]]
                best_route = all_routes[idx_sorted[0]]
                stagnation = 0
            else:
                stagnation += 1

            # Spread pheromone from top ants
            self._spread_pheromone(
                [all_routes[i] for i in idx_sorted[:self.n_best]],
                [costs[i] for i in idx_sorted[:self.n_best]]
            )
            # Evaporation
            self.pheromone *= (1 - self.decay)

            # Stagnation: reset or reinforce global best
            if stagnation >= self.stagnation_limit:
                # reheat: increase pheromone along best_route
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
            probs = []
            cand = [c for c in self.candidates[current] if c not in visited]
            # fallback to all if full
            if not cand:
                cand = [j for j in range(self.N) if j not in visited]
            for j in cand:
                pher = self.pheromone[current][j] ** self.alpha
                heuristic = (1.0 / self.dist_matrix[current][j]) ** self.beta
                probs.append(pher * heuristic)
            total = sum(probs)
            probs = [p/total for p in probs]
            next_city = np.random.choice(cand, p=probs)
            route.append(next_city)
            visited.add(next_city)
        # return to start
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
        generations: int = 500
    ):
        """
        distance_matrix: symmetric matrix of distances
        population_size: number of individuals per population
        elite_size: number of best individuals to carry over
        mutation_rate: probability of swapping two cities
        generations: number of generations to evolve
        """
        self.dist_matrix = distance_matrix
        self.N = distance_matrix.shape[0]
        self.pop_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def _create_route(self):
        # random permutation of city indices
        route = list(range(self.N))
        random.shuffle(route)
        route.append(route[0])  # complete the cycle
        return route

    def _initial_population(self):
        return [self._create_route() for _ in range(self.pop_size)]

    def _route_distance(self, route):
        return sum(
            self.dist_matrix[route[i]][route[i+1]]
            for i in range(len(route)-1)
        )

    def _rank_routes(self, population):
        # return list of (route, distance) sorted by distance
        fitness_results = [(route, self._route_distance(route)) for route in population]
        return sorted(fitness_results, key=lambda x: x[1])

    def _selection(self, ranked_routes):
        # elitism: keep top elite_size
        selection_results = [route for route, _ in ranked_routes[:self.elite_size]]
        # roulette wheel on remaining
        df = [(route, dist) for route, dist in ranked_routes]
        # compute cumulative probabilities
        total_fitness = sum(1.0/dist for route, dist in df[self.elite_size:])
        cum_probs = []
        cum_sum = 0
        for route, dist in df[self.elite_size:]:
            cum_sum += (1.0/dist) / total_fitness
            cum_probs.append((route, cum_sum))
        # select remaining
        for _ in range(self.pop_size - self.elite_size):
            pick = random.random()
            for route, cum_prob in cum_probs:
                if pick <= cum_prob:
                    selection_results.append(route)
                    break
        return selection_results

    def _crossover(self, parent1, parent2):
        # ordered crossover
        start, end = sorted(random.sample(range(1, self.N), 2))
        child_p1 = parent1[start:end]
        child = [None] * self.N
        child[start:end] = child_p1
        p2_iter = [c for c in parent2 if c not in child_p1]
        idx = 0
        for i in range(self.N):
            if child[i] is None:
                child[i] = p2_iter[idx]
                idx += 1
        child.append(child[0])
        return child

    def _breed_population(self, selection_results):
        children = selection_results[:self.elite_size]
        non_elite = selection_results[self.elite_size:]
        for i in range(len(non_elite)):
            parent1 = random.choice(selection_results)
            parent2 = random.choice(selection_results)
            child = self._crossover(parent1, parent2)
            children.append(child)
        return children

    def _mutate(self, route):
        for swapped in range(1, self.N):
            if random.random() < self.mutation_rate:
                swap_with = random.randint(1, self.N-1)
                route[swapped], route[swap_with] = route[swap_with], route[swapped]
        return route

    def _mutate_population(self, children):
        return [self._mutate(child) for child in children]

    def run(self):
        pop = self._initial_population()
        start_time = time.perf_counter()
        for generation in range(self.generations):
            ranked = self._rank_routes(pop)
            best_distance = ranked[0][1]
            print(f"Gen {generation+1}/{self.generations}, best dist: {best_distance}")
            selected = self._selection(ranked)
            children = self._breed_population(selected)
            pop = self._mutate_population(children)
        ranked = self._rank_routes(pop)
        best_route, best_dist = ranked[0]
        total_time = time.perf_counter() - start_time
        return best_route, best_dist, total_time





