"""Genetic Algorithm engine with tournament selection and spatial operators."""
import numpy as np
from .chromosome import Chromosome
from .fitness import evaluate, repair

class GeneticAlgorithm:
    def __init__(self, env, population_size=50, generations=100, mutation_rate=.12,
                 tournament_size=3, n_drones=3, n_waypoints=8, seed=23, elite_count=2):
        self.env, self.population_size, self.generations = env, population_size, generations
        self.mutation_rate, self.tournament_size = mutation_rate, tournament_size
        self.n_drones, self.n_waypoints, self.elite_count = n_drones, n_waypoints, elite_count
        self.rng = np.random.default_rng(seed)
        self.history = {'best': [], 'average': []}
        self.population = []

    def initialize(self):
        self.population = [repair(Chromosome.random(self.env, self.n_drones, self.n_waypoints, self.rng), self.env)
                           for _ in range(self.population_size)]
        for c in self.population: evaluate(c, self.env)

    def select(self):
        indices = self.rng.choice(len(self.population), size=min(self.tournament_size, len(self.population)), replace=False)
        candidates = [self.population[int(i)] for i in indices]
        return max(candidates, key=lambda c: c.fitness)

    def crossover(self, a, b):
        # Segment crossover: exchange a contiguous waypoint block for one drone.
        child = a.copy()
        d = int(self.rng.integers(0, self.n_drones))
        left = int(self.rng.integers(0, self.n_waypoints))
        right = int(self.rng.integers(left+1, self.n_waypoints+1))
        child.genes[d, left:right, :] = b.genes[d, left:right, :]
        # Occasional whole-drone inheritance preserves coherent subroutes.
        if self.rng.random() < .25:
            d2 = int(self.rng.integers(0, self.n_drones))
            child.genes[d2] = b.genes[d2]
        return child

    def mutate(self, chrom):
        g = chrom.genes
        for d in range(self.n_drones):
            for i in range(self.n_waypoints):
                if self.rng.random() < self.mutation_rate:
                    # Gaussian spatial perturbation; occasional re-sampling maintains diversity.
                    if self.rng.random() < .12:
                        g[d, i, 0:2] = self.rng.uniform(0, self.env.size-1, 2)
                    else:
                        g[d, i, 0:2] += self.rng.normal(0, 3.0, 2)
                if self.rng.random() < self.mutation_rate:
                    g[d, i, 2] += self.rng.normal(0, .18)
                if self.rng.random() < self.mutation_rate:
                    g[d, i, 3] += self.rng.normal(0, .18)
        return chrom

    def run(self):
        self.initialize()
        best_ever = max(self.population, key=lambda c: c.fitness).copy()
        stale = 0
        for generation in range(self.generations):
            self.population.sort(key=lambda c: c.fitness, reverse=True)
            current_best = self.population[0]
            self.history['best'].append(current_best.fitness)
            self.history['average'].append(float(np.mean([c.fitness for c in self.population])))
            if current_best.fitness > best_ever.fitness + 1e-8:
                best_ever = current_best.copy(); stale = 0
            else:
                stale += 1
            if stale >= 25:
                break
            next_pop = [c.copy() for c in self.population[:self.elite_count]]
            while len(next_pop) < self.population_size:
                p1, p2 = self.select(), self.select()
                child = self.crossover(p1, p2) if self.rng.random() < .85 else p1.copy()
                child = self.mutate(child)
                repair(child, self.env)
                evaluate(child, self.env)
                next_pop.append(child)
            self.population = next_pop
        # Include final population in best comparison.
        final_best = max(self.population, key=lambda c: c.fitness)
        if final_best.fitness > best_ever.fitness: best_ever = final_best.copy()
        return best_ever
