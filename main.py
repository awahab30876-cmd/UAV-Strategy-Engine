"""Run the complete UAV Strategy Engine demo and save plots/results."""
import json, os, sys
import numpy as np
from src.environment import Environment
from src.chromosome import Chromosome
from src.fitness import evaluate, repair
from src.genetic_algorithm import GeneticAlgorithm
from src.visualizer import save_all, plot_comparison

ROOT = os.path.dirname(__file__)
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)

def random_search(env, trials=250, seed=100):
    rng = np.random.default_rng(seed); scores=[]; best=None
    for _ in range(trials):
        c = repair(Chromosome.random(env, 3, 8, rng), env); evaluate(c, env); scores.append(c.fitness)
        if best is None or c.fitness > best.fitness: best = c.copy()
    return np.asarray(scores), best

def main():
    env = Environment.demo()
    ga = GeneticAlgorithm(env, population_size=35, generations=60, mutation_rate=.12,
                          tournament_size=3, n_drones=3, n_waypoints=8, seed=23)
    best = ga.run()
    # Match the random baseline to the GA's approximate number of fitness evaluations.
    ga_evaluations = ga.population_size + len(ga.history['best']) * (ga.population_size - ga.elite_count)
    random_scores, random_best = random_search(env, trials=ga_evaluations, seed=101)
    save_all(env, best, ga.history, OUT)
    plot_comparison([best.fitness], [float(np.max(random_scores))], OUT)
    report = {k: v for k, v in best.metrics.items() if k not in ('coverage_map',)}
    report.update({'generations_completed': len(ga.history['best']), 'population_size': ga.population_size,
                   'mutation_rate': ga.mutation_rate, 'random_search_trials': len(random_scores),
                   'random_search_best_fitness': float(np.max(random_scores)),
                   'ga_best_fitness': float(best.fitness), 'waypoints': best.genes.tolist()})
    with open(os.path.join(OUT, 'results.json'), 'w') as f: json.dump(report, f, indent=2)
    print('UAV Strategy Engine run complete.')
    print(f"Generations: {len(ga.history['best'])}")
    print(f"GA best fitness: {best.fitness:.4f}")
    print(f"Random-search best fitness ({len(random_scores)} trials): {np.max(random_scores):.4f}")
    print(f"Coverage: {best.metrics['coverage']:.3f}; detection: {best.metrics['detection']:.3f}")
    print(f"Energy: {best.metrics['energy']:.1f}; time: {best.metrics['time']:.1f}; hazard: {best.metrics['hazard']:.1f}")
    print(f"Plots and results saved to: {OUT}")

if __name__ == '__main__': main()
