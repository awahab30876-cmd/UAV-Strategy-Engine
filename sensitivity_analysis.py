"""Small reproducible sensitivity study for GA population and mutation settings."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.environment import Environment
from src.genetic_algorithm import GeneticAlgorithm

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, 'outputs'); os.makedirs(OUT, exist_ok=True)

def main():
    env = Environment.demo()
    configs = [
        ('population', 15, .12, 3), ('population', 25, .12, 3), ('population', 35, .12, 3),
        ('mutation', 50, .04, 3), ('mutation', 50, .12, 3), ('mutation', 50, .25, 3),
        ('selection_pressure', 50, .12, 2), ('selection_pressure', 50, .12, 3), ('selection_pressure', 50, .12, 5),
    ]
    rows=[]
    for i, (factor, pop, mut, tournament) in enumerate(configs):
        ga = GeneticAlgorithm(env, population_size=pop, generations=12, mutation_rate=mut,
                              tournament_size=tournament, seed=200+i)
        best = ga.run()
        rows.append({'factor':factor, 'population_size':pop, 'mutation_rate':mut,
                     'tournament_size':tournament, 'generations':len(ga.history['best']),
                     'best_fitness':best.fitness, 'coverage':best.metrics['coverage'],
                     'detection':best.metrics['detection'], 'energy':best.metrics['energy'],
                     'time':best.metrics['time'], 'hazard':best.metrics['hazard']})
        print(f"{factor:18s} pop={pop:2d} mutation={mut:.2f} tournament={tournament} fitness={best.fitness:.4f}")
    path=os.path.join(OUT,'sensitivity_results.csv')
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f'Saved {path}')
if __name__ == '__main__': main()
