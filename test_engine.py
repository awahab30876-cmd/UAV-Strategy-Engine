import unittest
import numpy as np
from src.environment import Environment
from src.chromosome import Chromosome
from src.fitness import evaluate, repair
from src.genetic_algorithm import GeneticAlgorithm

class StrategyEngineTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment.demo()
        self.rng = np.random.default_rng(42)

    def test_chromosome_shape_and_repair_bounds(self):
        chrom = Chromosome.random(self.env, 3, 8, self.rng)
        chrom.genes[0, 0, 0:2] = [-100, 200]
        chrom.genes[0, 0, 2] = 8
        chrom.genes[0, 0, 3] = -1
        repair(chrom, self.env)
        self.assertEqual(chrom.genes.shape, (3, 8, 4))
        self.assertTrue(np.all((chrom.genes[:, :, 0:2] >= 0) & (chrom.genes[:, :, 0:2] <= self.env.size-1)))
        self.assertTrue(np.all((chrom.genes[:, :, 2] >= .5) & (chrom.genes[:, :, 2] <= 2.0)))
        self.assertTrue(np.all((chrom.genes[:, :, 3] >= 0) & (chrom.genes[:, :, 3] <= 1)))
        for drone in chrom.genes:
            for x, y, *_ in drone:
                self.assertFalse(self.env.is_blocked(x, y))

    def test_fitness_metrics_are_finite(self):
        chrom = repair(Chromosome.random(self.env, 3, 8, self.rng), self.env)
        evaluate(chrom, self.env)
        self.assertTrue(np.isfinite(chrom.fitness))
        for key in ('coverage', 'detection', 'energy', 'time', 'hazard', 'violations'):
            self.assertIn(key, chrom.metrics)
            self.assertTrue(np.isfinite(chrom.metrics[key]))
        self.assertGreaterEqual(chrom.metrics['coverage'], 0)
        self.assertLessEqual(chrom.metrics['coverage'], 1)
        self.assertGreaterEqual(chrom.metrics['detection'], 0)
        self.assertLessEqual(chrom.metrics['detection'], 1)

    def test_ga_is_reproducible_for_fixed_seed(self):
        a = GeneticAlgorithm(self.env, population_size=8, generations=3, seed=7, n_drones=3, n_waypoints=4)
        b = GeneticAlgorithm(self.env, population_size=8, generations=3, seed=7, n_drones=3, n_waypoints=4)
        self.assertAlmostEqual(a.run().fitness, b.run().fitness, places=12)

if __name__ == '__main__':
    unittest.main()
