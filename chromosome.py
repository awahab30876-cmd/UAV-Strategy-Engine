"""Hybrid chromosome: per waypoint [x, y, speed, sensor_power]."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Chromosome:
    genes: np.ndarray  # shape: (drones, waypoints, 4)
    fitness: float = float('-inf')
    metrics: dict | None = None

    def copy(self):
        return Chromosome(self.genes.copy(), self.fitness, None if self.metrics is None else dict(self.metrics))

    @classmethod
    def random(cls, env, n_drones, n_waypoints, rng):
        g = np.zeros((n_drones, n_waypoints, 4), dtype=float)
        for d in range(n_drones):
            g[d, :, 0] = rng.uniform(0, env.size-1, n_waypoints)
            g[d, :, 1] = rng.uniform(0, env.size-1, n_waypoints)
            g[d, :, 2] = rng.uniform(.7, 1.8, n_waypoints)  # relative speed
            g[d, :, 3] = rng.uniform(.25, 1.0, n_waypoints) # sensor power
        return cls(g)
