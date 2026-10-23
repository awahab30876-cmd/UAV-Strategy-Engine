"""Synthetic 2-D disaster environment for UAV strategy experiments."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Environment:
    size: int
    blocked: np.ndarray
    hazard: np.ndarray
    priority: np.ndarray
    targets: list
    starts: list

    @classmethod
    def demo(cls, size=50, seed=17):
        rng = np.random.default_rng(seed)
        blocked = np.zeros((size, size), dtype=bool)
        hazard = np.zeros((size, size), dtype=float)
        priority = np.zeros((size, size), dtype=float)
        # Rectangular obstacles; leave corridors through the map.
        rectangles = [(8, 7, 14, 18), (22, 3, 27, 13), (32, 17, 38, 29),
                      (12, 29, 18, 40), (24, 36, 31, 43), (40, 5, 44, 12)]
        for x1, y1, x2, y2 in rectangles:
            blocked[y1:y2, x1:x2] = True
        # Hazard pockets are smooth-ish circular fields, with intensity 0..1.
        yy, xx = np.mgrid[0:size, 0:size]
        for cx, cy, radius, intensity in [(19, 22, 7, .95), (37, 36, 6, .8), (7, 42, 5, .65)]:
            d = np.sqrt((xx-cx)**2 + (yy-cy)**2)
            hazard = np.maximum(hazard, intensity * np.clip(1 - d/radius, 0, 1))
        # Priority zones (weights 1..3) and target locations.
        for cx, cy, radius, weight in [(6, 25, 5, 2.0), (28, 23, 6, 3.0), (43, 40, 5, 2.5), (40, 10, 4, 1.5)]:
            d = np.sqrt((xx-cx)**2 + (yy-cy)**2)
            priority = np.maximum(priority, np.where(d <= radius, weight, 0.0))
        targets = [(6, 25), (28, 23), (43, 40), (40, 10), (20, 45), (3, 8)]
        starts = [(2, 2), (47, 2), (2, 47)]
        # Ensure starts are clear.
        for x, y in starts:
            blocked[y, x] = False
        return cls(size, blocked, hazard, priority, targets, starts)

    def is_blocked(self, x, y):
        xi, yi = int(np.clip(round(x), 0, self.size-1)), int(np.clip(round(y), 0, self.size-1))
        return bool(self.blocked[yi, xi])

    def nearest_free(self, x, y):
        xi, yi = int(np.clip(round(x), 0, self.size-1)), int(np.clip(round(y), 0, self.size-1))
        if not self.blocked[yi, xi]:
            return float(xi), float(yi)
        free_y, free_x = np.where(~self.blocked)
        idx = np.argmin((free_x-x)**2 + (free_y-y)**2)
        return float(free_x[idx]), float(free_y[idx])

    def segment_samples(self, a, b, count=None):
        dist = float(np.linalg.norm(np.asarray(b)-np.asarray(a)))
        n = count or max(2, min(18, int(dist/3)+2))
        return [(a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t) for t in np.linspace(0, 1, n)]
