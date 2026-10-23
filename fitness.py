"""Fitness evaluation and repair for the synthetic UAV mission."""
import numpy as np

WEIGHTS = {'coverage': .35, 'detection': .25, 'energy': .15, 'time': .10, 'hazard': .15}

def repair(chrom, env, max_range=105.0):
    g = chrom.genes
    g[:, :, 0] = np.clip(g[:, :, 0], 0, env.size-1)
    g[:, :, 1] = np.clip(g[:, :, 1], 0, env.size-1)
    g[:, :, 2] = np.clip(g[:, :, 2], .5, 2.0)
    g[:, :, 3] = np.clip(g[:, :, 3], 0.0, 1.0)
    for d in range(g.shape[0]):
        for i in range(g.shape[1]):
            g[d, i, 0], g[d, i, 1] = env.nearest_free(g[d, i, 0], g[d, i, 1])
        # If path exceeds range, pull later waypoints toward start to enforce feasibility.
        points = [env.starts[d]] + [(x, y) for x, y in g[d, :, :2]]
        length = sum(np.linalg.norm(np.array(points[i+1])-points[i]) for i in range(len(points)-1))
        if length > max_range:
            scale = max_range / length
            start = np.array(env.starts[d], dtype=float)
            for i in range(g.shape[1]):
                p = start + (g[d, i, :2]-start)*scale
                g[d, i, 0], g[d, i, 1] = env.nearest_free(*p)
    return chrom

def evaluate(chrom, env, max_range=105.0, weights=None):
    w = weights or WEIGHTS
    g = chrom.genes
    coverage = np.zeros((env.size, env.size), dtype=float)
    energy = total_time = hazard_cost = violation = 0.0
    all_path_lengths = []
    for d in range(g.shape[0]):
        prev = np.array(env.starts[d], dtype=float)
        length = 0.0
        route = [(float(prev[0]), float(prev[1]))] + [(float(p[0]), float(p[1])) for p in g[d, :, :2]]
        for i, gene in enumerate(g[d]):
            point = gene[:2]
            speed, sensor = float(gene[2]), float(gene[3])
            dist = float(np.linalg.norm(point-prev))
            length += dist
            total_time += dist / max(speed, .5)
            energy += dist*(.65 + .35*speed*speed) + sensor*2.2
            # Exposure sampled along each segment; blocked cells are hard violations.
            samples = env.segment_samples(prev, point)
            for sx, sy in samples:
                xi, yi = int(np.clip(round(sx), 0, env.size-1)), int(np.clip(round(sy), 0, env.size-1))
                hazard_cost += float(env.hazard[yi, xi]) / max(len(samples), 1) * max(dist, 1.0)
            # Use dense, vectorized half-cell sampling for obstacle crossings. Hazard
            # integration remains coarser for runtime, but narrow blocked segments
            # should not be missed merely because of sparse hazard samples.
            dense = env.segment_samples(prev, point, count=max(2, int(dist * 2) + 2))
            dx = np.clip(np.rint([p[0] for p in dense]).astype(int), 0, env.size-1)
            dy = np.clip(np.rint([p[1] for p in dense]).astype(int), 0, env.size-1)
            violation += 2.0 * float(np.count_nonzero(env.blocked[dy, dx]))
            # Sensor footprint, radius depends mildly on power.
            radius = 2.0 + 2.2*sensor
            x0, y0 = int(max(0, point[0]-radius)), int(max(0, point[1]-radius))
            x1, y1 = int(min(env.size, point[0]+radius+1)), int(min(env.size, point[1]+radius+1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            mask = (xx-point[0])**2 + (yy-point[1])**2 <= radius**2
            coverage[y0:y1, x0:x1] = np.maximum(coverage[y0:y1, x0:x1], mask.astype(float)*sensor)
            prev = point
        all_path_lengths.append(length)
        if length > max_range:
            violation += (length-max_range)*2
    # Priority-weighted covered area, scaled to [0,1].
    weights_map = 1.0 + env.priority
    denom = weights_map.sum()
    coverage_score = float((np.minimum(coverage, 1.0)*weights_map).sum()/denom)
    # Detection is a smooth probability for each target based on nearest sensor waypoint.
    detected = []
    for tx, ty in env.targets:
        best = 0.0
        for drone in g:
            for gene in drone:
                dist = np.linalg.norm(gene[:2]-np.array([tx, ty]))
                radius = 2.0 + 2.2*gene[3]
                probability = float(gene[3] * np.clip((radius+1.5-dist)/1.5, 0, 1))
                best = max(best, probability)
        detected.append(best)
    detection_score = float(np.mean(detected))
    # Normalize costs to comparable 0..1-ish ranges.
    energy_n = float(np.clip(energy/(g.shape[0]*max_range*2.8), 0, 1.5))
    time_n = float(np.clip(total_time/(g.shape[0]*max_range/.5), 0, 1.5))
    hazard_n = float(np.clip(hazard_cost/(g.shape[0]*max_range), 0, 1.5))
    penalty = 0.08*violation
    fitness = (w['coverage']*coverage_score + w['detection']*detection_score
               - w['energy']*energy_n - w['time']*time_n - w['hazard']*hazard_n - penalty)
    metrics = {'coverage': coverage_score, 'detection': detection_score, 'energy': energy,
               'energy_norm': energy_n, 'time': total_time, 'time_norm': time_n,
               'hazard': hazard_cost, 'hazard_norm': hazard_n, 'violations': violation,
               'fitness': float(fitness), 'coverage_map': coverage, 'path_lengths': all_path_lengths}
    chrom.fitness, chrom.metrics = float(fitness), metrics
    return chrom
