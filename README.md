# UAV Strategy Engine — Genetic Algorithm Simulation

A modular, reproducible Python simulation for planning a fleet of three UAVs over a synthetic 50 × 50 disaster grid. The Genetic Algorithm (GA) evolves waypoint locations, relative speed, and sensor power while balancing priority-zone coverage, target detection, energy, time, and hazard exposure.

> **Scope:** This is an educational simulation only. The grid, target locations, hazard map, energy model, and detection model are synthetic and are not suitable for controlling real aircraft or making safety-critical decisions.

## Features
- Hybrid real-valued chromosome: `[x, y, speed, sensor_power]` per waypoint.
- Tournament selection, segment/whole-drone crossover, Gaussian spatial mutation, elitism, and early stopping.
- Constraint repair for map bounds, blocked waypoint cells, speed/sensor bounds, and approximate maximum path range; path segments crossing obstacles are penalized.
- Weighted multi-objective fitness function.
- Convergence curve, evolved flight paths, sensor coverage heatmap, and GA-vs-random-search plot.
- Hyperparameter sensitivity experiment for population size, mutation rate, and tournament size.

## Requirements
- Python 3.10+
- NumPy
- Matplotlib

## Run the main simulation
From the repository root:

```bash
python -m pip install -r requirements.txt
python main.py
```

Generated files are saved to `outputs/`:
- `convergence.png`
- `flight_paths.png`
- `coverage_heatmap.png`
- `random_vs_ga.png`
- `results.json`

## Run sensitivity analysis
```bash
python experiments/sensitivity_analysis.py
```
This writes `outputs/sensitivity_results.csv`. The experiment runs nine configurations with 12 generations each; it can take longer than the main demo.

## Project layout
```text
src/environment.py          Synthetic grid and hazard generation
src/chromosome.py           Chromosome data structure and initialization
src/fitness.py              Fitness, repair, and constraint penalties
src/genetic_algorithm.py    GA operators and termination criteria
src/visualizer.py            Plot generation
main.py                     Main demo and random-search comparison
experiments/sensitivity_analysis.py
report.md                   Technical report draft
outputs/                    Generated plots and experiment data
```

## Model details
For a chromosome gene `g = (x, y, v, s)`, `x,y` are grid coordinates, `v` is relative speed in `[0.5, 2.0]`, and `s` is sensor power in `[0,1]`. The initial scalar objective is:

`F = 0.35 C + 0.25 D - 0.15 E_norm - 0.10 T_norm - 0.15 H_norm - P`

where `C` is priority-weighted coverage, `D` is mean target-detection score, `E_norm` is normalized energy, `T_norm` is normalized flight time, `H_norm` is normalized hazard exposure, and `P` penalizes obstacle/range violations. This weighted sum is a practical single-score approximation; it does not guarantee a full Pareto front.

## Reproducibility and limitations
Fixed seeds make the demo reproducible in the same software environment. Results can vary slightly across library versions. The simplified sensor model uses circular waypoint footprints; the model does not include vehicle dynamics, wind vectors, communications, collision avoidance between UAVs, or certified path planning. The segment penalty is a soft barrier, not a proof that every path is obstacle-free. The GA-vs-random baseline uses approximately matched candidate-evaluation budgets.

## GitHub upload
Create an empty repository on GitHub, then from this folder run:

```bash
git init
git add .
git commit -m "Add UAV Strategy Engine GA simulation"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/UAV-Strategy-Engine.git
git push -u origin main
```
Replace the URL with your own repository URL. Do not commit virtual environments or personal credentials.

## Run verification tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py src experiments tests create_report_pdf.py
```

The included tests check chromosome shape and repaired gene bounds, finite fitness metrics, and repeatability with a fixed seed. The main experiment uses an approximately matched fitness-evaluation budget for GA and random search. Obstacle crossings are penalized using dense, vectorized segment sampling; this remains a simulation penalty rather than a certified collision-free route guarantee.
