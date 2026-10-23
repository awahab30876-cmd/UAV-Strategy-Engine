"""Plotting helpers; save publication-ready experiment figures."""
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_environment(ax, env):
    ax.imshow(env.hazard, origin='lower', cmap='YlOrRd', alpha=.7, vmin=0, vmax=1)
    obstacle = np.ma.masked_where(~env.blocked, env.blocked)
    ax.imshow(obstacle, origin='lower', cmap='Greys', alpha=.9, vmin=0, vmax=1)
    yy, xx = np.where(env.priority > 0)
    if len(xx): ax.scatter(xx, yy, s=4, c='cyan', alpha=.25, label='Priority zone')
    tx, ty = zip(*env.targets); ax.scatter(tx, ty, marker='*', s=100, c='lime', edgecolor='black', label='Targets')
    sx, sy = zip(*env.starts); ax.scatter(sx, sy, marker='s', s=55, c='white', edgecolor='black', label='UAV starts')
    ax.set_xlim(-.5, env.size-.5); ax.set_ylim(-.5, env.size-.5); ax.set_xlabel('X cell'); ax.set_ylabel('Y cell')

def save_all(env, best, history, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history['best'], label='Best fitness', linewidth=2)
    ax.plot(history['average'], label='Average fitness', linewidth=2)
    ax.set(title='GA fitness convergence', xlabel='Generation', ylabel='Fitness'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'convergence.png'), dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7)); plot_environment(ax, env)
    colors = ['deepskyblue', 'magenta', 'lime']
    for d, genes in enumerate(best.genes):
        pts = np.vstack([env.starts[d], genes[:, :2]])
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=colors[d % len(colors)], linewidth=2, markersize=3, label=f'UAV {d+1}')
    ax.set_title('Evolved UAV flight paths'); ax.legend(loc='upper left', fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'flight_paths.png'), dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7)); plot_environment(ax, env)
    cov = best.metrics['coverage_map']
    ax.imshow(cov, origin='lower', cmap='viridis', alpha=.75, vmin=0, vmax=1)
    ax.set_title('Evolved sensor coverage heatmap'); fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'coverage_heatmap.png'), dpi=170); plt.close(fig)

def plot_comparison(ga_scores, random_scores, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = [float(np.max(ga_scores)), float(np.max(random_scores))]
    ax.bar(['Genetic Algorithm', 'Random search'], values)
    ax.set_ylabel('Best fitness'); ax.set_title('Best fitness: GA vs random-search baseline'); ax.grid(axis='y', alpha=.25)
    for i, value in enumerate(values): ax.text(i, value, f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'random_vs_ga.png'), dpi=170); plt.close(fig)
