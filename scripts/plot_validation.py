"""
Load validation.csv and produce validation_plot.png:
  - 10 quantile bins of uncertainty (trace)
  - For each bin: mean and median 3D NN distance to LiDAR
  - Spearman correlation between trace and NN distance
  - Scatter density inset
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, binned_statistic
import os

IN_CSV  = '/home/ziyue_dang/sfm_uncertainty/output/validation.csv'
OUT_PNG = '/home/ziyue_dang/sfm_uncertainty/output/validation_plot.png'

# ── Load ──────────────────────────────────────────────────────────────────────
data = []
with open(IN_CSV) as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        data.append(list(map(float, line.split())))
data = np.array(data)

nn_dist = data[:, 3]
trace   = data[:, 4]

print(f'Loaded {len(data):,} points')
print(f'NN dist:  min={nn_dist.min():.3f}  median={np.median(nn_dist):.3f}  '
      f'p90={np.percentile(nn_dist,90):.3f}  max={nn_dist.max():.3f} m')
print(f'Trace:    min={trace.min():.2e}  median={np.median(trace):.2e}  '
      f'max={trace.max():.2e} m²')

# ── Spearman correlation ──────────────────────────────────────────────────────
rho, pval = spearmanr(trace, nn_dist)
print(f'Spearman ρ = {rho:.4f}  p = {pval:.2e}')

# ── Bin by uncertainty quantile ───────────────────────────────────────────────
n_bins = 10
quantile_edges = np.percentile(trace, np.linspace(0, 100, n_bins + 1))
# Deduplicate edges if necessary
quantile_edges = np.unique(quantile_edges)
n_bins = len(quantile_edges) - 1

bin_means,   edges, _ = binned_statistic(trace, nn_dist, statistic='mean',   bins=quantile_edges)
bin_medians, edges, _ = binned_statistic(trace, nn_dist, statistic='median', bins=quantile_edges)
bin_stds,    edges, _ = binned_statistic(trace, nn_dist, statistic='std',    bins=quantile_edges)
bin_counts,  edges, _ = binned_statistic(trace, nn_dist, statistic='count',  bins=quantile_edges)

bin_centers = 0.5 * (edges[:-1] + edges[1:])
bin_stderr  = bin_stds / np.sqrt(np.maximum(bin_counts, 1))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: binned uncertainty vs NN distance
ax = axes[0]
ax.errorbar(np.arange(1, n_bins + 1), bin_means, yerr=bin_stderr,
            fmt='o-', color='steelblue', capsize=4, label='Mean NN dist ± SE')
ax.plot(np.arange(1, n_bins + 1), bin_medians,
        's--', color='darkorange', label='Median NN dist')
ax.set_xlabel('Uncertainty quantile bin (1=lowest, 10=highest)', fontsize=11)
ax.set_ylabel('3D nearest-neighbour distance to LiDAR (m)', fontsize=11)
ax.set_title(f'SfM uncertainty vs LiDAR NN distance\n'
             f'Spearman ρ = {rho:.3f},  p = {pval:.1e}', fontsize=11)
ax.legend(fontsize=10)
ax.set_xticks(range(1, n_bins + 1))
ax.grid(alpha=0.3)

# Right: log-log scatter density
ax2 = axes[1]
# Hexbin for density
valid = (trace > 0) & (nn_dist > 0)
hb = ax2.hexbin(np.log10(trace[valid]), np.log10(nn_dist[valid]),
                gridsize=60, cmap='viridis', mincnt=1)
cb = fig.colorbar(hb, ax=ax2)
cb.set_label('Count')
ax2.set_xlabel('log₁₀(uncertainty trace) [log m²]', fontsize=11)
ax2.set_ylabel('log₁₀(NN dist) [log m]', fontsize=11)
ax2.set_title('Density scatter (log-log scale)', fontsize=11)

# Overlay linear fit in log space
log_t = np.log10(trace[valid])
log_d = np.log10(nn_dist[valid])
coeffs = np.polyfit(log_t, log_d, 1)
x_fit = np.linspace(log_t.min(), log_t.max(), 100)
ax2.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', lw=2,
         label=f'slope={coeffs[0]:.2f}')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f'Saved plot to {OUT_PNG}')
