"""
For each SfM point in reconstruction_aligned.csv, find the nearest LiDAR point
in Titan_data.las using a 3D cKDTree, then write validation.csv with columns:
  x_utm y_utm z_utm nn_dist_m uncertainty_trace_m2
Only processes SfM points that fall inside the LiDAR XY bounding box.
"""

import numpy as np
import laspy
from scipy.spatial import cKDTree
import time, os

LAS_PATH     = '/mnt/d/SFM/wfh/UH_campus/ALS/Titan_data.las'
SFM_CSV      = '/home/ziyue_dang/sfm_uncertainty/output/reconstruction_aligned.csv'
OUT_CSV      = '/home/ziyue_dang/sfm_uncertainty/output/validation.csv'

# ── Load LiDAR ────────────────────────────────────────────────────────────────
print('Loading LiDAR...', flush=True)
t0 = time.time()
las = laspy.read(LAS_PATH)
lidar_pts = np.column_stack([np.array(las.x), np.array(las.y), np.array(las.z)])
print(f'  {len(lidar_pts):,} LiDAR points loaded in {time.time()-t0:.1f}s')
print(f'  X [{lidar_pts[:,0].min():.1f}, {lidar_pts[:,0].max():.1f}]')
print(f'  Y [{lidar_pts[:,1].min():.1f}, {lidar_pts[:,1].max():.1f}]')
print(f'  Z [{lidar_pts[:,2].min():.1f}, {lidar_pts[:,2].max():.1f}]')

# ── Build KD-tree ─────────────────────────────────────────────────────────────
print('Building KD-tree...', flush=True)
t0 = time.time()
tree = cKDTree(lidar_pts)
print(f'  Done in {time.time()-t0:.1f}s')

# ── Load aligned SfM ──────────────────────────────────────────────────────────
print('Loading aligned SfM...', flush=True)
sfm_xyz  = []
sfm_covs = []
with open(SFM_CSV) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        vals = list(map(float, line.split()))
        sfm_xyz.append(vals[:3])
        sfm_covs.append(vals[3:])   # [c00,c01,c02,c11,c12,c22]

sfm_xyz  = np.array(sfm_xyz)   # Nx3
sfm_covs = np.array(sfm_covs)  # Nx6

# Trace of 3x3 cov = cov00 + cov11 + cov22
# Stored order: c00 c01 c02 c11 c12 c22  →  indices 0, 3, 5
uncertainty_trace = sfm_covs[:, 0] + sfm_covs[:, 3] + sfm_covs[:, 5]

print(f'  {len(sfm_xyz):,} SfM points')

# ── Filter SfM to LiDAR XY footprint ─────────────────────────────────────────
xmin, xmax = lidar_pts[:,0].min(), lidar_pts[:,0].max()
ymin, ymax = lidar_pts[:,1].min(), lidar_pts[:,1].max()
in_box = ((sfm_xyz[:,0] >= xmin) & (sfm_xyz[:,0] <= xmax) &
          (sfm_xyz[:,1] >= ymin) & (sfm_xyz[:,1] <= ymax))

sfm_in   = sfm_xyz[in_box]
trace_in = uncertainty_trace[in_box]
print(f'  {in_box.sum():,} SfM points inside LiDAR XY footprint')

# ── Nearest-neighbour query ───────────────────────────────────────────────────
print('Querying nearest neighbours...', flush=True)
t0 = time.time()
nn_dists, _ = tree.query(sfm_in, workers=-1)
print(f'  Done in {time.time()-t0:.1f}s')
print(f'  NN distance: min={nn_dists.min():.3f}  median={np.median(nn_dists):.3f}  '
      f'p90={np.percentile(nn_dists,90):.3f}  max={nn_dists.max():.3f} m')

# ── Write validation CSV ──────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, 'w') as f:
    f.write('# x_utm y_utm z_utm nn_dist_m uncertainty_trace_m2\n')
    for i in range(len(sfm_in)):
        f.write(f'{sfm_in[i,0]:.4f} {sfm_in[i,1]:.4f} {sfm_in[i,2]:.4f} '
                f'{nn_dists[i]:.6f} {trace_in[i]:.6e}\n')

print(f'Wrote {len(sfm_in):,} rows to {OUT_CSV}')
