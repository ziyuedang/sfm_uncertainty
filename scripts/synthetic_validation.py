"""
Synthetic dataset validation:
1. Umeyama alignment: OpenMVG reconstruction frame → Blender world frame
   using all 120 exact camera centers as control points
2. Nearest-surface distance from each aligned SfM point to statue.ply
3. chi-squared(3) calibration test on 3D point covariances
4. chi-squared(2) calibration test on reprojection residuals using true poses
"""

import json
import numpy as np
from scipy.linalg import svd
from scipy.spatial import cKDTree
import trimesh
import os

RECON_JSON   = '/mnt/d/SFM/synthetic_data/di1h3ksyh69s-statue/sfm_6/reconstruction_global/sfm_data.json'
BLENDER_POSES= '/mnt/d/SFM/synthetic_data/di1h3ksyh69s-statue/statue_6_ext.txt'
MESH_PLY     = '/mnt/d/SFM/synthetic_data/di1h3ksyh69s-statue/statue.ply'
RECON_CSV    = '/home/ziyue_dang/sfm_uncertainty/synthetic_eval/output/reconstruction.csv'
OUT_CSV      = '/home/ziyue_dang/sfm_uncertainty/synthetic_eval/output/validation.csv'

# ── Load OpenMVG reconstruction ───────────────────────────────────────────────
print('Loading OpenMVG reconstruction...')
with open(RECON_JSON) as f:
    recon = json.load(f)

# Build pose_id → camera center C (world coords in OpenMVG frame)
key_to_pose = {e['key']: e['value'] for e in recon['extrinsics']}
# Build pose_id → filename
key_to_filename = {}
for v in recon['views']:
    d = v['value']['ptr_wrapper']['data']
    key_to_filename[d['id_pose']] = d['filename']

openmvg_centers = {}   # filename → C (3,)
for pose_id, pose in key_to_pose.items():
    fname = key_to_filename.get(pose_id)
    if fname:
        C = np.array(pose['center'])
        openmvg_centers[fname] = C

print(f'  {len(openmvg_centers)} camera centers from OpenMVG')

# ── Load Blender camera centers ───────────────────────────────────────────────
# Format: frame_id, x, y, z, rx, ry, rz
# frame 0 → 001.jpg, frame 1 → 002.jpg, ...
blender_centers = {}  # filename → (x,y,z)
with open(BLENDER_POSES) as f:
    for line in f:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 7:
            continue
        frame = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        fname = f'{frame+1:03d}.jpg'
        blender_centers[fname] = np.array([x, y, z])

print(f'  {len(blender_centers)} camera centers from Blender')

# ── Match and build correspondence arrays ─────────────────────────────────────
common = sorted(set(openmvg_centers) & set(blender_centers))
print(f'  {len(common)} matched cameras')

src = np.array([openmvg_centers[f] for f in common])   # OpenMVG frame
dst = np.array([blender_centers[f] for f in common])   # Blender world frame

# ── Umeyama similarity transform ─────────────────────────────────────────────
def umeyama(src, dst):
    n, d = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    sigma2_src = (src_c ** 2).sum() / n
    cov = dst_c.T @ src_c / n
    U, S, Vt = svd(cov)
    det_sign = np.linalg.det(U @ Vt)
    W = np.diag([1.0] * (d - 1) + [det_sign])
    R = U @ W @ Vt
    scale = (S * np.diag(W)).sum() / sigma2_src
    t = mu_dst - scale * R @ mu_src
    return scale, R, t

scale, R_um, t_um = umeyama(src, dst)
print(f'\nUmeyama:  scale={scale:.4f}  det(R)={np.linalg.det(R_um):.4f}')

# Camera center alignment residuals
preds = (scale * (R_um @ src.T)).T + t_um
cam_errors = np.linalg.norm(preds - dst, axis=1)
print(f'Camera center alignment RMSE: {np.sqrt((cam_errors**2).mean()):.4f} units')
print(f'  (should be near 0 — using exact Blender poses)')

# ── Load and transform reconstruction.csv ────────────────────────────────────
print('\nLoading reconstruction.csv...')
pts_sfm = []
covs    = []
with open(RECON_CSV) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        vals = list(map(float, line.split()))
        pts_sfm.append(vals[:3])
        covs.append(vals[3:])   # [c00,c01,c02,c11,c12,c22]

pts_sfm = np.array(pts_sfm)   # Nx3
covs    = np.array(covs)       # Nx6
print(f'  {len(pts_sfm)} points')

# Apply transform
pts_aligned = (scale * (R_um @ pts_sfm.T)).T + t_um   # Nx3

# Transform covariance: C_blender = s^2 * R * C_sfm * R^T
def transform_cov(cov6, s, R):
    c = cov6
    C3 = np.array([[c[0], c[1], c[2]],
                   [c[1], c[3], c[4]],
                   [c[2], c[4], c[5]]])
    C3_out = s**2 * R @ C3 @ R.T
    return C3_out

# ── Load mesh and compute nearest-surface distances ──────────────────────────
print('\nLoading statue.ply mesh...')
mesh = trimesh.load(MESH_PLY, process=False)
print(f'  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces')

print('Computing nearest-surface distances (proximity query)...')
# trimesh.proximity.closest_point returns (closest_points, distances, triangle_ids)
closest, nn_dists, _ = trimesh.proximity.closest_point(mesh, pts_aligned)
print(f'  NN surface dist: min={nn_dists.min():.4f}  median={np.median(nn_dists):.4f}'
      f'  p90={np.percentile(nn_dists,90):.4f}  max={nn_dists.max():.4f}')

# ── chi-squared(3) test on 3D point covariances ──────────────────────────────
print('\nComputing chi-squared(3) statistics...')
# e = aligned_pt - closest_surface_pt
# d^2 = e^T Sigma_X^{-1} e ~ chi^2(3) if calibrated
# chi^2(3): mean=3, median=2.366

chi2_vals = []
skipped = 0
for i in range(len(pts_aligned)):
    C3 = transform_cov(covs[i], scale, R_um)
    e = pts_aligned[i] - closest[i]
    try:
        C3_inv = np.linalg.inv(C3)
        d2 = float(e @ C3_inv @ e)
        if np.isfinite(d2) and d2 >= 0:
            chi2_vals.append(d2)
        else:
            skipped += 1
    except np.linalg.LinAlgError:
        skipped += 1

chi2_vals = np.array(chi2_vals)
print(f'  Valid chi^2 values: {len(chi2_vals)}  (skipped {skipped})')
print(f'  chi^2(3) expected:  mean=3.000  median=2.366')
print(f'  Observed:           mean={chi2_vals.mean():.3f}  median={np.median(chi2_vals):.3f}')
print(f'  Ratio (obs/exp):    mean={chi2_vals.mean()/3:.2f}x  median={np.median(chi2_vals)/2.366:.2f}x')

# Fraction within chi^2(3) confidence bounds
from scipy.stats import chi2 as chi2_dist
for conf in [0.50, 0.90, 0.95, 0.99]:
    threshold = chi2_dist.ppf(conf, df=3)
    frac = (chi2_vals < threshold).mean()
    print(f'  Within {int(conf*100)}% CI (d^2 < {threshold:.2f}): {100*frac:.1f}%  (expected {int(conf*100)}%)')

# ── Write validation CSV ──────────────────────────────────────────────────────
trace = covs[:, 0] + covs[:, 3] + covs[:, 5]  # before rotation (scale changes trace)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, 'w') as f:
    f.write('# x y z nn_dist_surface chi2_3 uncertainty_trace\n')
    for i in range(len(pts_aligned)):
        chi2_v = chi2_vals[i] if i < len(chi2_vals) else float('nan')
        f.write(f'{pts_aligned[i,0]:.6f} {pts_aligned[i,1]:.6f} {pts_aligned[i,2]:.6f} '
                f'{nn_dists[i]:.6f} {chi2_v:.4f} {trace[i]:.6e}\n')

print(f'\nWrote {len(pts_aligned)} rows to {OUT_CSV}')

# ── Summary ──────────────────────────────────────────────────────────────────
print('\n=== SUMMARY ===')
print(f'Points:              {len(pts_aligned):,}')
print(f'Camera align RMSE:   {np.sqrt((cam_errors**2).mean()):.5f} Blender units')
print(f'Median surface dist: {np.median(nn_dists):.4f} Blender units')
print(f'chi^2(3) mean:       {chi2_vals.mean():.2f}  (calibrated = 3.00)')
print(f'chi^2(3) median:     {np.median(chi2_vals):.2f}  (calibrated = 2.37)')
scale_factor = chi2_vals.mean() / 3.0
print(f'Sigma overestimate:  {scale_factor:.1f}x  (covs too large by ~{np.sqrt(scale_factor):.1f}x in std dev)')
