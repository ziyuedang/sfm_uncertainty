"""
GCP-based geo-registration:
1. Triangulate each GCP in the OpenMVG SfM frame using pixel observations
   from the old project (full-res 8984x6732 → halved to match our 4492x3366)
2. Run Umeyama similarity transform: SfM_positions → UTM_coords
3. Apply transform to reconstruction_filtered.csv → reconstruction_aligned.csv
"""

import json
import numpy as np
from scipy.linalg import svd

# ── Camera intrinsics (half-resolution) ─────────────────────────────────────
f  = 5856.8
cx = 2243.49
cy = 1690.065
K  = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

# ── Load camera poses from our reconstruction ────────────────────────────────
with open('/home/ziyue_dang/sfm_uncertainty/openmvg_project/reconstruction/sfm_data_recon.json') as fh:
    recon = json.load(fh)

# filename → (R: 3x3, C: 3-vec) where R maps world→camera, C is camera center
cam_poses = {}
# Build key → pose
key_to_pose = {e['key']: e['value'] for e in recon['extrinsics']}
# Build key → filename
key_to_filename = {}
for v in recon['views']:
    vdata = v['value']['ptr_wrapper']['data']
    key_to_filename[vdata['id_pose']] = vdata['filename']

for pose_key, pose in key_to_pose.items():
    fname = key_to_filename.get(pose_key)
    if fname:
        R = np.array(pose['rotation'])  # 3x3, world→camera
        C = np.array(pose['center'])    # camera center in world
        cam_poses[fname] = (R, C)

print("Loaded camera poses:", sorted(cam_poses.keys()))

# ── GCP surveyed UTM coordinates ─────────────────────────────────────────────
# id → [X_utm, Y_utm, Z]
gcp_utm = {
    1:  [273584.6365, 3289693.1729, -2.3857],
    2:  [273597.4585, 3289611.8309, -17.4768],
    3:  [273575.4365, 3289504.4108, -17.1878],
    4:  [273428.6205, 3289500.8928, -16.9188],
    5:  [273469.7455, 3289614.7669, -17.4067],
    6:  [273281.6234, 3289531.3188, -17.5937],
    7:  [273263.2784, 3289628.8819, -17.5917],
    8:  [273243.1864, 3289764.0179, -17.3767],
    9:  [273380.3984, 3289766.6259, -16.5437],
    10: [273416.0504, 3289693.3469, -17.7427],
    11: [273564.8045, 3289773.9039, -2.5027],
}

# ── GCP pixel observations (full-res → halved) ───────────────────────────────
# Format: gcp_id → list of (filename, u_halfres, v_halfres)
gcp_pixels = {
    1:  [('168.JPG', 696/2, 804/2), ('169.JPG', 647/2, 6294/2),
         ('178.JPG', 7224/2, 2483/2), ('31.JPG', 5126/2, 3197/2),
         ('48.JPG', 711/2, 1574/2)],
    2:  [('11.JPG', 7527/2, 717/2), ('12.JPG', 7455/2, 5358/2),
         ('168.JPG', 2560/2, 522/2), ('169.JPG', 2578/2, 5891/2),
         ('178.JPG', 7471/2, 4125/2), ('31.JPG', 6954/2, 2946/2)],
    3:  [('11.JPG', 5341/2, 252/2), ('12.JPG', 5201/2, 4949/2),
         ('168.JPG', 4912/2, 854/2), ('169.JPG', 4965/2, 6317/2),
         ('177.JPG', 8374/2, 916/2), ('178.JPG', 7153/2, 6312/2)],
    4:  [('12.JPG', 5049/2, 1915/2), ('13.JPG', 4318/2, 6435/2),
         ('168.JPG', 5230/2, 4003/2), ('177.JPG', 5373/2, 1149/2),
         ('178.JPG', 4197/2, 6537/2), ('30.JPG', 8531/2, 1344/2)],
    5:  [('12.JPG', 7425/2, 2700/2), ('168.JPG', 2670/2, 3281/2),
         ('178.JPG', 4940/2, 4163/2), ('30.JPG', 6024/2, 429/2),
         ('31.JPG', 6916/2, 5854/2)],
    6:  [('13.JPG', 4915/2, 3346/2), ('167.JPG', 4682/2, 1953/2),
         ('177.JPG', 2397/2, 708/2), ('178.JPG', 1123/2, 6052/2),
         ('30.JPG', 7910/2, 4594/2)],
    7:  [('13.JPG', 6925/2, 2942/2), ('167.JPG', 2618/2, 2600/2),
         ('178.JPG', 695/2, 4038/2), ('30.JPG', 5722/2, 4999/2)],
    8:  [('178.JPG', 223/2, 1297/2), ('179.JPG', 350/2, 6096/2),
         ('30.JPG', 2647/2, 5447/2), ('50.JPG', 1191/2, 5133/2)],
    9:  [('178.JPG', 3033/2, 1178/2), ('179.JPG', 3192/2, 6094/2),
         ('30.JPG', 2639/2, 2340/2), ('49.JPG', 1170/2, 2563/2)],
    10: [('168.JPG', 985/2, 4597/2), ('30.JPG', 4287/2, 1576/2)],
    11: [('178.JPG', 6747/2, 880/2), ('179.JPG', 7003/2, 6064/2),
         ('31.JPG', 3272/2, 3646/2), ('48.JPG', 2360/2, 1082/2),
         ('49.JPG', 1394/2, 6504/2)],
}

# ── Linear triangulation (DLT) ────────────────────────────────────────────────
def triangulate_dlt(observations):
    """
    observations: list of (R, C, u, v) where
      R: 3x3 world→camera rotation, C: camera center, u,v: pixel coords
    Returns homogeneous 3D point (inhomogeneous).
    """
    rows = []
    for R, C, u, v in observations:
        t = -R @ C
        P = K @ np.hstack([R, t.reshape(3, 1)])  # 3x4 camera matrix
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.array(rows)
    _, _, Vt = svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

# ── Triangulate all GCPs ──────────────────────────────────────────────────────
sfm_pts = []
utm_pts = []
used_ids = []

for gcp_id, pixels in sorted(gcp_pixels.items()):
    obs = []
    for fname, u, v in pixels:
        if fname not in cam_poses:
            print(f"  GCP {gcp_id}: no pose for {fname}, skipping")
            continue
        R, C = cam_poses[fname]
        obs.append((R, C, u, v))

    if len(obs) < 2:
        print(f"GCP {gcp_id}: only {len(obs)} valid views, skipping")
        continue

    pt_sfm = triangulate_dlt(obs)
    pt_utm = np.array(gcp_utm[gcp_id])
    sfm_pts.append(pt_sfm)
    utm_pts.append(pt_utm)
    used_ids.append(gcp_id)
    print(f"GCP {gcp_id}: SfM {pt_sfm} | UTM {pt_utm[:2]}")

sfm_pts = np.array(sfm_pts)   # Nx3
utm_pts = np.array(utm_pts)   # Nx3

# Exclude GCPs 1 and 11: UTM Z ≈ -2.5m while all others are at -17m,
# but their triangulated SfM Z is identical to the ground-level GCPs (~1.23).
# They're on elevated structures (rooftops) — aerial SfM can't recover their
# height difference, so including them corrupts the Z component of Umeyama.
exclude_ids = {1, 11}
mask = np.array([gid not in exclude_ids for gid in used_ids])
sfm_pts_fit = sfm_pts[mask]
utm_pts_fit = utm_pts[mask]
print(f"\nFitting Umeyama with {mask.sum()} GCPs (excluding {exclude_ids})")

# ── Umeyama similarity transform (7-DOF) ─────────────────────────────────────
# Finds s, R, t such that utm ≈ s*R*sfm + t (minimise MSE)
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

scale, R_um, t_um = umeyama(sfm_pts_fit, utm_pts_fit)
print(f"\nUmeyama result:")
print(f"  Scale:       {scale:.4f} m/unit")
print(f"  Rotation:\n{R_um}")
print(f"  Translation: {t_um}")

# ── Compute GCP RMSE ──────────────────────────────────────────────────────────
residuals = []
for i, gid in enumerate(used_ids):
    if gid in exclude_ids:
        continue
    pred = scale * R_um @ sfm_pts[i] + t_um
    err  = np.linalg.norm(pred - utm_pts[i])
    residuals.append(err)
    print(f"  GCP {gid}: error {err:.4f} m")

rmse = np.sqrt(np.mean(np.array(residuals)**2))
print(f"\nGCP RMSE: {rmse:.4f} m  ({len(residuals)} GCPs used for fit)")

# ── Apply transform to reconstruction_filtered.csv ───────────────────────────
import csv, os

in_csv  = '/home/ziyue_dang/sfm_uncertainty/output/reconstruction_filtered.csv'
out_csv = '/home/ziyue_dang/sfm_uncertainty/output/reconstruction_aligned.csv'

pts_sfm = []
covs    = []

with open(in_csv) as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        vals = list(map(float, line.split()))
        pts_sfm.append(vals[:3])
        covs.append(vals[3:])

pts_sfm = np.array(pts_sfm)  # Nx3

# Transform positions
pts_utm_out = (scale * (R_um @ pts_sfm.T)).T + t_um  # Nx3

# Transform covariance matrices: Cov_utm = s^2 * R * Cov_sfm * R^T
# Stored as upper triangle: [c00, c01, c02, c11, c12, c22]
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, 'w') as fh:
    fh.write('# x_utm y_utm z_utm cov00 cov01 cov02 cov11 cov12 cov22\n')
    for i, (xyz, cov) in enumerate(zip(pts_utm_out, covs)):
        # Reconstruct 3x3 cov from upper triangle
        C3 = np.array([
            [cov[0], cov[1], cov[2]],
            [cov[1], cov[3], cov[4]],
            [cov[2], cov[4], cov[5]]
        ])
        C3_utm = scale**2 * R_um @ C3 @ R_um.T
        fh.write(
            f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
            f"{C3_utm[0,0]:.6e} {C3_utm[0,1]:.6e} {C3_utm[0,2]:.6e} "
            f"{C3_utm[1,1]:.6e} {C3_utm[1,2]:.6e} {C3_utm[2,2]:.6e}\n"
        )

print(f"\nWrote {len(pts_utm_out)} aligned points to {out_csv}")

# Quick sanity check: range of aligned coords
print(f"X range: [{pts_utm_out[:,0].min():.1f}, {pts_utm_out[:,0].max():.1f}]")
print(f"Y range: [{pts_utm_out[:,1].min():.1f}, {pts_utm_out[:,1].max():.1f}]")
print(f"Z range: [{pts_utm_out[:,2].min():.1f}, {pts_utm_out[:,2].max():.1f}]")
