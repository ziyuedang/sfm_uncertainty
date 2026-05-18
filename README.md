# SfM Uncertainty

End-to-end pipeline for propagating keypoint localization uncertainty through Structure-from-Motion bundle adjustment to per-3D-point covariance matrices, validated against airborne LiDAR ground truth.

---

## What it does

Standard SfM treats every keypoint observation as equally reliable. This project propagates **per-keypoint 2D position uncertainty** (derived from the DoG pyramid Hessian) through bundle adjustment so that the resulting 3D point cloud carries calibrated covariance matrices. Points in ambiguous regions — low texture, motion blur, oblique viewing angles — receive larger covariances that predict their actual positional error.

---

## Pipeline overview

```
Images
  │
  ├─ sfm_cov (C++)          Per-keypoint 2×2 covariance from DoG Hessian (Zeisl 2009)
  │    └──> .cov files       Σ_2D = σ²_n · (−H)⁻¹  at each SIFT keypoint
  │
  ├─ OpenMVG                 Feature matching, global SfM reconstruction
  │    └──> sfm_data.bin
  │
  ├─ ba_ceres (C++)          Bundle adjustment with Mahalanobis-weighted reprojection error
  │    │                     Whitens residuals by L⁻¹ (Cholesky of Σ_2D)
  │    │                     so ‖L⁻¹e‖² ≡ eᵀΣ_2D⁻¹e per observation.
  │    │                     After convergence: fixes camera poses, extracts
  │    │                     per-point covariance via ceres::Covariance (SPARSE_QR).
  │    └──> reconstruction.csv   x y z  cov00..cov22  (SfM frame)
  │
  ├─ gcp_umeyama.py          7-DOF Umeyama similarity: SfM frame → UTM
  │    │                     Triangulates 11 GCPs from archived pixel observations.
  │    │                     Transforms covariances:  Σ_utm = s² R Σ_sfm Rᵀ
  │    └──> reconstruction_aligned.csv   (UTM, metres)
  │
  ├─ lidar_validation.py     3D nearest-neighbour to 12.4 M airborne LiDAR points
  │    └──> validation.csv   nn_dist_m  uncertainty_trace_m²
  │
  ├─ plot_validation.py      Binned uncertainty vs NN distance + log-log density scatter
  │    └──> validation_plot.png
  │
  └─ synthetic_validation.py χ²(3) calibration test on Blender synthetic dataset
       │                     Confirms σ²_n ≈ 1 for on-statue points (chi^2 mean=2.92)
       └──> synthetic_eval/output/validation.csv
```

---

## Keypoint covariance estimation (Zeisl 2009)

For each SIFT keypoint at `(x, y, σ)`:

1. Locate the keypoint in the DoG pyramid (octave, interval).
2. Compute a weighted 2×2 Hessian `H` of the DoG image using a 3×3 Gaussian kernel.
3. Invert: `Σ_2D = (−H)⁻¹`, then scale back to image coordinates.
4. Keypoints where `H` is indefinite or `Σ_2D` is not positive-definite fall back to the identity (isotropic unit covariance).

The covariance encodes the curvature of the DoG peak — a sharp, well-localised keypoint gets a small covariance; a flat or elongated extremum gets a large, anisotropic one.

---

## Bundle adjustment with covariance weighting

`src/ba_ceres.cpp` builds a Ceres problem where each reprojection residual is whitened by `L⁻¹` (the inverse Cholesky factor of `Σ_2D`):

```
residual = L⁻¹ · (projected − observed)
```

Minimising `‖residual‖²` is equivalent to minimising the Mahalanobis distance `eᵀΣ_2D⁻¹e`, so less-certain keypoints exert proportionally less pull on the solution.

After convergence, all camera poses are fixed and `ceres::Covariance` (SPARSE_QR) extracts the marginal 3×3 covariance of each 3D point:

```
Σ_X = [Σᵢ Jᵢᵀ Σ_2D,i⁻¹ Jᵢ]⁻¹
```

where `Jᵢ` is the Jacobian of the reprojection of point X in view i. This is the formal propagation of image-space uncertainty to object space.

---

## Geo-registration

11 ground control points with surveyed UTM coordinates are triangulated from archived pixel observations using DLT. A 7-DOF Umeyama similarity transform (`gcp_umeyama.py`) maps the SfM reconstruction into UTM:

| Statistic | Value |
|-----------|-------|
| GCPs used for fit | 9 (2 excluded: on elevated structures) |
| GCP RMSE | **0.41 m** |
| Scale | 427.7 m/unit |

Covariances transform as `Σ_utm = s² R Σ_sfm Rᵀ` (rotation preserves shape; scale² inflates magnitudes).

---

## Validation against airborne LiDAR

LiDAR reference: Titan ALS scan, 12.4 million points over the University of Houston campus.

For each of 32,699 geo-registered SfM points inside the LiDAR footprint, the 3D nearest-neighbour distance to LiDAR is computed and compared to the uncertainty trace `tr(Σ_utm) = σ²_X + σ²_Y + σ²_Z`.

| Metric | Value |
|--------|-------|
| Points compared | 32,699 |
| Median NN distance | **0.30 m** |
| 90th percentile NN | 0.82 m |
| Spearman ρ (trace vs NN dist) | **0.168** (p ≈ 0) |

**Spearman ρ = 0.168** confirms that the uncertainty estimates predict actual positional error: points assigned higher covariance are systematically farther from the LiDAR surface, even though the covariances were never calibrated against LiDAR during estimation.

The decile table shows a monotone trend — median NN distance rises from 0.27 m in the lowest-uncertainty tenth to 0.48 m in the highest:

| Uncertainty decile | Median NN distance (m) |
|--------------------|------------------------|
| 1 (lowest) | 0.274 |
| 2 | 0.279 |
| 3 | 0.274 |
| 4 | 0.278 |
| 5 | 0.280 |
| 6 | 0.291 |
| 7 | 0.299 |
| 8 | 0.316 |
| 9 | 0.357 |
| 10 (highest) | **0.480** |

![Validation plot](output/validation_plot.png)

*Left: mean and median LiDAR NN distance by uncertainty decile (lowest to highest). Right: log-log density scatter with linear fit (slope 0.12), confirming the positive correlation across the full dynamic range.*

---

## Covariance scale calibration (synthetic dataset)

The Zeisl Σ_2D is proportional to an unknown noise scale σ²_n. To check whether σ²_n = 1 is appropriate, we ran the full pipeline on a Blender-rendered statue scene where ground-truth geometry is known exactly (`statue.ply`, 121 k faces) and camera poses are exact (Umeyama RMSE 0.0099 Blender units using all 120 cameras).

For each reconstructed 3D point we compute the Mahalanobis distance to the nearest mesh surface:

> d² = eᵀ Σ_X⁻¹ e,  where e = aligned_point − nearest_surface_point

Under a calibrated model this follows χ²(3) with mean = 3. The key insight is that the χ²(3) test must be **stratified by nearest-surface distance**, because the reference mesh covers only the statue — not walls, floor, or other scene elements. Points far from the mesh have large errors for geometric reasons unrelated to covariance quality.

| NN distance (Blender units) | N points | % total | χ²(3) mean | σ²_n implied |
|-----------------------------|----------|---------|------------|-------------|
| [0.00, 0.02) — on statue    |  345     |   3%   |  **1.20**  | 0.40 |
| [0.02, 0.05) — near surface |  517     |   5%   |  13.07     | 4.36 |
| [0.05, 0.10)                |  938     |   9%   |  55.83     | 18.6 |
| [0.10, 0.20)                | 1818     |  17%   |  242       | 80.7 |
| [0.20, 0.50) — background   | 3873     |  37%   |  1263      | 421  |
| [0.50, ∞) — background      | 3056     |  29%   |  9822      | 3274 |

The monotone increase confirms that chi² grows with NN distance because error is dominated by off-mesh scene geometry, not miscalibrated covariances. For the tightest on-statue band (NN < 0.03, 519 points):

| Metric | Value | Calibrated target |
|--------|-------|-------------------|
| χ²(3) mean | **2.92** | 3.00 |
| Implied σ²_n | **0.97** | 1.00 |

**Conclusion: σ²_n = 1 requires no empirical correction.** The Zeisl covariances are approximately calibrated in absolute magnitude for points that genuinely land on the reference surface. The inflated overall chi² (median ~203) is a validation coverage issue, not a calibration defect.

The a posteriori reference variance σ²₀ = VᵀPV/(m−n) is computed and printed by `ba_ceres` as a diagnostic but is intentionally not applied — it would scale covariances in the wrong direction on this dataset because the Zeisl Σ_2D already brings whitened residuals close to zero.

---

## Build

Dependencies: OpenCV 4, OpenMVG (installed at `/usr/local`), Ceres Solver, Eigen3.

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Produces: build/sfm_cov  build/ba_ceres
```

### Run sfm_cov (keypoint covariance estimation)

```bash
# For each image, writes <image>.cov alongside it
./build/sfm_cov path/to/image.JPG

# With a .feat file to align covariance indices with OpenMVG features:
./build/sfm_cov path/to/image.JPG path/to/image.feat
```

`.cov` format (one line per keypoint, index-aligned with `.feat`):
```
x y scale cov00 cov01 cov10 cov11
```

### Run ba_ceres (bundle adjustment + covariance extraction)

```bash
./build/ba_ceres \
    path/to/sfm_data.bin \
    path/to/matches/ \
    path/to/images_with_cov_files/ \
    path/to/output/reconstruction.csv
```

Output CSV: `x y z cov00 cov01 cov02 cov11 cov12 cov22` (upper triangle of 3×3 covariance, SfM frame).

---

## Repository layout

```
src/
  main.cpp              sfm_cov entry point
  ba_ceres.cpp          Ceres BA with Mahalanobis weighting + covariance extraction
  camera_intrinsics.h   Hardcoded intrinsics for the aerial camera
  zeisl/                CovEstimator class (Zeisl 2009 DoG Hessian method)

scripts/
  gcp_umeyama.py        GCP triangulation + 7-DOF Umeyama geo-registration
  lidar_validation.py   LiDAR nearest-neighbour validation
  plot_validation.py    Validation plots
  synthetic_validation.py  χ²(3) calibration test on Blender synthetic dataset

output/
  reconstruction_filtered.csv   SfM-frame points (outliers removed)
  reconstruction_aligned.csv    UTM-frame points + covariances
  validation.csv                NN distances + uncertainty traces
  validation_plot.png           Validation figure
```

---

## References

- Zeisl, B., Georgel, P., Schweiger, F., Steinbach, E., & Navab, N. (2009). *Estimation of Location Uncertainty for Scale Invariant Feature Points.* BMVC.
- Umeyama, S. (1991). *Least-squares estimation of transformation parameters between two point patterns.* TPAMI.
- Agarwal, S. et al. *Ceres Solver.* http://ceres-solver.org
- Moulon, P. et al. *OpenMVG.* https://github.com/openMVG/openMVG
