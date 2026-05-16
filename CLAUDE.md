# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

`sfm_uncertainty` computes per-keypoint 2D position covariance matrices for SIFT features detected in images. The covariance is derived from the Hessian of the Difference-of-Gaussian (DoG) pyramid at each keypoint location, following Zeisl's method. These covariances are intended for use in uncertainty-aware Structure-from-Motion (SfM) pipelines.

Two executables are built:
- **`sfm_cov`** — the main tool: takes an image path, detects SIFT keypoints via OpenMVG, builds a DoG pyramid, and writes a `.cov` file alongside the image.
- **`cov_test`** — a standalone test/demo that uses the original Zeisl file I/O and argument parsing.

## Build

Dependencies: OpenCV 4, OpenMVG (installed at `/usr/local`), Eigen3, OpenMP.

```bash
# From the repo root (build dir already exists and is configured)
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Binaries land in build/
./sfm_cov <image.JPG>     # writes <image.JPG>.cov
./cov_test -i <image.JPG> -d <image_dir>/ -show
```

To reconfigure OpenMVG from source (rarely needed):
```bash
cd openMVG_build
cmake ../openMVG/src -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON
make install -j$(nproc)
```

## Code Architecture

```
src/
  main.cpp                  # sfm_cov entry point
  camera_intrinsics.h       # hardcoded camera intrinsics (focal length, principal point, distortion)
  zeisl/
    covEstimate/
      covEstimate/src/
        covEstimator.h/.cpp # CovEstimator class — core algorithm
        tets.cpp            # cov_test entry point (standalone version)
        fileIO.cpp          # .cov / .feat file output helpers
      include/
        definitions.h       # SIFT_SIGMA, threshold constants, file extension macros
        covArgEvaluator.h   # CLI argument parsing for cov_test
        fileIO.h
      bundle_adjustment/    # stub (dataIO.h is empty — not yet implemented)
```

### Data flow in `sfm_cov` / `main.cpp`

1. Load grayscale image via OpenMVG `ReadImage`.
2. Build a 6-octave, 3-interval `HierarchicalGaussianScaleSpace` (DoG pyramid).
3. For each octave: detect & describe keypoints (`SIFT_KeypointExtractor` + `Sift_DescriptorExtractor`), and convert each DoG slice from `openMVG::Image<float>` to `cv::Mat`.
4. Pass the `cv::Mat` DoG pyramid to `CovEstimator`.
5. For each keypoint, call `CovEstimator::getCovAt(x, y, sigma)` → returns a 2×2 `cv::Mat` covariance.
6. Write results to `<image>.cov` (space-separated: `x y scale cov00 cov01 cov10 cov11`).

### `CovEstimator::getCovAt` algorithm

- Maps `(x, y, sigma)` → octave/interval/subinterval using `log2(sigma / SIFT_SIGMA) * intervals`.
- Scales pixel coords by `1 / 2^octave`.
- Computes a weighted 2×2 Hessian (`CovEstimator::hessian`) using a 3×3 Gaussian weight kernel over the DoG image.
- Inverts the (negated, positive-definite) Hessian via `cv::invert(..., CV_SVD_SYM)`.
- Scales covariance by `2^(octave + subinterval/intervals)` to map back to original image coordinates.
- Throws if keypoint is at image border, Hessian is indefinite, or resulting covariance is not positive definite.

## File Formats

| Extension | Format |
|-----------|--------|
| `.cov` (output) | Text: `# x y scale cov00 cov01 cov10 cov11` header, then one keypoint per line |
| `.feat` (OpenMVG) | Binary feature file (OpenMVG `SIOPointFeature`) |
| `.feat` (Zeisl) | Text: `x y scale octave slice` per line |
| `sfm_data.json` | OpenMVG SfM reconstruction: views, intrinsics, extrinsics, structure |

## Test Data

`test_data/` and `test_data/images/` contain sample JPEGs and their precomputed `.cov` files. `test_data/intrinsics.txt` documents the camera parameters mirrored in `camera_intrinsics.h`.

## Project Goal
End-to-end SfM with principled uncertainty propagation from SIFT keypoint localization (Zeisl 2009) through bundle adjustment to per-3D-point covariance matrices. Validated statistically against airborne LiDAR reference data.

## Cloud Storage
S3 bucket: `sfm-uncertainty-pipeline`
- `input/images/` — source imagery
- `input/intrinsics.txt` — camera parameters
- `output/covariances/` — per-image .cov files (already uploaded for all 14 images)
- `output/keypoints/` — OpenMVG feature files
- `output/pointcloud/` — final point cloud with uncertainty

## Dataset
14 aerial images at `~/sfm_uncertainty/test_data/images/*.JPG`, 8984x6732, all .cov files generated.
LiDAR reference data at `/mnt/d/SFM/UH_Data/` (copy to WSL when needed).

## Next Step: Feature Matching
Set up OpenMVG SfM project and run exhaustive matching:
```bash
mkdir -p ~/sfm_uncertainty/openmvg_project/matches

openMVG_main_SfMInit_ImageListing \
    -i ~/sfm_uncertainty/test_data/images/ \
    -o ~/sfm_uncertainty/openmvg_project/ \
    -f 11713.6 \
    -k "11713.6;0;4486.98;0;11713.6;3380.13;0;0;1"

openMVG_main_ComputeFeatures \
    -i ~/sfm_uncertainty/openmvg_project/sfm_data.json \
    -o ~/sfm_uncertainty/openmvg_project/matches/

openMVG_main_ComputeMatches \
    -i ~/sfm_uncertainty/openmvg_project/sfm_data.json \
    -o ~/sfm_uncertainty/openmvg_project/matches/ \
    -g e
```

## After Matching: Custom Ceres BA
- Load matched keypoints and their covariances from .cov files
- Build Ceres problem with per-keypoint covariance-weighted reprojection error
- Extract per-camera and per-3D-point covariance from Ceres after optimization

## Key Design Decision
Keypoint covariances are **inputs** to BA (weights on reprojection error), not computed inside BA. This follows Zeisl 2009 — the 2x2 covariance represents image-space localization uncertainty that propagates through to camera and point uncertainty.