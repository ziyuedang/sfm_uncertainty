#include <algorithm>
#include <array>
#include <filesystem>
#include <thread>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "openMVG/cameras/Camera_Pinhole_Radial.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"

namespace fs = std::filesystem;
using namespace openMVG;
using namespace openMVG::sfm;
using namespace openMVG::cameras;

// ── Covariance whitening ────────────────────────────────────────────────────

// Given symmetric 2x2 covariance [[a,b],[b,d]], compute the row-major
// elements of L^{-1} where L is the lower-triangular Cholesky factor.
// Returns false if Sigma is not positive-definite; caller should use identity.
static bool computeLInv(double a, double b, double d, double Linv[4]) {
    if (a <= 0.0) return false;
    double l00 = std::sqrt(a);
    double l10 = b / l00;
    double l11_sq = d - l10 * l10;
    if (l11_sq <= 0.0) return false;
    double l11 = std::sqrt(l11_sq);
    // L^{-1} = [[1/l00, 0], [-l10/(l00*l11), 1/l11]]
    Linv[0] = 1.0 / l00;
    Linv[1] = 0.0;
    Linv[2] = -l10 / (l00 * l11);
    Linv[3] = 1.0 / l11;
    return true;
}

// ── Cost functor ─────────────────────────────────────────────────────────────

// Pinhole + radial K3 reprojection error, whitened by per-keypoint covariance.
// Intrinsics block: [focal, cx, cy, k1, k2, k3]
// Extrinsics block: [rx, ry, rz, tx, ty, tz]  (angle-axis + translation)
// Point block:      [X, Y, Z]
struct MahalanobisReprojError {
    MahalanobisReprojError(double ox, double oy, const double Linv[4])
        : ox_(ox), oy_(oy) {
        std::copy(Linv, Linv + 4, Linv_);
    }

    template <typename T>
    bool operator()(const T* intr, const T* extr, const T* pt, T* res) const {
        Eigen::Matrix<T, 3, 1> p;
        ceres::AngleAxisRotatePoint(extr, pt, p.data());
        p[0] += extr[3];
        p[1] += extr[4];
        p[2] += extr[5];

        // Perspective division
        const T xn = p[0] / p[2];
        const T yn = p[1] / p[2];

        // Radial distortion
        const T r2 = xn * xn + yn * yn;
        const T r4 = r2 * r2;
        const T r6 = r4 * r2;
        const T rc = T(1) + intr[3] * r2 + intr[4] * r4 + intr[5] * r6;

        // Projected pixel
        const T px = intr[0] * xn * rc + intr[1];
        const T py = intr[0] * yn * rc + intr[2];

        // Raw reprojection error
        const T ex = px - T(ox_);
        const T ey = py - T(oy_);

        // Whiten by L^{-1}
        res[0] = T(Linv_[0]) * ex + T(Linv_[1]) * ey;
        res[1] = T(Linv_[2]) * ex + T(Linv_[3]) * ey;
        return true;
    }

    static ceres::CostFunction* Create(double ox, double oy, const double Linv[4]) {
        return new ceres::AutoDiffCostFunction<MahalanobisReprojError, 2, 6, 6, 3>(
            new MahalanobisReprojError(ox, oy, Linv));
    }

    double ox_, oy_, Linv_[4];
};

// ── .cov file loader ──────────────────────────────────────────────────────────
// The .cov file is index-aligned with the corresponding .feat file: entry i
// holds the covariance for feature i (zeros if the covariance was invalid).
// Format per line: x y scale cov00 cov01 cov10 cov11

struct KeypointCov {
    double cov00, cov01, cov11; // symmetric, cov10 == cov01
    bool valid;                 // false if covariance was zero/degenerate
};

static std::vector<KeypointCov> loadCovFile(const std::string& path) {
    std::vector<KeypointCov> covs;
    std::ifstream f(path);
    if (!f.is_open()) return covs;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        float x, y, scale;
        double c00, c01, c10, c11;
        if (!(ss >> x >> y >> scale >> c00 >> c01 >> c10 >> c11)) continue;
        bool valid = (c00 > 0.0 && c11 > 0.0);
        covs.push_back({c00, c01, c11, valid});
    }
    return covs;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const std::string sfm_bin  = (argc > 1) ? argv[1]
        : "/home/ziyue_dang/sfm_uncertainty/openmvg_project/reconstruction/sfm_data.bin";
    const std::string match_dir = (argc > 2) ? argv[2]
        : "/home/ziyue_dang/sfm_uncertainty/openmvg_project/matches/";
    const std::string cov_dir   = (argc > 3) ? argv[3]
        : "/home/ziyue_dang/sfm_uncertainty/test_data/images_half/";
    const std::string out_csv   = (argc > 4) ? argv[4]
        : "/home/ziyue_dang/sfm_uncertainty/output/reconstruction.csv";

    // ── Load reconstruction ───────────────────────────────────────────────
    SfM_Data sfm_data;
    if (!Load(sfm_data, sfm_bin, ESfM_Data::ALL)) {
        std::cerr << "Failed to load: " << sfm_bin << "\n";
        return 1;
    }
    std::cout << "Loaded: " << sfm_data.views.size() << " views, "
              << sfm_data.poses.size() << " poses, "
              << sfm_data.structure.size() << " landmarks\n";

    // ── Camera parameter blocks ───────────────────────────────────────────
    // Intrinsics: one shared block [focal, cx, cy, k1, k2, k3]
    if (sfm_data.intrinsics.empty()) { std::cerr << "No intrinsics\n"; return 1; }
    IndexT intr_id = sfm_data.intrinsics.begin()->first;
    auto intr_params = sfm_data.intrinsics.at(intr_id)->getParams(); // [focal,cx,cy,k1,k2,k3]
    if (intr_params.size() < 6) { std::cerr << "Need K3 intrinsic (6 params)\n"; return 1; }
    std::vector<double> intrinsics_block(intr_params.begin(), intr_params.end());

    // Extrinsics: one 6-element block per pose [rx,ry,rz,tx,ty,tz]
    std::unordered_map<IndexT, std::vector<double>> extr_blocks;
    for (auto& [pose_id, pose] : sfm_data.poses) {
        const Mat3& R = pose.rotation();
        const Vec3  t = pose.translation(); // t = -R*C
        std::vector<double> e(6);
        ceres::RotationMatrixToAngleAxis(R.data(), e.data());
        e[3] = t[0]; e[4] = t[1]; e[5] = t[2];
        extr_blocks[pose_id] = std::move(e);
    }

    // 3D points: one 3-element block per landmark
    std::unordered_map<IndexT, std::array<double, 3>> pt_blocks;
    for (auto& [lm_id, lm] : sfm_data.structure) {
        pt_blocks[lm_id] = {lm.X[0], lm.X[1], lm.X[2]};
    }

    // ── Load per-view covariances ─────────────────────────────────────────
    // Each .cov file is index-aligned with the matching .feat file, so
    // Observation::id_feat directly indexes into the covariance vector.
    std::unordered_map<IndexT, std::vector<KeypointCov>> view_covs;

    for (auto& [view_id, view_ptr] : sfm_data.views) {
        if (!sfm_data.IsPoseAndIntrinsicDefined(view_ptr.get())) continue;

        fs::path img_path(view_ptr->s_Img_path);
        std::string img_filename = img_path.filename().string(); // e.g. "11.JPG"
        std::string stem = img_path.stem().string();

        std::string cov_path = cov_dir + "/" + img_filename + ".cov";
        view_covs[view_id] = loadCovFile(cov_path);
        std::cout << "  View " << stem << ": " << view_covs[view_id].size() << " covs\n";
    }

    // ── Build Ceres problem ───────────────────────────────────────────────
    ceres::Problem problem;

    const double identity_Linv[4] = {1.0, 0.0, 0.0, 1.0};

    int n_with_cov = 0, n_no_cov = 0;

    for (auto& [lm_id, lm] : sfm_data.structure) {
        auto& pt = pt_blocks.at(lm_id);

        for (auto& [view_id, obs] : lm.obs) {
            auto view_it = sfm_data.views.find(view_id);
            if (view_it == sfm_data.views.end()) continue;
            IndexT pose_id = view_it->second->id_pose;
            auto extr_it = extr_blocks.find(pose_id);
            if (extr_it == extr_blocks.end()) continue;

            const double ox = obs.x[0];
            const double oy = obs.x[1];

            // Look up covariance by feature index (index-aligned with .feat file)
            double Linv[4];
            std::copy(identity_Linv, identity_Linv + 4, Linv);

            auto cov_it = view_covs.find(view_id);
            if (cov_it != view_covs.end()) {
                const auto& covvec = cov_it->second;
                IndexT fid = obs.id_feat;
                if (fid < covvec.size() && covvec[fid].valid) {
                    const auto& kc = covvec[fid];
                    if (!computeLInv(kc.cov00, kc.cov01, kc.cov11, Linv))
                        std::copy(identity_Linv, identity_Linv + 4, Linv);
                    else
                        ++n_with_cov;
                } else {
                    ++n_no_cov;
                }
            }

            problem.AddResidualBlock(
                MahalanobisReprojError::Create(ox, oy, Linv),
                nullptr,
                intrinsics_block.data(),
                extr_it->second.data(),
                pt.data());
        }
    }
    std::cout << "Residuals with keypoint cov: " << n_with_cov
              << ", using identity: " << n_no_cov << "\n";

    // Hold intrinsics fixed (already optimised by OpenMVG SfM)
    problem.SetParameterBlockConstant(intrinsics_block.data());

    // ── Solve ─────────────────────────────────────────────────────────────
    ceres::Solver::Options opts;
    opts.linear_solver_type           = ceres::SPARSE_SCHUR;
    opts.preconditioner_type          = ceres::SCHUR_JACOBI;
    opts.max_num_iterations           = 100;
    opts.minimizer_progress_to_stdout = true;
    opts.num_threads                  = std::max(1u, std::thread::hardware_concurrency());

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    // ── Per-point covariance via ceres::Covariance ────────────────────────
    // Fix all camera poses so the only free parameters are the 3D points.
    // This removes the 7 gauge DOF and gives the marginal covariance of each
    // 3D point conditioned on the estimated (fixed) camera poses — the right
    // quantity for downstream comparison against LiDAR reference data.
    std::cout << "Computing per-point covariance (cameras held fixed)...\n";
    for (auto& [pose_id, e] : extr_blocks)
        problem.SetParameterBlockConstant(e.data());

    std::vector<std::pair<const double*, const double*>> cov_blocks;
    cov_blocks.reserve(pt_blocks.size());
    for (auto& [lm_id, pt] : pt_blocks)
        cov_blocks.push_back({pt.data(), pt.data()});

    ceres::Covariance::Options cov_opts;
    cov_opts.algorithm_type  = ceres::SPARSE_QR;
    cov_opts.num_threads     = opts.num_threads;

    ceres::Covariance covariance(cov_opts);
    bool cov_ok = covariance.Compute(cov_blocks, &problem);
    if (!cov_ok)
        std::cerr << "Warning: covariance computation failed. "
                     "Writing zeros for covariance.\n";

    // ── Write output CSV ──────────────────────────────────────────────────
    fs::create_directories(fs::path(out_csv).parent_path());
    std::ofstream csv(out_csv);
    if (!csv.is_open()) { std::cerr << "Cannot open " << out_csv << "\n"; return 1; }

    csv << "# x y z cov00 cov01 cov02 cov11 cov12 cov22\n";
    csv << std::scientific;

    for (auto& [lm_id, pt] : pt_blocks) {
        double cov3[9] = {};
        if (cov_ok)
            covariance.GetCovarianceBlock(pt.data(), pt.data(), cov3);

        csv << pt[0] << " " << pt[1] << " " << pt[2] << " "
            << cov3[0] << " " << cov3[1] << " " << cov3[2] << " "
            << cov3[4] << " " << cov3[5] << " "
            << cov3[8] << "\n";
    }
    csv.close();
    std::cout << "Wrote " << pt_blocks.size() << " points to " << out_csv << "\n";
    return 0;
}
