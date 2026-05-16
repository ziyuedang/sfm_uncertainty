#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

// OpenCV
#include <opencv2/core.hpp>

// OpenMVG
#include "openMVG/image/image_container.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp"
#include "openMVG/features/sift/sift_KeypointExtractor.hpp"

// Zeisl
#include "covEstimator.h"

using namespace std;
using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::features;
using namespace openMVG::features::sift;

using MatCv = cv::Mat;

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cout << "Usage: sfm_cov <image_path> [feat_file]" << endl;
        cout << "  feat_file: optional path to OpenMVG .feat file; if given, covariances" << endl;
        cout << "             are computed at those exact feature positions (one cov per" << endl;
        cout << "             feature, in the same order) instead of re-detecting keypoints." << endl;
        return 1;
    }

    string imagePath = argv[1];
    string featFilePath = (argc >= 3) ? argv[2] : "";

    // Load image
    Image<unsigned char> imgGray;
    if (!ReadImage(imagePath.c_str(), &imgGray)) {
        cerr << "Failed to load image: " << imagePath << endl;
        return 1;
    }
    cout << "Image loaded: " << imgGray.Width() << "x" << imgGray.Height() << endl;

    // Build DoG pyramid
    const int supplementary_images = 3;
    int n_octaves = 6;
    int n_levels = 3;
    HierarchicalGaussianScaleSpace octave_gen(n_octaves, n_levels,
        GaussianScaleSpaceParams(1.6f, 1.0f, 0.5, supplementary_images));

    Image<float> image(imgGray.GetMat().cast<float>() / 255.0f);
    octave_gen.SetImage(image);

    // Build DoG pyramid (always needed for covariance estimation)
    vector<vector<MatCv>> dog_pyramid;

    // Load pre-computed features if a .feat file was given; otherwise detect keypoints.
    vector<SIOPointFeature> feat_features; // used when feat_file is provided
    vector<Keypoint>        keypoints;     // used when detecting from scratch

    Octave octave;
    uint8_t octave_id = 0;
    while (octave_gen.NextOctave(octave))
    {
        // Build DoG slices
        vector<MatCv> dog_slices;
        const int n = octave.slices.size();
        for (int i = 0; i < n - 1; ++i) {
            const Image<float>& P = octave.slices[i + 1];
            const Image<float>& M = octave.slices[i];
            Image<float> dog_img = P - M;

            MatCv dog_mat(dog_img.Height(), dog_img.Width(), CV_32FC1);
            for (int r = 0; r < dog_img.Height(); ++r)
                for (int c = 0; c < dog_img.Width(); ++c)
                    dog_mat.at<float>(r, c) = dog_img(r, c);

            dog_slices.push_back(dog_mat);
        }
        dog_pyramid.push_back(dog_slices);

        if (featFilePath.empty()) {
            // Auto-detect keypoints (original behaviour)
            vector<Keypoint> keys;
            SIFT_KeypointExtractor keypointDetector(0.04f / octave_gen.NbSlice(), 10.0f, 5);
            keypointDetector(octave, keys);
            Sift_DescriptorExtractor descriptorExtractor;
            descriptorExtractor(octave, keys);
            move(keys.begin(), keys.end(), back_inserter(keypoints));
        }

        ++octave_id;
    }

    if (!featFilePath.empty()) {
        if (!loadFeatsFromFile(featFilePath, feat_features)) {
            cerr << "Failed to load feat file: " << featFilePath << endl;
            return 1;
        }
        cout << "Loaded " << feat_features.size() << " features from " << featFilePath << endl;
    } else {
        cout << "Keypoints detected: " << keypoints.size() << endl;
    }

    // Initialize CovEstimator with DoG pyramid
    CovEstimator covEst(dog_pyramid, n_octaves, n_levels);

    // Compute covariance for each keypoint and write to file
    string outPath = imagePath + ".cov";
    ofstream outFile(outPath);
    if (!outFile.is_open()) {
        cerr << "Failed to open output file: " << outPath << endl;
        return 1;
    }

    outFile << "# x y scale cov00 cov01 cov10 cov11\n";
    int valid = 0, invalid = 0;

    auto writeCov = [&](float x, float y, float sigma) {
        try {
            MatCv cov = covEst.getCovAt(x, y, sigma);
            outFile << x << " " << y << " " << sigma << " "
                    << cov.at<float>(0,0) << " " << cov.at<float>(0,1) << " "
                    << cov.at<float>(1,0) << " " << cov.at<float>(1,1) << "\n";
            ++valid;
        } catch (...) {
            // Write zero covariance so the file stays index-aligned with the .feat file
            outFile << x << " " << y << " " << sigma
                    << " 0 0 0 0\n";
            ++invalid;
        }
    };

    if (!featFilePath.empty()) {
        for (const auto& f : feat_features)
            writeCov(f.x(), f.y(), f.scale());
    } else {
        for (const auto& kp : keypoints)
            writeCov(kp.x, kp.y, kp.sigma);
    }

    outFile.close();
    cout << "Covariances written: " << valid << " valid, " << invalid << " skipped" << endl;
    cout << "Output: " << outPath << endl;

    return 0;
}