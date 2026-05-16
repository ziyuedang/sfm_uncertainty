#include "covEstimator.h"
#include <cmath>
#include <stdexcept>
using namespace std;
using namespace cv;

#define PI 3.14159265f

MatCv CovEstimator::getImageFromPyramid(int octv, int intvls) {
    return detectPyr[octv][intvls];
}

MatCv CovEstimator::getCovAt(float x, float y, float scale) {
    int octv = 0, intv = 0, row = 0, col = 0;
    float subintv = 0.0f;

    float tmp = log(scale / SIFT_SIGMA) / log(2.0f) * intervals;
    int tmp2 = cvRound(tmp) - 1;
    intv = (tmp2 % intervals + intervals) % intervals + 1;
    octv = ((int)cvRound(tmp) - 1) / intervals;
    subintv = tmp - (octv * intervals + intv);

    col = cvRound(x / pow(2.0, octv));
    row = cvRound(y / pow(2.0, octv));

    if (octv < 0 || octv >= (int)detectPyr.size())
        throw std::runtime_error("Octave out of bounds");
    if (intv < 0 || intv >= (int)detectPyr[octv].size())
        throw std::runtime_error("Interval out of bounds");

    MatCv img = getImageFromPyramid(octv, intv);

    if (row < 1 || row >= img.rows - 1 || col < 1 || col >= img.cols - 1)
        throw std::runtime_error("Keypoint too close to image border");

    MatCv H = hessian(img, row, col);

    // Check eigenvalues of H using eigen() not SVD
    MatCv eigenvalues;
    eigen(H, eigenvalues);
    float h1 = eigenvalues.at<float>(0, 0);
    float h2 = eigenvalues.at<float>(1, 0);

    // H should be negative definite at a SIFT maximum - negate to make positive definite
    if (h1 < 0 && h2 < 0)
        H = -H;
    else if (h1 * h2 < 0)
        throw std::runtime_error("Hessian indefinite: saddle point");

    invert(H, cov, CV_SVD_SYM);
    cov.convertTo(cov, -1, pow(2.0f, (octv + subintv / intervals)));

    // Verify covariance is positive definite using eigen()
    MatCv covEigenvalues;
    eigen(cov, covEigenvalues);
    ev1 = covEigenvalues.at<float>(0, 0);
    ev2 = covEigenvalues.at<float>(1, 0);

    if (ev1 <= 0 || ev2 <= 0)
        throw std::runtime_error("Covariance not positive definite");

    return cov;
}

MatCv CovEstimator::hessian(MatCv dog, int row, int col) {
    int r, c;
    float v, dxx = 0, dyy = 0, dxy = 0;
    float w[3][3] = { 0.0449f, 0.1221f, 0.0449f,
                      0.1221f, 0.3319f, 0.1221f,
                      0.0449f, 0.1221f, 0.0449f };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            r = row + j - 1;
            c = col + i - 1;
            v = dog.at<float>(r, c);
            dxx += w[i][j] * (dog.at<float>(r, c + 1) + dog.at<float>(r, c - 1) - 2 * v);
            dyy += w[i][j] * (dog.at<float>(r + 1, c) + dog.at<float>(r - 1, c) - 2 * v);
            dxy += w[i][j] * ((dog.at<float>(r + 1, c + 1) -
                               dog.at<float>(r + 1, c - 1) -
                               dog.at<float>(r - 1, c + 1) +
                               dog.at<float>(r - 1, c - 1)) / 4.0f);
        }
    MatCv H(2, 2, CV_32FC1);
    H.at<float>(0, 0) = -dxx;
    H.at<float>(0, 1) = -dxy;
    H.at<float>(1, 0) = -dxy;
    H.at<float>(1, 1) = -dyy;
    return H;
}