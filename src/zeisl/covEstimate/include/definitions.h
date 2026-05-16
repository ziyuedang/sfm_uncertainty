#pragma once
// DETECTOR
#define SIFT_THRES					0.15
#define DETECTOR_SIFT				0
/** default sigma for initial gaussian smoothing */
#define SIFT_SIGMA 1.6f

// DESCRIPTOR
#define SIFT_DESCR_SIZE				128

// FILENAMES / FILE ENDINGS
// images
#define FE_INPUT_IMAGE				".jpg"
#define FE_FEATURES_IMAGE_SIFT		"_feat.jpg"
#define FE_COVARIANCE_IMAGE_SIFT	"_cov_sift.jpg"
// keypoints
#define FE_KEYS_SIFT				".feat"
#define FE_GENERAL_KEYS_SIFT_COV	".key_sift_cov"
