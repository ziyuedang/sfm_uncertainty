/*************************************************************************************************
* ----------------------------------COVARIANCE ESTIMATOR-----------------------------------------
***************************************************************************************************/
#define _USE_MATH_DEFINES
//c/c++
#include <iostream>
#include <string>
#include <cmath>

//opencv
#include <opencv2/imgcodecs.hpp>

//openMVG
#include "openMVG/image/image_container.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp"
#include "openMVG/features/sift/sift_KeypointExtractor.hpp" 

extern "C" {
#include "nonFree/sift/vl/sift.h"
}

// Zeisl's method
#include "covEstimator.h"

//own
#include "covArgEvaluator.h"
#include "fileIO.h"

using namespace std;
using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::features;
using namespace openMVG::features::sift;
using MatCv = cv::Mat;

// Define a feature and a container of features
using Feature_T = SIOPointFeature;
using Feats_T = vector<Feature_T>;
using Regions_type = SIFT_Regions;
int vl_sift(
	string filename, Image<unsigned char>& image
)
{
	//string filename_out = filename + ".vlsift";
	//ofstream& outfile = fileOut::initializeFileFeats(filename_out);
	const int w = image.Width(), h = image.Height();

	// parameters
	int num_octaves = 6;
	int num_scales = 3;
	int first_octave = 0;
	float edge_threshold = 10.0f;
	float peak_threshold = 0.04f;
	bool root_sift = true;

	// outputs
	vector<float> x, y, scale;
	vector<int> oct, slice;
	//Convert to float
	Image<float> If(image.GetMat().cast<float>());
	cout << "width = " << If.Width() << endl;
	VlSiftFilt *filt = vl_sift_new(w, h,
		num_octaves, num_scales, first_octave);
	cout << "filt = " << (filt) << endl;
	if (edge_threshold >= 0)
		vl_sift_set_edge_thresh(filt, edge_threshold);
	if (peak_threshold >= 0)
		vl_sift_set_peak_thresh(filt, 255 * double(peak_threshold / num_scales));
	if (!filt) {
		cout << "Could not create SIFT filter." << endl;
		exit(1);
	}
	Descriptor<vl_sift_pix, 128> descr;
	Descriptor<unsigned char, 128> descriptor;

	// Process SIFT computation
	int err = vl_sift_process_first_octave(filt, If.data());
	
	// Need to save data somehow

	while (true) {
		vl_sift_detect(filt);

		VlSiftKeypoint const* keys = vl_sift_get_keypoints(filt);
		const int nkeys = vl_sift_get_nkeypoints(filt);
		cout << "SIFT detected " << nkeys << "keypoints" << endl;
		int octave = vl_sift_get_octave_index(filt);
		cout << "octave =" << octave << endl;
		for (int i = 0; i < nkeys; ++i) {
			double angles[4] = { 0.0, 0.0, 0.0, 0.0 };
			int nangles = 1;
			nangles = vl_sift_calc_keypoint_orientations(filt, angles, keys + i);
			for (int q = 0; q < nangles; ++q) {
				vl_sift_calc_keypoint_descriptor(filt, &descr[0], keys + i, angles[q]);

				{
					x.push_back(keys[i].x);
					y.push_back(keys[i].y);
					scale.push_back(keys[i].sigma);
					oct.push_back(keys[i].o);
					slice.push_back(keys[i].is);
				}
			}
		}
		if (vl_sift_process_next_octave(filt))
			break; // Last octave
	}
	vl_sift_delete(filt);
	
	return 0;
}

int main(int argc, char* argv[])
{
	string filename;
	Feats_T vec_feats;

	/*** Parsing input arguments ***/
	covArgEvaluator arg;
	arg.evaluate(argc, argv);

	/*** Loading input images and according keys ***/
	// Loading image
	filename.clear();
	filename.append(arg.imgDir).append(arg.imgFile);	
	const string imageFile = filename;

	Image<unsigned char> in;

	int res = ReadImage(imageFile.c_str(), &in);
	/*** Check if image is loaded fine ***/
	if(!res){
		cout << " Cov Error: Unable to load image from " << filename << "\n";
		exit(1);
	}

	/*** Loading the keypoints ***/
	filename.clear();
	filename.append(arg.imgDir).append(arg.keyFile);
	if (arg.verbose)
		cout << "Loading key points from file " << filename << endl;

	loadFeatsFromFile(filename, vec_feats);
	
	// Above is tested and working correctly

	/*** Creating image pyramid from openMVG***/


	//// Create GSS
	//const int supplementary_images = 3;

	//HierarchicalGaussianScaleSpace octave_gen(6, 3, GaussianScaleSpaceParams(1.6f, 1.0f, 0.5f, supplementary_images));
	//Octave octave;
	//octave_gen.SetImage(image);

	///*** GSS output to image files ***/
	//cerr << "Octave computation started" << endl;
	//uint8_t octave_id = 0;
	//while (octave_gen.NextOctave(octave))
	//{
	//	cerr << "Computed octave : " << to_string(octave_id) << endl;
	//	for (int i = 0; i < octave.slices.size(); ++i)
	//	{
	//		stringstream str;
	//		str << "gaussian_octave_" << to_string(octave_id) << "_" << i << ".png";
	//		WriteImage(str.str().c_str(), Image<unsigned char>(octave.slices[i].cast<unsigned char>()));
	//	}
	//	
	//	++octave_id;
	//}

	/*** Read GSS and compute Difference of Gaussian (DoG) ***/

	// dog_pyr tested, dimension is correct
	
//	/*** estimating the covariances for the keypoints ***/
//	if (arg.verbose)
//		cout << "cov covariance estimation - writing data to file " << filename << endl;
//	CovEstimator estimator(dog_pyr, octaves, intervals);
//	for (int k = 0; k < vec_feats.size(); k++) {
//		cout << "k = " << k << endl;
//		MatCv cov = estimator.getCovAt(vec_feats[k].x(), vec_feats[k].y(), vec_feats[k].scale());	
////		// Output cov to file
//		fileOut::write(outfile, cov);
//	}
}

