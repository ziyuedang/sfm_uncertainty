#pragma once
//c++
#include <iostream>

//opencv
#include <opencv2/core/core_c.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

//covUtils
#include "definitions.h"
using MatCv = cv::Mat;

class CovEstimator {

public:
	CovEstimator(std::vector<std::vector<MatCv>> pyr, int octvs, int intvls) {
		detectPyr = pyr;
		octaves = octvs;
		intervals = intvls;
//		MatCv H(2, 2, CV_32FC1);
		MatCv cov(2, 2, CV_32FC1);
		MatCv evals(2, 1, CV_32FC1);
		MatCv evecs(2, 2, CV_32FC1);
	};

	MatCv getImageFromPyramid(int octv, int intvls);
	MatCv getCovAt(float x, float y, float scale);


private:
	/*** Methods ***/
	MatCv hessian(MatCv img, int r, int c);


	/*** Member variables ***/
	int type;
	int octaves, intervals;
	std::vector<std::vector<MatCv>> detectPyr;
//	MatCv H;
	MatCv cov;
	MatCv evals;
	float ev1, ev2;
	MatCv evecs;
};
