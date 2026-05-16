#pragma once
#include <string>
#include <fstream>
#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "openMVG/features/feature.hpp"

class fileOut;
using namespace std;
using MatCv = cv::Mat;

class fileOut {
public:
	static ofstream& initializeFile(char* filename);
	static ofstream& initializeFile(string fn);
	static void write(ofstream& outfile, MatCv cov);
	static void closeFile(ofstream& outfile);

	static ofstream& initializeFileFeats(char* filename);
	static ofstream& initializeFileFeats(string fn);
	static void writeFeats(ofstream& outfile, float x, float y, float scale, int oct, int slice);
};