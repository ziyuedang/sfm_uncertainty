#include "fileIO.h"

ofstream& fileOut::initializeFile(char* filename) {
	ofstream* outfile = new ofstream(filename);

	*outfile << "# format:\n# COVXX COVXY COVYY" << endl;

	return *outfile;
}

ofstream& fileOut::initializeFile(string fn) {
	return initializeFile((char*)fn.c_str());
}

void fileOut::write(ofstream& outfile, MatCv cov) {
	/* File format:
	covxx covxy covyy
	...
	*/


	// covariance information
	outfile << cov.at<float>(0, 0) << "\t";
	outfile << cov.at<float>(0, 1) << "\t";
	outfile << cov.at<float>(1, 1) << "\t";
	outfile << endl;
}

void fileOut::closeFile(ofstream& outfile) {
	outfile.close();
}

ofstream& fileOut::initializeFileFeats(char* filename) {
	ofstream* outfile = new ofstream(filename);
	*outfile << "# format: \n# x y scale octave slice" << endl;
	return *outfile;
}

ofstream& fileOut::initializeFileFeats(string fn) {
	return initializeFileFeats((char*)fn.c_str());
}

void fileOut::writeFeats(ofstream& outfile, float x, float y, float scale, int oct, int slice) {
	outfile << x << "\t";
	outfile << y << "\t";
	outfile << scale << "\t";
	outfile << oct << "\t";
	outfile << slice << "\t";
	outfile << endl;
}