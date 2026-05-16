#pragma once
#pragma once
#include <iostream>
#include <string>

#include "definitions.h"

using namespace std;


class covArgEvaluator {

public:
	/*** Constructor, Destructor ***/
	covArgEvaluator() :
		imgDir("./"), imgFile(NULL), keyFileEnding(NULL),
		verbose(true), saveCov(false), showCov(false),
		decType(-1) {};

	covArgEvaluator(int argc, char* argv[]) :
		decType(-1) {
		evaluate(argc, argv);
	};

	/*** Methods ***/

	void evaluate(int argc, char* argv[]);


	/*** Member variables ***/
	char* imgDir, *imgFile, *keyFileEnding;
	char imgFileStart[25];
	string keyFile;
	bool verbose, saveCov, showCov;
	int decType;

};


/*** Implementations - quick and dirty ***/

void covArgEvaluator::evaluate(int argc, char* argv[]) {
	// Information on usage
	if (argc == 1) {
		cout << "USAGE:\n"
			<< "  covarianceEstimate.exe -i <img> [OPTIONS]\n"
			<< endl
			<< "REQUIRED PARAMETERS:\n"
			<< "  -i <img>     Image file to process.\n"
			<< endl
			<< "OPTIONS:\n"
			<< "  -k <file>    File ending of the keypoint file to load. The name itself\n"
			<< "               is according to the name of the image.\n"
			<< "               (default is " << FE_KEYS_SIFT ")\n"
			<< "  -d <dir>     The directory containing the image file (default ist ./)\n"
			<< "  -q           Quite prossing. Don't print status information.\n"
			<< "  -show        Display the estimated uncertainty for feature points in the\n"
			<< "               image, each as ellipse according to the covariance matrix.\n"
			<< "  -save        Draw the covariance in the image and save to file.\n";

		exit(0);
	}

	// parse arguments
	int arg = 0;
	while (++arg < argc) {
		if (!strcmp(argv[arg], "-d"))
			imgDir = argv[++arg];
		else if (!strcmp(argv[arg], "-i"))
			imgFile = argv[++arg];
		else if (!strcmp(argv[arg], "-k"))
			keyFileEnding = argv[++arg];
		else if (!strcmp(argv[arg], "-q"))
			verbose = false;
		else if (!strcmp(argv[arg], "-save"))
			saveCov = true;
		else if (!strcmp(argv[arg], "-show"))
			showCov = true;
		else
			cout << "Warning: Skipping unknown parameter " << argv[arg] << "." << endl;
	}

	if (imgFile == NULL) {
		cout << "COV Error: No image file specified." << endl;
		exit(1);
	}
	else {
		int i = 0;
		while (imgFile[i] != '.') imgFileStart[i] = imgFile[i++];
		imgFileStart[i] = 0;
	}

	if (keyFileEnding == NULL) {

		keyFileEnding = FE_KEYS_SIFT;
	}

	keyFile.clear();
	keyFile.append(imgFileStart).append(keyFileEnding);

}