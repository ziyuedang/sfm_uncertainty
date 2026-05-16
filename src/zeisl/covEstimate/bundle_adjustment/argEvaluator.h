#pragma once
#include <iostream>
#include <string>

using namespace std;

class argEvaluator {

public:
	/*** Constructor, destructor ***/
	argEvaluator() :
		resultsDir("./") {};

	argEvaluator(int argc, char* argv[]) :
		 decType(-1){
		evaluate(argc, argv);
	};

	/*** Methods ***/

	void evaluate(int argc, char* argv[]);


	/*** Member variables ***/
	char* resultsDir;
	int decType;

};

/*** Implementations ***/

void argEvaluator::evaluate(int argc, char* argv[]) {
	// Information on usage
	if (argc == 1) {
		cout << "USAGE:\n"
			<< " blablablah.exe edit this later \n"
			<< endl;
		exit(0);
	}
}