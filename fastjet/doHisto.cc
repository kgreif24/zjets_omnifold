#include <iostream>
#include <string>
#include "TString.h"
#include "MakeOmni.h"
#include "TChain.h"

using namespace std;

int main(int argc, char* argv[]){

	// Read command line args
	TString fileName;
	string weights;
	TString outFile;
	bool isTruth = false;
	int maxEvents = 5000000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
		if (arg == "--file") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				fileName = TString(argv[i+1]);
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--file option requires one argument." << std::endl;
				return 1;
			}
		}
		if (arg == "--weights") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				weights = string(argv[i+1]);
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--weights option requires one argument." << std::endl;
				return 1;
			}
		}
		if (arg == "--outFile") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				outFile = TString(argv[i+1]);
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--outFile option requires one argument." << std::endl;
				return 1;
			}
		}
        if (arg == "--truth") {
            isTruth = true;
        }
		if (arg == "--maxEvents") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				maxEvents = std::stoi(argv[i+1]);
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--maxEvents option requires one argument." << std::endl;
				return 1;
			}
		}
    }

	// Set up the chain
	TChain* myChain = new TChain("OmniTree");
	myChain->Add(fileName);
	cout << "Building hists from file: " << fileName << endl;
	cout << "Using weights: " << weights << endl;
	cout << "Using truth: " << isTruth << endl;
	cout << "Has entries: " << myChain->GetEntries() << endl;
	cout << "Max events: " << maxEvents << endl;

	// Run the analysis
	MakeOmni* myAnalysis = new MakeOmni(myChain, weights, outFile, isTruth);
	myAnalysis->Loop(maxEvents);

	return 0;

}
