#include <iostream>
#include <string>
#include <vector>
#include "TString.h"
#include "MakeOmni.h"
#include "TChain.h"

using namespace std;

int main(int argc, char* argv[]){

	// Read command line args
	TString fileName;
	string weights;
	vector<string> ens_weights;
	TString outFile;
	bool isTruth = false;
	int maxEvents = 5000000;
	int nEns = 0;
	vector<string> syst_weights;
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
		if (arg == "--syst") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				string systList = string(argv[i+1]);
				// Parse comma separated list into vector
				size_t pos = 0;
				string token;
				while ((pos = systList.find(",")) != string::npos) {
					token = systList.substr(0, pos);
					syst_weights.push_back(token);
					systList.erase(0, pos + 1);
				}
				syst_weights.push_back(systList); // Add the last token
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--syst option requires one argument." << std::endl;
				return 1;
			}
		}
		if (arg == "--nEns") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				nEns = std::stoi(argv[i+1]);
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--nEns option requires one argument." << std::endl;
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
	if (ens_weights.size() > 0) {
		cout << "Using ens_weights: ";
		for (const auto& ens_weight : ens_weights) {
			cout << ens_weight << " ";
		}
		cout << endl;
	}
	cout << "Using truth: " << isTruth << endl;
	cout << "Has entries: " << myChain->GetEntries() << endl;
	cout << "Max events: " << maxEvents << endl;

	// Run the analysis
	MakeOmni* myAnalysis = new MakeOmni(myChain, weights, outFile, isTruth, nEns, syst_weights);
	myAnalysis->Loop(maxEvents);

	return 0;

}
