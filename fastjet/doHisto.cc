#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include "TString.h"
#include "MakeOmni.h"
#include "TChain.h"

using namespace std;

int main(int argc, char* argv[]){

	// Read command line args
	TString fileName;
	string weight_file;
	vector<string> weight_names;
	TString outFile;
	bool isTruth = false;
	int maxEvents = 5000000;
	int nEns = 0;
	int nBootstrapData = 0;
	int kinematic_region = 0;
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
		if (arg == "--weight_file") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				weight_file = string(argv[i+1]);
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--weight_file option requires one argument." << std::endl;
				return 1;
			}
		}
		if (arg == "--weight_names") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				string systList = string(argv[i+1]);
				// Parse comma separated list into vector
				size_t pos = 0;
				string token;
				while ((pos = systList.find(",")) != string::npos) {
					token = systList.substr(0, pos);
					weight_names.push_back(token);
					systList.erase(0, pos + 1);
				}
				weight_names.push_back(systList); // Add the last token
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--weight_names option requires one argument." << std::endl;
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
		if (arg == "--nBootstrapData") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				nBootstrapData = std::stoi(argv[i+1]);
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--nBootstrapData option requires one argument." << std::endl;
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
		if (arg == "--kinematic_region") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				kinematic_region = std::stoi(argv[i+1]);
				i++; // Move to the next arg
			} else { // Throw error if no argument provided
				std::cerr << "--kinematic_region option requires one argument." << std::endl;
				return 1;
			}
		}
    }

	// Auto-detect weight names if not provided
	if (weight_names.size() == 0 && weight_file != "") {
		cout << "No weight names specified, auto-detecting from weight file..." << endl;
		cout << "Attempting to read from: " << weight_file << endl;
		try {
			weight_names = MakeOmni::DetectWeightNames(weight_file);
			cout << "Detected " << weight_names.size() << " weight arrays:" << endl;
			for (const auto& weight_name : weight_names) {
				cout << "  - " << weight_name << endl;
			}
		} catch (const std::exception& e) {
			std::cerr << "Error auto-detecting weight names: " << e.what() << std::endl;
			std::cerr << "Please check that the weight file exists and is readable." << std::endl;
			std::cerr << "You can also manually specify weight names using --weight_names" << std::endl;
			return 1;
		}
	}

	// Set up the chain
	TChain* myChain = new TChain("OmniTree");
	myChain->Add(fileName);
	cout << "Building hists from file: " << fileName << endl;
	cout << "Using weights from file: " << weight_file << endl;
	if (weight_names.size() > 0) {
		cout << "Using weight_names: ";
		for (const auto& weight_name : weight_names) {
			cout << weight_name << " ";
		}
		cout << endl;
	}
	cout << "Using truth: " << isTruth << endl;
	cout << "Has entries: " << myChain->GetEntries() << endl;
	cout << "Max events: " << maxEvents << endl;
	cout << "Kinematic region: " << kinematic_region << endl;

	// Run the analysis
	MakeOmni* myAnalysis = new MakeOmni(myChain, weight_file, weight_names, outFile, isTruth, nEns, nBootstrapData, kinematic_region);
	myAnalysis->Loop(maxEvents);

	return 0;

}
