#include <iostream>
#include <string>
#include <vector>
#include "TString.h"
#include "MakeOmni.h"
#include "TChain.h"

using namespace std;

int main(int argc, char* argv[]){

	// Read command line args
	vector<TString> fileNames;
	string weight_file = "None";
	vector<string> weight_names;
  vector<string> trackVariations = {""}; //"" is nominal
	TString outFile;
	bool isTruth = false;
	int maxEvents = 5000000;
	int nEns = 0;
	int kinematic_region = 0;
  bool do_IBU = false;
  bool is_data = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
		if (arg == "--file") {
      if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				string systList = string(argv[i+1]);
				// Parse comma separated list into vector
				size_t pos = 0;
				string token;
				while ((pos = systList.find(",")) != string::npos) {
					token = systList.substr(0, pos);
					fileNames.push_back(token);
					systList.erase(0, pos + 1);
				}
				fileNames.push_back(systList);  // Add the last token
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
        if (weight_file == "None") nEns =0;
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
    if (arg == "--do_IBU") {
        do_IBU = true;
    }
    if (arg == "--is_data") {
        is_data = true;
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
    if (arg == "--track_variations") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				string systList = string(argv[i+1]);
				// Parse comma separated list into vector
				size_t pos = 0;
				string token;
				while ((pos = systList.find(",")) != string::npos) {
					token = systList.substr(0, pos);
					trackVariations.push_back(token+"_");
					systList.erase(0, pos + 1);
				}
				trackVariations.push_back(systList+"_"); // Add the last token
				i++; // Move to the next arg
			} else {
        trackVariations = {"", "syst_pTScale_","syst_Fake_","syst_TrackFilter_","syst_JetTrackFilter_"};
			}
		}
    }

	// Set up the chain
	TChain* myChain = new TChain("OmniTree");
  for (auto file:fileNames) myChain->Add(file);
	cout << "Using event from files: " << endl;
  for (auto file:fileNames) cout << " - " << file << endl;
  cout << "Using weights from file: " << weight_file << endl;
	if (weight_names.size() > 0) {
		cout << "Using weight_names: ";
		for (const auto& weight_name : weight_names) {
			cout << weight_name << " ";
		}
		cout << endl;
	}
  if (trackVariations.size() > 1) {
		cout << "Evaluating track systematic variations: ";
		for (long unsigned int i=0;i< trackVariations.size();i++) {
      if (i == 0) cout << "nominal ";
			else cout << trackVariations[i] << " ";
		}
		cout << endl;
	}
  if (do_IBU) cout << "Making branches for IBU "<< endl;
	else cout << "Use truth: " << isTruth << endl;
  if (is_data) cout << "Input is data "<< endl;
	cout << "Has entries: " << myChain->GetEntries() << endl;
	cout << "Max events: " << maxEvents << endl;
	cout << "Kinematic region: " << kinematic_region << endl;

	// Run the analysis
	MakeOmni* myAnalysis = new MakeOmni(myChain, weight_file, weight_names, outFile, isTruth, nEns, kinematic_region, trackVariations, do_IBU, is_data);
	myAnalysis->Loop(maxEvents);

	return 0;

}
