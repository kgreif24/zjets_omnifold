#include <iostream>
#include "TString.h"
#include "MakeOmni.h"
#include "DoPlots.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"
#include <string>
#include <vector>
#include <TLorentzVector.h>
#include "AtlasStyle.h"
#include "AtlasLabels.h"
#include "AtlasUtils.h"

using namespace std ;

int main(int argc, char* argv[]){

	std::string omniOrtrue(argv[1]);
	TString OmniOrTruth = TString(omniOrtrue.c_str());
	TString theLink;
	TChain * myChain;
	// file for omni : /global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root
	// but for omni take tmp_mc.root since omni_test_weights are added as a branch 
	// file for truth : /global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_TruthPseudodata_Mar12_Combined_1_40_shuffled.root
	
	// Option to run the final plotting once histograms have been filled
	if (OmniOrTruth=="plots"){

		SetAtlasStyle();
		TFile* outputFileOmni  = TFile::Open("out/output_omni.root");
		TFile* outputFileTruth  = TFile::Open("out/output_truth.root");
	
		FinalPlots(outputFileOmni, outputFileTruth, "hpT_R04");
		FinalPlots(outputFileOmni, outputFileTruth, "hpT_R06");
		FinalPlots(outputFileOmni, outputFileTruth, "hpT_R10");
		FinalPlots(outputFileOmni, outputFileTruth, "hpT_CA04");
		FinalPlots(outputFileOmni, outputFileTruth, "hLund_z_R04");
		FinalPlots(outputFileOmni, outputFileTruth, "hLund_z_R06");
		FinalPlots(outputFileOmni, outputFileTruth, "hLund_z_R10");
		FinalPlots(outputFileOmni, outputFileTruth, "hLund_z_CA04");
		FinalPlots(outputFileOmni, outputFileTruth, "hLund_dR_R04");
		FinalPlots(outputFileOmni, outputFileTruth, "hLund_dR_R06");
		FinalPlots(outputFileOmni, outputFileTruth, "hLund_dR_R10");
		FinalPlots(outputFileOmni, outputFileTruth, "hLund_dR_CA04");
		FinalPlots(outputFileOmni, outputFileTruth, "hEEC_R04");
		FinalPlots(outputFileOmni, outputFileTruth, "hEEC_R06");
		FinalPlots(outputFileOmni, outputFileTruth, "hEEC_R10");
		FinalPlots(outputFileOmni, outputFileTruth, "hEEC_CA04");
		FinalPlots(outputFileOmni, outputFileTruth, "hTEEC");
		FinalPlots(outputFileOmni, outputFileTruth, "h_fracpT_ring");

	// Else we need to build histograms from the MC / truth pseudodata
	} else {

		// Set input file
		bool isTruth = false;
		Long64_t maxEvents = 0;
		if (OmniOrTruth == "reco") {
			std::cout << " do plots for omni ... " << std::endl;
			theLink = "./plotting_mc/test.root";
		} else if (OmniOrTruth == "truth") {
			std::cout << " do plots for truth ... " << std::endl;
			theLink = "/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_TruthPseudodata_Mar12_Combined_1_50_Top_shuffled.root";
			isTruth = true;
			maxEvents = 5000000; // Set a limit for truth events
		} else {
			std::cout << "Invalid option. Please choose 'reco'/'truth' or 'plots'." << std::endl;
			return 1;
		}

		// Set up the chain
		myChain = new TChain( "OmniTree" );
		myChain->Add( theLink );
		cout << "my link = " << theLink << endl ;
		cout << "my chain = " << myChain->GetEntries() << endl ;

		// Run the analysis
		MakeOmni* myAnalysis = new MakeOmni( myChain, isTruth );
		myAnalysis->Loop(maxEvents);

	}

	return 0;

}
