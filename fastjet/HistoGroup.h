/*

HistoGroup.h - Header file for the HistoGroup struct, which is used to manage histograms in the MakeOmni analysis.
One set of histograms is created for each reweighting passed to MakeOmni, whether it be the central value or a variation.

*/

#ifndef HISTOGROUP_H
#define HISTOGROUP_H

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <string>
#include <vector>
#include <iostream>
using namespace std;

class HistoGroup {

    public:

        HistoGroup(string name = "default", int kinematic_region = 0);
        ~HistoGroup();
        void WriteHistoMap(TFile& foutput);
        void WriteProfMap(TFile& foutput);
        void WriteGroup(TFile& foutput);
        void MergeHistoMaps(const HistoGroup& other);
        void MergeProfMaps(const HistoGroup& other);
        void MergeGroup(const HistoGroup& other);

        // Name for the histogram group, prepended to each histogram name
        string name;

        map<string,shared_ptr<TH1D>> h_map;
        map<string,shared_ptr<TH1D>> h2d_map;
        map<string,shared_ptr<TProfile>> prof_map;

        vector<Double_t> pTj1_bins;
        vector<Double_t> yj1_bins;
        vector<Double_t> antikt_jetR;
        vector<Double_t> antikt_jetR_bins;
        vector<Double_t> ca_jetR;
        vector<Double_t> ca_jetR_bins;

        // Kinematic region
        int kinematic_region;

};

#endif // HISTOGROUP_H