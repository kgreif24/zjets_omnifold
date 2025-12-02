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
#include <TEnv.h>
using namespace std;

class HistoGroup {

    public:

        HistoGroup(string name = "default", int kinematic_region = 0, bool is_truth = true, bool do_IBU = false, bool is_data = false);
        ~HistoGroup();
        void WriteHistoMap(TFile& foutput);
        void WriteProfMap(TFile& foutput);
        void WriteGroup(TFile& foutput);
        void MergeHistoMaps(const HistoGroup& other);
        void MergeProfMaps(const HistoGroup& other);
        void MergeGroup(const HistoGroup& other);
        TEnv *openSettingsFile(TString fileName);
        vector<TString> vectorize(TString str, TString sep= " ");
        vector<double>   numberize(TString str, TString sep= " ");
        TString getStr(TEnv *settings, TString key);

        // Name for the histogram group, prepended to each histogram name
        TEnv *settings;
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
        vector<Double_t> jetshape_edges;

        // Kinematic region
        int kinematic_region;
        bool is_truth;
        bool do_IBU;
        bool is_data;
        

};

#endif // HISTOGROUP_H