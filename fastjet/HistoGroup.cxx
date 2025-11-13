#include <iostream>
#include <iomanip>
#include <vector>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TMath.h>
#include "HistoGroup.h"
using namespace std;

HistoGroup::HistoGroup(string name, int kinematic_region) : name(name), kinematic_region(kinematic_region) {

    // binning for <m/pT> study
    pTj1_bins = {10, 75, 125, 200, 1000};
    yj1_bins  = {0, 0.7, 1.3, 1.8, 2.5};
    antikt_jetR = {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85};
    ca_jetR     = {0.2, 0.4, 0.6, 0.8};

    antikt_jetR_bins.push_back(antikt_jetR.front() - 0.5 * (antikt_jetR[1] - antikt_jetR[0]));
    for (size_t i = 1; i < antikt_jetR.size(); ++i) antikt_jetR_bins.push_back(0.5 * (antikt_jetR[i - 1] + antikt_jetR[i]));

    ca_jetR_bins.push_back(ca_jetR.front() - 0.5 * (ca_jetR[1] - ca_jetR[0]));
    for (size_t i = 1; i < ca_jetR.size(); ++i) ca_jetR_bins.push_back(0.5 * (ca_jetR[i - 1] + ca_jetR[i]));

    // Define log spaced bins for EEC plots
    const Long64_t nbins = 20;
    Double_t logxmin_b2b = -2;
    Double_t logxmin_all = -6;
    Double_t logxmax_ak4 = -2;
    Double_t logxmax_ak6 = TMath::Log10(0.05);
    Double_t logxmax_ak10 = -1;
    Double_t logxmax_all = TMath::Log10(0.5);
    Double_t ak4_edges[nbins+1];
    Double_t ak6_edges[nbins+1];
    Double_t ak10_edges[nbins+1];
    Double_t b2b_edges[nbins+1];
    Double_t all_edges[nbins+1];

    for (Long64_t i = 0; i <= nbins; ++i) {
       ak4_edges[i] = pow(10, logxmin_all + i*(logxmax_ak4-logxmin_all)/nbins);
       ak6_edges[i] = pow(10, logxmin_all + i*(logxmax_ak6-logxmin_all)/nbins);
       ak10_edges[i] = pow(10, logxmin_all + i*(logxmax_ak10-logxmin_all)/nbins);
       b2b_edges[i] = pow(10, logxmin_b2b + i*(logxmax_all-logxmin_b2b)/nbins);
       all_edges[i] = pow(10, logxmin_all + i*(logxmax_all-logxmin_all)/nbins);
    }

    const Long64_t nbins_symlog_coll = 30;
    Double_t logxmin_symlog_coll = -5;
    Double_t symlog_center = TMath::Log10(0.5);
    const Long64_t nbins_symlog_b2b = 11;
    const Long64_t nbins_symlog = nbins_symlog_coll + nbins_symlog_b2b;
    Double_t logxmin_symlog_b2b = -2;
    Double_t symlog_edges[nbins_symlog+1];
    for (Long64_t i = 0; i <= nbins_symlog_coll; ++i) {
       symlog_edges[i] = pow(10, logxmin_symlog_coll + i*(symlog_center-logxmin_symlog_coll)/nbins_symlog_coll);
    }
    for (Long64_t i = 0; i <= nbins_symlog_b2b; ++i) {
       symlog_edges[nbins_symlog - i] = 1 - pow(10, logxmin_symlog_b2b + i*(symlog_center-logxmin_symlog_b2b)/nbins_symlog_b2b);
    }

    // Initialize histograms
    Double_t m_edges[] = {0, 2.5, 5.0, 10.0, 20.0, 30.0};
    Double_t m1_edges[] = {0, 8.0, 16.0, 24.0, 32.0, 42.0, 70.0};
    Double_t m2_edges[] = {0, 5.0, 10.0, 20.0, 40.0};
    Double_t pT_edges[] = {5.0, 50.0, 100.0, 150.0, 200.0, 300.0, 1000.0};
    Double_t pi = TMath::Pi();
    Double_t dphijj_edges[] = {-pi, -7*pi/8, -3*pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3*pi/4, 7*pi/8, pi};
    
    // Define arrays for different kinematic regions
    Double_t mjj_edges_default[] = {0, 200., 400., 600., 800., 1000.};
    Double_t dyjj_edges_default[] = {0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0};
    Double_t dRjj_edges_default[] = {0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0};
    
    Double_t mjj_edges_region2[] = {200., 300., 400., 600., 800., 1000.};
    Double_t dyjj_edges_region2[] = {2.0, 2.4, 2.8, 3.2, 3.6, 4.0};
    Double_t dRjj_edges_region2[] = {2.0, 2.4, 2.8, 3.2, 3.6, 4.0};
    
    // Choose which arrays to use based on kinematic region
    Double_t* mjj_edges;
    Double_t* dyjj_edges;
    Double_t* dRjj_edges;
    int mjj_nbins, dyjj_nbins, dRjj_nbins;
    
    if (kinematic_region == 2) {
        mjj_edges = mjj_edges_region2;
        dyjj_edges = dyjj_edges_region2;
        dRjj_edges = dRjj_edges_region2;
        mjj_nbins = 5;  // 6 edges = 5 bins
        dyjj_nbins = 5; // 6 edges = 5 bins
        dRjj_nbins = 5; // 6 edges = 5 bins
    } else {
        mjj_edges = mjj_edges_default;
        dyjj_edges = dyjj_edges_default;
        dRjj_edges = dRjj_edges_default;
        mjj_nbins = 5;   // 6 edges = 5 bins
        dyjj_nbins = 10; // 11 edges = 10 bins
        dRjj_nbins = 10; // 11 edges = 10 bins
    }


    h_map["hm1_R04"] = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    h_map["hm2_R04"] = make_shared<TH1D>(TH1D("", "", 4, m2_edges));
    h_map["hm3_R04"] = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    h_map["hm4_R04"] = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    h_map["hmjj_R04"] = make_shared<TH1D>(TH1D("", "", mjj_nbins, mjj_edges));
    h_map["hdyjj_R04"] = make_shared<TH1D>(TH1D("", "", dyjj_nbins, dyjj_edges));
    h_map["hEEC_R04"] = make_shared<TH1D>(TH1D("", "", nbins, ak4_edges));
    // hLund_z_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    // hLund_dR_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    // hLund_plane_R04 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 11, 0.5, 6));

    h_map["hm1_R06"] = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    h_map["hpT_R06"] = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    h_map["hEEC_R06"] = make_shared<TH1D>(TH1D("", "", nbins, ak6_edges));
    // hLund_z_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    // hLund_dR_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    // hLund_plane_R06 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    h_map["hm1_R10"] = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    h_map["hpT_R10"] = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    h_map["hEEC_R10"] = make_shared<TH1D>(TH1D("", "", nbins, ak10_edges));
    // hLund_z_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    // hLund_dR_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    // hLund_plane_R10 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    h_map["hm1_CA04"] = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    h_map["hpT_CA04"] = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    h_map["hEEC_CA04"] = make_shared<TH1D>(TH1D("", "", nbins, ak4_edges));
    h_map["hmjj_CA04"] = make_shared<TH1D>(TH1D("", "", mjj_nbins, mjj_edges));
    h_map["hdRjj_CA04"] = make_shared<TH1D>(TH1D("", "", dRjj_nbins, dRjj_edges));
    h_map["hdyjj_CA04"] = make_shared<TH1D>(TH1D("", "", dyjj_nbins, dyjj_edges));
    h_map["hdphijj_CA04"] = make_shared<TH1D>(TH1D("", "", 10, dphijj_edges));

    h_map["hm1_CA06"] = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    h_map["hpT_CA06"] = make_shared<TH1D>(TH1D("", "", 6, pT_edges));

    h_map["hm1_KT04"] = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    h_map["hpT_KT04"] = make_shared<TH1D>(TH1D("", "", 6, pT_edges));

    h_map["hTEEC_collinear"] = make_shared<TH1D>(TH1D("", "", nbins, all_edges));
    h_map["hTEEC_full_nolog"] = make_shared<TH1D>(TH1D("", "", 20, 0.0, 1.0));
    h_map["hTEEC_b2b"] = make_shared<TH1D>(TH1D("", "", nbins, b2b_edges));
    h_map["hTEEC_full"] = make_shared<TH1D>(TH1D("", "", 2*nbins, symlog_edges));
    h_map["hTEEC_z_collinear"] = make_shared<TH1D>(TH1D("", "", nbins, b2b_edges));
    h_map["hTEEC_z_full_nolog"] = make_shared<TH1D>(TH1D("", "", 20, 0.0, 1.0));
    h_map["hTEEC_z_b2b"] = make_shared<TH1D>(TH1D("", "", nbins, all_edges));
    h_map["hTEEC_z_full"] = make_shared<TH1D>(TH1D("", "", 2*nbins, symlog_edges));

    prof_map["prof_aktRVaried_M1OverpT_All"] = std::make_shared<TProfile>( TProfile("", "", antikt_jetR_bins.size() - 1, antikt_jetR_bins.data()));
    prof_map["prof_caRVaried_M1OverpT_All"]  = std::make_shared<TProfile>( TProfile("", "", ca_jetR_bins.size() - 1, ca_jetR_bins.data()));
    for (long unsigned int i = 0; i < pTj1_bins.size()-1; i++){
        for (long unsigned int j = 0; j < yj1_bins.size()-1; j++){
            string prof_name;
            prof_name           =  "prof_aktRVaried_M1OverpT_pTj1bin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
            prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", antikt_jetR_bins.size() - 1, antikt_jetR_bins.data()));

            prof_name           = "prof_caRVaried_M1OverpT_pTj1bin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
            prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", ca_jetR_bins.size() - 1, ca_jetR_bins.data()));
        }
    }

    // prof_map["h2d_m1OverpT_JetR_CA"] = make_shared<TProfile>(TH1D("", "", 2*nbins, symlog_edges));
    // prof_map["h2d_m1OverpT_pT_CA"]   = make_shared<TProfile>(TH1D("", "", 2*nbins, symlog_edges));


}

HistoGroup::~HistoGroup() {
    // Destructor
    // Histograms will be automatically deleted when the unique_ptr goes out of scope
}

void HistoGroup::WriteHistoMap(TFile& foutput) {
    // Write histograms to the output file
    foutput.cd();
    string key; std::shared_ptr<TH1D> h;
    for (auto elmnt:h_map){
        key = elmnt.first;
        h = elmnt.second;
        h.get()->SetName((name + key).c_str());
        h.get()->Write((name + key).c_str(),TObject::kOverwrite);
    }
}

void HistoGroup::WriteProfMap(TFile& foutput) {
    foutput.cd();
    string key; std::shared_ptr<TProfile> p;
    for (auto elmnt:prof_map){
        key = elmnt.first;
        p = elmnt.second;
        p->SetName((name + key).c_str());
        p->Write((name + key).c_str(),TObject::kOverwrite);
    }
}

void HistoGroup::WriteGroup(TFile& foutput) {
    // Merge all histograms from other into this group
    WriteHistoMap(foutput);
    WriteProfMap(foutput);
}

void HistoGroup::MergeHistoMaps(const HistoGroup& other) {
    // Merge all histograms from other into this group
    for (auto& [key, h] : h_map){
        h->Add(other.h_map.at(key).get());
    }
}

void HistoGroup::MergeProfMaps(const HistoGroup& other) {
    // Merge all histograms from other into this group
    for (auto& [key, p] : prof_map){
        p->Add(other.prof_map.at(key).get());
    }
}

void HistoGroup::MergeGroup(const HistoGroup& other) {
    // Merge all histograms from other into this group
    MergeHistoMaps(other);
    MergeProfMaps(other);
}
