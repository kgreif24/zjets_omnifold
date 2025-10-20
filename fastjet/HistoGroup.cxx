#include <iostream>
#include <iomanip>
#include <vector>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include "HistoGroup.h"
using namespace std;

HistoGroup::HistoGroup(string name, int kinematic_region) : name(name), kinematic_region(kinematic_region) {

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

    hm1_R04 = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    hm2_R04 = make_shared<TH1D>(TH1D("", "", 4, m2_edges));
    hm3_R04 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hm4_R04 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hmjj_R04 = make_shared<TH1D>(TH1D("", "", mjj_nbins, mjj_edges));
    hdyjj_R04 = make_shared<TH1D>(TH1D("", "", dyjj_nbins, dyjj_edges));
    hEEC_R04 = make_shared<TH1D>(TH1D("", "", nbins, ak4_edges));
    // hLund_z_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    // hLund_dR_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    // hLund_plane_R04 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 11, 0.5, 6));

    hm1_R06 = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    hpT_R06 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_R06 = make_shared<TH1D>(TH1D("", "", nbins, ak6_edges));
    // hLund_z_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    // hLund_dR_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    // hLund_plane_R06 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    hm1_R10 = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    hpT_R10 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_R10 = make_shared<TH1D>(TH1D("", "", nbins, ak10_edges));
    // hLund_z_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    // hLund_dR_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    // hLund_plane_R10 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    hm1_CA04 = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    hpT_CA04 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_CA04 = make_shared<TH1D>(TH1D("", "", nbins, ak4_edges));
    hmjj_CA04 = make_shared<TH1D>(TH1D("", "", mjj_nbins, mjj_edges));
    hdRjj_CA04 = make_shared<TH1D>(TH1D("", "", dRjj_nbins, dRjj_edges));
    hdyjj_CA04 = make_shared<TH1D>(TH1D("", "", dyjj_nbins, dyjj_edges));
    hdphijj_CA04 = make_shared<TH1D>(TH1D("", "", 10, dphijj_edges));

    hm1_CA06 = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    hpT_CA06 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));

    hm1_KT04 = make_shared<TH1D>(TH1D("", "", 6, m1_edges));
    hpT_KT04 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));

    hTEEC_collinear = make_shared<TH1D>(TH1D("", "", nbins, all_edges));
    hTEEC_full_nolog = make_shared<TH1D>(TH1D("", "", 20, 0.0, 1.0));
    hTEEC_b2b = make_shared<TH1D>(TH1D("", "", nbins, b2b_edges));
    hTEEC_full = make_shared<TH1D>(TH1D("", "", 2*nbins, symlog_edges));
    hTEEC_z_collinear = make_shared<TH1D>(TH1D("", "", nbins, b2b_edges));
    hTEEC_z_full_nolog = make_shared<TH1D>(TH1D("", "", 20, 0.0, 1.0));
    hTEEC_z_b2b = make_shared<TH1D>(TH1D("", "", nbins, all_edges));
    hTEEC_z_full = make_shared<TH1D>(TH1D("", "", 2*nbins, symlog_edges));

}

HistoGroup::~HistoGroup() {
    // Destructor
    // Histograms will be automatically deleted when the unique_ptr goes out of scope
}

void HistoGroup::WriteHistos(TFile& foutput) {
    // Write histograms to the output file
    foutput.cd();
    hm1_R04->Write((name + "hm1_R04").c_str());
    hm2_R04->Write((name + "hm2_R04").c_str());
    hm3_R04->Write((name + "hm3_R04").c_str());
    hm4_R04->Write((name + "hm4_R04").c_str());
    hmjj_R04->Write((name + "hmjj_R04").c_str());
    hdyjj_R04->Write((name + "hdyjj_R04").c_str());
    hEEC_R04->Write((name + "hEEC_R04").c_str());
    // hLund_z_R04->Write((name + "hLund_z_R04").c_str());
    // hLund_dR_R04->Write((name + "hLund_dR_R04").c_str());
    // hLund_plane_R04->Write((name + "hLund_plane_R04").c_str());

    hm1_R06->Write((name + "hm1_R06").c_str());
    hpT_R06->Write((name + "hpT_R06").c_str());
    hEEC_R06->Write((name + "hEEC_R06").c_str());
    // hLund_z_R06->Write((name + "hLund_z_R06").c_str());
    // hLund_dR_R06->Write((name + "hLund_dR_R06").c_str());
    // hLund_plane_R06->Write((name + "hLund_plane_R06").c_str());

    hm1_R10->Write((name + "hm1_R10").c_str());
    hpT_R10->Write((name + "hpT_R10").c_str());
    hEEC_R10->Write((name + "hEEC_R10").c_str());
    // hLund_z_R10->Write((name + "hLund_z_R10").c_str());
    // hLund_dR_R10->Write((name + "hLund_dR_R10").c_str());
    // hLund_plane_R10->Write((name + "hLund_plane_R10").c_str());

    hm1_CA04->Write((name + "hm1_CA04").c_str());
    hpT_CA04->Write((name + "hpT_CA04").c_str());
    hEEC_CA04->Write((name + "hEEC_CA04").c_str());
    hmjj_CA04->Write((name + "hmjj_CA04").c_str());
    hdRjj_CA04->Write((name + "hdRjj_CA04").c_str());
    hdyjj_CA04->Write((name + "hdyjj_CA04").c_str());
    hdphijj_CA04->Write((name + "hdphijj_CA04").c_str());

    hm1_CA06->Write((name + "hm1_CA06").c_str());
    hpT_CA06->Write((name + "hpT_CA06").c_str());

    hm1_KT04->Write((name + "hm1_KT04").c_str());
    hpT_KT04->Write((name + "hpT_KT04").c_str());

    hTEEC_collinear->Write((name + "hTEEC_collinear").c_str());
    hTEEC_full_nolog->Write((name + "hTEEC_full_nolog").c_str());
    hTEEC_b2b->Write((name + "hTEEC_b2b").c_str());
    hTEEC_full->Write((name + "hTEEC_full").c_str());
    hTEEC_z_collinear->Write((name + "hTEEC_z_collinear").c_str());
    hTEEC_z_full_nolog->Write((name + "hTEEC_z_full_nolog").c_str());
    hTEEC_z_b2b->Write((name + "hTEEC_z_b2b").c_str());
    hTEEC_z_full->Write((name + "hTEEC_z_full").c_str());

}

void HistoGroup::MergeHistos(const HistoGroup& other) {
    // Merge all histograms from other into this group
    hm1_R04->Add(other.hm1_R04.get());
    hm2_R04->Add(other.hm2_R04.get());
    hm3_R04->Add(other.hm3_R04.get());
    hm4_R04->Add(other.hm4_R04.get());
    hmjj_R04->Add(other.hmjj_R04.get());
    hdyjj_R04->Add(other.hdyjj_R04.get());
    hEEC_R04->Add(other.hEEC_R04.get());
    
    hm1_R06->Add(other.hm1_R06.get());
    hpT_R06->Add(other.hpT_R06.get());
    hEEC_R06->Add(other.hEEC_R06.get());
    
    hm1_R10->Add(other.hm1_R10.get());
    hpT_R10->Add(other.hpT_R10.get());
    hEEC_R10->Add(other.hEEC_R10.get());

    hm1_CA04->Add(other.hm1_CA04.get());
    hpT_CA04->Add(other.hpT_CA04.get());
    hEEC_CA04->Add(other.hEEC_CA04.get());
    hmjj_CA04->Add(other.hmjj_CA04.get());
    hdRjj_CA04->Add(other.hdRjj_CA04.get());
    hdyjj_CA04->Add(other.hdyjj_CA04.get());
    hdphijj_CA04->Add(other.hdphijj_CA04.get());
    
    hm1_CA06->Add(other.hm1_CA06.get());
    hpT_CA06->Add(other.hpT_CA06.get());
    
    hm1_KT04->Add(other.hm1_KT04.get());
    hpT_KT04->Add(other.hpT_KT04.get());
    
    hTEEC_collinear->Add(other.hTEEC_collinear.get());
    hTEEC_full_nolog->Add(other.hTEEC_full_nolog.get());
    hTEEC_b2b->Add(other.hTEEC_b2b.get());
    hTEEC_full->Add(other.hTEEC_full.get());
    hTEEC_z_collinear->Add(other.hTEEC_z_collinear.get());
    hTEEC_z_full_nolog->Add(other.hTEEC_z_full_nolog.get());
    hTEEC_z_b2b->Add(other.hTEEC_z_b2b.get());
    hTEEC_z_full->Add(other.hTEEC_z_full.get());
}