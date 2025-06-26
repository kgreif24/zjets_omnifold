#include <iostream>
#include <vector>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include "HistoGroup.h"
using namespace std;

HistoGroup::HistoGroup(string name) : name(name) {

    // Define log spaced bins for EEC plots
    Long64_t nbins = 20;
    Double_t logxmin = -8;
    Double_t logxmax = 0.5;
    Double_t binEdges[nbins+1];
    for (Long64_t i = 0; i <= nbins; ++i) {
       binEdges[i] = pow(10, logxmin + i*(logxmax-logxmin)/nbins);
    }

    // Initialize histograms
    Double_t m_edges[] = {0, 2.5, 5.0, 10.0, 20.0, 30.0};
    Double_t pT_edges[] = {5.0, 50.0, 100.0, 150.0, 200.0, 300.0, 1000.0};
    Double_t mjj_edges[] = {0, 200., 400., 600., 800., 1000.};
    Double_t dRjj_edges[] = {0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0};
    Double_t dyjj_edges[] = {0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0};

    hm3_R04 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hm4_R04 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hEEC_R04 = make_shared<TH1D>(TH1D("", "", nbins, binEdges));
    hLund_z_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    hLund_dR_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    hLund_plane_R04 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 11, 0.5, 6));

    hpT_R06 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_R06 = make_shared<TH1D>(TH1D("", "", nbins, binEdges));
    hLund_z_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    hLund_dR_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    hLund_plane_R06 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    hm1_R10 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hpT_R10 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_R10 = make_shared<TH1D>(TH1D("", "", nbins, binEdges));
    hLund_z_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    hLund_dR_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    hLund_plane_R10 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    hpT_CA04 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_CA04 = make_shared<TH1D>(TH1D("", "", nbins, binEdges));
    hmjj_CA04 = make_shared<TH1D>(TH1D("", "", 5, mjj_edges));
    hdRjj_CA04 = make_shared<TH1D>(TH1D("", "", 10, dRjj_edges));
    hdyjj_CA04 = make_shared<TH1D>(TH1D("", "", 10, dyjj_edges));

    hm1_CA06 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hpT_CA06 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));

    hm1_KT04 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hpT_KT04 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));

}

HistoGroup::~HistoGroup() {
    // Destructor
    // Histograms will be automatically deleted when the unique_ptr goes out of scope
}

void HistoGroup::WriteHistos(TFile& foutput) {
    // Write histograms to the output file
    foutput.cd();
    hm3_R04->Write((name + "hm3_R04").c_str());
    hm4_R04->Write((name + "hm4_R04").c_str());
    hEEC_R04->Write((name + "hEEC_R04").c_str());
    hLund_z_R04->Write((name + "hLund_z_R04").c_str());
    hLund_dR_R04->Write((name + "hLund_dR_R04").c_str());
    hLund_plane_R04->Write((name + "hLund_plane_R04").c_str());

    hpT_R06->Write((name + "hpT_R06").c_str());
    hEEC_R06->Write((name + "hEEC_R06").c_str());
    hLund_z_R06->Write((name + "hLund_z_R06").c_str());
    hLund_dR_R06->Write((name + "hLund_dR_R06").c_str());
    hLund_plane_R06->Write((name + "hLund_plane_R06").c_str());

    hm1_R10->Write((name + "hm1_R10").c_str());
    hpT_R10->Write((name + "hpT_R10").c_str());
    hEEC_R10->Write((name + "hEEC_R10").c_str());
    hLund_z_R10->Write((name + "hLund_z_R10").c_str());
    hLund_dR_R10->Write((name + "hLund_dR_R10").c_str());
    hLund_plane_R10->Write((name + "hLund_plane_R10").c_str());

    hpT_CA04->Write((name + "hpT_CA04").c_str());
    hEEC_CA04->Write((name + "hEEC_CA04").c_str());
    hmjj_CA04->Write((name + "hmjj_CA04").c_str());
    hdRjj_CA04->Write((name + "hdRjj_CA04").c_str());
    hdyjj_CA04->Write((name + "hdyjj_CA04").c_str());

    hm1_CA06->Write((name + "hm1_CA06").c_str());
    hpT_CA06->Write((name + "hpT_CA06").c_str());

    hm1_KT04->Write((name + "hm1_KT04").c_str());
    hpT_KT04->Write((name + "hpT_KT04").c_str());

}