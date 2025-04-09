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
    hm3_R04 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hm4_R04 = make_shared<TH1D>(TH1D("", "", 5, m_edges));
    hEEC_R04 = make_shared<TH1D>(TH1D("", "", nbins, binEdges));
    hLund_z_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    hLund_dR_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    hLund_plane_R04 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    hpT_R06 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_R06 = make_shared<TH1D>(TH1D("", "", nbins, binEdges));
    hLund_z_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    hLund_dR_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    hLund_plane_R06 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    hpT_R10 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_R10 = make_shared<TH1D>(TH1D("", "", nbins, binEdges));
    hLund_z_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
    hLund_dR_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
    hLund_plane_R10 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

    hpT_CA04 = make_shared<TH1D>(TH1D("", "", 6, pT_edges));
    hEEC_CA04 = make_shared<TH1D>(TH1D("", "", nbins, binEdges));

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

    hpT_R10->Write((name + "hpT_R10").c_str());
    hEEC_R10->Write((name + "hEEC_R10").c_str());
    hLund_z_R10->Write((name + "hLund_z_R10").c_str());
    hLund_dR_R10->Write((name + "hLund_dR_R10").c_str());
    hLund_plane_R10->Write((name + "hLund_plane_R10").c_str());

    hpT_CA04->Write((name + "hpT_CA04").c_str());
    hEEC_CA04->Write((name + "hEEC_CA04").c_str());

}