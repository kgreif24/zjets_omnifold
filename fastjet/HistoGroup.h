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

        HistoGroup(string name = "default");
        ~HistoGroup();
        void WriteHistos(TFile& foutput);

        // Name for the histogram group, prepended to each histogram name
        string name;

        // Define pointers to histograms
        // R=0.4 jets
        shared_ptr<TH1D> hm3_R04;
        shared_ptr<TH1D> hm4_R04;
        shared_ptr<TH1D> hEEC_R04;
        shared_ptr<TH1D> hLund_z_R04;
        shared_ptr<TH1D> hLund_dR_R04;
        shared_ptr<TH2D> hLund_plane_R04;

        // R=0.6 jets
        shared_ptr<TH1D> hpT_R06;
        shared_ptr<TH1D> hEEC_R06;
        shared_ptr<TH1D> hLund_z_R06;
        shared_ptr<TH1D> hLund_dR_R06;
        shared_ptr<TH2D> hLund_plane_R06;

        // R=1.0 jets
        shared_ptr<TH1D> hpT_R10;
        shared_ptr<TH1D> hEEC_R10;
        shared_ptr<TH1D> hLund_z_R10;
        shared_ptr<TH1D> hLund_dR_R10;
        shared_ptr<TH2D> hLund_plane_R10;

        // CA R=0.4 jets
        shared_ptr<TH1D> hpT_CA04;
        shared_ptr<TH1D> hEEC_CA04;

};

#endif // HISTOGROUP_H