//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sun Sep  8 17:40:14 2024 by ROOT version 6.30/02
// from TTree OmniTree/
// found on file: /data/jmsardain/Zjets/data/tmp_mc.root
//////////////////////////////////////////////////////////

#ifndef MakeOmni_h
#define MakeOmni_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TString.h>
#include <TH1D.h>
#include <TH2D.h>
#include <vector>
#include <string>
#include <iostream>
#include "HistoGroup.h"
using namespace std;

// Header file for the classes stored in the TTree if any.

class MakeOmni {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   bool isTruth; // Flag to indicate if we are processing truth data
   // bool loadSystematics; // Flag to indicate if we are loading systematics
   string weightFilename;
   vector<string> weightBranchNames;
   int nEns;
   TString saveName;

   // HistoGroups
   vector<HistoGroup> centralHistoGroups;
   vector<HistoGroup> ensHistoGroups;

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Float_t         weight;
   Int_t           pass190;
   Int_t           truth_pass190;
   Float_t         weight_mc;
   Int_t           pass190_syst_ID_Up;
   Int_t           pass190_syst_ID_Down;
   Int_t           pass190_syst_MS_Up;
   Int_t           pass190_syst_MS_Down;
   Int_t           pass190_syst_MSResbias_Up;
   Int_t           pass190_syst_MSResbias_Down;
   Int_t           pass190_syst_MSRho_Up;
   Int_t           pass190_syst_MSRho_Down;
   Int_t           pass190_syst_Scale_Up;
   Int_t           pass190_syst_Scale_Down;
   Float_t         pT_ll;
   Float_t         pT_l1;
   Float_t         pT_l2;
   Float_t         eta_l1;
   Float_t         eta_l2;
   Float_t         phi_l1;
   Float_t         phi_l2;
   Float_t         y_ll;
   Float_t         pT_trackj1;
   Float_t         y_trackj1;
   Float_t         phi_trackj1;
   Float_t         m_trackj1;
   Float_t         tau1_trackj1;
   Float_t         tau2_trackj1;
   Float_t         tau3_trackj1;
   Float_t         pT_trackj2;
   Float_t         y_trackj2;
   Float_t         phi_trackj2;
   Float_t         m_trackj2;
   Float_t         tau1_trackj2;
   Float_t         tau2_trackj2;
   Float_t         tau3_trackj2;
   Int_t           EventNumber;
   Int_t           RunNumber;
   Int_t           Ntracks;
   Int_t           Ntracks_trackj1;
   Int_t           Ntracks_trackj2;
   Int_t           NtrackJets20;
   Int_t           nweight_bs;
   Int_t           weight_bs[100];   //[nweight_bs]
   Int_t           npT_tracks;
   Double_t        pT_tracks[309];   //[npT_tracks]
   Int_t           neta_tracks;
   Double_t        eta_tracks[309];   //[neta_tracks]
   Int_t           nphi_tracks;
   Double_t        phi_tracks[309];   //[nphi_tracks]
   Int_t           ntrackJetIndex_tracks;
   Double_t        trackJetIndex_tracks[309];   //[ntrackJetIndex_tracks]

   // List of branches
   TBranch        *b_weight;   //!
   TBranch        *b_pass190;   //!
   TBranch        *b_truth_pass190;   //!
   TBranch        *b_weight_mc;   //!
   TBranch        *b_pass190_syst_ID_Up;   //!
   TBranch        *b_pass190_syst_ID_Down;   //!
   TBranch        *b_pass190_syst_MS_Up;   //!
   TBranch        *b_pass190_syst_MS_Down;   //!
   TBranch        *b_pass190_syst_MSResbias_Up;   //!
   TBranch        *b_pass190_syst_MSResbias_Down;   //!
   TBranch        *b_pass190_syst_MSRho_Up;   //!
   TBranch        *b_pass190_syst_MSRho_Down;   //!
   TBranch        *b_pass190_syst_Scale_Up;   //!
   TBranch        *b_pass190_syst_Scale_Down;   //!
   TBranch        *b_pT_ll;   //!
   TBranch        *b_pT_l1;   //!
   TBranch        *b_pT_l2;   //!
   TBranch        *b_eta_l1;   //!
   TBranch        *b_eta_l2;   //!
   TBranch        *b_phi_l1;   //!
   TBranch        *b_phi_l2;   //!
   TBranch        *b_y_ll;   //!
   TBranch        *b_pT_trackj1;   //!
   TBranch        *b_y_trackj1;   //!
   TBranch        *b_phi_trackj1;   //!
   TBranch        *b_m_trackj1;   //!
   TBranch        *b_tau1_trackj1;   //!
   TBranch        *b_tau2_trackj1;   //!
   TBranch        *b_tau3_trackj1;   //!
   TBranch        *b_pT_trackj2;   //!
   TBranch        *b_y_trackj2;   //!
   TBranch        *b_phi_trackj2;   //!
   TBranch        *b_m_trackj2;   //!
   TBranch        *b_tau1_trackj2;   //!
   TBranch        *b_tau2_trackj2;   //!
   TBranch        *b_tau3_trackj2;   //!
   TBranch        *b_EventNumber;   //!
   TBranch        *b_RunNumber;   //!
   TBranch        *b_Ntracks;   //!
   TBranch        *b_Ntracks_trackj1;   //!
   TBranch        *b_Ntracks_trackj2;   //!
   TBranch        *b_NtrackJets20;   //!
   TBranch        *b_nweight_bs;   //!
   TBranch        *b_weight_bs;   //!
   TBranch        *b_npT_tracks;   //!
   TBranch        *b_pT_tracks;   //!
   TBranch        *b_neta_tracks;   //!
   TBranch        *b_eta_tracks;   //!
   TBranch        *b_nphi_tracks;   //!
   TBranch        *b_phi_tracks;   //!
   TBranch        *b_ntrackJetIndex_tracks;   //!
   TBranch        *b_trackJetIndex_tracks;   //!

   MakeOmni(TTree*, string, vector<string>, TString, bool runTruth = false, int nEnsembles = 0);
   virtual ~MakeOmni();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(Long64_t maxEvents = 0);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
   virtual vector<float> LoadWeights(string, string);
   virtual void     FillEEC(shared_ptr<TH1D>& h, const vector<double>& esum, const vector<double>& z, double Q2, double weight);
   virtual void     FillLund(shared_ptr<TH1D>& hz, shared_ptr<TH1D>& hdr, shared_ptr<TH2D>& h2, const vector<double>& z, const vector<double>& dR, double weight);
};

#endif

#ifdef MakeOmni_cxx
MakeOmni::MakeOmni(TTree *tree, string weightFile, vector<string> weightNames, TString outFile, bool runTruth, int nEnsembles) : fChain(0) 
{

   // Store instance variables
   weightFilename = weightFile; // Store the weight file name
   weightBranchNames = weightNames; // Store the weight branch names
   nEns = nEnsembles; // Store the number of ensembles
   saveName = outFile; // Store the output file name
   isTruth = runTruth; // Store the truth flag

   // Initialize histogram groups
   for (const auto& weight_name : weightBranchNames) {
      if (weight_name != "weight" && weight_name != "weight_mc") {
         centralHistoGroups.push_back(HistoGroup(weight_name + "-"));
      } else {
         centralHistoGroups.push_back(HistoGroup(""));
      }
   }

   // Initialize the ensemble histograms for the nominal ensemble
   for (int i = 0; i < nEns; ++i) {
      ensHistoGroups.push_back(HistoGroup(weightBranchNames[0] + "-" + to_string(i) + "-"));
   }

   // Initialize the tree
   Init(tree);

}

MakeOmni::~MakeOmni()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t MakeOmni::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t MakeOmni::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void MakeOmni::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   // Set branch addresses that are used for both truth and reco data
   fChain->SetBranchAddress("weight", &weight, &b_weight);
   fChain->SetBranchAddress("weight_mc", &weight_mc, &b_weight_mc);
   fChain->SetBranchAddress("EventNumber", &EventNumber, &b_EventNumber);
   fChain->SetBranchAddress("RunNumber", &RunNumber, &b_RunNumber);

   // Set branch addresses depending on whether we want to use truth or reco
   if (isTruth) {
      fChain->SetBranchAddress("truth_pass190", &pass190, &b_pass190);
      fChain->SetBranchAddress("truth_pT_ll", &pT_ll, &b_pT_ll);
      fChain->SetBranchAddress("truth_pT_l1", &pT_l1, &b_pT_l1);
      fChain->SetBranchAddress("truth_pT_l2", &pT_l2, &b_pT_l2);
      fChain->SetBranchAddress("truth_eta_l1", &eta_l1, &b_eta_l1);
      fChain->SetBranchAddress("truth_eta_l2", &eta_l2, &b_eta_l2);
      fChain->SetBranchAddress("truth_phi_l1", &phi_l1, &b_phi_l1);
      fChain->SetBranchAddress("truth_phi_l2", &phi_l2, &b_phi_l2);
      fChain->SetBranchAddress("truth_y_ll", &y_ll, &b_y_ll);
      fChain->SetBranchAddress("truth_pT_trackj1", &pT_trackj1, &b_pT_trackj1);
      fChain->SetBranchAddress("truth_y_trackj1", &y_trackj1, &b_y_trackj1);
      fChain->SetBranchAddress("truth_phi_trackj1", &phi_trackj1, &b_phi_trackj1);
      fChain->SetBranchAddress("truth_m_trackj1", &m_trackj1, &b_m_trackj1);
      fChain->SetBranchAddress("truth_tau1_trackj1", &tau1_trackj1, &b_tau1_trackj1);
      fChain->SetBranchAddress("truth_tau2_trackj1", &tau2_trackj1, &b_tau2_trackj1);
      fChain->SetBranchAddress("truth_tau3_trackj1", &tau3_trackj1, &b_tau3_trackj1);
      fChain->SetBranchAddress("truth_pT_trackj2", &pT_trackj2, &b_pT_trackj2);
      fChain->SetBranchAddress("truth_y_trackj2", &y_trackj2, &b_y_trackj2);
      fChain->SetBranchAddress("truth_phi_trackj2", &phi_trackj2, &b_phi_trackj2);
      fChain->SetBranchAddress("truth_m_trackj2", &m_trackj2, &b_m_trackj2);
      fChain->SetBranchAddress("truth_tau1_trackj2", &tau1_trackj2, &b_tau1_trackj2);
      fChain->SetBranchAddress("truth_tau2_trackj2", &tau2_trackj2, &b_tau2_trackj2);
      fChain->SetBranchAddress("truth_tau3_trackj2", &tau3_trackj2, &b_tau3_trackj2);
      fChain->SetBranchAddress("truth_pT_tracks", &pT_tracks, &b_pT_tracks);
      fChain->SetBranchAddress("truth_eta_tracks", &eta_tracks, &b_eta_tracks);
      fChain->SetBranchAddress("truth_phi_tracks", &phi_tracks, &b_phi_tracks);
      fChain->SetBranchAddress("truth_trackJetIndex_tracks", &trackJetIndex_tracks, &b_trackJetIndex_tracks);
      fChain->SetBranchAddress("truth_Ntracks", &Ntracks, &b_Ntracks);
      fChain->SetBranchAddress("truth_Ntracks_trackj1", &Ntracks_trackj1, &b_Ntracks_trackj1);
      fChain->SetBranchAddress("truth_Ntracks_trackj2", &Ntracks_trackj2, &b_Ntracks_trackj2);
      fChain->SetBranchAddress("truth_NtrackJets20", &NtrackJets20, &b_NtrackJets20);
      fChain->SetBranchAddress("ntruth_pT_tracks", &npT_tracks, &b_npT_tracks);
      fChain->SetBranchAddress("ntruth_eta_tracks", &neta_tracks, &b_neta_tracks);
      fChain->SetBranchAddress("ntruth_phi_tracks", &nphi_tracks, &b_nphi_tracks);
      fChain->SetBranchAddress("ntruth_trackJetIndex_tracks", &ntrackJetIndex_tracks, &b_ntrackJetIndex_tracks);
   } else {
      fChain->SetBranchAddress("pass190", &pass190, &b_pass190);
      fChain->SetBranchAddress("pT_ll", &pT_ll, &b_pT_ll);
      fChain->SetBranchAddress("pT_l1", &pT_l1, &b_pT_l1);
      fChain->SetBranchAddress("pT_l2", &pT_l2, &b_pT_l2);
      fChain->SetBranchAddress("eta_l1", &eta_l1, &b_eta_l1);
      fChain->SetBranchAddress("eta_l2", &eta_l2, &b_eta_l2);
      fChain->SetBranchAddress("phi_l1", &phi_l1, &b_phi_l1);
      fChain->SetBranchAddress("phi_l2", &phi_l2, &b_phi_l2);
      fChain->SetBranchAddress("y_ll", &y_ll, &b_y_ll);
      fChain->SetBranchAddress("pT_trackj1", &pT_trackj1, &b_pT_trackj1);
      fChain->SetBranchAddress("y_trackj1", &y_trackj1, &b_y_trackj1);
      fChain->SetBranchAddress("phi_trackj1", &phi_trackj1, &b_phi_trackj1);
      fChain->SetBranchAddress("m_trackj1", &m_trackj1, &b_m_trackj1);
      fChain->SetBranchAddress("tau1_trackj1", &tau1_trackj1, &b_tau1_trackj1);
      fChain->SetBranchAddress("tau2_trackj1", &tau2_trackj1, &b_tau2_trackj1);
      fChain->SetBranchAddress("tau3_trackj1", &tau3_trackj1, &b_tau3_trackj1);
      fChain->SetBranchAddress("pT_trackj2", &pT_trackj2, &b_pT_trackj2);
      fChain->SetBranchAddress("y_trackj2", &y_trackj2, &b_y_trackj2);
      fChain->SetBranchAddress("phi_trackj2", &phi_trackj2, &b_phi_trackj2);
      fChain->SetBranchAddress("m_trackj2", &m_trackj2, &b_m_trackj2);
      fChain->SetBranchAddress("tau1_trackj2", &tau1_trackj2, &b_tau1_trackj2);
      fChain->SetBranchAddress("tau2_trackj2", &tau2_trackj2, &b_tau2_trackj2);
      fChain->SetBranchAddress("tau3_trackj2", &tau3_trackj2, &b_tau3_trackj2);
      fChain->SetBranchAddress("Ntracks", &Ntracks, &b_Ntracks);
      fChain->SetBranchAddress("Ntracks_trackj1", &Ntracks_trackj1, &b_Ntracks_trackj1);
      fChain->SetBranchAddress("Ntracks_trackj2", &Ntracks_trackj2, &b_Ntracks_trackj2);
      fChain->SetBranchAddress("NtrackJets20", &NtrackJets20, &b_NtrackJets20);
      fChain->SetBranchAddress("npT_tracks", &npT_tracks, &b_npT_tracks);
      fChain->SetBranchAddress("neta_tracks", &neta_tracks, &b_neta_tracks);
      fChain->SetBranchAddress("nphi_tracks", &nphi_tracks, &b_nphi_tracks);
      fChain->SetBranchAddress("ntrackJetIndex_tracks", &ntrackJetIndex_tracks, &b_ntrackJetIndex_tracks);
      fChain->SetBranchAddress("pT_tracks", &pT_tracks, &b_pT_tracks);
      fChain->SetBranchAddress("eta_tracks", &eta_tracks, &b_eta_tracks);
      fChain->SetBranchAddress("phi_tracks", &phi_tracks, &b_phi_tracks);
      fChain->SetBranchAddress("trackJetIndex_tracks", &trackJetIndex_tracks, &b_trackJetIndex_tracks);
   }
   
   Notify();

}

Bool_t MakeOmni::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void MakeOmni::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t MakeOmni::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef MakeOmni_cxx
