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
#include <TProfile.h>
#include <TLeaf.h>
#include <vector>
#include <string>
#include <iostream>
#include "HistoGroup.h"
using namespace std;


// Create a structure to hold event data
struct EventData {
  map<string, Float_t> floats;
  map<string, Int_t> ints;
  map<string, vector<Double_t>> f_vecs;
  map<string, vector<int>> i_vecs;
  map<string,Float_t> w_theory;
  map<string, Float_t> out_floats;

  Long64_t entry;
};

// Header file for the classes stored in the TTree if any.

class MakeOmni {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   bool isTruth; // Flag to indicate if we are processing truth data
   bool hasTruth;
   bool is_data;
   // bool loadSystematics; // Flag to indicate if we are loading systematics
   string weightFilename;
   vector<string> weightBranchNames;
   int nEns;
   TString saveName;
   int kinematicRegion;
   map<string,bool> is_multiplicative;
   vector<string> trackVariations;
   bool do_IBU;

   // HistoGroups
   vector<HistoGroup> centralHistoGroups;
   vector<HistoGroup> ensHistoGroups;

   // Fixed size dimensions of array or collections stored in the TTree if any.
   // Declaration of leaf types
   bool has_kevin_branches = true;
   bool has_theory_weights = true;
   bool theory_prefix      = false;
   map<string,Float_t> w_theory;
   Float_t w_QCD_uu;
   Float_t w_QCD_dd;
   Float_t w_QCD_un;
   Float_t w_QCD_nu;
   Float_t w_QCD_nd;
   Float_t w_QCD_dn;
   Float_t w_PDF_CT14nnlo;
   Float_t w_PDF_MMHT2014;
   Float_t w_PDF_MSHT2020;
   Float_t w_PDF_CT18nnlo;
   Float_t w_Alpha_s1;
   Float_t w_Alpha_s2;
   Float_t w_Var2Up;
   Float_t w_Var2Down;
   Float_t w_Var1Up;
   Float_t w_Var1Down;
   Float_t w_MPIUp;
   Float_t w_MPIDown;
   Float_t w_RenUp;
   Float_t w_RenDown;

   Float_t         weight = 0;
   Int_t           pass190 = 0;
   Float_t         target_dd = 0;
   Int_t           pass190_syst_ID_Up = 0;
   Int_t           pass190_syst_ID_Down = 0;
   Int_t           pass190_syst_MS_Up = 0;
   Int_t           pass190_syst_MS_Down = 0;
   Int_t           pass190_syst_MSResbias_Up = 0;
   Int_t           pass190_syst_MSResbias_Down = 0;
   Int_t           pass190_syst_MSRho_Up = 0;
   Int_t           pass190_syst_MSRho_Down = 0;
   Int_t           pass190_syst_Scale_Up = 0;
   Int_t           pass190_syst_Scale_Down = 0;
   Float_t         pT_ll = 0;
   Float_t         pT_l1 = 0;
   Float_t         pT_l2 = 0;
   Float_t         eta_l1 = 0;
   Float_t         eta_l2 = 0;
   Float_t         phi_l1 = 0;
   Float_t         phi_l2 = 0;
   Float_t         y_ll = 0;
   Float_t         pT_trackj1 = 0;
   Float_t         y_trackj1 = 0;
   Float_t         phi_trackj1 = 0;
   Float_t         m_trackj1 = 0;
   Float_t         tau1_trackj1 = 0;
   Float_t         tau2_trackj1 = 0;
   Float_t         tau3_trackj1 = 0;
   Float_t         pT_trackj2 = 0;
   Float_t         y_trackj2 = 0;
   Float_t         phi_trackj2 = 0;
   Float_t         m_trackj2 = 0;
   Float_t         tau1_trackj2 = 0;
   Float_t         tau2_trackj2 = 0;
   Float_t         tau3_trackj2 = 0;
   Int_t           EventNumber = 0;
   Int_t           RunNumber = 0;
   Int_t           Ntracks = 0;
   Int_t           Ntracks_trackj1 = 0;
   Int_t           Ntracks_trackj2 = 0;
   Int_t           NtrackJets20 = 0;
   Int_t           nweight_bs = 0;
   Int_t           weight_bs[100];   //[nweight_bs]
   Int_t           npT_tracks = 0;
   Double_t        pT_tracks[309];   //[npT_tracks]
   Int_t           neta_tracks = 0;
   Double_t        eta_tracks[309];   //[neta_tracks]
   Int_t           nphi_tracks = 0;
   Double_t        phi_tracks[309];   //[nphi_tracks]
   Int_t           ntrackJetIndex_tracks = 0;
   Double_t        trackJetIndex_tracks[309];   //[ntrackJetIndex_tracks]
   Int_t           npdgId_tracks = 0;
   Long_t          pdgId_tracks[309];   //[npdgId_tracks]


   vector<float>*   pT_tracks_vec = nullptr; 
   vector<float>*   eta_tracks_vec = nullptr;
   vector<float>*   phi_tracks_vec = nullptr;
   vector<float>*   trackJetIndex_tracks_vec = nullptr;

   Int_t           truth_pass190 = 0;
   Float_t         weight_mc = 0;
   Float_t         truth_pT_ll = 0;
   Float_t         truth_pT_l1 = 0;
   Float_t         truth_pT_l2 = 0;
   Float_t         truth_eta_l1 = 0;
   Float_t         truth_eta_l2 = 0;
   Float_t         truth_phi_l1 = 0;
   Float_t         truth_phi_l2 = 0;
   Float_t         truth_y_ll = 0;
   Float_t         truth_pT_trackj1 = 0;
   Float_t         truth_y_trackj1 = 0;
   Float_t         truth_phi_trackj1 = 0;
   Float_t         truth_m_trackj1 = 0;
   Float_t         truth_tau1_trackj1 = 0;
   Float_t         truth_tau2_trackj1 = 0;
   Float_t         truth_tau3_trackj1 = 0;
   Float_t         truth_pT_trackj2 = 0;
   Float_t         truth_y_trackj2 = 0;
   Float_t         truth_phi_trackj2 = 0;
   Float_t         truth_m_trackj2 = 0;
   Float_t         truth_tau1_trackj2 = 0;
   Float_t         truth_tau2_trackj2 = 0;
   Float_t         truth_tau3_trackj2 = 0;
   Int_t           truth_Ntracks = 0;
   Int_t           truth_Ntracks_trackj1 = 0;
   Int_t           truth_Ntracks_trackj2 = 0;
   Int_t           truth_NtrackJets20 = 0;

   Int_t           ntruth_weight_bs = 0;
   Int_t           truth_weight_bs[100];   //[nweight_bs]
   Int_t           ntruth_pT_tracks = 0;
   Double_t        truth_pT_tracks[309];   //[npT_tracks]
   Int_t           ntruth_eta_tracks = 0;
   Double_t        truth_eta_tracks[309];   //[neta_tracks]
   Int_t           ntruth_phi_tracks = 0;
   Double_t        truth_phi_tracks[309];   //[nphi_tracks]
   Double_t        truth_phi_tracks2[309];   //[nphi_tracks]
   Int_t           ntruth_trackJetIndex_tracks = 0;
   Long64_t        truth_trackJetIndex_tracks[309];   //[ntrackJetIndex_tracks]
   Int_t           ntruth_pdgId_tracks = 0;
   Long64_t        truth_pdgId_tracks[309];   //[npdgId_tracks]
   vector<float>*  truth_pT_tracks_vec = nullptr; 
   vector<float>*  truth_eta_tracks_vec = nullptr;
   vector<float>*  truth_phi_tracks_vec = nullptr;
   vector<float>*  truth_trackJetIndex_tracks_vec = nullptr;
   vector<int>*    truth_pdgId_tracks_vec = nullptr;

   vector<int>*    syst_correctedpT_tracks = nullptr; 
   vector<int>*    syst_passTrackTruthFilter_tracks = nullptr; 
   vector<int>*    syst_passJetTrackFilter_tracks = nullptr; 
   vector<int>*    syst_passTrackFake_tracks = nullptr; 

   // List of branches
   TBranch        *b_weight;   //!
   TBranch        *b_pass190;   //!
   TBranch        *b_target_dd;   //!
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
   TBranch        *b_npdgId_tracks;   //!
   TBranch        *b_pdgId_tracks;   //!
   TBranch        *b_QCD_uu;
   TBranch        *b_QCD_dd;
   TBranch        *b_QCD_un;
   TBranch        *b_QCD_nu;
   TBranch        *b_QCD_nd;
   TBranch        *b_QCD_dn;
   TBranch        *b_PDF_CT14nnlo;
   TBranch        *b_PDF_MMHT2014;
   TBranch        *b_PDF_MSHT2020;
   TBranch        *b_PDF_CT18nnlo;
   TBranch        *b_Alpha_s1;
   TBranch        *b_Alpha_s2;
   TBranch        *b_Var2Up;
   TBranch        *b_Var2Down;
   TBranch        *b_Var1Up;
   TBranch        *b_Var1Down;
   TBranch        *b_MPIUp;
   TBranch        *b_MPIDown;
   TBranch        *b_RenUp;
   TBranch        *b_RenDown;
   
   TBranch        *b_weight_mc;   //!
   TBranch        *b_truth_pass190;   //!
   TBranch        *b_truth_pT_ll;   //!
   TBranch        *b_truth_pT_l1;   //!
   TBranch        *b_truth_pT_l2;   //!
   TBranch        *b_truth_eta_l1;   //!
   TBranch        *b_truth_eta_l2;   //!
   TBranch        *b_truth_phi_l1;   //!
   TBranch        *b_truth_phi_l2;   //!
   TBranch        *b_truth_y_ll;   //!
   TBranch        *b_truth_pT_trackj1;   //!
   TBranch        *b_truth_y_trackj1;   //!
   TBranch        *b_truth_phi_trackj1;   //!
   TBranch        *b_truth_m_trackj1;   //!
   TBranch        *b_truth_tau1_trackj1;   //!
   TBranch        *b_truth_tau2_trackj1;   //!
   TBranch        *b_truth_tau3_trackj1;   //!
   TBranch        *b_truth_pT_trackj2;   //!
   TBranch        *b_truth_y_trackj2;   //!
   TBranch        *b_truth_phi_trackj2;   //!
   TBranch        *b_truth_m_trackj2;   //!
   TBranch        *b_truth_tau1_trackj2;   //!
   TBranch        *b_truth_tau2_trackj2;   //!
   TBranch        *b_truth_tau3_trackj2;   //!
   TBranch        *b_truth_Ntracks;   //!
   TBranch        *b_truth_Ntracks_trackj1;   //!
   TBranch        *b_truth_Ntracks_trackj2;   //!
   TBranch        *b_truth_NtrackJets20;   //!
   TBranch        *b_ntruth_weight_bs;   //!
   TBranch        *b_truth_weight_bs;   //!
   TBranch        *b_ntruth_pT_tracks;   //!
   TBranch        *b_ntruth_eta_tracks;   //!
   TBranch        *b_ntruth_phi_tracks;   //!
   TBranch        *b_ntruth_trackJetIndex_tracks;   //!
   TBranch        *b_ntruth_pdgId_tracks;   //!
   TBranch        *b_truth_pT_tracks;   //!
   TBranch        *b_truth_eta_tracks;   //!
   TBranch        *b_truth_phi_tracks;   //!
   TBranch        *b_truth_phi_tracks2;
   TBranch        *b_truth_trackJetIndex_tracks;   //!
   TBranch        *b_truth_pdgId_tracks;   //!

   TBranch        *b_syst_correctedpT_tracks;
   TBranch        *b_syst_passTrackTruthFilter_tracks;
   TBranch        *b_syst_passJetTrackFilter_tracks;
   TBranch        *b_syst_passTrackFake_tracks;


   MakeOmni(TTree*, string, vector<string>, TString, bool runTruth = false, int nEnsembles = 0, int kinematic_region = 0, vector<string> trackvariations = {""}, bool do_ibu = false, bool is_Data = false);
   virtual ~MakeOmni();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(Long64_t maxEvents = 0);
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
   virtual vector<float> LoadWeights(string, string);
   virtual void     FillEEC(shared_ptr<TH1D>& h, const vector<double>& esum, const vector<double>& z, double Q2, double weight, bool flip_z = false);
   virtual void     FillLund(shared_ptr<TH1D>& hz, shared_ptr<TH1D>& hdr, shared_ptr<TH2D>& h2, const vector<double>& z, const vector<double>& dR, double weight);
   virtual float    GetMassFromPID(int pdgId);
   virtual bool     inBin(Float_t val, double low, double high);
   virtual bool     inBin(Int_t val, int low, int high);
   virtual string   get_jetR_id(string prefix, Double_t jetR);
   virtual void     WriteIBUTree(vector<EventData> event_data, TFile& foutput);

};

#endif

#ifdef MakeOmni_cxx
MakeOmni::MakeOmni(TTree *tree, string weightFile, vector<string> weightNames, TString outFile, bool runTruth, int nEnsembles, int kinematic_region, vector<string> trackvariations, bool do_ibu, bool is_Data) : fChain(0) 
{

   // Store instance variables
   weightFilename = weightFile; // Store the weight file name
   weightBranchNames = weightNames; // Store the weight branch names
   nEns = nEnsembles; // Store the number of ensembles
   saveName = outFile; // Store the output file name
   isTruth = runTruth; // Store the truth flag
   kinematicRegion = kinematic_region; // Store the kinematic region
   trackVariations = trackvariations;
   do_IBU = do_ibu;
   is_data = is_Data;
   

   // Initialize histogram groups
   for (auto& weight_name : weightBranchNames) {
      size_t pos = weight_name.find('*');
      if (pos != string::npos) {
          weight_name.erase(pos, 1);
          is_multiplicative[weight_name] = true;
      } else {
          is_multiplicative[weight_name] = false;
      }
      if (weight_name != "weight" && weight_name != "weight_mc") {
         centralHistoGroups.push_back(HistoGroup(weight_name + "-", kinematicRegion));
      } else {
         centralHistoGroups.push_back(HistoGroup("nominal-", kinematicRegion));
      }
   }

   // Initialize the ensemble histograms for the nominal ensemble
   for (int i = 0; i < nEns; ++i) {
      ensHistoGroups.push_back(HistoGroup(weightBranchNames[0] + "-" + to_string(i) + "-", kinematicRegion));
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
  //  fChain->SetMakeClass(1);
   TObjArray* branches = fChain->GetListOfBranches();
   has_kevin_branches  = branches->FindObject("npT_tracks");
   hasTruth            = branches->FindObject("truth_pass190");
   if (!branches->FindObject("w_QCD_uu") && !branches->FindObject("QCD_uu")) has_theory_weights = false;

   if (has_theory_weights){ 
      TString prefix = "";
      if (branches->FindObject("w_QCD_uu")) {theory_prefix = true; prefix = "w_";}  //if w_... then its MGFxFx, else its sherpa
      fChain->SetBranchAddress(prefix+"QCD_uu", &w_QCD_uu, &b_QCD_uu);
      fChain->SetBranchAddress(prefix+"QCD_dd", &w_QCD_dd, &b_QCD_dd);
      fChain->SetBranchAddress(prefix+"QCD_un", &w_QCD_un, &b_QCD_un);
      fChain->SetBranchAddress(prefix+"QCD_nu", &w_QCD_nu, &b_QCD_nu);
      fChain->SetBranchAddress(prefix+"QCD_nd", &w_QCD_nd, &b_QCD_nd);
      fChain->SetBranchAddress(prefix+"QCD_dn", &w_QCD_dn, &b_QCD_dn);
      // fChain->SetBranchAddress(prefix+"PDF_CT14nnlo", &w_PDF_CT14nnlo, &b_PDF_CT14nnlo);
      // fChain->SetBranchAddress(prefix+"PDF_MMHT2014", &w_PDF_MMHT2014, &b_PDF_MMHT2014);
      fChain->SetBranchAddress(prefix+"PDF_MSHT2020", &w_PDF_MSHT2020, &b_PDF_MSHT2020);
      fChain->SetBranchAddress(prefix+"PDF_CT18nnlo", &w_PDF_CT18nnlo, &b_PDF_CT18nnlo);
      fChain->SetBranchAddress(prefix+"Alpha_s1", &w_Alpha_s1, &b_Alpha_s1);
      fChain->SetBranchAddress(prefix+"Alpha_s2", &w_Alpha_s2, &b_Alpha_s2);

      fChain->SetBranchAddress("w_Var2Up", &w_Var2Up, &b_Var2Up);
      fChain->SetBranchAddress("w_Var2Down", &w_Var2Down, &b_Var2Down);
      fChain->SetBranchAddress("w_Var1Up", &w_Var1Up, &b_Var1Up);
      fChain->SetBranchAddress("w_Var1Down", &w_Var1Down, &b_Var1Down);
      fChain->SetBranchAddress("w_MPIUp", &w_MPIUp, &b_MPIUp);
      fChain->SetBranchAddress("w_MPIDown", &w_MPIDown, &b_MPIDown);
      fChain->SetBranchAddress("w_RenUp", &w_RenUp, &b_RenUp);
      fChain->SetBranchAddress("w_RenDown", &w_RenDown, &b_RenDown);
   }
  //  hasTruth =false;
   
   // Set branch addresses depending on whether we have truth 

   
   if (hasTruth && !is_data) {
      fChain->SetBranchAddress("weight_mc", &weight_mc, &b_weight_mc);
      fChain->SetBranchAddress("truth_pass190", &truth_pass190, &b_truth_pass190);
      fChain->SetBranchAddress("truth_pT_ll", &truth_pT_ll, &b_truth_pT_ll);
      fChain->SetBranchAddress("truth_pT_l1", &truth_pT_l1, &b_truth_pT_l1);
      fChain->SetBranchAddress("truth_pT_l2", &truth_pT_l2, &b_truth_pT_l2);
      fChain->SetBranchAddress("truth_eta_l1", &truth_eta_l1, &b_truth_eta_l1);
      fChain->SetBranchAddress("truth_eta_l2", &truth_eta_l2, &b_truth_eta_l2);
      fChain->SetBranchAddress("truth_phi_l1", &truth_phi_l1, &b_truth_phi_l1);
      fChain->SetBranchAddress("truth_phi_l2", &truth_phi_l2, &b_truth_phi_l2);
      fChain->SetBranchAddress("truth_y_ll", &truth_y_ll, &b_truth_y_ll);
      fChain->SetBranchAddress("truth_pT_trackj1", &truth_pT_trackj1, &b_truth_pT_trackj1);
      fChain->SetBranchAddress("truth_y_trackj1", &truth_y_trackj1, &b_truth_y_trackj1);
      fChain->SetBranchAddress("truth_phi_trackj1", &truth_phi_trackj1, &b_truth_phi_trackj1);
      fChain->SetBranchAddress("truth_m_trackj1", &truth_m_trackj1, &b_truth_m_trackj1);
      fChain->SetBranchAddress("truth_tau1_trackj1", &truth_tau1_trackj1, &b_truth_tau1_trackj1);
      fChain->SetBranchAddress("truth_tau2_trackj1", &truth_tau2_trackj1, &b_truth_tau2_trackj1);
      fChain->SetBranchAddress("truth_tau3_trackj1", &truth_tau3_trackj1, &b_truth_tau3_trackj1);
      fChain->SetBranchAddress("truth_pT_trackj2", &truth_pT_trackj2, &b_truth_pT_trackj2);
      fChain->SetBranchAddress("truth_y_trackj2", &truth_y_trackj2, &b_truth_y_trackj2);
      fChain->SetBranchAddress("truth_phi_trackj2", &truth_phi_trackj2, &b_truth_phi_trackj2);
      fChain->SetBranchAddress("truth_m_trackj2", &truth_m_trackj2, &b_truth_m_trackj2);
      fChain->SetBranchAddress("truth_tau1_trackj2", &truth_tau1_trackj2, &b_truth_tau1_trackj2);
      fChain->SetBranchAddress("truth_tau2_trackj2", &truth_tau2_trackj2, &b_truth_tau2_trackj2);
      fChain->SetBranchAddress("truth_tau3_trackj2", &truth_tau3_trackj2, &b_truth_tau3_trackj2);
      fChain->SetBranchAddress("truth_Ntracks", &truth_Ntracks, &b_truth_Ntracks);
      fChain->SetBranchAddress("truth_Ntracks_trackj1", &truth_Ntracks_trackj1, &b_truth_Ntracks_trackj1);
      fChain->SetBranchAddress("truth_Ntracks_trackj2", &truth_Ntracks_trackj2, &b_truth_Ntracks_trackj2);
      fChain->SetBranchAddress("truth_NtrackJets20", &truth_NtrackJets20, &b_truth_NtrackJets20); 
      if (has_kevin_branches) {

        fChain->SetBranchAddress("truth_pT_tracks", &truth_pT_tracks, &b_truth_pT_tracks);
        fChain->SetBranchAddress("truth_eta_tracks", &truth_eta_tracks, &b_truth_eta_tracks);
        fChain->SetBranchAddress("truth_phi_tracks", &truth_phi_tracks2, &b_truth_phi_tracks2); // somewhere, something is grabbing truth_phi_tracks and causing ROOT TTRee read error. the fis is to juse assign a new variable?
        fChain->SetBranchAddress("truth_trackJetIndex_tracks", &truth_trackJetIndex_tracks, &b_truth_trackJetIndex_tracks);
        fChain->SetBranchAddress("truth_pdgId_tracks", &truth_pdgId_tracks, &b_truth_pdgId_tracks);

        fChain->SetBranchAddress("ntruth_pT_tracks", &ntruth_pT_tracks, &b_ntruth_pT_tracks);
        fChain->SetBranchAddress("ntruth_eta_tracks", &ntruth_eta_tracks, &b_ntruth_eta_tracks);
        fChain->SetBranchAddress("ntruth_phi_tracks", &ntruth_phi_tracks, &b_ntruth_phi_tracks);
        fChain->SetBranchAddress("ntruth_trackJetIndex_tracks", &ntruth_trackJetIndex_tracks, &b_ntruth_trackJetIndex_tracks);
        fChain->SetBranchAddress("ntruth_pdgId_tracks", &ntruth_pdgId_tracks, &b_ntruth_pdgId_tracks);
      }
      else {
        fChain->SetBranchAddress("truth_pT_tracks", &truth_pT_tracks_vec, &b_truth_pT_tracks);
        fChain->SetBranchAddress("truth_eta_tracks", &truth_eta_tracks_vec, &b_truth_eta_tracks);
        fChain->SetBranchAddress("truth_phi_tracks", &truth_phi_tracks_vec, &b_truth_phi_tracks);
        fChain->SetBranchAddress("truth_trackJetIndex_tracks", &truth_trackJetIndex_tracks_vec, &b_truth_trackJetIndex_tracks);
        fChain->SetBranchAddress("truth_pdgId_tracks", &truth_pdgId_tracks_vec, &b_truth_pdgId_tracks);
      }
    } 
    fChain->SetBranchAddress("weight", &weight, &b_weight);
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
    if (has_kevin_branches) {
      fChain->SetBranchAddress("npT_tracks", &npT_tracks, &b_npT_tracks);
      fChain->SetBranchAddress("neta_tracks", &neta_tracks, &b_neta_tracks);
      fChain->SetBranchAddress("nphi_tracks", &nphi_tracks, &b_nphi_tracks);
      fChain->SetBranchAddress("ntrackJetIndex_tracks", &ntrackJetIndex_tracks, &b_ntrackJetIndex_tracks);

      fChain->SetBranchAddress("pT_tracks", &pT_tracks, &b_pT_tracks);
      fChain->SetBranchAddress("eta_tracks", &eta_tracks, &b_eta_tracks);
      fChain->SetBranchAddress("phi_tracks", &phi_tracks, &b_phi_tracks);
      fChain->SetBranchAddress("trackJetIndex_tracks", &trackJetIndex_tracks, &b_trackJetIndex_tracks);
    }
    else{
      fChain->SetBranchAddress("pT_tracks", &pT_tracks_vec, &b_pT_tracks);
      fChain->SetBranchAddress("eta_tracks", &eta_tracks_vec, &b_eta_tracks);
      fChain->SetBranchAddress("phi_tracks", &phi_tracks_vec, &b_phi_tracks);
      fChain->SetBranchAddress("trackJetIndex_tracks", &trackJetIndex_tracks_vec, &b_trackJetIndex_tracks);
    }
    if (std::find(trackVariations.begin(), trackVariations.end(), "syst_pTScale_") != trackVariations.end()) {
      fChain->SetBranchAddress("syst_correctedpT_tracks", &syst_correctedpT_tracks, &b_syst_correctedpT_tracks);
    }
    if (std::find(trackVariations.begin(), trackVariations.end(), "syst_Fake_") != trackVariations.end()) {
      fChain->SetBranchAddress("syst_passTrackFake_tracks", &syst_passTrackFake_tracks, &b_syst_passTrackFake_tracks);
    }
    if (std::find(trackVariations.begin(), trackVariations.end(), "syst_TrackFilter_") != trackVariations.end()) {
      fChain->SetBranchAddress("syst_passTrackTruthFilter_tracks", &syst_passTrackTruthFilter_tracks, &b_syst_passTrackTruthFilter_tracks);
    }
    if (std::find(trackVariations.begin(), trackVariations.end(), "syst_JetTrackFilter_") != trackVariations.end()) {
      fChain->SetBranchAddress("syst_passJetTrackFilter_tracks", &syst_passJetTrackFilter_tracks, &b_syst_passJetTrackFilter_tracks);
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
