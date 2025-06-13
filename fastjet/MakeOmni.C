#define MakeOmni_cxx
#include "MakeOmni.h"
#include "jetHelpers.h"
#include "HistoGroup.h"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/Recluster.hh" 
#include <TLorentzVector.h>
#include "jetHelpers.h"
#include "cnpy.h"
#include <memory>
#include <string>
using namespace fastjet;
using namespace std;

vector<float> MakeOmni::LoadWeights(string filename, string key) {

   // Access weights, assume we want the test weights
   cnpy::npz_t npz_file = cnpy::npz_load(filename);
   cnpy::NpyArray array = npz_file[key];

   // Convert to vector<float> and return
   vector<float> vec(array.data<float>(), array.data<float>() + array.num_vals);
   return vec;

}

void MakeOmni::Loop(Long64_t maxEvents) {

   // Load needed weights
   vector<float> central_weights;
   vector<vector<float>> ens_weights;
   vector<vector<float>> syst_weights;

   if (weightName != "weight_mc" && weightName != "weight") {

      central_weights = LoadWeights(weightName, "nominal-ensemble-central");

      for (int i = 0; i < nEns; ++i) {
         ens_weights.push_back(LoadWeights(weightName, "nominal-ensemble-" + to_string(i)));
      }

      for (int i = 0; i < syst_weight_names.size(); ++i) {
         syst_weights.push_back(LoadWeights(weightName, syst_weight_names[i] + "-central"));
      }

   }

   // // Define histograms here
   // // Later add code for reading binning information from config file

   // // R=0.4 jets
   // unique_ptr<TH1D> hpT_R04 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
   // unique_ptr<TH1D> hm1_R04 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 100));
   // unique_ptr<TH1D> hm2_R04 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 80));
   // unique_ptr<TH1D> hm3_R04 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 50));
   // unique_ptr<TH1D> hm4_R04 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 30));
   // unique_ptr<TH1D> hEEC_R04 = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
   // unique_ptr<TH1D> hLund_z_R04 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 10));
   // unique_ptr<TH1D> hLund_dR_R04 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
   // unique_ptr<TH2D> hLund_plane_R04 = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 12, 0, 6));

   // // R=0.6 jets
   // unique_ptr<TH1D> hpT_R06 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
   // unique_ptr<TH1D> hEEC_R06 = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
   // unique_ptr<TH1D> hLund_z_R06 = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
   // unique_ptr<TH1D> hLund_dR_R06 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
   // unique_ptr<TH2D> hLund_plane_R06 = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 12, 0, 6));

   // // R=1.0 jets
   // unique_ptr<TH1D> hpT_R10 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
   // unique_ptr<TH1D> hEEC_R10 = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
   // unique_ptr<TH1D> hLund_z_R10 = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
   // unique_ptr<TH1D> hLund_dR_R10 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
   // unique_ptr<TH2D> hLund_plane_R10 = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 12, 0, 6));

   // // CA R=0.4 jets
   // unique_ptr<TH1D> hpT_CA04 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
   // unique_ptr<TH1D> hEEC_CA04 = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
   // unique_ptr<TH1D> hLund_z_CA04 = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
   // unique_ptr<TH1D> hLund_dR_CA04 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
   // unique_ptr<TH2D> hLund_plane_CA04 = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 12, 0, 6));

   // // ring around 0.6
   // unique_ptr<TH1D> h_fracpT_ring = unique_ptr<TH1D>(new TH1D("", "", 100, 0, 1));
   // unique_ptr<TH1D> h_fracE_ring = unique_ptr<TH1D>(new TH1D("", "", 100, 0, 1));

   // // TEEC
   // unique_ptr<TH1D> hTEEC = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));

   // Jet definitions to consider
   JetDefinition jetdef_r04(antikt_algorithm, 0.4);
   JetDefinition jetdef_r06(antikt_algorithm, 0.6);
   JetDefinition jetdef_r10(antikt_algorithm, 1.0);
   JetDefinition jetdef_ca04(cambridge_algorithm, 0.4);


   // ----------------------- Loop over events -----------------------
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();
   if (maxEvents > 0) {
      nentries = maxEvents;
   }

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;

      if (jentry%100000==0) cout << " Entry #" << jentry << endl;

      // Apply event selection, filtering on pass190 flag
      if (pass190 == 0) {
         continue;
      }

      // Skip events that have no tracks stored
      if (npT_tracks == 0) {
         continue;
      }

      // Create PseudoJet object of the ll system (Z-boson)
      TLorentzVector m1_tlv, m2_tlv;
      m1_tlv.SetPtEtaPhiM(pT_l1, eta_l1, phi_l1, 0.10566);
      m2_tlv.SetPtEtaPhiM(pT_l2, eta_l2, phi_l2, 0.10566);
      TLorentzVector zboson_tlv = m1_tlv + m2_tlv;
      PseudoJet zboson;
      zboson.reset_PtYPhiM(zboson_tlv.Pt(), zboson_tlv.Rapidity(), zboson_tlv.Phi(), zboson_tlv.M());

      // Create vector of PseudoJets from all tracks in event
      vector<PseudoJet> particles;
      for (int i=0; i<npT_tracks; i++){ 
         TLorentzVector constit_tlv;
         // Which track 3 vectors to use depend on whether we are processing truth or reco
         // Truth files use double precision while reco files use float precision
         constit_tlv.SetPtEtaPhiM(pT_tracks[i], eta_tracks[i], phi_tracks[i], 0.13957);
         PseudoJet constit_pj;
         constit_pj.reset_PtYPhiM(constit_tlv.Pt(), constit_tlv.Rapidity(), constit_tlv.Phi(), constit_tlv.M());
         particles.push_back(constit_pj);
      }

      // Build anti-kt jets w/ R=0.4, 0.6, 1.0 and CA
      ClusterSequence cs_seq_r04(particles, jetdef_r04);
      ClusterSequence cs_seq_r06(particles, jetdef_r06);
      ClusterSequence cs_seq_r10(particles, jetdef_r10);
      ClusterSequence cs_seq_ca04(particles, jetdef_ca04);
      vector<PseudoJet> R04_jets = sorted_by_pt(cs_seq_r04.inclusive_jets());
      vector<PseudoJet> R06_jets = sorted_by_pt(cs_seq_r06.inclusive_jets());
      vector<PseudoJet> R10_jets = sorted_by_pt(cs_seq_r10.inclusive_jets());
      vector<PseudoJet> CA04_jets = sorted_by_pt(cs_seq_ca04.inclusive_jets());

      // Get Lund variables 
      vector<double> R04_lundz;
      vector<double> R04_lundkt;
      vector<double> R04_lundDr;
      processJets(R04_jets[0], 0.4, R04_lundz, R04_lundkt, R04_lundDr);

      vector<double> R06_lundz;
      vector<double> R06_lundkt;
      vector<double> R06_lundDr;
      processJets(R06_jets[0], 0.6, R06_lundz, R06_lundkt, R06_lundDr);

      vector<double> R10_lundz;
      vector<double> R10_lundkt;
      vector<double> R10_lundDr;
      processJets(R10_jets[0], 1.0, R10_lundz, R10_lundkt, R10_lundDr);

      vector<double> CA04_lundz;
      vector<double> CA04_lundkt;
      vector<double> CA04_lundDr;
      processJets(CA04_jets[0], 0.4, CA04_lundz, CA04_lundkt, CA04_lundDr);

      // Get EEC variables
      double R04_Q2;
      vector<double> R04_esum;
      vector<double> R04_z;
      R04_Q2 = GetEEC(R04_jets[0], R04_esum, R04_z);

      double R06_Q2;
      vector<double> R06_esum;
      vector<double> R06_z;
      R06_Q2 = GetEEC(R06_jets[0], R06_esum, R06_z);

      double R10_Q2;
      vector<double> R10_esum;
      vector<double> R10_z;
      R10_Q2 = GetEEC(R10_jets[0], R10_esum, R10_z);

      double CA04_Q2;
      vector<double> CA04_esum;
      vector<double> CA04_z;
      CA04_Q2 = GetEEC(CA04_jets[0], CA04_esum, CA04_z);

      // Get TEEC variables
      double ETransTotal;
      vector<double> etrans;
      vector<double> tau;
      ETransTotal = GetTEEC(particles, zboson, etrans, tau);

      // -------------------- Fill Histograms --------------------
      // Note we are filling with values only from leading jet for now
      // Can also use the FillEEC function for the TEEC

      // Loop through all groups
      for (unsigned int i = 0; i < histoGroups.size(); ++i) {

         // Get weight
         float use_weight;
         if (weightName == "weight") {
            use_weight = weight;
         } else if (weightName == "weight_mc") {
            use_weight = weight_mc;
         } else {
            if (i == 0) {
               use_weight = central_weights[jentry] * weight_mc;
            } else if (i <= nEns) {
               use_weight = ens_weights[i-1][jentry] * weight_mc;
            } else {
               use_weight = syst_weights[i-nEns-1][jentry] * weight_mc;
            }
         } 

         // R=0.4 jets
         histoGroups[i].hm3_R04->Fill(R04_jets[2].m(), use_weight);
         histoGroups[i].hm4_R04->Fill(R04_jets[3].m(), use_weight);
         FillEEC(histoGroups[i].hEEC_R04, R04_esum, R04_z, R04_Q2, use_weight);
         FillLund(histoGroups[i].hLund_z_R04, histoGroups[i].hLund_dR_R04, histoGroups[i].hLund_plane_R04, R04_lundz, R04_lundDr, use_weight);

         // R=0.6 jets
         histoGroups[i].hpT_R06->Fill(R06_jets[0].pt(), use_weight);
         FillEEC(histoGroups[i].hEEC_R06, R06_esum, R06_z, R06_Q2, use_weight);
         FillLund(histoGroups[i].hLund_z_R06, histoGroups[i].hLund_dR_R06, histoGroups[i].hLund_plane_R06, R06_lundz, R06_lundDr, use_weight);

         // R=1.0 jets
         histoGroups[i].hpT_R10->Fill(R10_jets[0].pt(), use_weight);
         FillEEC(histoGroups[i].hEEC_R10, R10_esum, R10_z, R10_Q2, use_weight);
         FillLund(histoGroups[i].hLund_z_R10, histoGroups[i].hLund_dR_R10, histoGroups[i].hLund_plane_R10, R10_lundz, R10_lundDr, use_weight);

         // CA R=0.4 jets
         histoGroups[i].hpT_CA04->Fill(CA04_jets[0].pt(), use_weight);
         FillEEC(histoGroups[i].hEEC_CA04, CA04_esum, CA04_z, CA04_Q2, use_weight);

      }

      // // R=0.4 jets

      // FillEEC(hEEC_R04, R04_esum, R04_z, R04_Q2, use_weight);
      // FillLund(hLund_z_R04, hLund_dR_R04, hLund_plane_R04, R04_lundz, R04_lundDr, use_weight);

      // // R=0.6 jets
      // hpT_R06->Fill(R06_jets[0].pt(), use_weight);
      // FillEEC(hEEC_R06, R06_esum, R06_z, R06_Q2, use_weight);
      // FillLund(hLund_z_R06, hLund_dR_R06, hLund_plane_R06, R06_lundz, R06_lundDr, use_weight);
      
      // // R=1.0 jets
      // hpT_R10->Fill(R10_jets[0].pt(), use_weight);
      // FillEEC(hEEC_R10, R10_esum, R10_z, R10_Q2, use_weight);
      // FillLund(hLund_z_R10, hLund_dR_R10, hLund_plane_R10, R10_lundz, R10_lundDr, use_weight);

      // // CA R=0.4 jets
      // hpT_CA04->Fill(CA04_jets[0].pt(), use_weight);
      // FillEEC(hEEC_CA04, CA04_esum, CA04_z, CA04_Q2, use_weight);
      // FillLund(hLund_z_CA04, hLund_dR_CA04, hLund_plane_CA04, CA04_lundz, CA04_lundDr, use_weight);

      // // Ring
      // h_fracpT_ring->Fill(GetRing(R06_jets[0]), use_weight);
      // h_fracE_ring->Fill(0., use_weight);

      // // TEEC
      // FillEEC(hTEEC, etrans, tau, ETransTotal, use_weight);
      
   }

   std::cout << " === create output ROOT file === " << std::endl;
   TFile foutput(saveName, "recreate");

   std::cout << " === write in file === " << std::endl;
   for (auto& histoGroup : histoGroups) {
      histoGroup.WriteHistos(foutput);
   }

   // Close file
   foutput.Close();

}

void MakeOmni::FillEEC(shared_ptr<TH1D>& hEEC, const vector<double>& esum, const vector<double>& z, double Q2, double weight) {
   for (size_t i = 0; i < esum.size(); ++i) {
      hEEC->Fill(z[i], esum[i] * weight / Q2);
   }
}

void MakeOmni::FillLund(shared_ptr<TH1D>& hLund_z, shared_ptr<TH1D>& hLund_dR, shared_ptr<TH2D>& hLund_plane, const vector<double>& lundz, const vector<double>& lundDr, double weight) {
   for (size_t i = 0; i < lundz.size(); ++i) {
      hLund_z->Fill(lundz[i], weight);
      hLund_dR->Fill(lundDr[i], weight);
      hLund_plane->Fill(lundDr[i], lundz[i], weight);
   }
}


