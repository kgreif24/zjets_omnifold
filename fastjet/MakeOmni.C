#define MakeOmni_cxx
#include "MakeOmni.h"
#include "jetHelpers.h"
#include "HistoGroup.h"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/Recluster.hh" 
#include <TLorentzVector.h>
#include <TMath.h>
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
   vector<vector<float>> central_weights;
   vector<vector<float>> ens_weights;

   // Load the needed weights
   for (const auto& weight_name : weightBranchNames) {
      if (weight_name != "weight" && weight_name != "weight_mc") {
         central_weights.push_back(LoadWeights(weightFilename, weight_name + "-central"));
      } else {
         central_weights.push_back(vector<float>());
      }
   }

   // Load the ensemble weights
   for (int i = 0; i < nEns; ++i) {
      ens_weights.push_back(LoadWeights(weightFilename, weightBranchNames[0] + "-" + to_string(i)));
   }

   // Jet definitions to consider
   JetDefinition jetdef_kt(kt_algorithm, 0.4);
   JetDefinition jetdef_r04(antikt_algorithm, 0.4);
   JetDefinition jetdef_r06(antikt_algorithm, 0.6);
   JetDefinition jetdef_r10(antikt_algorithm, 1.0);
   JetDefinition jetdef_ca04(cambridge_algorithm, 0.4);
   JetDefinition jetdef_ca06(cambridge_algorithm, 0.6);


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

      if (jentry%10000==0) cout << " Entry #" << jentry << endl;

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
      ClusterSequence cs_seq_kt(particles, jetdef_kt);
      ClusterSequence cs_seq_r04(particles, jetdef_r04);
      ClusterSequence cs_seq_r06(particles, jetdef_r06);
      ClusterSequence cs_seq_r10(particles, jetdef_r10);
      ClusterSequence cs_seq_ca04(particles, jetdef_ca04);
      ClusterSequence cs_seq_ca06(particles, jetdef_ca06);
      vector<PseudoJet> KT_jets = sorted_by_pt(cs_seq_kt.inclusive_jets());
      vector<PseudoJet> R04_jets = sorted_by_pt(cs_seq_r04.inclusive_jets());
      vector<PseudoJet> R06_jets = sorted_by_pt(cs_seq_r06.inclusive_jets());
      vector<PseudoJet> R10_jets = sorted_by_pt(cs_seq_r10.inclusive_jets());
      vector<PseudoJet> CA04_jets = sorted_by_pt(cs_seq_ca04.inclusive_jets());
      vector<PseudoJet> CA06_jets = sorted_by_pt(cs_seq_ca06.inclusive_jets());

      // Calculate mjj, dRjj, dyjj for CA04 jets
      double CA04_mjj, CA04_dRjj, CA04_dyjj;
      if (CA04_jets.size() > 1) {
         CA04_mjj = (CA04_jets[0] + CA04_jets[1]).m();
         CA04_dRjj = CA04_jets[0].delta_R(CA04_jets[1]);
         CA04_dyjj = TMath::Abs(CA04_jets[0].rap() - CA04_jets[1].rap());
      } else {
         CA04_mjj = -999;
         CA04_dRjj = -999;
         CA04_dyjj = -999;
      }

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
      for (unsigned int i = 0; i < centralHistoGroups.size() + ensHistoGroups.size(); ++i) {

         // Get weight
         float use_weight;
         if (weightBranchNames[i] == "weight") {
            use_weight = weight;
         } else if (weightBranchNames[i] == "weight_mc") {
            use_weight = weight_mc;
         } else if (i < centralHistoGroups.size()) {
            use_weight = central_weights[i][jentry] * weight_mc;
         } else {
            use_weight = ens_weights[i - centralHistoGroups.size()][jentry] * weight_mc;
         }

         // Get histogram group
         HistoGroup histoGroup;
         if (i < centralHistoGroups.size()) {
            histoGroup = centralHistoGroups[i];
         } else {
            histoGroup = ensHistoGroups[i - centralHistoGroups.size()];
         }

         // KT R=0.4 jets
         histoGroup.hm1_KT04->Fill(KT_jets[0].m(), use_weight);
         histoGroup.hpT_KT04->Fill(KT_jets[0].pt(), use_weight);

         // R=0.4 jets
         histoGroup.hm3_R04->Fill(R04_jets[2].m(), use_weight);
         histoGroup.hm4_R04->Fill(R04_jets[3].m(), use_weight);
         FillEEC(histoGroup.hEEC_R04, R04_esum, R04_z, R04_Q2, use_weight);
         FillLund(histoGroup.hLund_z_R04, histoGroup.hLund_dR_R04, histoGroup.hLund_plane_R04, R04_lundz, R04_lundDr, use_weight);

         // R=0.6 jets
         histoGroup.hpT_R06->Fill(R06_jets[0].pt(), use_weight);
         FillEEC(histoGroup.hEEC_R06, R06_esum, R06_z, R06_Q2, use_weight);
         FillLund(histoGroup.hLund_z_R06, histoGroup.hLund_dR_R06, histoGroup.hLund_plane_R06, R06_lundz, R06_lundDr, use_weight);

         // R=1.0 jets
         histoGroup.hm1_R10->Fill(R10_jets[0].m(), use_weight);
         histoGroup.hpT_R10->Fill(R10_jets[0].pt(), use_weight);
         FillEEC(histoGroup.hEEC_R10, R10_esum, R10_z, R10_Q2, use_weight);
         FillLund(histoGroup.hLund_z_R10, histoGroup.hLund_dR_R10, histoGroup.hLund_plane_R10, R10_lundz, R10_lundDr, use_weight);

         // CA R=0.4 jets
         histoGroup.hpT_CA04->Fill(CA04_jets[0].pt(), use_weight);
         FillEEC(histoGroup.hEEC_CA04, CA04_esum, CA04_z, CA04_Q2, use_weight);
         histoGroup.hmjj_CA04->Fill(CA04_mjj, use_weight);
         histoGroup.hdRjj_CA04->Fill(CA04_dRjj, use_weight);
         histoGroup.hdyjj_CA04->Fill(CA04_dyjj, use_weight);

         // CA R=0.6 jets
         histoGroup.hm1_CA06->Fill(CA06_jets[0].m(), use_weight);
         histoGroup.hpT_CA06->Fill(CA06_jets[0].pt(), use_weight);

      }

   }

   std::cout << " === create output ROOT file === " << std::endl;
   TFile foutput(saveName, "recreate");

   std::cout << " === write in file === " << std::endl;
   for (auto& histoGroup : centralHistoGroups) {
      histoGroup.WriteHistos(foutput);
   }

   for (auto& histoGroup : ensHistoGroups) {
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


