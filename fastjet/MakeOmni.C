#define MakeOmni_cxx
#include "MakeOmni.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include "jetHelpers.h"
#include "analysisHelpers.h"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/Recluster.hh" 
#include <TLorentzVector.h>
#include "jetHelpers.h"
#include <memory>
using namespace fastjet;
using namespace std;

void MakeOmni::Loop( Long64_t maxEvents ) {

   // Define log-spaced bin edges for EEC plots
   Long64_t nbins = 80;
   Double_t logxmin = -8;
   Double_t logxmax = 0.5;
   Double_t binEdges[nbins+1];
   for (Long64_t i = 0; i <= nbins; ++i) {
      binEdges[i] = pow(10, logxmin + i*(logxmax-logxmin)/nbins);
   }

   // Histograms for reco or truth

   // R=0.4 jets
   unique_ptr<TH1D> hpT_R04 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
   unique_ptr<TH1D> hEEC_R04 = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
   unique_ptr<TH1D> hLund_z_R04 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 10));
   unique_ptr<TH1D> hLund_dR_R04 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
   unique_ptr<TH2D> hLund_plane_R04 = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 10, 0, 10));

   // R=0.6 jets
   unique_ptr<TH1D> hpT_R06 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
   unique_ptr<TH1D> hEEC_R06 = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
   unique_ptr<TH1D> hLund_z_R06 = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
   unique_ptr<TH1D> hLund_dR_R06 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
   unique_ptr<TH2D> hLund_plane_R06 = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 10, 0, 10));

   // R=1.0 jets
   unique_ptr<TH1D> hpT_R10 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
   unique_ptr<TH1D> hEEC_R10 = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
   unique_ptr<TH1D> hLund_z_R10 = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
   unique_ptr<TH1D> hLund_dR_R10 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
   unique_ptr<TH2D> hLund_plane_R10 = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 10, 0, 10));

   // CA R=0.4 jets
   unique_ptr<TH1D> hpT_CA04 = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
   unique_ptr<TH1D> hEEC_CA04 = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
   unique_ptr<TH1D> hLund_z_CA04 = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
   unique_ptr<TH1D> hLund_dR_CA04 = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
   unique_ptr<TH2D> hLund_plane_CA04 = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 10, 0, 10));

   // ring around 0.6
   unique_ptr<TH1D> h_fracpT_ring = unique_ptr<TH1D>(new TH1D("", "", 100, 0, 1));
   unique_ptr<TH1D> h_fracE_ring = unique_ptr<TH1D>(new TH1D("", "", 100, 0, 1));

   // TEEC
   unique_ptr<TH1D> hTEEC = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));

   // Histograms for omni if this is reco
   unique_ptr<TH1D> hpT_R04_omni;
   unique_ptr<TH1D> hEEC_R04_omni;
   unique_ptr<TH1D> hLund_z_R04_omni;
   unique_ptr<TH1D> hLund_dR_R04_omni;
   unique_ptr<TH2D> hLund_plane_R04_omni;
   unique_ptr<TH1D> hpT_R06_omni;
   unique_ptr<TH1D> hEEC_R06_omni;
   unique_ptr<TH1D> hLund_z_R06_omni;
   unique_ptr<TH1D> hLund_dR_R06_omni;
   unique_ptr<TH2D> hLund_plane_R06_omni;
   unique_ptr<TH1D> hpT_R10_omni;
   unique_ptr<TH1D> hEEC_R10_omni;
   unique_ptr<TH1D> hLund_z_R10_omni;
   unique_ptr<TH1D> hLund_dR_R10_omni;
   unique_ptr<TH2D> hLund_plane_R10_omni;
   unique_ptr<TH1D> hpT_CA04_omni;
   unique_ptr<TH1D> hEEC_CA04_omni;
   unique_ptr<TH1D> hLund_z_CA04_omni;
   unique_ptr<TH1D> hLund_dR_CA04_omni;
   unique_ptr<TH2D> hLund_plane_CA04_omni;
   unique_ptr<TH1D> h_fracpT_ring_omni;
   unique_ptr<TH1D> h_fracE_ring_omni;
   unique_ptr<TH1D> hTEEC_omni;

   if (!isTruth) {

      // R=0.4 jets
      hpT_R04_omni = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
      hEEC_R04_omni = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
      hLund_z_R04_omni = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 10));
      hLund_dR_R04_omni = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
      hLund_plane_R04_omni = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 10, 0, 10));

      // R=0.6 jets
      hpT_R06_omni = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
      hEEC_R06_omni = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
      hLund_z_R06_omni = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
      hLund_dR_R06_omni = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
      hLund_plane_R06_omni = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 10, 0, 10));

      // R=1.0 jets
      hpT_R10_omni = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
      hEEC_R10_omni = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
      hLund_z_R10_omni = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
      hLund_dR_R10_omni = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
      hLund_plane_R10_omni = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 10, 0, 10));

      // CA R=0.4 jets
      hpT_CA04_omni = unique_ptr<TH1D>(new TH1D("", "", 150, 0, 1000));
      hEEC_CA04_omni = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));
      hLund_z_CA04_omni = unique_ptr<TH1D>(new TH1D("", "",  10, 0, 10));
      hLund_dR_CA04_omni = unique_ptr<TH1D>(new TH1D("", "", 10, 0, 5));
      hLund_plane_CA04_omni = unique_ptr<TH2D>(new TH2D("", "", 10, 0, 5, 10, 0, 10));

      // Ring
      h_fracpT_ring_omni = unique_ptr<TH1D>(new TH1D("", "", 100, 0, 1));
      h_fracE_ring_omni = unique_ptr<TH1D>(new TH1D("", "", 100, 0, 1));

      // TEEC
      hTEEC_omni = unique_ptr<TH1D>(new TH1D("", "", nbins, binEdges));

   }

   // Jet definitions to consider
   JetDefinition jetdef_r04(antikt_algorithm, 0.4);
   JetDefinition jetdef_r06(antikt_algorithm, 0.6);
   JetDefinition jetdef_r10(antikt_algorithm, 1.0);
   JetDefinition jetdef_ca04(cambridge_algorithm, 0.4);

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

      // Apply event selection, filtering on truth pass 190
      if (truth_pass190==0) {
         continue;
      }

      // Which weight we use depends on whether we are processing truth or reco
      float use_weight;
      if (isTruth) {
         use_weight = weight_mc;
      } else {
         use_weight = weight;
      }

      // Create PseudoJet object of the ll system (Z-boson)
      TLorentzVector m1_tlv, m2_tlv;
      m1_tlv.SetPtEtaPhiM(truth_pT_l1, truth_eta_l1, truth_phi_l1, 0.10566);
      m2_tlv.SetPtEtaPhiM(truth_pT_l2, truth_eta_l2, truth_phi_l2, 0.10566);
      TLorentzVector zboson_tlv = m1_tlv + m2_tlv;
      PseudoJet zboson;
      cout << "\n Z boson phi: " << zboson_tlv.Phi() << endl;
      zboson.reset_PtYPhiM(zboson_tlv.Pt(), zboson_tlv.Rapidity(), zboson_tlv.Phi(), zboson_tlv.M());

      // Create vector of PseudoJets from all tracks in event
      vector<PseudoJet> particles;
      for (int i=0; i<ntruth_pT_tracks; i++){ 
         TLorentzVector constit_tlv;
         // Which track 3 vectors to use depend on whether we are processing truth or reco
         // Truth files use double precision while reco files use float precision
         if (isTruth) { 
            constit_tlv.SetPtEtaPhiM(dtruth_pT_tracks[i], dtruth_eta_tracks[i], dtruth_phi_tracks[i], 0.13957);
         } else {
            constit_tlv.SetPtEtaPhiM(ftruth_pT_tracks[i], ftruth_eta_tracks[i], ftruth_phi_tracks[i], 0.13957);
         }
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

      // R=0.4 jets
      hpT_R04->Fill(R04_jets[0].pt(), use_weight);
      FillEEC(hEEC_R04, R04_esum, R04_z, R04_Q2, use_weight);
      FillLund(hLund_z_R04, hLund_dR_R04, hLund_plane_R04, R04_lundz, R04_lundDr, use_weight);

      // R=0.6 jets
      hpT_R06->Fill(R06_jets[0].pt(), use_weight);
      FillEEC(hEEC_R06, R06_esum, R06_z, R06_Q2, use_weight);
      FillLund(hLund_z_R06, hLund_dR_R06, hLund_plane_R06, R06_lundz, R06_lundDr, use_weight);
      
      // R=1.0 jets
      hpT_R10->Fill(R10_jets[0].pt(), use_weight);
      FillEEC(hEEC_R10, R10_esum, R10_z, R10_Q2, use_weight);
      FillLund(hLund_z_R10, hLund_dR_R10, hLund_plane_R10, R10_lundz, R10_lundDr, use_weight);

      // CA R=0.4 jets
      hpT_CA04->Fill(CA04_jets[0].pt(), use_weight);
      FillEEC(hEEC_CA04, CA04_esum, CA04_z, CA04_Q2, use_weight);
      FillLund(hLund_z_CA04, hLund_dR_CA04, hLund_plane_CA04, CA04_lundz, CA04_lundDr, use_weight);

      // Ring
      h_fracpT_ring->Fill(GetRing(R06_jets[0]), use_weight);
      h_fracE_ring->Fill(0., use_weight);

      // TEEC
      FillEEC(hTEEC, etrans, tau, ETransTotal, use_weight);

      // Fill histograms for omni if this is reco
      if (!isTruth) {

         // R=0.4 jets
         hpT_R04_omni->Fill(R04_jets[0].pt(), omni_weight);
         FillEEC( hEEC_R04_omni, R04_esum, R04_z, R04_Q2, omni_weight);
         FillLund(hLund_z_R04_omni, hLund_dR_R04_omni, hLund_plane_R04_omni, R04_lundz, R04_lundDr, omni_weight);

         // R=0.6 jets
         hpT_R06_omni->Fill(R06_jets[0].pt(), omni_weight);
         FillEEC( hEEC_R06_omni, R06_esum, R06_z, R06_Q2, omni_weight);
         FillLund(hLund_z_R06_omni, hLund_dR_R06_omni, hLund_plane_R06_omni, R06_lundz, R06_lundDr, omni_weight);

         // R=1.0 jets
         hpT_R10_omni->Fill(R10_jets[0].pt(), omni_weight);
         FillEEC( hEEC_R10_omni, R10_esum, R10_z, R10_Q2, omni_weight);
         FillLund(hLund_z_R10_omni, hLund_dR_R10_omni, hLund_plane_R10_omni, R10_lundz, R10_lundDr, omni_weight);

         // CA R=0.4 jets
         hpT_CA04_omni->Fill(CA04_jets[0].pt(), omni_weight);
         FillEEC( hEEC_CA04_omni, CA04_esum, CA04_z, CA04_Q2, omni_weight);
         FillLund(hLund_z_CA04_omni, hLund_dR_CA04_omni, hLund_plane_CA04_omni, CA04_lundz, CA04_lundDr, omni_weight);

         // Ring
         h_fracpT_ring_omni->Fill(GetRing(R06_jets[0]), omni_weight);
         h_fracE_ring_omni->Fill(0., omni_weight);

         // TEEC
         FillEEC(hTEEC_omni, etrans, tau, ETransTotal, omni_weight);

      }
      
   }


   std::cout << " === normalize histos === " << std::endl;
   normalizeHisto(hpT_R04); normalizeHisto(hpT_R06); normalizeHisto(hpT_R10); normalizeHisto(hpT_CA04);
   normalizeHisto(hEEC_R04); normalizeHisto(hEEC_R06); normalizeHisto(hEEC_R10); normalizeHisto(hEEC_CA04);
   normalizeHisto(hLund_z_R04); normalizeHisto(hLund_z_R06); normalizeHisto(hLund_z_R10); normalizeHisto(hLund_z_CA04);
   normalizeHisto(hLund_dR_R04); normalizeHisto(hLund_dR_R06); normalizeHisto(hLund_dR_R10); normalizeHisto(hLund_dR_CA04);
   normalizeHisto2D(hLund_plane_R04); normalizeHisto2D(hLund_plane_R06); normalizeHisto2D(hLund_plane_R10); normalizeHisto2D(hLund_plane_CA04);
   normalizeHisto(h_fracpT_ring); normalizeHisto(h_fracE_ring);
   normalizeHisto(hTEEC);

   std::cout << " === y axis range for 1D histos === " << std::endl;
   YAxisRangeUserName(hpT_R04); YAxisRangeUserName(hpT_R06); YAxisRangeUserName(hpT_R10); YAxisRangeUserName(hpT_CA04);
   SetEECAxisRange(hEEC_R04, "z", "EEC"); SetEECAxisRange(hEEC_R06, "z", "EEC"); SetEECAxisRange(hEEC_R10, "z", "EEC"); SetEECAxisRange(hEEC_CA04, "z", "EEC");
   YAxisRangeUserName(hLund_z_R04); YAxisRangeUserName(hLund_z_R06); YAxisRangeUserName(hLund_z_R10); YAxisRangeUserName(hLund_z_CA04);
   YAxisRangeUserName(hLund_dR_R04); YAxisRangeUserName(hLund_dR_R06); YAxisRangeUserName(hLund_dR_R10); YAxisRangeUserName(hLund_dR_CA04);
   YAxisRangeUserName(h_fracpT_ring); YAxisRangeUserName(h_fracE_ring);
   SetEECAxisRange(hTEEC, "tau", "TEEC");

   // Do the same again for omni histograms if this is reco
   if (!isTruth) {

      normalizeHisto(hpT_R04_omni); normalizeHisto(hpT_R06_omni); normalizeHisto(hpT_R10_omni); normalizeHisto(hpT_CA04_omni);
      normalizeHisto(hEEC_R04_omni); normalizeHisto(hEEC_R06_omni); normalizeHisto(hEEC_R10_omni); normalizeHisto(hEEC_CA04_omni);
      normalizeHisto(hLund_z_R04_omni); normalizeHisto(hLund_z_R06_omni); normalizeHisto(hLund_z_R10_omni); normalizeHisto(hLund_z_CA04_omni);
      normalizeHisto(hLund_dR_R04_omni); normalizeHisto(hLund_dR_R06_omni); normalizeHisto(hLund_dR_R10_omni); normalizeHisto(hLund_dR_CA04_omni);
      normalizeHisto2D(hLund_plane_R04_omni); normalizeHisto2D(hLund_plane_R06_omni); normalizeHisto2D(hLund_plane_R10_omni); normalizeHisto2D(hLund_plane_CA04_omni);
      normalizeHisto(h_fracpT_ring_omni); normalizeHisto(h_fracE_ring_omni); 
      normalizeHisto(hTEEC_omni);

      YAxisRangeUserName(hpT_R04_omni); YAxisRangeUserName(hpT_R06_omni); YAxisRangeUserName(hpT_R10_omni); YAxisRangeUserName(hpT_CA04_omni);
      SetEECAxisRange(hEEC_R04_omni, "z", "EEC"); SetEECAxisRange(hEEC_R06_omni, "z", "EEC"); SetEECAxisRange(hEEC_R10_omni, "z", "EEC"); SetEECAxisRange(hEEC_CA04_omni, "z", "EEC");
      YAxisRangeUserName(hLund_z_R04_omni); YAxisRangeUserName(hLund_z_R06_omni); YAxisRangeUserName(hLund_z_R10_omni); YAxisRangeUserName(hLund_z_CA04_omni);
      YAxisRangeUserName(h_fracpT_ring_omni); YAxisRangeUserName(h_fracE_ring_omni);
      SetEECAxisRange(hTEEC_omni, "tau", "TEEC");
   
   }

   std::cout << " === create output ROOT file === " << std::endl;
   TString outputFileName = "out/output_omni.root";
   if (isTruth) {
      outputFileName = "out/output_truth.root";
   }
   TFile* foutput = new TFile(outputFileName, "recreate");

   std::cout << " === write in file === " << std::endl;
   // R=0.4 jets
   hpT_R04->Write("hpT_R04");
   hEEC_R04->Write("hEEC_R04");
   hLund_z_R04->Write("hLund_z_R04");
   hLund_dR_R04->Write("hLund_dR_R04");
   hLund_plane_R04->Write("hLund_plane_R04");

   // R=0.6 jets
   hpT_R06->Write("hpT_R06");
   hEEC_R06->Write("hEEC_R06");
   hLund_z_R06->Write("hLund_z_R06");
   hLund_dR_R06->Write("hLund_dR_R06");
   hLund_plane_R06->Write("hLund_plane_R06");

   // R=1.0 jets
   hpT_R10->Write("hpT_R10");
   hEEC_R10->Write("hEEC_R10");
   hLund_z_R10->Write("hLund_z_R10");
   hLund_dR_R10->Write("hLund_dR_R10");
   hLund_plane_R10->Write("hLund_plane_R10");

   // CA R=0.4 jets
   hpT_CA04->Write("hpT_CA04");
   hEEC_CA04->Write("hEEC_CA04");
   hLund_z_CA04->Write("hLund_z_CA04");
   hLund_dR_CA04->Write("hLund_dR_CA04");
   hLund_plane_CA04->Write("hLund_plane_CA04");

   // ring around 0.6
   h_fracpT_ring->Write("h_fracpT_ring");
   h_fracE_ring->Write("h_fracE_ring");

   // TEEC
   hTEEC->Write("hTEEC");

   // Write omni histograms if this is reco
   if (!isTruth) {

      hpT_R04_omni->Write("hpT_R04_omni");
      hEEC_R04_omni->Write("hEEC_R04_omni");
      hLund_z_R04_omni->Write("hLund_z_R04_omni");
      hLund_dR_R04_omni->Write("hLund_dR_R04_omni");
      hLund_plane_R04_omni->Write("hLund_plane_R04_omni");

      hpT_R06_omni->Write("hpT_R06_omni");
      hEEC_R06_omni->Write("hEEC_R06_omni");
      hLund_z_R06_omni->Write("hLund_z_R06_omni");
      hLund_dR_R06_omni->Write("hLund_dR_R06_omni");
      hLund_plane_R06_omni->Write("hLund_plane_R06_omni");

      hpT_R10_omni->Write("hpT_R10_omni");
      hEEC_R10_omni->Write("hEEC_R10_omni");
      hLund_z_R10_omni->Write("hLund_z_R10_omni");
      hLund_dR_R10_omni->Write("hLund_dR_R10_omni");
      hLund_plane_R10_omni->Write("hLund_plane_R10_omni");

      hpT_CA04_omni->Write("hpT_CA04_omni");
      hEEC_CA04_omni->Write("hEEC_CA04_omni");
      hLund_z_CA04_omni->Write("hLund_z_CA04_omni");
      hLund_dR_CA04_omni->Write("hLund_dR_CA04_omni");
      hLund_plane_CA04_omni->Write("hLund_plane_CA04_omni");

      h_fracpT_ring_omni->Write("h_fracpT_ring_omni");
      h_fracE_ring_omni->Write("h_fracE_ring_omni");

      hTEEC_omni->Write("hTEEC_omni");

   }

   foutput->Close();

}

void MakeOmni::FillEEC(unique_ptr<TH1D>& hEEC, const vector<double>& esum, const vector<double>& z, double Q2, double weight) {
   for (size_t i = 0; i < esum.size(); ++i) {
      hEEC->Fill(z[i], esum[i] * weight / Q2);
   }
}

void MakeOmni::FillLund(unique_ptr<TH1D>& hLund_z, unique_ptr<TH1D>& hLund_dR, unique_ptr<TH2D>& hLund_plane, const vector<double>& lundz, const vector<double>& lundDr, double weight) {
   for (size_t i = 0; i < lundz.size(); ++i) {
      hLund_z->Fill(lundz[i], weight);
      hLund_dR->Fill(lundDr[i], weight);
      hLund_plane->Fill(lundDr[i], lundz[i], weight);
   }
}


