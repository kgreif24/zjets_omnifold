#define MakeOmni_cxx
#include "MakeOmni.h"
#include "jetHelpers.h"
#include "HistoGroup.h"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/Recluster.hh" 
#include <TLorentzVector.h>
#include <TMath.h>
#include "jetHelpers.h"
#include "cnpy/cnpy.h"
#include <memory>
#include <string>
#include "indicators/progress_bar.hpp"
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>
using namespace fastjet;
using namespace std;
using namespace indicators;

float MakeOmni::GetMassFromPID(int pdgId) {
   int absPdgId = TMath::Abs(pdgId);
   if (absPdgId == 11) {
      return 0.00051;
   } else if (absPdgId == 13) {
      return 0.10566;
   } else if (absPdgId == 211) {
      return 0.13957;
   } else if (absPdgId == 321) {
      return 0.493677;
   } else if (absPdgId == 2212) {
      return 0.938272;
   } else if (absPdgId == 3112) {
      return 1.11794;
   } else if (absPdgId == 3222) {
      return 1.18937;
   } else if (absPdgId == 3312) {
      return 1.3217;
   } else if (absPdgId == 3334) {
      return 1.67243;
   } else {
      cout << "Unknown PDG ID: " << pdgId << endl;
      return -999;
   }
}

vector<float> MakeOmni::LoadWeights(string filename, string key) {

   try {
      // Load a single array from the npz file (more reliable than loading entire file)
      // This uses the single-array version of npz_load which is more robust for large files
      cnpy::NpyArray array = cnpy::npz_load(filename, key);

      // Convert to vector<float> and return
      vector<float> vec(array.data<float>(), array.data<float>() + array.num_vals);
      return vec;
      
   } catch (const std::runtime_error& e) {
      throw std::runtime_error("LoadWeights: Error loading '" + key + "' from " + filename + ": " + e.what());
   } catch (const std::exception& e) {
      throw std::runtime_error("LoadWeights: Unexpected error loading '" + key + "' from " + filename + ": " + e.what());
   }

}

vector<string> MakeOmni::DetectWeightNames(string filename) {
   vector<string> detected_names;
   
   // Check if file exists
   ifstream file_check(filename);
   if (!file_check.good()) {
      throw std::runtime_error("DetectWeightNames: Cannot open file " + filename + ". Please check the file path.");
   }
   file_check.close();
   
   try {
      // Load the npz file to get all keys
      cnpy::npz_t npz_file = cnpy::npz_load(filename);
      
      // Iterate over all keys in the npz file
      for (const auto& pair : npz_file) {
         string key = pair.first;
         
         // Skip ensemble weights (handled separately via --nEns)
         if (key.find("weights_ensemble_") == 0) {
            continue;
         }
         
         // Skip bootstrap data weights (handled separately via --nBootstrapData)
         if (key.find("weights_bootstrap_data_") == 0) {
            continue;
         }
         
      // Skip bootstrap MC weights (handled separately if needed)
      if (key.find("weights_bootstrap_mc_") == 0) {
         continue;
      }
      
      // Skip weights_hv - this re-weights a different dataset and should be
      // specified explicitly via --weight_names if needed
      if (key == "weights_hv") {
         continue;
      }
      
      // Include all other weights (weights_nominal, weights_dd, 
      // weights_trackEffMain, etc., and target_dd)
      detected_names.push_back(key);
      }
      
      // Sort for consistent ordering
      sort(detected_names.begin(), detected_names.end());
      
   } catch (const std::runtime_error& e) {
      throw std::runtime_error("DetectWeightNames: Error reading npz file " + filename + ": " + e.what());
   }
   
   return detected_names;
}

void MakeOmni::Loop(Long64_t maxEvents) {

   // Load needed weights
   vector<vector<float>> central_weights;
   vector<vector<float>> ens_weights;
   vector<vector<float>> bootstrap_data_weights;

   // Load weights individually (more reliable for large files with many arrays)
   // Loading the entire npz file at once can fail with large files due to cnpy limitations

   // Load the needed weights individually (more reliable for large files)
   cout << "Loading weights from file: " << weightFilename << endl;
   for (const auto& weight_name : weightBranchNames) {
      if (weight_name != "weight" && weight_name != "weight_mc") {
         // Load from npz file using weight name directly (new format)
         // target_dd can now be loaded from npz if specified
         cout << "  Loading: " << weight_name << "..." << flush;
         try {
            central_weights.push_back(LoadWeights(weightFilename, weight_name));
            cout << " done" << endl;
         } catch (const std::exception& e) {
            cout << " FAILED" << endl;
            cerr << "Error loading weight '" << weight_name << "': " << e.what() << endl;
            throw;
         }
      } else {
         // weight and weight_mc come from ROOT tree, not npz file
         central_weights.push_back(vector<float>());
      }
   }

   // Load the ensemble weights
   for (int i = 0; i < nEns; ++i) {
      string key = "weights_ensemble_" + to_string(i);
      cout << "  Loading: " << key << "..." << flush;
      try {
         ens_weights.push_back(LoadWeights(weightFilename, key));
         cout << " done" << endl;
      } catch (const std::exception& e) {
         cout << " FAILED" << endl;
         cerr << "Error loading weight '" << key << "': " << e.what() << endl;
         throw;
      }
   }

   // Load the bootstrap data weights
   for (int i = 0; i < nBootstrapData; ++i) {
      string key = "weights_bootstrap_data_" + to_string(i);
      cout << "  Loading: " << key << "..." << flush;
      try {
         bootstrap_data_weights.push_back(LoadWeights(weightFilename, key));
         cout << " done" << endl;
      } catch (const std::exception& e) {
         cout << " FAILED" << endl;
         cerr << "Error loading weight '" << key << "': " << e.what() << endl;
         throw;
      }
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
   
   // Initialize progress bar with ETA
   ProgressBar bar{
      option::BarWidth{50},
      option::Start{"["},
      option::Fill{"="},
      option::Lead{">"},
      option::Remainder{" "},
      option::End{"]"},
      option::PostfixText{"Processing Events"},
      option::ForegroundColor{Color::green},
      option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
      option::ShowElapsedTime{true},
      option::ShowRemainingTime{true}
   };
   
   // Set the total number of events for ETA calculation
   bar.set_option(option::MaxProgress{nentries});
   
   // Create thread-local histogram groups for parallel processing
   int num_threads = omp_get_max_threads();
   vector<vector<HistoGroup>> thread_central_histos(num_threads);
   vector<vector<HistoGroup>> thread_ens_histos(num_threads);
   vector<vector<HistoGroup>> thread_bootstrap_histos(num_threads);
   
   // Calculate total number of histogram groups per thread
   int total_central = weightBranchNames.size();
   int total_per_thread = total_central + nEns + nBootstrapData;
   int total_groups = total_per_thread * num_threads;
   
   std::cout << " === initializing " << num_threads << " thread-local histogram groups ===" << std::endl;
   std::cout << "  Per thread: " << total_central << " central + " << nEns << " ensemble + " << nBootstrapData << " bootstrap = " << total_per_thread << " groups" << std::endl;
   std::cout << "  Total: " << total_groups << " histogram groups" << std::endl;
   
   // Initialize thread-local histogram groups
   int group_count = 0;
   int progress_interval = (total_groups > 20) ? (total_groups / 20) : 1;
   int thread_report_interval = (num_threads > 10) ? (num_threads / 10) : 1;
   
   for (int t = 0; t < num_threads; ++t) {
      if (t % thread_report_interval == 0 || t == num_threads - 1) {
         std::cout << "  Initializing thread " << (t + 1) << "/" << num_threads << "..." << std::endl;
      }
      
      // Initialize central weight histograms
      for (size_t w = 0; w < weightBranchNames.size(); ++w) {
         const auto& weight_name = weightBranchNames[w];
         if (weight_name != "weight" && weight_name != "weight_mc") {
            // Drop "weights_" prefix from weight name
            string weight_name_stripped = weight_name.substr(7);
            thread_central_histos[t].push_back(HistoGroup(weight_name_stripped + "-", kinematicRegion));
         } else {
            thread_central_histos[t].push_back(HistoGroup("nominal-", kinematicRegion));
         }
         group_count++;
         if (group_count % progress_interval == 0) {
            std::cout << "    Progress: " << group_count << "/" << total_groups << " groups initialized (" 
                      << (100 * group_count / total_groups) << "%)" << std::endl;
         }
      }
      
      // Initialize ensemble weight histograms
      for (int i = 0; i < nEns; ++i) {
         thread_ens_histos[t].push_back(HistoGroup("ensemble_" + to_string(i) + "-", kinematicRegion));
         group_count++;
         if (group_count % progress_interval == 0) {
            std::cout << "    Progress: " << group_count << "/" << total_groups << " groups initialized (" 
                      << (100 * group_count / total_groups) << "%)" << std::endl;
         }
      }
      
      // Initialize bootstrap data weight histograms
      for (int i = 0; i < nBootstrapData; ++i) {
         thread_bootstrap_histos[t].push_back(HistoGroup("bootstrap_data_" + to_string(i) + "-", kinematicRegion));
         group_count++;
         if (group_count % progress_interval == 0) {
            std::cout << "    Progress: " << group_count << "/" << total_groups << " groups initialized (" 
                      << (100 * group_count / total_groups) << "%)" << std::endl;
         }
      }
   }
   std::cout << "  Completed initialization of all " << total_groups << " histogram groups" << std::endl;
   
   // Pre-load all events sequentially (ROOT trees are not thread-safe)
   std::cout << " === pre-loading events === " << std::endl;
   vector<Long64_t> valid_entries;
   
   // Create a structure to hold event data
   struct EventData {
      Long64_t entry;
      // Store all the variables we need for processing
      Float_t weight, weight_mc, target_dd;
      Int_t pass190;
      Float_t pT_ll, pT_l1, pT_l2, eta_l1, eta_l2, phi_l1, phi_l2, y_ll;
      Float_t pT_trackj1, y_trackj1, phi_trackj1, m_trackj1;
      Float_t pT_trackj2, y_trackj2, phi_trackj2, m_trackj2;
      Int_t Ntracks, npT_tracks;
      vector<Double_t> pT_tracks_vec, eta_tracks_vec, phi_tracks_vec;
      vector<Long_t> pdgId_tracks_vec;
   };
   
   vector<EventData> event_data;
   
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) continue;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      
      // Store the event data
      EventData evt;
      evt.entry = jentry;
      evt.weight = weight;
      evt.weight_mc = weight_mc;
      evt.target_dd = target_dd;
      evt.pass190 = pass190;
      evt.pT_ll = pT_ll;
      evt.pT_l1 = pT_l1;
      evt.pT_l2 = pT_l2;
      evt.eta_l1 = eta_l1;
      evt.eta_l2 = eta_l2;
      evt.phi_l1 = phi_l1;
      evt.phi_l2 = phi_l2;
      evt.y_ll = y_ll;
      evt.pT_trackj1 = pT_trackj1;
      evt.y_trackj1 = y_trackj1;
      evt.phi_trackj1 = phi_trackj1;
      evt.m_trackj1 = m_trackj1;
      evt.pT_trackj2 = pT_trackj2;
      evt.y_trackj2 = y_trackj2;
      evt.phi_trackj2 = phi_trackj2;
      evt.m_trackj2 = m_trackj2;
      evt.Ntracks = Ntracks;
      evt.npT_tracks = npT_tracks;
      
      // Copy track arrays
      evt.pT_tracks_vec.assign(pT_tracks, pT_tracks + npT_tracks);
      evt.eta_tracks_vec.assign(eta_tracks, eta_tracks + npT_tracks);
      evt.phi_tracks_vec.assign(phi_tracks, phi_tracks + npT_tracks);
      if (isTruth) {
         evt.pdgId_tracks_vec.assign(pdgId_tracks, pdgId_tracks + npT_tracks);
      }
      
      event_data.push_back(evt);
      
      // Update progress bar for loading
      if (jentry % 1000 == 0) {
         bar.set_option(option::PostfixText{"Loading Events: " + to_string(jentry) + "/" + to_string(nentries)});
         bar.set_progress(static_cast<size_t>(jentry));
      }
   }
   
   std::cout << " === processing " << event_data.size() << " valid events in parallel === " << std::endl;
   
   // Parallel event processing loop
   #pragma omp parallel
   {
      int thread_id = omp_get_thread_num();
      
      #pragma omp for schedule(dynamic, 100)
      for (size_t idx=0; idx<event_data.size(); idx++) {
         const EventData& evt = event_data[idx];
         Long64_t jentry = evt.entry;

         // Update progress bar every 100 events for better performance (thread-safe)
         if (idx % 100 == 0) {
            #pragma omp critical
            {
               bar.set_option(option::PostfixText{"Processing Events: " + to_string(idx) + "/" + to_string(event_data.size())});
               // Update progress for ETA calculation
               bar.set_progress(static_cast<size_t>(idx));
            }
         }

         // Filter on pass 190 flag here
         if (evt.pass190 == 0) {
            continue;
         }

         // Skip events that have no tracks stored
         if (evt.Ntracks == 0) {
            continue;
         }

         // Create PseudoJet object of the ll system (Z-boson)
         TLorentzVector m1_tlv, m2_tlv;
         m1_tlv.SetPtEtaPhiM(evt.pT_l1, evt.eta_l1, evt.phi_l1, 0.10566);
         m2_tlv.SetPtEtaPhiM(evt.pT_l2, evt.eta_l2, evt.phi_l2, 0.10566);
         TLorentzVector zboson_tlv = m1_tlv + m2_tlv;
         PseudoJet zboson;
         zboson.reset_PtYPhiM(zboson_tlv.Pt(), zboson_tlv.Rapidity(), zboson_tlv.Phi(), zboson_tlv.M());

         // Create vector of PseudoJets from all tracks in event
         vector<PseudoJet> particles;
         for (int i=0; i<evt.npT_tracks; i++){ 
            TLorentzVector constit_tlv;
            // Which track 3 vectors to use depend on whether we are processing truth or reco
            // Truth files use double precision while reco files use float precision
            float constit_mass = 0.13957;
            if (isTruth) {
               constit_mass = GetMassFromPID(evt.pdgId_tracks_vec[i]);
            }
            constit_tlv.SetPtEtaPhiM(evt.pT_tracks_vec[i], evt.eta_tracks_vec[i], evt.phi_tracks_vec[i], constit_mass);
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

         // Calculate mjj, dyjj for R04 jets
         double R04_mjj, R04_dyjj;
         if (R04_jets.size() > 1) {
            R04_mjj = (R04_jets[0] + R04_jets[1]).m();
            R04_dyjj = TMath::Abs(R04_jets[0].rap() - R04_jets[1].rap());
         } else {
            R04_mjj = -999;
            R04_dyjj = -999;
         }

         // Calculate mjj, dRjj, dyjj for CA04 jets
         double CA04_mjj, CA04_dRjj, CA04_dyjj, CA04_dphijj;
         if (CA04_jets.size() > 1) {
            CA04_mjj = (CA04_jets[0] + CA04_jets[1]).m();
            CA04_dRjj = CA04_jets[0].delta_R(CA04_jets[1]);
            CA04_dyjj = TMath::Abs(CA04_jets[0].rap() - CA04_jets[1].rap());
            CA04_dphijj = (CA04_jets[0].rapidity() > CA04_jets[1].rapidity()) ? CA04_jets[0].delta_phi_to(CA04_jets[1]) : CA04_jets[1].delta_phi_to(CA04_jets[0]);
         } else {
            CA04_mjj = -999;
            CA04_dRjj = -999;
            CA04_dyjj = -999;
            CA04_dphijj = -999;
         }

         // Apply kinematic region cuts here
         if (kinematicRegion == 1 && (evt.pT_trackj2 < 50 || evt.pT_ll < 350)) {
            continue;
         } else if (kinematicRegion == 2 && (R04_mjj < 200 || R04_dyjj < 2)) {
            continue;
         } else if (kinematicRegion == 3 && evt.m_trackj1 < 32) {
            continue;
         }

         // Get EEC variables in jets
         double R04_Q2 = 0.0;
         vector<double> R04_esum;
         vector<double> R04_z;
         if (R04_jets.size() > 0) {
            R04_Q2 = GetEEC(R04_jets[0].constituents(), R04_esum, R04_z);
         }

         double R06_Q2 = 0.0;
         vector<double> R06_esum;
         vector<double> R06_z;
         if (R06_jets.size() > 0) {
            R06_Q2 = GetEEC(R06_jets[0].constituents(), R06_esum, R06_z);
         }

         double R10_Q2 = 0.0;
         vector<double> R10_esum;
         vector<double> R10_z;
         if (R10_jets.size() > 0) {
            R10_Q2 = GetEEC(R10_jets[0].constituents(), R10_esum, R10_z);
         }

         double CA04_Q2 = 0.0;
         vector<double> CA04_esum;
         vector<double> CA04_z;
         if (CA04_jets.size() > 0) {
            CA04_Q2 = GetEEC(CA04_jets[0].constituents(), CA04_esum, CA04_z);
         }

         // Get event level EEC variables
         double EEC_Q2;
         vector<double> EEC_esum;
         vector<double> EEC_z;
         EEC_Q2 = GetEEC(particles, EEC_esum, EEC_z);

         // Get TEEC variables
         double ETransTotal;
         vector<double> etrans;
         vector<double> tau;
         ETransTotal = GetTEEC(particles, zboson, etrans, tau);

         // -------------------- Fill Histograms --------------------
         // Note we are filling with values only from leading jet for now
         // Can also use the FillEEC function for the TEEC

         // Loop through all groups (central, ensemble, bootstrap)
         unsigned int total_groups = centralHistoGroups.size() + ensHistoGroups.size() + bootstrapHistoGroups.size();
         for (unsigned int i = 0; i < total_groups; ++i) {

            // Get weight
            float use_weight;
            if (i < centralHistoGroups.size()) {
               // Central weights
               if (weightBranchNames[i] == "weight") {
                  use_weight = evt.weight;
               } else if (weightBranchNames[i] == "weight_mc") {
                  use_weight = evt.weight_mc;
               } else if (weightBranchNames[i] == "target_dd") {
                  // target_dd can come from npz file (if specified) or ROOT tree
                  if (central_weights[i].size() > 0) {
                     use_weight = central_weights[i][jentry] * evt.weight_mc;
                  } else {
                     use_weight = evt.target_dd;
                  }
               } else {
                  use_weight = central_weights[i][jentry] * evt.weight_mc;
               }
            } else if (i < centralHistoGroups.size() + ensHistoGroups.size()) {
               // Ensemble weights
               unsigned int ens_idx = i - centralHistoGroups.size();
               use_weight = ens_weights[ens_idx][jentry] * evt.weight_mc;
            } else {
               // Bootstrap data weights
               unsigned int bootstrap_idx = i - centralHistoGroups.size() - ensHistoGroups.size();
               use_weight = bootstrap_data_weights[bootstrap_idx][jentry] * evt.weight_mc;
            }

            // Get thread-local histogram group
            HistoGroup& histoGroup = (i < centralHistoGroups.size()) ? 
               thread_central_histos[thread_id][i] : 
               (i < centralHistoGroups.size() + ensHistoGroups.size()) ?
               thread_ens_histos[thread_id][i - centralHistoGroups.size()] :
               thread_bootstrap_histos[thread_id][i - centralHistoGroups.size() - ensHistoGroups.size()];

            // // KT R=0.4 jets
            if (KT_jets.size() > 0) {
               histoGroup.hm1_KT04->Fill(KT_jets[0].m(), use_weight);
               histoGroup.hpT_KT04->Fill(KT_jets[0].pt(), use_weight);
            }

            // R=0.4 jets
            if (R04_jets.size() > 0) {
               histoGroup.hm1_R04->Fill(R04_jets[0].m(), use_weight);
            }
            if (R04_jets.size() > 1) {
               histoGroup.hm2_R04->Fill(R04_jets[1].m(), use_weight);
            }
            if (R04_jets.size() > 2) {
               histoGroup.hm3_R04->Fill(R04_jets[2].m(), use_weight);
            }
            if (R04_jets.size() > 3) {
               histoGroup.hm4_R04->Fill(R04_jets[3].m(), use_weight);
            }
            histoGroup.hmjj_R04->Fill(R04_mjj, use_weight);
            histoGroup.hdyjj_R04->Fill(R04_dyjj, use_weight);
            FillEEC(histoGroup.hEEC_R04, R04_esum, R04_z, R04_Q2, use_weight);
            // FillLund(histoGroup.hLund_z_R04, histoGroup.hLund_dR_R04, histoGroup.hLund_plane_R04, R04_lundz, R04_lundDr, use_weight);

            // // R=0.6 jets
            if (R06_jets.size() > 0) {
               histoGroup.hm1_R06->Fill(R06_jets[0].m(), use_weight);
               histoGroup.hpT_R06->Fill(R06_jets[0].pt(), use_weight);
            }
            FillEEC(histoGroup.hEEC_R06, R06_esum, R06_z, R06_Q2, use_weight);
            // FillLund(histoGroup.hLund_z_R06, histoGroup.hLund_dR_R06, histoGroup.hLund_plane_R06, R06_lundz, R06_lundDr, use_weight);

            // R=1.0 jets
            if (R10_jets.size() > 0) {
               histoGroup.hm1_R10->Fill(R10_jets[0].m(), use_weight);
               histoGroup.hpT_R10->Fill(R10_jets[0].pt(), use_weight);
            }
            FillEEC(histoGroup.hEEC_R10, R10_esum, R10_z, R10_Q2, use_weight);
            // FillLund(histoGroup.hLund_z_R10, histoGroup.hLund_dR_R10, histoGroup.hLund_plane_R10, R10_lundz, R10_lundDr, use_weight);

            // CA R=0.4 jets
            if (CA04_jets.size() > 0) {
               histoGroup.hm1_CA04->Fill(CA04_jets[0].m(), use_weight);
               histoGroup.hpT_CA04->Fill(CA04_jets[0].pt(), use_weight);
            }
            FillEEC(histoGroup.hEEC_CA04, CA04_esum, CA04_z, CA04_Q2, use_weight);
            histoGroup.hmjj_CA04->Fill(CA04_mjj, use_weight);
            histoGroup.hdRjj_CA04->Fill(CA04_dRjj, use_weight);
            histoGroup.hdyjj_CA04->Fill(CA04_dyjj, use_weight);
            histoGroup.hdphijj_CA04->Fill(CA04_dphijj, use_weight);

            // CA R=0.6 jets
            if (CA06_jets.size() > 0) {
               histoGroup.hm1_CA06->Fill(CA06_jets[0].m(), use_weight);
               histoGroup.hpT_CA06->Fill(CA06_jets[0].pt(), use_weight);
            }

            // Event-level EEC
            FillEEC(histoGroup.hTEEC_collinear, EEC_esum, EEC_z, EEC_Q2, use_weight);
            FillEEC(histoGroup.hTEEC_full_nolog, EEC_esum, EEC_z, EEC_Q2, use_weight);
            FillEEC(histoGroup.hTEEC_b2b, EEC_esum, EEC_z, EEC_Q2, use_weight, true);
            FillEEC(histoGroup.hTEEC_full, EEC_esum, EEC_z, EEC_Q2, use_weight);
            FillEEC(histoGroup.hTEEC_z_collinear, etrans, tau, ETransTotal, use_weight, true);
            FillEEC(histoGroup.hTEEC_z_full_nolog, etrans, tau, ETransTotal, use_weight);
            FillEEC(histoGroup.hTEEC_z_b2b, etrans, tau, ETransTotal, use_weight);
            FillEEC(histoGroup.hTEEC_z_full, etrans, tau, ETransTotal, use_weight);

         }

      } // End of parallel region

   } // End of OpenMP parallel block
   
   // Merge thread-local histograms into main histogram groups
   std::cout << " === merging thread-local histograms === " << std::endl;
   for (int t = 0; t < num_threads; ++t) {
      for (unsigned int i = 0; i < centralHistoGroups.size(); ++i) {
         centralHistoGroups[i].MergeHistos(thread_central_histos[t][i]);
      }
      for (unsigned int i = 0; i < ensHistoGroups.size(); ++i) {
         ensHistoGroups[i].MergeHistos(thread_ens_histos[t][i]);
      }
      for (unsigned int i = 0; i < bootstrapHistoGroups.size(); ++i) {
         bootstrapHistoGroups[i].MergeHistos(thread_bootstrap_histos[t][i]);
      }
   }
   
   // Complete the progress bar
   bar.set_option(option::PostfixText{"Processing Events: " + to_string(event_data.size()) + "/" + to_string(event_data.size()) + " - Complete!"});
   bar.set_progress(static_cast<size_t>(event_data.size()));
   std::cout << std::endl;

   std::cout << " === create output ROOT file === " << std::endl;
   TFile foutput(saveName, "recreate");

   std::cout << " === write in file === " << std::endl;
   for (auto& histoGroup : centralHistoGroups) {
      histoGroup.WriteHistos(foutput);
   }

   for (auto& histoGroup : ensHistoGroups) {
      histoGroup.WriteHistos(foutput);
   }

   for (auto& histoGroup : bootstrapHistoGroups) {
      histoGroup.WriteHistos(foutput);
   }

   // Close file
   foutput.Close();

}

void MakeOmni::FillEEC(shared_ptr<TH1D>& hEEC, const vector<double>& esum, const vector<double>& z, double Q2, double weight, bool flip_z) {
   for (size_t i = 0; i < esum.size(); ++i) {
      if (flip_z) {
         hEEC->Fill(1.0 - z[i], esum[i] * weight / Q2);
      } else {
         hEEC->Fill(z[i], esum[i] * weight / Q2);
      }
   }
}

void MakeOmni::FillLund(shared_ptr<TH1D>& hLund_z, shared_ptr<TH1D>& hLund_dR, shared_ptr<TH2D>& hLund_plane, const vector<double>& lundz, const vector<double>& lundDr, double weight) {
   for (size_t i = 0; i < lundz.size(); ++i) {
      hLund_z->Fill(lundz[i], weight);
      hLund_dR->Fill(lundDr[i], weight);
      hLund_plane->Fill(lundDr[i], lundz[i], weight);
   }
}


