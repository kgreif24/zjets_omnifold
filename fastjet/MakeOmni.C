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
#include "indicators/progress_bar.hpp"
#include <omp.h>
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

   // Access weights, assume we want the test weights
   cnpy::npz_t npz_file = cnpy::npz_load(filename);
   cnpy::NpyArray array = npz_file[key];

   // Convert to vector<float> and return
   vector<float> vec(array.data<float>(), array.data<float>() + array.num_vals);
   return vec;

}

void MakeOmni::WriteIBUTree(std::vector<EventData> event_data, TFile& foutput){

    TTree* t_IBU = new TTree("add_to_IBU", "Add-to-IBU Tree");

    // --- Determine branch structure from the first event ---
    const EventData& first = event_data[0];

    // Storage for branch variables
    std::map<std::string, Float_t> b_outfloats;
    Long64_t b_entry;

    // Book branches for out_floats
    for (const auto& kv : first.out_floats) {
        const std::string& name = kv.first;
        b_outfloats[name] = 0.0f;  // allocate storage
        t_IBU->Branch(name.c_str(), &b_outfloats[name], (name + "/F").c_str());
    }

    t_IBU->Branch("entry", &b_entry, "entry/L");

    // --- Fill the tree ---
    for (const auto& evt : event_data) {
        for (const auto& kv : evt.out_floats) {
            const std::string& name = kv.first;
            b_outfloats[name] = kv.second;
        }
        b_entry = evt.entry;
        t_IBU->Fill();
    }

    // Write tree into the output file
    foutput.cd();
    t_IBU->Write();
}


void MakeOmni::Loop(Long64_t maxEvents) { 
     // Load needed weights
    vector<vector<float>> central_weights;
    vector<vector<float>> ens_weights;

    // Load the needed weights
    if (weightFilename == "None"){
        for (long unsigned int i=0; i< weightBranchNames.size();i++) {
            central_weights.push_back(vector<float>());
        }
    }
    else{
        for (const auto& weight_name : weightBranchNames) {
            if (weight_name != "weight" && weight_name != "weight_mc" && weight_name != "target_dd") {
              central_weights.push_back(LoadWeights(weightFilename, weight_name + "-central"));
            } else {
              central_weights.push_back(vector<float>());
            }
        }
    }

    // Load the ensemble weights
    for (int i = 0; i < nEns; ++i) {
        ens_weights.push_back(LoadWeights(weightFilename, weightBranchNames[0] + "-" + to_string(i)));
    }



   // ----------------------- Loop over events -----------------------
   if (fChain == 0) return;
   fChain->SetCacheSize(0);

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
   
   // Initialize thread-local histogram groups
   for (int t = 0; t < num_threads; ++t) {
      for (const auto& weight_name : weightBranchNames) {
         if (weight_name != "weight" && weight_name != "weight_mc") {
            thread_central_histos[t].push_back(HistoGroup(weight_name + "-", kinematicRegion, isTruth, do_IBU, is_data));
         } else {
            thread_central_histos[t].push_back(HistoGroup("nominal-", kinematicRegion, isTruth, do_IBU, is_data)); 
         }
      }
      
      for (int i = 0; i < nEns; ++i) {
         thread_ens_histos[t].push_back(HistoGroup(weightBranchNames[0] + "-" + to_string(i) + "-", kinematicRegion, isTruth, do_IBU, is_data));
      }
   }

    // ----------------------- Levels to consider ----------------------- 
    vector<string> levels;
    if (do_IBU && !is_data)        levels = {"truth_", ""};
    else if (isTruth && !is_data)  levels = {"truth_"};
    else                           levels = {""};

    // ----------------------- Jet definitions to consider ----------------------- 
    map<string,JetDefinition> m_jetdef;
     // hardcoded for now.currently need to separately modify in HistoGroup.cxx aswell.  Will go into a config file. 
    for (string pre: levels){
      cout << pre << endl;
      m_jetdef[pre+"jetdef_kt"]   = JetDefinition(kt_algorithm, 0.4);

      for (auto jetR: thread_central_histos[0][0].antikt_jetR){
        string id = get_jetR_id("_r",jetR);// _r for antikT, _kt for kT, _ca for Cambridge aachen
        m_jetdef[pre+"jetdef"+id]   = JetDefinition(antikt_algorithm, jetR);
      }
      for (auto jetR: thread_central_histos[0][0].ca_jetR){
        string id = get_jetR_id("_ca",jetR);// _r for antikT, _kt for kT, _ca for Cambridge aachen
        m_jetdef[pre+"jetdef"+id]   = JetDefinition(cambridge_algorithm, jetR);
      }
    }
    


   
   // Pre-load all events sequentially (ROOT trees are not thread-safe)
   std::cout << " === pre-loading events === " << std::endl; 
   vector<Long64_t> valid_entries;


   vector<EventData> event_data;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {

      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) continue;

      nb = fChain->GetEntry(jentry);   nbytes += nb;

      // Store the event data 
      EventData evt;
      evt.entry = jentry;
      evt.floats["weight"] = weight;
      evt.floats["weight_mc"] = weight_mc;

      evt.floats["target_dd"] = target_dd;
      evt.ints["pass190"] = pass190;
      evt.floats["pT_ll"] = pT_ll;
      evt.floats["pT_l1"] = pT_l1;
      evt.floats["pT_l2"] = pT_l2;
      evt.floats["eta_l1"] = eta_l1;
      evt.floats["eta_l2"] = eta_l2; 
      evt.floats["phi_l1"] = phi_l1;
      evt.floats["phi_l2"] = phi_l2;
      evt.floats["y_ll"] = y_ll;
      evt.floats["pT_trackj1"] = pT_trackj1;
      evt.floats["y_trackj1"] = y_trackj1; 
      evt.floats["phi_trackj1"] = phi_trackj1;
      evt.floats["m_trackj1"] = m_trackj1;
      evt.floats["pT_trackj2"] = pT_trackj2;
      evt.floats["y_trackj2"] = y_trackj2;
      evt.floats["phi_trackj2"] = phi_trackj2;
      evt.floats["m_trackj2"] = m_trackj2;
      evt.ints["Ntracks"] = Ntracks;

      if (hasTruth && !is_data) {
        evt.ints["truth_pass190"] = truth_pass190;
        evt.floats["truth_pT_ll"] = truth_pT_ll;
        evt.floats["truth_pT_l1"] = truth_pT_l1;
        evt.floats["truth_pT_l2"] = truth_pT_l2;
        evt.floats["truth_eta_l1"] = truth_eta_l1;
        evt.floats["truth_eta_l2"] = truth_eta_l2;
        evt.floats["truth_phi_l1"] = truth_phi_l1;
        evt.floats["truth_phi_l2"] = truth_phi_l2;
        evt.floats["truth_y_ll"] = truth_y_ll;
        evt.floats["truth_pT_trackj1"] = truth_pT_trackj1;
        evt.floats["truth_y_trackj1"] = truth_y_trackj1;
        evt.floats["truth_phi_trackj1"] = truth_phi_trackj1;
        evt.floats["truth_m_trackj1"] = truth_m_trackj1;      
        evt.floats["truth_pT_trackj2"] = truth_pT_trackj2;
        evt.floats["truth_y_trackj2"] = truth_y_trackj2;
        evt.floats["truth_phi_trackj2"] = truth_phi_trackj2;
        evt.floats["truth_m_trackj2"] = truth_m_trackj2;
        evt.ints["truth_Ntracks"] = truth_Ntracks;
      }
      

      if (std::find(trackVariations.begin(), trackVariations.end(), "syst_pTScale_") != trackVariations.end()) {
        evt.i_vecs["syst_pTScale_"].assign(syst_correctedpT_tracks->begin(), syst_correctedpT_tracks->end());
      }
      if (std::find(trackVariations.begin(), trackVariations.end(), "syst_Fake_") != trackVariations.end()) {
        evt.i_vecs["syst_Fake_"].assign(syst_passTrackFake_tracks->begin(), syst_passTrackFake_tracks->end());
      }
      if (std::find(trackVariations.begin(), trackVariations.end(), "syst_TrackFilter_") != trackVariations.end()) {
        evt.i_vecs["syst_TrackFilter_"].assign(syst_passTrackTruthFilter_tracks->begin(), syst_passTrackTruthFilter_tracks->end());
      }
      if (std::find(trackVariations.begin(), trackVariations.end(), "syst_JetTrackFilter_") != trackVariations.end()) {
        evt.i_vecs["syst_JetTrackFilter_"].assign(syst_passJetTrackFilter_tracks->begin(), syst_passJetTrackFilter_tracks->end());
      }

      if (has_kevin_branches) {   
        evt.ints["npT_tracks"] = npT_tracks;
        evt.f_vecs["pT_tracks_vec"].assign(pT_tracks, pT_tracks + npT_tracks);
        evt.f_vecs["eta_tracks_vec"].assign(eta_tracks, eta_tracks + npT_tracks);
        evt.f_vecs["phi_tracks_vec"].assign(phi_tracks, phi_tracks + npT_tracks);

        evt.ints["ntruth_pT_tracks"] = ntruth_pT_tracks;   
        evt.f_vecs["truth_pT_tracks_vec"].assign(truth_pT_tracks, truth_pT_tracks + ntruth_pT_tracks);
        evt.f_vecs["truth_eta_tracks_vec"].assign(truth_eta_tracks, truth_eta_tracks + ntruth_pT_tracks) ;
        evt.f_vecs["truth_phi_tracks_vec"].assign(truth_phi_tracks2, truth_phi_tracks2 + ntruth_pT_tracks);
        evt.i_vecs["truth_pdgId_tracks_vec"].assign(truth_pdgId_tracks, truth_pdgId_tracks + ntruth_pT_tracks);
      }
      else {
        evt.ints["npT_tracks"] = Ntracks;   
        evt.f_vecs["pT_tracks_vec"].assign(pT_tracks_vec->begin(), pT_tracks_vec->end());
        evt.f_vecs["eta_tracks_vec"].assign(eta_tracks_vec->begin(), eta_tracks_vec->end());
        evt.f_vecs["phi_tracks_vec"].assign(phi_tracks_vec->begin(), phi_tracks_vec->end()); 


        if (hasTruth && !is_data) {      
          evt.f_vecs["truth_pT_tracks_vec"].assign(truth_pT_tracks_vec->begin(), truth_pT_tracks_vec->end());
          evt.f_vecs["truth_eta_tracks_vec"].assign(truth_eta_tracks_vec->begin(), truth_eta_tracks_vec->end());
          evt.f_vecs["truth_phi_tracks_vec"].assign(truth_phi_tracks_vec->begin(), truth_phi_tracks_vec->end());
          evt.i_vecs["truth_pdgId_tracks_vec"].assign(truth_pdgId_tracks_vec->begin(), truth_pdgId_tracks_vec->end());
        }
      }


      if (has_theory_weights){
        string pre = "";
        if (theory_prefix) pre = "w_";
         evt.w_theory[pre+"QCD_uu"] = w_QCD_uu;
         evt.w_theory[pre+"QCD_dd"] = w_QCD_dd;
         evt.w_theory[pre+"QCD_un"] = w_QCD_un;
         evt.w_theory[pre+"QCD_nu"] = w_QCD_nu;
         evt.w_theory[pre+"QCD_nd"] = w_QCD_nd;
         evt.w_theory[pre+"QCD_dn"] = w_QCD_dn;
        //  evt.w_theory[pre+"PDF_CT14nnlo"] = w_PDF_CT14nnlo;
        //  evt.w_theory[pre+"PDF_MMHT2014"] = w_PDF_MMHT2014;
         evt.w_theory[pre+"PDF_MSHT2020"] = w_PDF_MSHT2020;
         evt.w_theory[pre+"PDF_CT18nnlo"] = w_PDF_CT18nnlo;
         evt.w_theory[pre+"Alpha_s1"] = w_Alpha_s1;
         evt.w_theory[pre+"Alpha_s2"] = w_Alpha_s2;
         evt.w_theory["w_Var2Up"] = w_Var2Up;
         evt.w_theory["w_Var2Down"] = w_Var2Down;
         evt.w_theory["w_Var1Up"] = w_Var1Up;
         evt.w_theory["w_Var1Down"] = w_Var1Down;
         evt.w_theory["w_MPIUp"] = w_MPIUp;
         evt.w_theory["w_MPIDown"] = w_MPIDown;
         evt.w_theory["w_RenUp"] = w_RenUp;
         evt.w_theory["w_RenDown"] = w_RenDown;
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
         EventData& evt = event_data[idx];
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
        for (string trackVar: trackVariations){
          for (string pre:levels){ 
            if (pre == "truth_" && trackVar!= "") continue; // need some way to mske sure this is cirrec
            

            // Filter on pass 190 flag here
            if (evt.ints.at(pre+"pass190") == 0 && kinematicRegion > 0) { 
                continue;  
            }

            // Skip events that have no tracks stored
            if (evt.ints.at(pre+"Ntracks") == 0 && kinematicRegion > 0) {
                continue;
            }

            // Create PseudoJet object of the ll system (Z-boson)
            TLorentzVector m1_tlv, m2_tlv;
            m1_tlv.SetPtEtaPhiM(evt.floats.at(pre+"pT_l1"), evt.floats.at(pre+"eta_l1"), evt.floats.at(pre+"phi_l1"), 0.10566);
            m2_tlv.SetPtEtaPhiM(evt.floats.at(pre+"pT_l2"), evt.floats.at(pre+"eta_l2"), evt.floats.at(pre+"phi_l2"), 0.10566);
            TLorentzVector zboson_tlv = m1_tlv + m2_tlv;
            PseudoJet zboson;
            zboson.reset_PtYPhiM(zboson_tlv.Pt(), zboson_tlv.Rapidity(), zboson_tlv.Phi(), zboson_tlv.M());

            // Create vector of PseudoJets from all tracks in event
            vector<PseudoJet> particles;
            for (long unsigned int i=0; i<evt.f_vecs.at(pre+"pT_tracks_vec").size(); i++){ 
              if (trackVar!=""){ if (!evt.i_vecs.at(trackVar)[i]) continue;}
              TLorentzVector constit_tlv;
              // Which track 3 vectors to use depend on whether we are processing truth or reco
              // Truth files use double precision while reco files use float precision
              float constit_mass = 0.13957;
              if (isTruth && pre=="truth_") {
                constit_mass = GetMassFromPID(evt.i_vecs.at(pre+"pdgId_tracks_vec")[i]);
              }
              constit_tlv.SetPtEtaPhiM(evt.f_vecs.at(pre+"pT_tracks_vec")[i], evt.f_vecs.at(pre+"eta_tracks_vec")[i], evt.f_vecs.at(pre+"phi_tracks_vec")[i], constit_mass);
              PseudoJet constit_pj;
              constit_pj.reset_PtYPhiM(constit_tlv.Pt(), constit_tlv.Rapidity(), constit_tlv.Phi(), constit_tlv.M());
              particles.push_back(constit_pj);
            }

            // Build anti-kt jets w/ R=0.4, 0.6, 1.0 and CA
            map<string,ClusterSequence> m_clustseq;
            map<string,vector<PseudoJet>> m_jets;
            m_clustseq[trackVar+pre+"cs_seq_kt"] = ClusterSequence(particles, m_jetdef[pre+"jetdef_kt"]);
            m_jets[trackVar+pre+"KT_jets"]       = sorted_by_pt(m_clustseq[trackVar+pre+"cs_seq_kt"].inclusive_jets());

              // generalized for lists of jet radii. Includes R=0.4, 0.6, 1.0
            for (auto jetR: centralHistoGroups[0].antikt_jetR){
                string id = get_jetR_id("_r",jetR);// _r for antikT, _kt for kT, _ca for Cambridge aachen
                m_clustseq[trackVar+pre+"cs_seq"+id] = ClusterSequence(particles, m_jetdef[pre+"jetdef"+id]);
                m_jets[trackVar+pre+"jets"+id]       = sorted_by_pt(m_clustseq[trackVar+pre+"cs_seq"+id].inclusive_jets());
            }
            for (auto jetR: centralHistoGroups[0].ca_jetR){
                string id = get_jetR_id("_ca",jetR);// _r for antikT, _kt for kT, _ca for Cambridge aachen
                m_clustseq[trackVar+pre+"cs_seq"+id] = ClusterSequence(particles, m_jetdef[pre+"jetdef"+id]);
                m_jets[trackVar+pre+"jets"+id]       = sorted_by_pt(m_clustseq[trackVar+pre+"cs_seq"+id].inclusive_jets());
            }

            // Calculate mjj, dyjj for R04 jets
            double R04_mjj, R04_dyjj;
            if (m_jets[trackVar+pre+"jets_r04"].size() > 1) {
                R04_mjj = (m_jets[trackVar+pre+"jets_r04"][0] + m_jets[trackVar+pre+"jets_r04"][1]).m();
                R04_dyjj = TMath::Abs(m_jets[trackVar+pre+"jets_r04"][0].rap() - m_jets[trackVar+pre+"jets_r04"][1].rap());
            } else {
                R04_mjj = -999;
                R04_dyjj = -999;
            }

            // Calculate mjj, dRjj, dyjj for CA04 jets
            double CA04_mjj, CA04_dRjj, CA04_dyjj, CA04_dphijj;
            if (m_jets[trackVar+pre+"jets_ca04"].size() > 1) {
                CA04_mjj = (m_jets[trackVar+pre+"jets_ca04"][0] + m_jets[trackVar+pre+"jets_ca04"][1]).m();
                CA04_dRjj = m_jets[trackVar+pre+"jets_ca04"][0].delta_R(m_jets[trackVar+pre+"jets_ca04"][1]);
                CA04_dyjj = TMath::Abs(m_jets[trackVar+pre+"jets_ca04"][0].rap() - m_jets[trackVar+pre+"jets_ca04"][1].rap());
                CA04_dphijj = (m_jets[trackVar+pre+"jets_ca04"][0].rapidity() > m_jets[trackVar+pre+"jets_ca04"][1].rapidity()) ? m_jets[trackVar+pre+"jets_ca04"][0].delta_phi_to(m_jets[trackVar+pre+"jets_ca04"][1]) : m_jets[trackVar+pre+"jets_ca04"][1].delta_phi_to(m_jets[trackVar+pre+"jets_ca04"][0]);
            } else {
                CA04_mjj = -999;
                CA04_dRjj = -999;
                CA04_dyjj = -999;
                CA04_dphijj = -999;
            }
 
            // Apply kinematic region cuts here 
            if (kinematicRegion == 1 && (evt.floats.at(pre+"pT_trackj2")< 50 || evt.floats.at(pre+"pT_ll") < 350)) { 
                continue;
            } else if (kinematicRegion == 2 && (R04_mjj < 200 || R04_dyjj < 2)) {
                continue;
            } else if (kinematicRegion == 3 && evt.floats.at(pre+"m_trackj1") < 32) {
                continue;
            }

            if (do_IBU){
                          // -------------------- If saving IBU branches --------------------
              if (m_jets[trackVar+pre+"jets_r04"].size() > 0){
                evt.out_floats[trackVar+pre+"antikT_r04_j1_pT"]  = m_jets[trackVar+pre+"jets_r04"][0].pt();
                evt.out_floats[trackVar+pre+"antikT_r04_j1_rapidity"] = m_jets[trackVar+pre+"jets_r04"][0].rapidity();
                evt.out_floats[trackVar+pre+"antikT_r04_j1_phi"] = m_jets[trackVar+pre+"jets_r04"][0].phi() > std::numbers::pi? m_jets[trackVar+pre+"jets_r04"][0].phi() - 2*std::numbers::pi: m_jets[trackVar+pre+"jets_r04"][0].phi();
                evt.out_floats[trackVar+pre+"antikT_r04_j1_m"]   = m_jets[trackVar+pre+"jets_r04"][0].m();
              } else {
                evt.out_floats[trackVar+pre+"antikT_r04_j1_pT"]  = -99;
                evt.out_floats[trackVar+pre+"antikT_r04_j1_rapidity"] = -99;
                evt.out_floats[trackVar+pre+"antikT_r04_j1_phi"] = -99;
                evt.out_floats[trackVar+pre+"antikT_r04_j1_m"]   = -99;
              }

              if (m_jets[trackVar+pre+"jets_r06"].size() > 0){
                evt.out_floats[trackVar+pre+"antikT_r06_j1_pT"]  = m_jets[trackVar+pre+"jets_r06"][0].pt();
                evt.out_floats[trackVar+pre+"antikT_r06_j1_rapidity"] = m_jets[trackVar+pre+"jets_r06"][0].rapidity();
                evt.out_floats[trackVar+pre+"antikT_r06_j1_phi"] = m_jets[trackVar+pre+"jets_r06"][0].phi() > std::numbers::pi ? m_jets[trackVar+pre+"jets_r06"][0].phi() - 2*std::numbers::pi: m_jets[trackVar+pre+"jets_r06"][0].phi();
                evt.out_floats[trackVar+pre+"antikT_r06_j1_m"]   = m_jets[trackVar+pre+"jets_r06"][0].m();
              } else {
                evt.out_floats[trackVar+pre+"antikT_r06_j1_pT"]  = -99;
                evt.out_floats[trackVar+pre+"antikT_r06_j1_rapidity"] = -99;
                evt.out_floats[trackVar+pre+"antikT_r06_j1_phi"] = -99;
                evt.out_floats[trackVar+pre+"antikT_r06_j1_m"]   = -99;
              }

              if (m_jets[trackVar+pre+"jets_r10"].size() > 0){
                evt.out_floats[trackVar+pre+"antikT_r10_j1_pT"]  = m_jets[trackVar+pre+"jets_r10"][0].pt();
                evt.out_floats[trackVar+pre+"antikT_r10_j1_rapidity"] = m_jets[trackVar+pre+"jets_r10"][0].rapidity();
                evt.out_floats[trackVar+pre+"antikT_r10_j1_phi"] = m_jets[trackVar+pre+"jets_r10"][0].phi() > std::numbers::pi? m_jets[trackVar+pre+"jets_r10"][0].phi() - 2*std::numbers::pi: m_jets[trackVar+pre+"jets_r10"][0].phi();
                evt.out_floats[trackVar+pre+"antikT_r10_j1_m"]   = m_jets[trackVar+pre+"jets_r10"][0].m();
              } else {
                evt.out_floats[trackVar+pre+"antikT_r10_j1_pT"]  = -99;
                evt.out_floats[trackVar+pre+"antikT_r10_j1_rapidity"] = -99;
                evt.out_floats[trackVar+pre+"antikT_r10_j1_phi"] = -99;
                evt.out_floats[trackVar+pre+"antikT_r10_j1_m"]   = -99;
              }

              if (m_jets[trackVar+pre+"jets_ca04"].size() > 0){
                evt.out_floats[trackVar+pre+"ca_r04_j1_m"]       = m_jets[trackVar+pre+"jets_ca04"][0].m();
              } else {
                evt.out_floats[trackVar+pre+"ca_r04_j1_m"]       = -99;
              }

              if (m_jets[trackVar+pre+"jets_ca06"].size() > 0){
                evt.out_floats[trackVar+pre+"ca_r06_j1_m"]       = m_jets[trackVar+pre+"jets_ca06"][0].m();
              } else {
                evt.out_floats[trackVar+pre+"ca_r06_j1_m"]       = -99;
              }

              if (m_jets[trackVar+pre+"jets_ca10"].size() > 0){
                evt.out_floats[trackVar+pre+"ca_r10_j1_m"]       = m_jets[trackVar+pre+"jets_ca10"][0].m();
              } else {
                evt.out_floats[trackVar+pre+"ca_r10_j1_m"]       = -99;
              }

              evt.out_floats[trackVar+pre+"ca_r04_mjj"]        = CA04_mjj;
              evt.out_floats[trackVar+pre+"ca_r04_dRjj"]       = CA04_dRjj;
              evt.out_floats[trackVar+pre+"ca_r04_dphijj"]     = CA04_dphijj;    

            }
            else {

              // // Get Lund variables 
              // vector<double> R04_lundz;
              // vector<double> R04_lundkt;
              // vector<double> R04_lundDr;
              // if (m_jets["jets_r04"].size() > 0) {
              //    processJets(m_jets["jets_r04"][0], 0.4, R04_lundz, R04_lundkt, R04_lundDr);
              // }

              // vector<double> R06_lundz;
              // vector<double> R06_lundkt;
              // vector<double> R06_lundDr;
              // if (m_jets["jets_r06"].size() > 0) {
              //    processJets(m_jets["jets_r06"][0], 0.6, R06_lundz, R06_lundkt, R06_lundDr);
              // }

              // vector<double> R10_lundz;
              // vector<double> R10_lundkt;
              // vector<double> R10_lundDr;
              // if (m_jets["jets_r10"].size() > 0) {
              //    processJets(m_jets["jets_r10"][0], 1.0, R10_lundz, R10_lundkt, R10_lundDr);
              // }

              // vector<double> CA04_lundz;
              // vector<double> CA04_lundkt;
              // vector<double> CA04_lundDr;
              // if (m_jets[trackVar+pre+"jets_ca04"].size() > 0) {
              //    processJets(m_jets[trackVar+pre+"jets_ca04"][0], 0.4, CA04_lundz, CA04_lundkt, CA04_lundDr);
              // }

              // Get EEC variables in jets
              double R04_Q2 = 0.0;
              vector<double> R04_esum;
              vector<double> R04_z;
              if (m_jets[trackVar+pre+"jets_r04"].size() > 0) {
                  R04_Q2 = GetEEC(m_jets[trackVar+pre+"jets_r04"][0].constituents(), R04_esum, R04_z);
              }
              double R06_Q2 = 0.0;
              vector<double> R06_esum;
              vector<double> R06_z;
              if (m_jets[trackVar+pre+"jets_r06"].size() > 0) {
                  R06_Q2 = GetEEC(m_jets[trackVar+pre+"jets_r06"][0].constituents(), R06_esum, R06_z);
              }

              double R10_Q2 = 0.0;
              vector<double> R10_esum;
              vector<double> R10_z;
              if (m_jets[trackVar+pre+"jets_r10"].size() > 0) {
                  R10_Q2 = GetEEC(m_jets[trackVar+pre+"jets_r10"][0].constituents(), R10_esum, R10_z);
              }

              double CA04_Q2 = 0.0;
              vector<double> CA04_esum;
              vector<double> CA04_z;
              if (m_jets[trackVar+pre+"jets_ca04"].size() > 0) {
                  CA04_Q2 = GetEEC(m_jets[trackVar+pre+"jets_ca04"][0].constituents(), CA04_esum, CA04_z);
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

              // Note we are filling with values only from leading jet for now
              // Can also use the FillEEC function for the TEEC
              // Loop through all groups  
              for (unsigned int i = 0; i < centralHistoGroups.size() + ensHistoGroups.size(); ++i) {

                // Get weight
                float use_weight; 
                if (weightBranchNames[i] == "weight") {
                  use_weight = evt.floats.at("weight");
                } else if (weightBranchNames[i] == "weight_mc") {
                  use_weight = evt.floats.at("weight_mc");
                } else if (weightBranchNames[i] == "target_dd") {
                  use_weight = evt.floats.at("target_dd");
                } else if (evt.w_theory.find(weightBranchNames[i]) != evt.w_theory.end()) {
                  if (is_multiplicative[weightBranchNames[i]]){ // assumes if no weight file, then no ensembling.
                    if (isTruth && pre=="truth_") use_weight = evt.w_theory.at(weightBranchNames[i]) * evt.floats.at("weight_mc");
                    else                          use_weight = evt.w_theory.at(weightBranchNames[i]) * evt.floats.at("weight");
                  }
                  else { 
                    use_weight = evt.w_theory.at(weightBranchNames[i]);
                  }
                } else if (i < centralHistoGroups.size()) { 
                  use_weight = central_weights[i][jentry] * evt.floats.at("weight_mc") ;
                } else {
                  use_weight = ens_weights[i - centralHistoGroups.size()][jentry] * evt.floats.at("weight_mc");
                }

                // Get thread-local histogram group
                HistoGroup& histoGroup = (i < centralHistoGroups.size()) ? 
                  thread_central_histos[thread_id][i] : 
                  thread_ens_histos[thread_id][i - centralHistoGroups.size()];
                // // KT R=0.4 jets
                if (m_jets[trackVar+pre+"KT_jets"].size() > 0) {
                  histoGroup.h_map[trackVar+pre+"hm1_KT04"]->Fill(m_jets[trackVar+pre+"KT_jets"][0].m(), use_weight);
                  histoGroup.h_map[trackVar+pre+"hpT_KT04"]->Fill(m_jets[trackVar+pre+"KT_jets"][0].pt(), use_weight);
                }

                // R=0.4 jets
                if (m_jets[trackVar+pre+"jets_r04"].size() > 0) {
                  histoGroup.h_map[trackVar+pre+"hm1_R04"]->Fill(m_jets[trackVar+pre+"jets_r04"][0].m(), use_weight);
                }
                if (m_jets[trackVar+pre+"jets_r04"].size() > 1) {
                  histoGroup.h_map[trackVar+pre+"hm2_R04"]->Fill(m_jets[trackVar+pre+"jets_r04"][1].m(), use_weight);
                }
                if (m_jets[trackVar+pre+"jets_r04"].size() > 2) {
                  histoGroup.h_map[trackVar+pre+"hm3_R04"]->Fill(m_jets[trackVar+pre+"jets_r04"][2].m(), use_weight);
                }
                if (m_jets[trackVar+pre+"jets_r04"].size() > 3) {
                  histoGroup.h_map[trackVar+pre+"hm4_R04"]->Fill(m_jets[trackVar+pre+"jets_r04"][3].m(), use_weight);
                }
                histoGroup.h_map[trackVar+pre+"hmjj_R04"]->Fill(R04_mjj, use_weight);
                histoGroup.h_map[trackVar+pre+"hdyjj_R04"]->Fill(R04_dyjj, use_weight);
                FillEEC(histoGroup.h_map[trackVar+pre+"hEEC_R04"], R04_esum, R04_z, R04_Q2, use_weight);

                // FillLund(histoGroup.hLund_z_R04, histoGroup.hLund_dR_R04, histoGroup.hLund_plane_R04, R04_lundz, R04_lundDr, use_weight);
                // // R=0.6 jets
                if (m_jets[trackVar+pre+"jets_r06"].size() > 0) {
                  histoGroup.h_map[trackVar+pre+"hm1_R06"]->Fill(m_jets[trackVar+pre+"jets_r06"][0].m(), use_weight);
                  histoGroup.h_map[trackVar+pre+"hpT_R06"]->Fill(m_jets[trackVar+pre+"jets_r06"][0].pt(), use_weight);
                }
                FillEEC(histoGroup.h_map["hEEC_R06"], R06_esum, R06_z, R06_Q2, use_weight);
                // FillLund(histoGroup.hLund_z_R06, histoGroup.hLund_dR_R06, histoGroup.hLund_plane_R06, R06_lundz, R06_lundDr, use_weight);

                // R=1.0 jets
                if (m_jets[trackVar+pre+"jets_r10"].size() > 0) {
                  histoGroup.h_map[trackVar+pre+"hm1_R10"]->Fill(m_jets[trackVar+pre+"jets_r10"][0].m(), use_weight);
                  histoGroup.h_map[trackVar+pre+"hpT_R10"]->Fill(m_jets[trackVar+pre+"jets_r10"][0].pt(), use_weight);
                }
                FillEEC(histoGroup.h_map["hEEC_R10"], R10_esum, R10_z, R10_Q2, use_weight);
                // FillLund(histoGroup.hLund_z_R10, histoGroup.hLund_dR_R10, histoGroup.hLund_plane_R10, R10_lundz, R10_lundDr, use_weight);

                // CA R=0.4 jets
                if (m_jets[trackVar+pre+"jets_ca04"].size() > 0) {
                  histoGroup.h_map[trackVar+pre+"hm1_CA04"]->Fill(m_jets[trackVar+pre+"jets_ca04"][0].m(), use_weight);
                  histoGroup.h_map[trackVar+pre+"hpT_CA04"]->Fill(m_jets[trackVar+pre+"jets_ca04"][0].pt(), use_weight);
                }
                FillEEC(histoGroup.h_map[trackVar+pre+"hEEC_CA04"], CA04_esum, CA04_z, CA04_Q2, use_weight);
                histoGroup.h_map[trackVar+pre+"hmjj_CA04"]->Fill(CA04_mjj, use_weight);
                histoGroup.h_map[trackVar+pre+"hdRjj_CA04"]->Fill(CA04_dRjj, use_weight);
                histoGroup.h_map[trackVar+pre+"hdyjj_CA04"]->Fill(CA04_dyjj, use_weight);
                histoGroup.h_map[trackVar+pre+"hdphijj_CA04"]->Fill(CA04_dphijj, use_weight);

                // CA R=0.6 jets
                if (m_jets[trackVar+pre+"jets_ca06"].size() > 0) {
                  histoGroup.h_map[trackVar+pre+"hm1_CA06"]->Fill(m_jets[trackVar+pre+"jets_ca06"][0].m(), use_weight);
                  histoGroup.h_map[trackVar+pre+"hpT_CA06"]->Fill(m_jets[trackVar+pre+"jets_ca06"][0].pt(), use_weight);
                }

                // Event-level EEC
                FillEEC(histoGroup.h_map[trackVar+pre+"hTEEC_collinear"], EEC_esum, EEC_z, EEC_Q2, use_weight);
                FillEEC(histoGroup.h_map[trackVar+pre+"hTEEC_full_nolog"], EEC_esum, EEC_z, EEC_Q2, use_weight);
                FillEEC(histoGroup.h_map[trackVar+pre+"hTEEC_b2b"], EEC_esum, EEC_z, EEC_Q2, use_weight, true);
                FillEEC(histoGroup.h_map[trackVar+pre+"hTEEC_full"], EEC_esum, EEC_z, EEC_Q2, use_weight);
                FillEEC(histoGroup.h_map[trackVar+pre+"hTEEC_z_collinear"], etrans, tau, ETransTotal, use_weight, true);
                FillEEC(histoGroup.h_map[trackVar+pre+"hTEEC_z_full_nolog"], etrans, tau, ETransTotal, use_weight);
                FillEEC(histoGroup.h_map[trackVar+pre+"hTEEC_z_b2b"], etrans, tau, ETransTotal, use_weight);
                FillEEC(histoGroup.h_map[trackVar+pre+"hTEEC_z_full"], etrans, tau, ETransTotal, use_weight);

                // Fill TProfiles varying jet radius of antikt jets, in slices of leading jet pT and y
                for (auto jetR : histoGroup.antikt_jetR){
                    string id = get_jetR_id("_r",jetR);
                    double M1OverpT = m_jets[trackVar+pre+"jets"+id][0].m() / m_jets[trackVar+pre+"jets"+id][0].pt();

                    histoGroup.prof_map[trackVar+pre+"prof_aktRVaried_M1OverpT_All"]->Fill(jetR,M1OverpT, use_weight);

                    bool fill = false;
                    for (long unsigned int i = 0; i < histoGroup.pTj1_bins.size()-1; i++){
                        fill = inBin(evt.floats.at("truth_pT_trackj1"), histoGroup.pTj1_bins[i], histoGroup.pTj1_bins[i+1]); // alwayse make phase space cuts based of truth info!
                        for (long unsigned int j = 0; j < histoGroup.yj1_bins.size()-1; j++){
                            fill = fill && inBin(evt.floats.at("truth_y_trackj1"), histoGroup.yj1_bins[i], histoGroup.yj1_bins[i+1]);
                            if (!fill) continue;
                            string prof_name =  trackVar+pre+"prof_aktRVaried_M1OverpT_pTj1bin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
                            histoGroup.prof_map[prof_name]->Fill(jetR, M1OverpT, use_weight);
                        }
                    }
                }

                // Fill TProfiles varying jet radius of cajets, in slices of leading jet pT and y
                for (auto jetR : histoGroup.ca_jetR){
                    string id = get_jetR_id("_ca",jetR);
                    double M1OverpT = m_jets[trackVar+pre+"jets"+id][0].m() / m_jets[trackVar+pre+"jets"+id][0].pt();
                    histoGroup.prof_map[trackVar+pre+"prof_caRVaried_M1OverpT_All"]->Fill(jetR,M1OverpT, use_weight);

                    bool fill = false;
                    for (long unsigned int k = 0; k < histoGroup.pTj1_bins.size()-1; k++){
                        fill = inBin(evt.floats.at("truth_pT_trackj1"), histoGroup.pTj1_bins[k], histoGroup.pTj1_bins[k+1]); //  alwayse make phase space cuts based of truth info!
                        for (long unsigned int j = 0; j < histoGroup.yj1_bins.size()-1; j++){
                            fill = fill && inBin(evt.floats.at("truth_y_trackj1"), histoGroup.yj1_bins[k], histoGroup.yj1_bins[k+1]);
                            if (!fill) continue;
                            string prof_name =  trackVar+pre+"prof_caRVaried_M1OverpT_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
                            histoGroup.prof_map[prof_name]->Fill(jetR, M1OverpT, use_weight);
                        }
                    }
                }
                if (m_jets[trackVar+pre+"jets_r04"].size() > 0){
                  for (long unsigned int a = 0; a < histoGroup.jetshape_edges.size(); a++){
                    float jetR = histoGroup.jetshape_edges[a];
                    vector<float> annulusRange = {0,jetR}; 
                    if (a==0) annulusRange[0] = 0;
                    else      annulusRange[0] = histoGroup.jetshape_edges[a-1];
                    PseudoJet inRadius(0,0,0,0);
                    PseudoJet inAnnulus(0,0,0,0);
                    for (auto constit:m_jets[trackVar+pre+"jets_r04"][0].constituents()){
                      if (constit.delta_R(m_jets[trackVar+pre+"jets_r04"][0]) < jetR){
                        inRadius = inRadius + constit;
                      }
                      if (constit.delta_R(m_jets[trackVar+pre+"jets_r04"][0]) > annulusRange[0] && constit.delta_R(m_jets[trackVar+pre+"jets_r04"][0]) <= annulusRange[1] ){
                        inAnnulus = inAnnulus + constit;
                      }
                    }
      
                    double jet_pT       = m_jets[trackVar+pre+"jets_r04"][0].pt();
                    double jet_Area     = 3.14159265 *(jetR*jetR);
                    double Annulus_Area = 3.14159265 *(annulusRange[1]*annulusRange[1] - annulusRange[0]*annulusRange[0]);
                    double pT_frac_inRadius           = inRadius.pt() / jet_pT;
                    double pT_frac_inRadius_areaNorm  = inRadius.pt() / (jet_pT * jet_Area);
                    double pT_frac_inAnnulus          = inAnnulus.pt() / jet_pT;
                    double pT_frac_inAnnulus_areaNorm = inAnnulus.pt() / (jet_pT * Annulus_Area);
                    for (long unsigned int j = 0; j < histoGroup.yj1_bins.size()-1; j++){
                      if (!inBin(evt.floats.at("truth_y_trackj1"),   histoGroup.yj1_bins[j], histoGroup.yj1_bins[j+1])) continue;

                      string prof_name;
                      prof_name           =  trackVar+pre+"prof_antikt04_pT_frac_in_jetRbin"+to_string(a+1)+"_yj1bin"+to_string(j+1);
                      histoGroup.prof_map[prof_name]->Fill(jet_pT, pT_frac_inRadius, use_weight);

                      prof_name           =  trackVar+pre+"prof_antikt04_pT_density_in_jetRbin"+to_string(a+1)+"_yj1bin"+to_string(j+1);
                      histoGroup.prof_map[prof_name]->Fill(jet_pT, pT_frac_inRadius_areaNorm, use_weight);

                      prof_name           =  trackVar+pre+"prof_antikt04_pT_frac_in_AnnulusRbin"+to_string(a+1)+"_yj1bin"+to_string(j+1);
                      histoGroup.prof_map[prof_name] ->Fill(jet_pT, pT_frac_inAnnulus, use_weight);

                      prof_name           =  trackVar+pre+"prof_antikt04_pT_density_in_AnnulusRbin"+to_string(a+1)+"_yj1bin"+to_string(j+1);
                      histoGroup.prof_map[prof_name]->Fill(jet_pT, pT_frac_inAnnulus_areaNorm, use_weight);

                      for (long unsigned int k = 0; k < histoGroup.pTj1_bins.size()-1; k++){
                        if (!inBin(evt.floats.at("truth_pT_trackj1"),   histoGroup.pTj1_bins[k], histoGroup.pTj1_bins[k+1])) continue;
                        prof_name           =  trackVar+pre+"prof_antikt04_frac_per_R_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
                        histoGroup.prof_map[prof_name]->Fill(jetR-0.01, pT_frac_inRadius, use_weight);

                        prof_name           =  trackVar+pre+"prof_antikt04_density_per_R_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
                        histoGroup.prof_map[prof_name]->Fill(jetR-0.01, pT_frac_inRadius_areaNorm, use_weight);

                        prof_name           =  trackVar+pre+"prof_antikt04_frac_per_Annulus_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
                        histoGroup.prof_map[prof_name] ->Fill(jetR-0.01, pT_frac_inAnnulus, use_weight);

                        prof_name           =  trackVar+pre+"prof_antikt04_density_per_Annulus_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
                        histoGroup.prof_map[prof_name]->Fill(jetR-0.01, pT_frac_inAnnulus_areaNorm, use_weight);    
                      }           
                    }   
                  }
                }
              }
            }
          }
        }
      } // End of parallel region
   } // End of OpenMP parallel block
   
   // Merge thread-local histograms into main histogram groups
   std::cout << " === merging thread-local histograms === " << std::endl;
   if (!do_IBU){
    for (int t = 0; t < num_threads; ++t) {
        for (unsigned int i = 0; i < centralHistoGroups.size(); ++i) {
          //  centralHistoGroups[i].MergeHistos(thread_central_histos[t][i]);
          centralHistoGroups[i].MergeGroup(thread_central_histos[t][i]);
        }
        for (unsigned int i = 0; i < ensHistoGroups.size(); ++i) {
          //  ensHistoGroups[i].MergeHistos(thread_ens_histos[t][i]);
          ensHistoGroups[i].MergeGroup(thread_ens_histos[t][i]);
        }
    }
   }
   
   // Complete the progress bar
   bar.set_option(option::PostfixText{"Processing Events: " + to_string(event_data.size()) + "/" + to_string(event_data.size()) + " - Complete!"});
   bar.set_progress(static_cast<size_t>(event_data.size()));
   std::cout << std::endl;

   std::cout << " === create output ROOT file === " << std::endl;
   TFile foutput(saveName, "recreate"); 

   if (do_IBU) WriteIBUTree(event_data, foutput);
   else{
      std::cout << " === write in file === " << std::endl;
      for (auto& histoGroup : centralHistoGroups) {
        // histoGroup.WriteHistos(foutput);
        histoGroup.WriteGroup(foutput);
      }

      for (auto& histoGroup : ensHistoGroups) {
        // histoGroup.WriteHistos(foutput);
        histoGroup.WriteGroup(foutput);
      }
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

bool MakeOmni::inBin(Float_t val, double low, double high) {
    return val >= low && val < high;
}

bool MakeOmni::inBin(Int_t val, int low, int high) {
    return val >= low && val < high;
}

// to keep compatability with kevin's plotting macro.
// _r for antikT, _kt for kT, _ca for Cambridge aachen
string MakeOmni::get_jetR_id(string prefix, Double_t jetR){
    string id = prefix + std::to_string(jetR);
    id.erase(remove(id.begin(), id.end(), '.'), id.end());
    while (!id.empty() && id.back() == '0') id.pop_back();
    if (fmod(jetR,1.)==0) id = id+"0";
    return id;
}
