#include <iostream>
#include <iomanip>
#include <vector>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TMath.h>
#include "HistoGroup.h"
#include <TObjString.h>
#include <TObjArray.h>
using namespace std;

HistoGroup::HistoGroup(string name, int kinematic_region, bool is_truth, bool do_IBU, bool is_data) : name(name), kinematic_region(kinematic_region), is_truth(is_truth), do_IBU(do_IBU), is_data(is_data) {

    TString configFile = "/afs/cern.ch/user/m/mbsmith/zjets_omnifold/fastjet/config.data";
    settings = HistoGroup::openSettingsFile(configFile);
    
    pTj1_bins      = numberize(getStr(settings, "pTj1.bins"));
    yj1_bins       = numberize(getStr(settings, "yj1.bins"));
    antikt_jetR    = numberize(getStr(settings, "jetR.antikt"));
    ca_jetR        = numberize(getStr(settings, "jetR.ca"));
    jetshape_edges = numberize(getStr(settings, "jetR.jetshape_edges"));


    antikt_jetR_bins.push_back(antikt_jetR.front() - 0.5 * (antikt_jetR[1] - antikt_jetR[0]));
    for (size_t i = 1; i < antikt_jetR.size(); ++i) antikt_jetR_bins.push_back(0.5 * (antikt_jetR[i - 1] + antikt_jetR[i]));

    ca_jetR_bins.push_back(ca_jetR.front() - 0.5 * (ca_jetR[1] - ca_jetR[0]));
    for (size_t i = 1; i < ca_jetR.size(); ++i) ca_jetR_bins.push_back(0.5 * (ca_jetR[i - 1] + ca_jetR[i]));

    // Define log spaced bins for EEC plots


    const Long64_t nbins = settings->GetValue("EEC_Binning.nbins",0);
    Double_t logxmin_b2b = settings->GetValue("EEC_Binning.b2b.logxmin",0.0f);
    Double_t logxmin_all = settings->GetValue("EEC_Binning.all.logxmin",0.0f);
    Double_t logxmax_all = settings->GetValue("EEC_Binning.all.logxmax",0.0f);
    Double_t logxmax_ak4 = settings->GetValue("EEC_Binning.ak4.logxmax",0.0f);
    Double_t logxmax_ak6 = settings->GetValue("EEC_Binning.ak6.logxmax",0.0f);
    Double_t logxmax_ak10 = settings->GetValue("EEC_Binning.ak10.logxmax",0.0f);
    std::vector<Double_t> ak4_edges(nbins + 1);
    std::vector<Double_t> ak6_edges(nbins + 1);
    std::vector<Double_t> ak10_edges(nbins + 1);
    std::vector<Double_t> b2b_edges(nbins + 1);
    std::vector<Double_t> all_edges(nbins + 1);

    for (Long64_t i = 0; i <= nbins; ++i) {
      ak4_edges[i]  = pow(10, logxmin_all + i*(logxmax_ak4-logxmin_all)/nbins);
      ak6_edges[i]  = pow(10, logxmin_all + i*(logxmax_ak6-logxmin_all)/nbins);
      ak10_edges[i] = pow(10, logxmin_all + i*(logxmax_ak10-logxmin_all)/nbins);
      b2b_edges[i]  = pow(10, logxmin_b2b + i*(logxmax_all-logxmin_b2b)/nbins);
      all_edges[i]  = pow(10, logxmin_all + i*(logxmax_all-logxmin_all)/nbins);
    }


    const Long64_t nbins_symlog_coll = settings->GetValue("symlog.col.nbins",0);
    Double_t logxmin_symlog_coll     = settings->GetValue("symlog.col.logxmin",0.0f);
    Double_t symlog_center           = settings->GetValue("symlog.center",0.0f);
    const Long64_t nbins_symlog_b2b  = settings->GetValue("symlog.b2b.nbins",0);
    Double_t logxmin_symlog_b2b      = settings->GetValue("symlog.b2b.logxmin",0.0f);
    const Long64_t nbins_symlog      = nbins_symlog_coll + nbins_symlog_b2b;

    std::vector<Double_t> symlog_edges(nbins_symlog+1);
    for (Long64_t i = 0; i <= nbins_symlog_coll; ++i) {
       symlog_edges[i] = pow(10, logxmin_symlog_coll + i*(symlog_center-logxmin_symlog_coll)/nbins_symlog_coll);
    }
    for (Long64_t i = 0; i <= nbins_symlog_b2b; ++i) {
       symlog_edges[nbins_symlog - i] = 1 - pow(10, logxmin_symlog_b2b + i*(symlog_center-logxmin_symlog_b2b)/nbins_symlog_b2b);
    }


    // Initialize histograms
    std::vector<Double_t> m_edges  = numberize(getStr(settings, "m.bins"));
    std::vector<Double_t> m1_edges = numberize(getStr(settings, "m1.bins"));
    std::vector<Double_t> m2_edges = numberize(getStr(settings, "m2.bins"));
    std::vector<Double_t> pT_edges = numberize(getStr(settings, "pT.bins"));
    std::vector<Double_t> dphijj_edges = numberize(getStr(settings, "dphijj.bins"));

    // Define arrays for different kinematic regions
    std::vector<Double_t> mjj_edges_default  = numberize(getStr(settings, "mjj.bins"));
    std::vector<Double_t> dyjj_edges_default = numberize(getStr(settings, "dyjj.bins"));
    std::vector<Double_t> dRjj_edges_default = numberize(getStr(settings, "dRjj.bins"));
    
    std::vector<Double_t> mjj_edges_region2  = numberize(getStr(settings, "region2.mjj.bins"));
    std::vector<Double_t> dyjj_edges_region2 = numberize(getStr(settings, "region2.dyjj.bins"));
    std::vector<Double_t> dRjj_edges_region2 = numberize(getStr(settings, "region2.dRjj.bins"));
    
    // Choose which arrays to use based on kinematic region
    std::vector<Double_t> mjj_edges;
    std::vector<Double_t> dyjj_edges;
    std::vector<Double_t> dRjj_edges;
    
    if (kinematic_region == 2) {
      mjj_edges  = mjj_edges_region2;
      dyjj_edges = dyjj_edges_region2;
      dRjj_edges = dRjj_edges_region2;
    } else {
      mjj_edges = mjj_edges_default;
      dyjj_edges = dyjj_edges_default;
      dRjj_edges = dRjj_edges_default;
    }

    vector<string> levels;
    if (do_IBU && !is_data)        levels = {"truth_", ""};
    else if (is_truth && !is_data) levels = {"truth_"};
    else                           levels = {""};

    for(string pre:levels){
      h_map[pre+"hm1_R04"]   = make_shared<TH1D>(TH1D("", "", m1_edges.size()-1,   m1_edges.data()));
      h_map[pre+"hm2_R04"]   = make_shared<TH1D>(TH1D("", "", m2_edges.size()-1,   m2_edges.data()));
      h_map[pre+"hm3_R04"]   = make_shared<TH1D>(TH1D("", "", m_edges.size()-1,    m_edges.data()));
      h_map[pre+"hm4_R04"]   = make_shared<TH1D>(TH1D("", "", m_edges.size()-1,    m_edges.data()));
      h_map[pre+"hmjj_R04"]  = make_shared<TH1D>(TH1D("", "", mjj_edges.size()-1,  mjj_edges.data()));
      h_map[pre+"hdyjj_R04"] = make_shared<TH1D>(TH1D("", "", dyjj_edges.size()-1, dyjj_edges.data()));
      h_map[pre+"hEEC_R04"]  = make_shared<TH1D>(TH1D("", "", ak4_edges.size()-1,  ak4_edges.data()));
      // hLund_z_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
      // hLund_dR_R04 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
      // hLund_plane_R04 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 11, 0.5, 6));

      h_map[pre+"hm1_R06"] = make_shared<TH1D>(TH1D("", "",  m1_edges.size()-1,  m1_edges.data()));
      h_map[pre+"hpT_R06"] = make_shared<TH1D>(TH1D("", "",  pT_edges.size()-1,  pT_edges.data()));
      h_map[pre+"hEEC_R06"] = make_shared<TH1D>(TH1D("", "", ak6_edges.size()-1, ak6_edges.data()));
      // hLund_z_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
      // hLund_dR_R06 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
      // hLund_plane_R06 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

      h_map[pre+"hm1_R10"] = make_shared<TH1D>(TH1D("", "",  m1_edges.size()-1,   m1_edges.data()));
      h_map[pre+"hpT_R10"] = make_shared<TH1D>(TH1D("", "",  pT_edges.size()-1,   pT_edges.data()));
      h_map[pre+"hEEC_R10"] = make_shared<TH1D>(TH1D("", "", ak10_edges.size()-1, ak10_edges.data()));
      // hLund_z_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 10));
      // hLund_dR_R10 = make_shared<TH1D>(TH1D("", "", 10, 0, 5));
      // hLund_plane_R10 = make_shared<TH2D>(TH2D("", "", 10, 0, 5, 12, 0, 6));

      h_map[pre+"hm1_CA04"] = make_shared<TH1D>(TH1D("", "",     m1_edges.size()-1,     m1_edges.data()));
      h_map[pre+"hpT_CA04"] = make_shared<TH1D>(TH1D("", "",     pT_edges.size()-1,     pT_edges.data()));
      h_map[pre+"hEEC_CA04"] = make_shared<TH1D>(TH1D("", "",    ak4_edges.size()-1,    ak4_edges.data()));
      h_map[pre+"hmjj_CA04"] = make_shared<TH1D>(TH1D("", "",    mjj_edges.size()-1,    mjj_edges.data()));
      h_map[pre+"hdRjj_CA04"] = make_shared<TH1D>(TH1D("", "",   dRjj_edges.size()-1,   dRjj_edges.data()));
      h_map[pre+"hdyjj_CA04"] = make_shared<TH1D>(TH1D("", "",   dyjj_edges.size()-1,   dyjj_edges.data()));
      h_map[pre+"hdphijj_CA04"] = make_shared<TH1D>(TH1D("", "", dphijj_edges.size()-1, dphijj_edges.data()));

      h_map[pre+"hm1_CA06"] = make_shared<TH1D>(TH1D("", "", m1_edges.size()-1, m1_edges.data()));
      h_map[pre+"hpT_CA06"] = make_shared<TH1D>(TH1D("", "", pT_edges.size()-1, pT_edges.data()));

      h_map[pre+"hm1_KT04"] = make_shared<TH1D>(TH1D("", "", m1_edges.size()-1, m1_edges.data()));
      h_map[pre+"hpT_KT04"] = make_shared<TH1D>(TH1D("", "", pT_edges.size()-1, pT_edges.data()));

      h_map[pre+"hTEEC_collinear"]    = make_shared<TH1D>(TH1D("", "", all_edges.size()-1,    all_edges.data()));
      h_map[pre+"hTEEC_full_nolog"]   = make_shared<TH1D>(TH1D("", "", 20, 0.0, 1.0));
      h_map[pre+"hTEEC_b2b"]          = make_shared<TH1D>(TH1D("", "", b2b_edges.size()-1,    b2b_edges.data()));
      h_map[pre+"hTEEC_full"]         = make_shared<TH1D>(TH1D("", "", symlog_edges.size()-1, symlog_edges.data()));
      h_map[pre+"hTEEC_z_collinear"]  = make_shared<TH1D>(TH1D("", "", b2b_edges.size()-1,    b2b_edges.data()));
      h_map[pre+"hTEEC_z_full_nolog"] = make_shared<TH1D>(TH1D("", "", 20, 0.0, 1.0));
      h_map[pre+"hTEEC_z_b2b"]        = make_shared<TH1D>(TH1D("", "", all_edges.size()-1,    all_edges.data()));
      h_map[pre+"hTEEC_z_full"]       = make_shared<TH1D>(TH1D("", "", symlog_edges.size()-1, symlog_edges.data()));

      prof_map[pre+"prof_aktRVaried_M1OverpT_All"] = std::make_shared<TProfile>( TProfile("", "", antikt_jetR_bins.size() - 1, antikt_jetR_bins.data()));
      prof_map[pre+"prof_caRVaried_M1OverpT_All"]  = std::make_shared<TProfile>( TProfile("", "", ca_jetR_bins.size() - 1, ca_jetR_bins.data()));
      for (long unsigned int i = 0; i < pTj1_bins.size()-1; i++){
        for (long unsigned int j = 0; j < yj1_bins.size()-1; j++){
          string prof_name;
          prof_name           = pre+"prof_aktRVaried_M1OverpT_pTj1bin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", antikt_jetR_bins.size() - 1, antikt_jetR_bins.data()));

          prof_name           = pre+"prof_caRVaried_M1OverpT_pTj1bin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", ca_jetR_bins.size() - 1, ca_jetR_bins.data()));
        }
      }

      prof_map[pre+"prof_aktRVaried_M1OverpT_All"] = std::make_shared<TProfile>( TProfile("", "", antikt_jetR_bins.size() - 1, antikt_jetR_bins.data()));
      prof_map[pre+"prof_caRVaried_M1OverpT_All"]  = std::make_shared<TProfile>( TProfile("", "", ca_jetR_bins.size() - 1, ca_jetR_bins.data()));


      for (long unsigned int j = 0; j < yj1_bins.size()-1; j++){
        string prof_name;
        for (long unsigned int i = 0; i < jetshape_edges.size(); i++){
          prof_name           =  pre+"prof_antikt04_pT_frac_in_jetRbin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", pTj1_bins.size() - 1, pTj1_bins.data()));

          prof_name           =  pre+"prof_antikt04_pT_density_in_jetRbin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", pTj1_bins.size() - 1, pTj1_bins.data()));

          prof_name           =  pre+"prof_antikt04_pT_frac_in_AnnulusRbin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", pTj1_bins.size() - 1, pTj1_bins.data()));

          prof_name           =  pre+"prof_antikt04_pT_density_in_AnnulusRbin"+to_string(i+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", pTj1_bins.size() - 1, pTj1_bins.data()));
        }

        for (long unsigned int k = 0; k < pTj1_bins.size()-1; k++){
          vector<Double_t> corr_binEdges = {0};
          for (auto bin:jetshape_edges) corr_binEdges.push_back(bin); 
          prof_name           =  pre+"prof_antikt04_frac_per_R_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", corr_binEdges.size() - 1, corr_binEdges.data()));
          prof_name           =  pre+"prof_antikt04_density_per_R_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", corr_binEdges.size() - 1, corr_binEdges.data()));
          prof_name           =  pre+"prof_antikt04_frac_per_Annulus_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", corr_binEdges.size() - 1, corr_binEdges.data()));
          prof_name           =  pre+"prof_antikt04_density_per_Annulus_pTj1bin"+to_string(k+1)+"_yj1bin"+to_string(j+1);
          prof_map[prof_name] = std::make_shared<TProfile>( TProfile("", "", corr_binEdges.size() - 1, corr_binEdges.data()));
        }
      }
    }


    // Branches needed for IBU Undolfing. 
    //  syst_pTScale_variable
    //  syst_Fake
    //  syst_JetTrackFilter
    //  syst_TrackFilter
    //  EntryNumber, then i can append this info to the nominal files, and run directly through the 

}

HistoGroup::~HistoGroup() {
    // Destructor
    // Histograms will be automatically deleted when the unique_ptr goes out of scope
}

void HistoGroup::WriteHistoMap(TFile& foutput) {
    // Write histograms to the output file
    foutput.cd();
    string key; std::shared_ptr<TH1D> h;
    for (auto elmnt:h_map){
        key = elmnt.first;
        h = elmnt.second;
        h.get()->SetName((name + key).c_str());
        h.get()->Write((name + key).c_str(),TObject::kOverwrite);
    }
}

void HistoGroup::WriteProfMap(TFile& foutput) {
    foutput.cd();
    string key; std::shared_ptr<TProfile> p;
    for (auto elmnt:prof_map){
        key = elmnt.first;
        p = elmnt.second;
        p->SetName((name + key).c_str());
        p->Write((name + key).c_str(),TObject::kOverwrite);
    }
}

void HistoGroup::WriteGroup(TFile& foutput) {
    // Merge all histograms from other into this group
    WriteHistoMap(foutput);
    WriteProfMap(foutput);
}

void HistoGroup::MergeHistoMaps(const HistoGroup& other) {
    // Merge all histograms from other into this group
    for (auto& [key, h] : h_map){
        h->Add(other.h_map.at(key).get());
    }
}

void HistoGroup::MergeProfMaps(const HistoGroup& other) {
    // Merge all histograms from other into this group
    for (auto& [key, p] : prof_map){
        p->Add(other.prof_map.at(key).get());
    }
}

void HistoGroup::MergeGroup(const HistoGroup& other) {
    // Merge all histograms from other into this group
    MergeHistoMaps(other);
    MergeProfMaps(other);
}

TEnv* HistoGroup::openSettingsFile(TString fileName) {
  if (fileName == "")
    cout << "No config file name specified. Cannot open file!" << endl;
  TEnv *settings = new TEnv();
  int status = settings->ReadFile(fileName.Data(), EEnvLevel(0));
  if (status != 0)
    cout <<Form("Cannot read file %s", fileName.Data())  << endl;
  return settings;
}

vector<TString> HistoGroup::vectorize(TString str, TString sep) {
  vector<TString> result;
  TObjArray *strings = str.Tokenize(sep.Data());
  if (strings->GetEntries() == 0)
    return result;
  TIter istr(strings);
  while (TObjString* os = (TObjString*) istr()){
    if (os->GetString()[0] != '#')
      result.push_back(os->GetString());
    else
      break;
  }
  return result;
}

vector<double> HistoGroup::numberize(TString str, TString sep) {
  vector<double>  result; 
  vector<TString> words = HistoGroup::vectorize(str,sep);
  for (auto s:words) result.push_back(s.Atof());
  return result;
}

TString HistoGroup::getStr(TEnv *settings, TString key) {
  TString val = settings->GetValue(key,"");
  if (val=="") cout << "No value found for TEnv key: "+key << endl;
  return val;
}