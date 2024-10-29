#include "DoPlots.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TLine.h"
#include <iostream>
using namespace std;

#ifndef DoPlots_cxx
#define DoPlots_cxx

void FinalPlots(TFile* outFileOmni, TFile* outFileTruth, TString variable){
  
  
  TCanvas* c = new TCanvas("", "", 800, 800);

  TPad* pad1 = new TPad("pad1", "", 0, 0.3, 1, 1);
  TPad* pad2 = new TPad("pad2", "", 0, 0, 1, 0.3);

  pad1->SetBottomMargin(0.); 
  pad1->SetTopMargin(0.05);   
  pad2->SetTopMargin(0.);   
  pad2->SetBottomMargin(0.3); 

  c->cd();
  pad1->Draw();
  pad2->Draw();

  pad1->cd();
  gPad->SetLogy();
  if (variable.Contains("EEC")) {
    gPad->SetLogx();
  }
  TH1D* hReco = (TH1D*)outFileOmni->Get(variable);
  TH1D* hOmni = (TH1D*)outFileOmni->Get(variable+"_omni");
  TH1D* hTrue      = (TH1D*)outFileTruth->Get(variable);
  double maxVal = max({hReco->GetMaximum(), hOmni->GetMaximum(), hTrue->GetMaximum()});
  if (variable.Contains("EEC")) {
    maxVal *= 10;
  }
  double minVal = min({hReco->GetMinimum(), hOmni->GetMinimum(), hTrue->GetMinimum()});
  hTrue->GetYaxis()->SetRangeUser(minVal, maxVal);
  hReco->GetYaxis()->SetRangeUser(minVal, maxVal);
  hOmni->GetYaxis()->SetRangeUser(minVal, maxVal);
  hReco->SetMarkerStyle(20);
  hOmni->SetMarkerStyle(20);
  hTrue->SetMarkerStyle(20);
  hReco->SetLineColor(kBlue);
  hOmni->SetLineColor(kRed);
  hTrue->SetLineColor(kBlack);
  hReco->SetMarkerColor(kBlue);
  hOmni->SetMarkerColor(kRed);
  hTrue->SetMarkerColor(kBlack);
  
  TLegend* l;
  if (variable.Contains("EEC" || "Lund")) {
    l = new TLegend(0.3, 0.3, 0.5, 0.5);
  } else {
    l = new TLegend(0.7, 0.7, 0.9, 0.9);
  }
  l->AddEntry(hReco, "reco", "l");
  l->AddEntry(hOmni, "omnifold", "l");
  l->AddEntry(hTrue, "truth", "l");
  hTrue->Draw("HIST");
  hReco->Draw("HIST same");
  hOmni->Draw("HIST same");
  l->Draw("same");
  pad1->Update();

  // Create the ratio histogram
  pad2->cd();
  if (variable.Contains("EEC")) {
    gPad->SetLogx();
  }
  TH1D* ratioReco = (TH1D*)hReco->Clone("ratio1");
  ratioReco->Divide(hTrue);
  TH1D* ratioOmni = (TH1D*)hOmni->Clone("ratio2");
  ratioOmni->Divide(hTrue);

  ratioReco->SetLineColor(kBlue);
  ratioOmni->SetLineColor(kRed);
  ratioReco->SetTitle("");
  ratioReco->GetYaxis()->SetTitle("Omni / Truth");
  ratioReco->GetYaxis()->SetNdivisions(505);
  ratioOmni->GetYaxis()->SetNdivisions(505);
  ratioReco->GetYaxis()->SetTitleSize(0.1);
  ratioReco->GetYaxis()->SetTitleOffset(0.5);
  ratioReco->GetYaxis()->SetLabelSize(0.1);
  ratioReco->GetXaxis()->SetTitleSize(0.1);
  ratioReco->GetXaxis()->SetLabelSize(0.1);
  ratioReco->GetYaxis()->SetRangeUser(0.75, 1.25);
  ratioOmni->GetYaxis()->SetRangeUser(0.75, 1.25);
  if (variable.Contains("TEEC")) {
    ratioReco->GetXaxis()->SetTitle("Tau");
  } else if (variable.Contains("EEC")) {
    ratioReco->GetXaxis()->SetTitle("z");
  } else {
    ratioReco->GetXaxis()->SetTitle(variable);
  }
  ratioReco->Draw("HIST");
  ratioOmni->Draw("HIST same");
  double xmin = ratioReco->GetXaxis()->GetXmin();
  double xmax = ratioReco->GetXaxis()->GetXmax();
  TLine* line = new TLine(xmin, 1, xmax, 1);
  line->SetLineStyle(4);
  line->SetLineColor(kBlack);
  line->Draw("same");
  c->Update();
  c->SaveAs("out/plots/"+variable+".png");
  c->SaveAs("out/plots/"+variable+".pdf");

}


#endif
