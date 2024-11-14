#include "DoPlots.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"
#include <iostream>
using namespace std;

#ifndef DoPlots_cxx
#define DoPlots_cxx

void FinalPlots1D(TFile* outFileOmni, TFile* outFileTruth, TString variable){
  
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
  ratioReco->GetYaxis()->SetTitle("Ratio to truth");
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


void FinalPlots2D(TFile* outFileOmni, TFile* outFileTruth, TString variable){
  
  TCanvas* cOmni = new TCanvas("", "", 800, 800);
  TCanvas* cTrue = new TCanvas("", "", 800, 800);
  TCanvas* cStartRes = new TCanvas("", "", 800, 800);
  TCanvas* cEndRes = new TCanvas("", "", 800, 800);

  TPad* pOmni = new TPad("pad1", "", 0, 0, 1, 1);
  TPad* pTrue = new TPad("pad1", "", 0, 0, 1, 1);
  TPad* pStartRes = new TPad("pad1", "", 0, 0, 1, 1);
  TPad* pEndRes = new TPad("pad1", "", 0, 0, 1, 1);

  pOmni->SetBottomMargin(0.15);
  pOmni->SetTopMargin(0.1);
  pOmni->SetRightMargin(0.2);
  pTrue->SetBottomMargin(0.15);
  pTrue->SetTopMargin(0.1);
  pTrue->SetRightMargin(0.2);
  pStartRes->SetBottomMargin(0.15);
  pStartRes->SetTopMargin(0.1);
  pStartRes->SetRightMargin(0.2);
  pEndRes->SetBottomMargin(0.15);
  pEndRes->SetTopMargin(0.1);
  pEndRes->SetRightMargin(0.2);

  TH2D* hReco = (TH2D*)outFileOmni->Get(variable);
  TH2D* hOmni = (TH2D*)outFileOmni->Get(variable+"_omni");
  TH2D* hTrue = (TH2D*)outFileTruth->Get(variable);

  // Draw Omni
  cOmni->cd();
  pOmni->Draw();
  pOmni->cd();

  hOmni->SetTitle("Omnifold;log (R / #Delta R);log (1 / z)");
  hOmni->Draw("COLZ");
  TLatex* lOmni = new TLatex();
  lOmni->SetNDC();
  lOmni->SetTextSize(0.05);
  lOmni->DrawLatex(0.2, 0.92, hOmni->GetTitle());
  cOmni->Update();

  // Draw True
  cTrue->cd();
  pTrue->Draw();
  pTrue->cd();

  hTrue->SetTitle("Truth;log (R / #Delta R);log (1 / z)");
  hTrue->Draw("COLZ");
  TLatex* lTrue = new TLatex();
  lTrue->SetNDC();
  lTrue->SetTextSize(0.05);
  lTrue->DrawLatex(0.2, 0.92, hTrue->GetTitle());
  cTrue->Update();

  // Draw start residuals
  cStartRes->cd();
  pStartRes->Draw();
  pStartRes->cd();

  TH2D* hStartRes = (TH2D*)hReco->Clone("hStartRes");
  hStartRes->Add(hTrue, -1);
  hStartRes->SetTitle("Reco - Truth;log (R / #Delta R);log (1 / z)");
  hStartRes->Draw("COLZ");
  TLatex* lStartRes = new TLatex();
  lStartRes->SetNDC();
  lStartRes->SetTextSize(0.05);
  lStartRes->DrawLatex(0.2, 0.92, hStartRes->GetTitle());
  cStartRes->Update();

  // Get start residual min and max
  double minStart = hStartRes->GetMinimum();
  double maxStart = hStartRes->GetMaximum();

  // Draw end residuals
  cEndRes->cd();
  pEndRes->Draw();
  pEndRes->cd();

  TH2D* hEndRes = (TH2D*)hOmni->Clone("hEndRes");
  hEndRes->Add(hTrue, -1);
  hEndRes->SetMinimum(minStart);
  hEndRes->SetMaximum(maxStart);
  hEndRes->SetTitle("Omni - Truth;log (R / #Delta R);log (1 / z)");
  hEndRes->Draw("COLZ");
  TLatex* lEndRes = new TLatex();
  lEndRes->SetNDC();
  lEndRes->SetTextSize(0.05);
  lEndRes->DrawLatex(0.2, 0.92, hEndRes->GetTitle());
  cEndRes->Update();
  
  // Save plots
  cOmni->SaveAs("out/plots/"+variable+"_omni.png");
  cOmni->SaveAs("out/plots/"+variable+"_omni.pdf");
  cTrue->SaveAs("out/plots/"+variable+"_true.png");
  cTrue->SaveAs("out/plots/"+variable+"_true.pdf");
  cStartRes->SaveAs("out/plots/"+variable+"_startRes.png");
  cStartRes->SaveAs("out/plots/"+variable+"_startRes.pdf");
  cEndRes->SaveAs("out/plots/"+variable+"_endRes.png");
  cEndRes->SaveAs("out/plots/"+variable+"_endRes.pdf");

}


#endif
