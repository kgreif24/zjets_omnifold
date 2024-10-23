#include "analysisHelpers.h"
#include "TH2.h"

using namespace std;

#ifndef analysisHelpers_cxx
#define analysisHelpers_cxx

void normalizeHisto(unique_ptr<TH1D>& hist){
  hist->Scale(1./hist->Integral());
}

void normalizeHisto2D(unique_ptr<TH2D>& hist){
  hist->Scale(1./hist->Integral());
}
void XaxisName(unique_ptr<TH1D>& hist, TString xLabel){
  hist->GetXaxis()->SetTitle(xLabel);
}

void YAxisRangeUserName(unique_ptr<TH1D>& hist){
  hist->GetYaxis()->SetRangeUser(1e-6, 1.2);
  hist->GetYaxis()->SetTitle("Normalised events");
}

void SetEECAxisRange(unique_ptr<TH1D>& hist, TString xtitle, TString ytitle){
  hist->GetXaxis()->SetRangeUser(0, 1);
  hist->GetXaxis()->SetTitle(xtitle);
  hist->GetYaxis()->SetRangeUser(1e-5, 0.1);
  hist->GetYaxis()->SetTitle(ytitle);
}

TH1D* MakeRatioPlot(unique_ptr<TH1D>& hnum, unique_ptr<TH1D>& hden, TString yLabel){
  TH1D*hRatio = (TH1D*)hnum->Clone();
  for(int i=1; i<hRatio->GetNbinsX()+1; i++){
    if (hden->GetBinContent(i) != 0 ){
      hRatio->SetBinContent(i, hnum->GetBinContent(i) / hden->GetBinContent(i));
    } else {
      hRatio->SetBinContent(i, 0.00000000000001);
    }
  }
  hRatio->GetYaxis()->SetTitle(yLabel);
  hRatio->GetYaxis()->SetRangeUser(0.8, 1.2);
  return hRatio;
}
#endif
