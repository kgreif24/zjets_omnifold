#include "TH1.h"
#include "TH2.h"


using namespace std;
#ifndef analysisHelpers_h
#define analysisHelpers_h

void normalizeHisto(unique_ptr<TH1D>& hist);
void normalizeHisto2D(unique_ptr<TH2D>& hist);
void SetEECAxisRange(unique_ptr<TH1D>& hist, TString xtitle, TString ytitle);
void YAxisRangeUserName(unique_ptr<TH1D>& hist);
void XaxisName(unique_ptr<TH1D>& hist, TString xLabel);
TH1D* MakeRatioPlot(unique_ptr<TH1D>& hnum, unique_ptr<TH1D>& hden, TString yLabel);
#endif
