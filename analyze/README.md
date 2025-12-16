
# Draft public facing python package

This is a draft of the public facing python codebase that will accompany the un-binned spectra.
Under active development!
For now implements a correlation dimension measurement in `calc_correlation_dimension.ipynb`.

## Input data files

All of the data files to use as input are in the following location: `/pscratch/sd/k/kgreif/zjets_plot_staging/`:

- `Pseudodata_SherpaDY_PowhegPythiaTop_June2025_shuffled.root`
- `TruthPseudodata_Sherpa2211DY_Dibo_EW_PowhegPythiaTop_PosWeights_WithTracks_June2025_shuffled.root`
- `ZjetOmnifold_30Jul2025_Sherpa2211Truth_shuffled.root`
- `ZjetOmnifold_5Jul2025_MGPy8FxFxPlusNonStrong_syst_Test_withdd.root`
- `ZjetOmnifold_Mar10_Sherpa2211_LookLike_MgFxFx_Test_V5.root`
- `ZjetOmnifold_May19_MGPy8FxFx_All_shuffled.root`
- `ZjetOmnifold_Nov11_data_WithTracks_slim_Systematics_shuffled.root`

## Input weights

The current generation weights for the Omnifold pseudodata and data measurements are at: `/global/cfs/cdirs/m3246/ZjetOmnifold/weights/zjets-v4/`:

- `data-weights.npz`
- `pd-weights.npz`