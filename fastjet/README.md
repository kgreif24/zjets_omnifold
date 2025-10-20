# Z+Jets Histogram Plotting Code

A high-performance, parallelized C++ application for generating histograms from Z+jets events using ROOT, FastJet, and OpenMP. This code processes particle physics data to create various jet and event-level observables with support for systematic uncertainties and ensemble variations.

## Features

- **Parallel Processing**: OpenMP-based parallelization for histogram filling
- **Jet Analysis**: Support for multiple jet algorithms (anti-kT, Cambridge-Aachen, kT)
- **Systematic Uncertainties**: Built-in support for multiple weight variations
- **Ensemble Analysis**: Support for ensemble-based uncertainty quantification
- **Progress Tracking**: Real-time progress bars with ETA estimation
- **Memory Efficient**: Optimized data loading and processing pipeline

## Dependencies

### Required Libraries
- **ROOT** (6.30+): For histogram management and data I/O
- **FastJet** (3.5+): For jet clustering algorithms
- **OpenMP**: For parallel processing
- **zlib**: For compressed data handling

### Optional Libraries
- **indicators**: For progress bar display (included in repository)

## Installation

It is easiest and recommended to install all of the sofware using a LCG release from CVMFS.
On lxplus this is trivial:

```
setupATLAS
lsetup "views LCG_108 x86_64-el9-gcc15-opt"
```

The code can then be compiled with a simple `make` command.
On perlmutter, there are no LCG releases for SUSE linux so you need to run using an image and shifter.
See the NERSC documentation for more details but the following recipe should work fine:

```
shifter --module=cvmfs --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9:latest /bin/bash
setupATLAS
lsetup "views LCG_108 x86_64-el9-gcc15-opt"
```

## Usage

### Basic Command Line Interface
```bash
./doHisto.out [OPTIONS]
```

### Required Arguments
- `--file <path>`: Path to input ROOT file containing the TTree
- `--weight_file <path>`: Path to NumPy .npz file containing event weights
- `--weight_names <list>`: Comma-separated list of weight names to process
- `--outFile <path>`: Output ROOT file path

### Optional Arguments
- `--truth`: Process truth-level data (default: reconstruction-level)
- `--maxEvents <N>`: Maximum number of events to process (default: 5,000,000)
- `--nEns <N>`: Number of ensemble variations (default: 0)
- `--kinematic_region <N>`: Kinematic region selection (0=all, 1=high-pT, 2=high-mass, 3=high-mass-jet)

### Example Commands

#### Basic Analysis
```bash
./doHisto.out \
  --file /path/to/data.root \
  --weight_file /path/to/weights.npz \
  --weight_names nominal,hv,dd \
  --outFile output.root
```

#### Truth-Level Analysis with Ensemble
```bash
./doHisto.out \
  --file /path/to/truth_data.root \
  --weight_file /path/to/weights.npz \
  --weight_names nominal \
  --nEns 10 \
  --truth \
  --maxEvents 100000 \
  --outFile truth_output.root
```

#### High-Mass Region Analysis
```bash
./doHisto.out \
  --file /path/to/data.root \
  --weight_file /path/to/weights.npz \
  --weight_names nominal,track-eff,jet-track-eff \
  --kinematic_region 2 \
  --outFile highmass_output.root
```

## Parallel Processing

The application uses OpenMP for parallel processing. Control the number of threads using:

```bash
# Set number of threads
export OMP_NUM_THREADS=8

# Run the analysis
./doHisto.out [options]
```
On perlmutter I find using 50 threads is a good balance.

## Input Data Format

### ROOT Tree Structure
The input ROOT file must contain a TTree named "OmniTree" with the following branches:

#### Event-Level Variables
- `weight`, `weight_mc`, `target_dd`: Event weights
- `pass190`: Event selection flag
- `EventNumber`, `RunNumber`: Event identification

#### Lepton Variables
- `pT_l1`, `pT_l2`: Lepton transverse momenta
- `eta_l1`, `eta_l2`: Lepton pseudorapidities  
- `phi_l1`, `phi_l2`: Lepton azimuthal angles
- `pT_ll`, `y_ll`: Dilepton system variables

#### Track Jet Variables
- `pT_trackj1`, `pT_trackj2`: Track jet transverse momenta
- `y_trackj1`, `y_trackj2`: Track jet rapidities
- `phi_trackj1`, `phi_trackj2`: Track jet azimuthal angles
- `m_trackj1`, `m_trackj2`: Track jet masses

#### Track Arrays
- `npT_tracks`: Number of tracks
- `pT_tracks[]`: Track transverse momenta array
- `eta_tracks[]`: Track pseudorapidities array
- `phi_tracks[]`: Track azimuthal angles array
- `pdgId_tracks[]`: Track PDG IDs (truth only)

### Weight File Format
The weight file should be a NumPy .npz file containing arrays for each systematic variation:
```
weights.npz
├── nominal-central
├── hv-central  
├── dd-central
├── track-eff-central
├── jet-track-eff-central
└── nominal-0, nominal-1, ..., nominal-N (for ensembles)
```

## Output Format

### Histogram Structure
The output ROOT file contains histograms organized by weight variation:

```
output.root
├── nominal-hm1_R04
├── nominal-hm2_R04
├── nominal-hpT_R10
├── hv-hm1_R04
├── hv-hm2_R04
└── ...
```

### Available Histograms

#### Jet Mass Distributions
- `hm1_R04`, `hm2_R04`, `hm3_R04`, `hm4_R04`: Leading jet masses (R=0.4)
- `hmjj_R04`: Dijet invariant mass (R=0.4)
- `hmjj_CA04`: Dijet invariant mass (Cambridge-Aachen R=0.4)

#### Jet Kinematics
- `hpT_R10`: Jet transverse momentum (R=1.0)
- `hpT_CA04`, `hpT_CA06`: Cambridge-Aachen jet pT

#### Jet Correlations
- `hdyjj_R04`, `hdyjj_CA04`: Dijet rapidity difference
- `hdRjj_CA04`: Dijet angular separation
- `hdphijj_CA04`: Dijet azimuthal angle difference

#### Energy-Energy Correlators (EEC)
- `hEEC_R04`, `hEEC_R10`: Jet-level EEC
- `hTEEC_*`: Event-level transverse EEC variants

## Code Architecture

### Main Components

#### `doHisto.cc`
- Command-line interface and argument parsing
- TChain setup and analysis initialization

#### `MakeOmni.C/h`
- Main analysis class with parallel event processing
- ROOT tree data loading and histogram filling
- OpenMP parallelization implementation

#### `HistoGroup.C/h`
- Histogram management and organization
- Thread-safe histogram merging
- Output file writing

#### `jetHelpers.C/h`
- Jet clustering utilities
- Energy-energy correlator calculations
- Lund plane analysis functions

#### `cnpy.C/h`
- NumPy .npz file reading utilities
- Weight loading and management

### Parallel Processing Strategy

1. **Sequential Data Loading**: All ROOT tree data is loaded sequentially to avoid thread safety issues
2. **Thread-Local Histograms**: Each thread maintains its own set of histograms
3. **Parallel Processing**: Event processing (jet clustering, histogram filling) happens in parallel
4. **Histogram Merging**: Thread-local histograms are merged sequentially after parallel processing

## Performance Considerations

### Memory Usage
- **Base Memory**: For a small ROOT file, e.g. the reco level data, the memory usage is manageable. For a large ROOT file, e.g. the truth pseudodata, the memory required is large but fits easily into RAM on a perlmutter CPU node. For running on lxplus, memory constraints might be an issue.
- **Per Thread**: Additional ~50-100 MB for thread-local histograms
- **Event Data**: ~1-2 MB per 1000 events (stored in memory during processing)

### Speed Optimization
- **Thread Count**: Optimal performance typically at 4-8 threads
- **Event Batching**: Dynamic scheduling with 100-event chunks
- **Memory Access**: Sequential data loading minimizes I/O bottlenecks

## Troubleshooting

### Common Issues

#### Segmentation Faults
- **Cause**: Multiple threads accessing ROOT trees simultaneously
- **Solution**: Ensure `OMP_NUM_THREADS=1` for debugging, or use the fixed parallel version

#### Empty Histograms
- **Cause**: Incorrect weight file format or missing branches
- **Solution**: Verify weight file contains expected arrays and ROOT tree has required branches

#### Compilation Errors
- **Cause**: Missing dependencies or incorrect ROOT/FastJet setup
- **Solution**: Check environment variables and library paths

### Debug Mode
For debugging, run with single thread and with few events:
```bash
export OMP_NUM_THREADS=1
./doHisto.out --maxEvents 1000 [options]
```
