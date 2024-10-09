""" unfolding_compare.py - This script generates plots comparing the performance of a given
ensemble of omnifold runs to the multifold and IBU results. The multifold comparison is
only possible for the 24 multifold observables. The IBU comparison is only possible for those
dimensions in which we have run IBU.

Author: Kevin Greif
Last updated 09/06/2024
python3
"""

import sys, argparse
import glob
import scipy
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import matplotlib.pyplot as plt

sys.path.append('./utils')
import plotting_utils as pu


# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate unfolding comparison plots')
parser.add_argument("--store", type=str, default="plots", help="Directory to store plots")
parser.add_argument("--weight_card", type=str, help="Path containing wildcard for Omnifold weights")
parser.add_argument("--err_multiple", type=float, default=1.0, help="Error multiple for plotting")
parser.add_argument("--compare", type=str, default='multifold', choices=['multifold', 'ibu'], help="Which comparison to make")
args = parser.parse_args()

# Load Omnifold measurement data
f_mctest = uproot.open('./dataloc/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root')
t_mctest = f_mctest['OmniTree']
f_truthpd = uproot.open("./dataloc/WithTracks_TruthPseudodata_Mar12_Combined_1_50_Top_shuffled.root")
t_truthpd = f_truthpd['OmniTree']

# Get the Omnifold filters
filter_of = ak.to_numpy(t_mctest['truth_pass190'].array())
filter_of_target = ak.to_numpy(t_truthpd['truth_pass190'].array())

# If we are comparing to multifold, need to restrict dimuon pT to > 200 GeV
if args.compare == 'multifold':
    of_ptll = ak.to_numpy(t_mctest['truth_pT_ll'].array())
    of_ptll_target = ak.to_numpy(t_truthpd['truth_pT_ll'].array())
    filter_of = np.logical_and(filter_of, of_ptll > 200)
    filter_of_target = np.logical_and(filter_of_target, of_ptll_target > 200)

# Get the Omnifold weights
weight_files = sorted(glob.glob(args.weight_card))
ensemble_weights_of = [np.load(weight_file)['test'][filter_of == 1] for weight_file in weight_files]
mean_weights_of = np.mean(ensemble_weights_of, axis=0)
target_weights_of = ak.to_numpy(t_truthpd['weight_mc'].array())[filter_of_target == 1]

# Load multifold measurement data
if args.compare == 'multifold':
    multifold = pd.read_hdf("/global/cfs/cdirs/m3246/ZjetOmnifold/data/multifold/multifold.h5")
    multifold_target = pd.read_hdf("/global/cfs/cdirs/m3246/ZjetOmnifold/data/multifold/target.h5")

    # Take the first N network ensemble weights
    ensemble_names = [col for col in multifold.keys() if col.startswith("weights_ensemble_")][:len(weight_files)]
    ensemble_weights_mf = [multifold[ensemble_name] for ensemble_name in ensemble_names]
    mean_weights_mf = np.mean(ensemble_weights_mf, axis=0)
    target_weights_mf = multifold_target['weight_mc']

    # Counter for keeping track of multifold W1 distances
    w1_counter = 0

# Else load IBU data
else:
    ibu = np.load('/global/cfs/cdirs/m3246/ZjetOmnifold/data/ibu/ibu.npy', allow_pickle=True)

# Loop through the 24 multifold observables + Ntracks
for key, obs_dict in pu.default_settings.items():

    print("Plotting observable", key)

    # Set IBU binning if we are comparing to IBU
    if args.compare == 'ibu':
        obs_dict.update({'bins': np.array(pu.ibu_bins[key])})

    # Get the Omnifold data, take care to take the truth data
    obs_of = ak.to_numpy(t_mctest["truth_" + key].array())[filter_of == 1]
    obs_of_target = ak.to_numpy(t_truthpd["truth_" + key].array())[filter_of_target == 1]

    # Make Omnifold plot
    obs_dict.update({'color': 'purple', 'name': 'Omnifold'})
    fig = pu.unfold_performance_plot(obs_of, mean_weights_of, obs_of_target, target_weights_of, ensemble_weights_of, plot_params=obs_dict, err_multiple=args.err_multiple)
    fig.savefig(f"{args.store}/of_{key}.png", dpi=300)
    plt.close()

    # Repeat for comparison if this is not Ntracks
    if key != "Ntracks":

        # For multifold
        if args.compare == 'multifold':

            # Get the multifold data
            obs_mf = multifold[key]
            obs_mf_target = multifold_target[key]

            # Calculate W1 distance for this dimension
            w1 = scipy.stats.wasserstein_distance(obs_mf, obs_mf_target, u_weights=mean_weights_mf, v_weights=target_weights_mf)
            w1_counter += w1

            # Make multifold plot
            obs_dict.update({'color': 'green', 'name': 'Multifold'})
            fig = pu.unfold_performance_plot(obs_mf, mean_weights_mf, obs_mf_target, target_weights_mf, ensemble_weights_mf, plot_params=obs_dict, err_multiple=args.err_multiple)
            fig.savefig(f"{args.store}/mf_{key}.png", dpi=300)
            plt.close()

        # For IBU
        else:
            
            # Loop through IBU results and find the one that matches the observables
            for ibu_dict in ibu:
                if ibu_dict['file_label'] == key:
                    obs_data  = ibu_dict
                    break

            # Make IBU plot
            obs_dict.update({'color': 'blue', 'name': 'IBU'})
            fig = pu.ibu_performance_plot(obs_data, obs_mf_target, target_weights_mf, obs_dict)
            fig.savefig(f"{args.store}/ibu_{key}.png", dpi=300)
            plt.close()

# Print the total W1 distance for multifold
if args.compare == 'multifold':
    print(f"Total W1 distance for multifold: {w1_counter}")

# Make omnifold plot for H_T
track_pt = t_mctest['truth_pT_tracks'].array()[filter_of == 1]
track_pt_target = t_truthpd['truth_pT_tracks'].array()[filter_of_target == 1]
ht = ak.sum(track_pt, axis=1)
ht_target = ak.sum(track_pt_target, axis=1)

obs_dict = pu.track_hists['alltrack_Ht']
obs_dict.update({'color': 'purple', 'name': 'Omnifold'})
fig = pu.unfold_performance_plot(ht, mean_weights_of, ht_target, target_weights_of, ensemble_weights_of, plot_params=obs_dict, err_multiple=args.err_multiple)
fig.savefig(f"{args.store}/of_alltrack_Ht.png", dpi=300)
plt.close()