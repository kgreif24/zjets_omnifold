"""unfolding_comparisons.py - This script generates plots illustrating
the performance of Multifold and IBU (baseline methods).
Both comparisons are only possible in the 24 multifold dimensions at the moment.

Author: Kevin Greif
Last updated 04/04/2024
python3
"""

import argparse
import yaml
import scipy
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import utils.plotting_utils as pu


# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate unfolding comparison plots")
parser.add_argument(
    "--store", type=str, default="plots", help="Directory to store plots"
)
parser.add_argument(
    "--compare",
    type=str,
    default="multifold",
    choices=["multifold", "ibu"],
    help="Which comparison to make",
)
parser.add_argument(
    "--region",
    type=int,
    default=0,
    help=(
        "Integer which sets which region of the phase space to apply cuts to, if any."
        "0: No cuts, 1: high pT_Z, 2: electroweak enhanced, 3, diboson enhanced"
    ),
)
args = parser.parse_args()

# Set max PD events
max_pd_events = 1000000

# Load Omnifold target data
f_truthpd = uproot.open(
    ("/pscratch/sd/k/kgreif/data/WithTracks_TruthPseudodata_Mar12"
     "_Combined_1_50_Top_shuffled.root")
)
t_truthpd = f_truthpd["OmniTree"]

# Get the Omnifold target filters
filter_of_target = ak.to_numpy(
    t_truthpd["truth_pass190"].array(entry_stop=max_pd_events)
)

# If we are comparing to multifold, need to restrict dimuon pT to > 200 GeV
if args.compare == "multifold":
    of_ptll_target = ak.to_numpy(
        t_truthpd["truth_pT_ll"].array(entry_stop=max_pd_events)
    )
    filter_of_target = np.logical_and(filter_of_target, of_ptll_target > 200)
    print("N target events:", np.sum(filter_of_target))

# # If we are applying a region cut, do so here
# if args.region == 1:
#     of_ptll = ak.to_numpy(t_mctest["truth_pT_ll"].array())
#     of_ptll_target = ak.to_numpy(t_truthpd["truth_pT_ll"].array())
#     of_pt_j2 = ak.to_numpy(t_mctest["truth_pT_trackj2"].array())
#     of_pt_j2_target = ak.to_numpy(t_truthpd["truth_pT_trackj2"].array())
#     filter_of = np.logical_and(filter_of, of_ptll > 350, of_pt_j2 > 50)
#     filter_of_target = np.logical_and(
#         filter_of_target, of_ptll_target > 350, of_pt_j2_target > 50
#     )

# Get the target weights
target_weights_of = ak.to_numpy(t_truthpd["weight_mc"].array(entry_stop=max_pd_events))[
    filter_of_target == 1
]

# Load multifold measurement data
if args.compare == "multifold":
    multifold = pd.read_hdf(
        "/global/cfs/cdirs/m3246/ZjetOmnifold/data/multifold/multifold.h5"
    )

    # Get nominal multifold weights
    mean_weights_mf = multifold["weights_nominal"]
    ensemble_weights_mf = np.array([
        multifold[key] for key in multifold.keys() if "weights_ensemble" in key
    ])

    # Counter for keeping track of multifold W1 distances
    w1_counter = 0

# Else load IBU data
else:
    ibu = np.load(
        "/global/cfs/cdirs/m3246/ZjetOmnifold/data/ibu/ibu.npy", allow_pickle=True
    )

# Get config from yaml, and keep only multifold observables
with open("./utils/plots_config.yml", "r") as stream:
    config = yaml.safe_load(stream)
plots = [
    config["plots"][plot]
    for plot in config["plots"]
    if config["plots"][plot]["multifold"]
]

# Loop through the observables
for obs_dict in plots:

    # if obs_dict["key"] != "pT_ll":
    #     continue

    key = obs_dict["key"]
    print("Plotting observable", key)

    # Get the Omnifold data, take care to take the truth data
    obs_of_target = ak.to_numpy(
        t_truthpd["truth_" + key].array(entry_stop=max_pd_events)
    )[filter_of_target == 1]

    # For multifold
    if args.compare == "multifold":

        # Get the multifold data
        obs_mf = multifold[key]

        # Calculate W1 distance for this dimension
        w1 = scipy.stats.wasserstein_distance(
            obs_mf,
            obs_of_target,
            u_weights=mean_weights_mf,
            v_weights=target_weights_of,
        )
        w1_counter += w1

        # Make multifold plot
        obs_dict.update({"color": "green", "name": "Multifold"})
        fig = pu.unfold_performance_plot(
            obs_mf,
            mean_weights_mf,
            obs_of_target,
            target_weights_of,
            ensemble_weights_mf,
            plot_params=obs_dict,
        )
        fig.savefig(f"{args.store}/mf_{key}.png", dpi=300)
        fig.savefig(f"{args.store}/mf_{key}.pdf", dpi=300)
        plt.close()

    # For IBU
    else:

        # Loop through IBU results and find the one that matches the observables
        for ibu_dict in ibu:
            if ibu_dict["file_label"] == key:
                obs_data = ibu_dict
                break

        # Make IBU plot
        obs_dict.update({"color": "red", "name": "IBU"})
        fig = pu.ibu_performance_plot(
            obs_data, obs_of_target, target_weights_of, obs_dict
        )
        fig.savefig(f"{args.store}/ibu_{key}.png", dpi=300)
        fig.savefig(f"{args.store}/ibu_{key}.pdf", dpi=300)
        plt.close()

# Print the total W1 distance for multifold
if args.compare == "multifold":
    print(f"Total W1 distance for multifold: {w1_counter}")
    print("NOTE: this distance does not include the Ntracks or HT observables")

