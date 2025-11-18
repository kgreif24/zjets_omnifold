"""ensemble_weights.py - This is a script for calculating central value weights
over an ensemble of runs of Omnifold.

Author: Kevin Greif
Last updated 08/27/2024
python3
"""

import argparse
import glob
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def pull_weights(campaign_path, run_group, iteration, indices=None):
    """pull_weights - This function will pull the weights produced by a given
    run group. For example, if the nominal run group is titled "nominal-run-[1-10],
    the function will build a numpy array of weights from all of the step 2 trainings
    for a given iteration within the run group.

    Args:
        campaign_path (str): The path to the campaign directory.
        run_group (str): The name of the run group to pull weights from.
        iteration (int): The iteration number to pull weights for.
        indices (np.ndarray, optional): Indices to reorder the weights. Default is None.

    Returns:
        np.ndarray: A numpy array of the weights with shape (n_runs, n_test_events)
    """

    # Get the weight files for this run group
    weight_card = (
        f"./{campaign_path}/{run_group}_*/weights/iteration_{iteration}_step_2.npz"
    )
    weight_files = sorted(glob.glob(weight_card))
    if not weight_files:
        raise FileNotFoundError(
            f"No weight files found for run group '{run_group}'"
            " at iteration {iteration}."
        )

    # Place weights in a numpy array
    iteration_weights = []
    for file in weight_files:
        weights = np.load(file)["test"]
        iteration_weights.append(weights)
    iteration_weights = np.stack(iteration_weights, axis=0, dtype=np.float32)

    # If indices are provided, reorder the weights
    if indices is not None:
        iteration_weights = iteration_weights[:, indices]

    return iteration_weights


# Parse arguments
parser = argparse.ArgumentParser(
    description="Calculate central value weights over an ensemble of Omnifold runs.",
)
parser.add_argument(
    "--campaign_path",
    type=str,
    help="Path to the directory containing all of the data from a campaign",
)
parser.add_argument(
    "--iterations",
    type=int,
    nargs="+",
    help="The iterations to pull weights for, in order of the groups",
)
parser.add_argument(
    "--use_data", action="store_true", help="Use the data indices", default=False
)
parser.add_argument(
    "--group_names",
    type=str,
    nargs="+",
    help="The names of the run groups to pull weights for",
)
parser.add_argument("--output", type=str, help="Output file path")
args = parser.parse_args()

# Load indices for unshuffling MC test events and HV events
indices_nominal = np.load("/pscratch/sd/k/kgreif/data/unshuffle_indices.npy")
indices_hv = np.load("/pscratch/sd/k/kgreif/data/unshuffle_indices_hv.npy")

# Load trees and raw MC weights (multiply by Omnifold weightst to get final results)
t = uproot.open(
    "/pscratch/sd/k/kgreif/zjets_plot_staging/"
    "ZjetOmnifold_5Jul2025_MGPy8FxFxPlusNonStrong_syst_Test_withdd.root"
)["OmniTree"]
t_hv = uproot.open(
    "/pscratch/sd/k/kgreif/zjets_plot_staging/"
    "ZjetOmnifold_Mar10_Sherpa2211_LookLike_MgFxFx_Test_V5.root"
)["OmniTree"]
nominal_root_weights = ak.to_numpy(t["weight_mc"].array())
hv_root_weights = ak.to_numpy(t_hv["weight_mc"].array())
dd_target_weights = ak.to_numpy(t["target_dd"].array())

# Define the names of the various run groups in a campaign
assert len(args.iterations) == len(
    args.group_names
), "Number of iterations must match number of groups"
all_weights = {}
print(f"Pulling weights for groups {args.group_names} at iterations {args.iterations}")

# Loop through the run groups
for gn, it in zip(args.group_names, args.iterations):

    if args.use_data and gn != "dd":
        pull_gn = f"{gn}-data"
    else:
        pull_gn = gn
    print(f"Pulling weights for {pull_gn}")
    # Set the indices to use
    if gn == "hv":
        use_indices = indices_hv
    else:
        use_indices = indices_nominal
    pulled_weights = pull_weights(
        args.campaign_path,
        pull_gn,
        it,
        indices=use_indices,
    )
    print(f"Got {len(pulled_weights)} weights for group {pull_gn}")

    # Calculate the central value weights
    if gn not in ["dbootstrap", "mcbootstrap"]:
        central_weights = np.mean(pulled_weights.clip(min=0, max=100), axis=0)
        all_weights[f"{gn}-central"] = central_weights

    # Only save ensemble weights for specific group names
    if gn in ["nominal", "dbootstrap", "mcbootstrap"]:
        # Loop over pulled weights and add each to the all_weights dictionary
        # Change the name of nominal weights to "nn-init"
        if gn == "nominal":
            gn = "nn-init"
        for i, weight in enumerate(pulled_weights):
            all_weights[f"{gn}-{i}"] = weight

# Loop through the all_weights dictionary and do the following:
# 1. Multiply by the nominal root weights
# 2. Normalize to the nominal weights
# 3. Split off the HV weights since they have a different shape
hv_weights = {}
other_weights = {}
nominal_weights_norm = np.sum(all_weights["nominal-central"] * nominal_root_weights)
for key, value in all_weights.items():
    if key.startswith("hv-"):
        value *= hv_root_weights
        value *= nominal_weights_norm / np.sum(value)
        hv_weights[key] = value
    else:
        value *= nominal_root_weights
        value *= nominal_weights_norm / np.sum(value)
        other_weights[key] = value

# Add in dd-target weights, note this are already multiplied by the nominal root weights
dd_target_weights *= nominal_weights_norm / np.sum(dd_target_weights)
other_weights["dd-target"] = dd_target_weights

# Create output directory if it doesn't exist
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Save results in HDF5 format compatible with pd.read_hdf()
# Save non-hv weights first (creates the file with mode='w')
if other_weights:
    df_other = pd.DataFrame(other_weights)
    df_other.to_hdf(args.output, key="weights", mode="w", format="table")
    # Save hv weights to the same file with append mode
    if hv_weights:
        df_hv = pd.DataFrame(hv_weights)
        df_hv.to_hdf(args.output, key="hv_weights", mode="a", format="table")
elif hv_weights:
    # If only hv weights exist, create file with hv weights
    df_hv = pd.DataFrame(hv_weights)
    df_hv.to_hdf(args.output, key="hv_weights", mode="w", format="table")
