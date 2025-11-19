"""ensemble_weights.py - This is a script for calculating central value weights
over an ensemble of runs of Omnifold.

Author: Kevin Greif
Last updated 08/27/2024
python3
"""

import os
import sys
import argparse
import glob
import uproot
import awkward as ak
import numpy as np
import pandas as pd

sys.path.append("./utils")
import data_utils as du  # noqa: E402

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


def group_name_to_write_name(gn, idx=None):
    """ group_name_to_write_name - Utility function to convert a group name,
    which is how a given group is referred to in the Omnifold results repository,
    to a write name, which is how a given group is referred to in the final weight
    files provided for publication.
    """
    if gn == "nominal" and idx is None:
        return "weights_nominal"
    elif gn == "hv":  # Not a bug! HV uncertainty re-weights the sherpa sample
        return "weights_nominal"
    elif gn == "nominal" and idx is not None:
        return f"weights_ensemble_{idx}"
    elif gn == "dd":
        return "weights_dd"
    elif gn == "dbootstrap":
        return f"weights_bootstrap_data_{idx}"
    elif gn == "mcbootstrap":
        return f"weights_bootstrap_mc_{idx}"
    elif gn == "nn-init":
        return f"weights_ensemble_{idx}"
    elif gn == "track-eff":
        return "weights_trackEffMain"
    elif gn == "jet-track-eff":
        return "weights_trackEffJet"
    elif gn == "track-fake":
        return "weights_trackFake"
    elif gn == "track-scale":
        return "weights_trackPtScale"
    elif gn == "muon-id":
        return "weights_muonCalID"
    elif gn == "muon-ms":
        return "weights_muonCalMS"
    elif gn == "muon-resbias":
        return "weights_muonCalResBias"
    elif gn == "muon-scale":
        return "weights_muonCalScale"
    elif gn == "muon-effreco":
        return "weights_muonEffReco"
    elif gn == "muon-effiso":
        return "weights_muonEffIso"
    elif gn == "muon-efftrk":
        return "weights_muonEffTrack"
    elif gn == "muon-efftrig":
        return "weights_muonEffTrig"
    else:
        raise ValueError(f"Group name {gn} not recognized!")


def get_truth_to_reco_ratio(gn, t_mc):
    """ get_truth_to_reco_ratio - This function will calculate the ratio of the truth
    to the reconstructed events for a given run group. Typically this is just the
    sum of the `weight_mc` branch divided by the sum of the `weight` branch, unless
    either of these things are modified by the systematic applied to the run group.

    Note we don't consider the HV systematic here, it is handled separately.
    """

    nominal_weight_mc = ak.to_numpy(t_mc["weight_mc"].array())
    nominal_weight = ak.to_numpy(t_mc["weight"].array())
    nominal_pass190 = ak.to_numpy(t_mc["pass190"].array())
    nominal_truth_pass190 = ak.to_numpy(t_mc["truth_pass190"].array())

    nominal_weight_mc = nominal_weight_mc[nominal_truth_pass190 == 1]
    nominal_weight = nominal_weight[nominal_pass190 == 1]

    nominal_numerator = np.sum(nominal_weight_mc)
    nominal_denominator = np.sum(nominal_weight)
    nominal_ratio = nominal_numerator / nominal_denominator

    if gn == "hv":
        raise ValueError("HV systematic is handled separately")
    elif "muon" in gn:
        pass190 = du.calc_muon_syst_pass190(t_mc, syst_kw=gn)
        if gn == "muon-effreco":
            weight = ak.to_numpy(t_mc["syst_recoSFDown"].array())
        elif gn == "muon-effiso":
            weight = ak.to_numpy(t_mc["syst_isoSFDown"].array())
        elif gn == "muon-efftrk":
            weight = ak.to_numpy(t_mc["syst_trkSFDown"].array())
        elif gn == "muon-efftrig":
            weight = ak.to_numpy(t_mc["syst_trigSFDown"].array())
        else:
            weight = nominal_weight
        weight = weight[pass190 == 1]
        numerator = np.sum(weight)
        return numerator / nominal_denominator
    elif gn == "mcbootstrap":
        raise NotImplementedError("MC bootstrap weights are not implemented yet")
    else:
        return nominal_ratio


def get_bs_n_data(gn, campaign_path):
    """ get_bs_data_weights - This function will return the weights for a given
    data bootstrap run group. It will re-create the data sample and sum the weights
    to get the number of data events.

    Args:
        gn (str): The name of the run group.
        campaign_path (str): The path to the campaign directory.

    Returns:
        int: The number of data events.
    """
    sample_files = glob.glob(f"./{campaign_path}/{gn}/bootstrap_*.npy")
    assert len(sample_files) == 1, "Expected exactly one sample file for data bootstrap"
    sample = np.load(sample_files[0])
    return np.sum(sample)


def norm_weights(weights, ratio, n_data):
    """ norm_weights - This function will normalize a set of weights to restore the
    event yield predicted by the MC given the number of data events.
    """
    return weights * n_data * ratio / np.sum(weights)


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

# Load trees, n_data, and raw MC weights
t = uproot.open(
    "/pscratch/sd/k/kgreif/zjets_plot_staging/"
    "ZjetOmnifold_5Jul2025_MGPy8FxFxPlusNonStrong_syst_Test_withdd.root"
)["OmniTree"]
t_hv = uproot.open(
    "/pscratch/sd/k/kgreif/zjets_plot_staging/"
    "ZjetOmnifold_Mar10_Sherpa2211_LookLike_MgFxFx_Test_V5.root"
)["OmniTree"]
if args.use_data:
    t_data = uproot.open(
        "/pscratch/sd/k/kgreif/zjets_plot_staging/"
        "ZjetOmnifold_Nov11_data_WithTracks_slim_Systematics_shuffled.root"
    )["OmniTree"]
    n_data_nominal = int(t_data.num_entries)
else:
    t_data = uproot.open(
        "/pscratch/sd/k/kgreif/zjets_plot_staging/"
        "Pseudodata_SherpaDY_PowhegPythiaTop_June2025_shuffled.root"
    )["OmniTree"]
    n_data_nominal = int(t_data.num_entries)
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

    # Skip the HV group, it is handled separately
    if gn == "hv":
        continue

    # Pull the weights for a given group
    if args.use_data and gn != "dd":
        pull_gn = f"{gn}-data"
    else:
        pull_gn = gn
    print(f"Pulling weights for {pull_gn}")
    pulled_weights = pull_weights(
        args.campaign_path,
        pull_gn,
        it,
        indices=indices_nominal,
    )
    print(f"Got {len(pulled_weights)} weights for group {pull_gn}")

    # Calculate the central value weights
    if gn not in ["dbootstrap", "mcbootstrap"]:
        central_weights = np.mean(pulled_weights.clip(min=0, max=100), axis=0)
        write_name = group_name_to_write_name(gn)
        all_weights[write_name] = central_weights

    # Only save ensemble weights for specific group names
    if gn in ["nominal", "dbootstrap", "mcbootstrap"]:
        # Loop over pulled weights and add each to the all_weights dictionary
        for i, weight in enumerate(pulled_weights):
            write_name = group_name_to_write_name(gn, i)
            all_weights[write_name] = weight

# Loop through the all_weights dictionary and do the following:
# 1. Multiply by the nominal root weights
# 2. Normalize to the event yield predicted by the MC
mc_weights = {}
for key, value in all_weights.items():
    value *= nominal_root_weights
    ratio_mc = get_truth_to_reco_ratio(key, t)
    if "dbootstrap" in key:
        n_data = get_bs_n_data(key, args.campaign_path)
        print(f"Number of data events for {key}: {n_data}")
    else:
        n_data = n_data_nominal
    value = norm_weights(value, ratio_mc, n_data)
    mc_weights[key] = value

# Add in dd-target weights, note this are already multiplied by the nominal root weights
ratio_mc = get_truth_to_reco_ratio("target_dd", t)
dd_target_weights = norm_weights(dd_target_weights, ratio_mc, n_data_nominal)
mc_weights["target_dd"] = dd_target_weights

# Now handle the HV weights
hv_weights = {}
if "hv" in args.group_names:
    print("Pulling HV weights")
    pulled_weights = pull_weights(
        args.campaign_path,
        "hv",
        args.iterations[0],
        indices=indices_hv,
    )
    print(f"Got {len(pulled_weights)} weights for group hv")
    # Calculate the central value weights
    central_weights = np.mean(pulled_weights.clip(min=0, max=100), axis=0)
    central_weights *= hv_root_weights
    # Normalize the HV weights
    ratio_hv = np.sum(hv_root_weights) / np.sum(ak.to_numpy(t_hv["weight"].array()))
    central_weights = norm_weights(central_weights, ratio_hv, n_data_nominal)
    hv_weights["weights_nominal"] = central_weights

# Create output directory if it doesn't exist
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Save results in HDF5 format compatible with pd.read_hdf()
# Save non-hv weights first (creates the file with mode='w')
df_mc = pd.DataFrame(mc_weights)
df_mc.to_hdf(args.output, key="weights", mode="w", format="table")

df_hv = pd.DataFrame(hv_weights)
df_hv.to_hdf(args.output, key="hv_weights", mode="a", format="table")
