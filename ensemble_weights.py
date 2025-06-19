"""ensemble_weights.py - This is a script for calculating central value weights
over an ensemble of runs of Omnifold.

Author: Kevin Greif
Last updated 08/27/2024
python3
"""

import argparse
import glob
import numpy as np


def pull_weights(campaign_path, run_group, iteration, indices=None, max_ens=10):
    """pull_weights - This function will pull the weights produced by a given
    run group. For example, if the nominal run group is titled "nominal-run-[1-10],
    the function will build a numpy array of weights from all of the step 2 trainings
    for a given iteration within the run group.

    Args:
        campaign_path (str): The path to the campaign directory.
        run_group (str): The name of the run group to pull weights from.
        iteration (int): The iteration number to pull weights for.
        indices (np.ndarray, optional): Indices to reorder the weights. Default is None.
            Note this is only used for the zjets-pretrain-v3 nominal run group.
        max_ens (int, optional): The maximum number of ensembles to pull weights for.
            Default is 10.

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

    # If max_ens is provided, limit the number of weight files to max_ens
    if max_ens is not None:
        weight_files = weight_files[:max_ens]

    # Place weights in a numpy array
    iteration_weights = np.zeros(
        (len(weight_files), np.load(weight_files[0])["test"].shape[0]), dtype=np.float32
    )
    for i, file in enumerate(weight_files):
        weights = np.load(file)["test"]
        iteration_weights[i] = weights

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
    "--max_ens",
    type=int,
    default=10,
    help="The maximum number of ensembles to pull weights for",
)
parser.add_argument("--output", type=str, help="Output file path")
args = parser.parse_args()

# Load indices for applying syst shuffling to nominal events
indices = np.load("/pscratch/sd/k/kgreif/data/matching_indices.npy")

# Define the names of the various run groups in a campaign
group_names = ["nominal-ensemble", "track-eff", "hidden-variable"]
iteration_number = [6, 8, 5]
all_weights = {}

for gn, it in zip(group_names, iteration_number):

    if gn == "nominal-ensemble":
        pulled_weights = pull_weights(
            args.campaign_path,
            gn,
            it,
            indices=indices,
            max_ens=args.max_ens,
        )
    else:
        pulled_weights = pull_weights(
            args.campaign_path, gn, it, max_ens=args.max_ens
        )
    print(f"Got {len(pulled_weights)} weights for group {gn}")

    # Calculate the central value weights
    central_weights = np.mean(pulled_weights.clip(min=0, max=100), axis=0)
    all_weights[f"{gn}-central"] = central_weights

    # Loop over pulled weights and add each to the all_weights dictionary
    for i, weight in enumerate(pulled_weights):
        all_weights[f"{gn}-{i}"] = weight

# Save results
np.savez(args.output, **all_weights)
