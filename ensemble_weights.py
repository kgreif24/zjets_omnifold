"""ensemble_weights.py - This is a script for calculating central value weights
over an ensemble of runs of Omnifold.

Author: Kevin Greif
Last updated 12/22/2025
python3
"""

import os
import sys
import argparse
import glob
import tempfile
import zipfile
import uproot
import awkward as ak
import numpy as np
import pandas as pd

sys.path.append("./utils")
import data_utils as du  # noqa: E402

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def pull_weights(
    campaign_path,
    run_group,
    iteration,
):
    """pull_weights - This function will pull the weights produced by a given
    run group. For example, if the nominal run group is titled "nominal-run-[1-10],
    the function will build a numpy array of weights from all of the step 2 trainings
    for a given iteration within the run group.

    Args:
        campaign_path (str): The path to the campaign directory.
        run_group (str): The name of the run group to pull weights from.
        iteration (int): The iteration number to pull weights for.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A numpy array of the weights with shape
              (n_runs, n_test_events)
            - list: A list of full run names (e.g., "dbootstrap_1", "dbootstrap_2")
              in the same order as weights
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

    # Place weights in a numpy array and extract run names
    iteration_weights = []
    run_names = []
    for file in weight_files:
        weights = np.load(file)["test"]
        iteration_weights.append(weights)
        # Extract full run name from path:
        # {campaign_path}/{run_name}/weights/...
        # Get the directory name that contains the weights file
        dir_path = os.path.dirname(file)
        # Go up two levels to get run directory
        run_name = os.path.basename(os.path.dirname(dir_path))
        run_names.append(run_name)
    iteration_weights = np.stack(iteration_weights, axis=0, dtype=np.float32)

    return iteration_weights, run_names


def group_name_to_write_name(gn, idx=None):
    """group_name_to_write_name - Utility function to convert a group name,
    which is how a given group is referred to in the Omnifold results repository,
    to a write name, which is how a given group is referred to in the final weight
    files provided for publication.
    """
    if gn == "nominal" and idx is None:
        return "weights_nominal"
    elif (
        gn == "hv" or gn == "hv2"
    ):  # Not a bug! HV uncertainty re-weights the sherpa sample
        return "weights_nominal"
    elif gn == "hvhad":
        return "weights_hvhad"
    elif gn == "nominal" and idx is not None:
        return f"weights_ensemble_{idx}"
    elif gn == "dd":
        return "weights_dd"
    elif gn == "dbootstrap":
        assert idx is not None
        return f"weights_bootstrap_data_{idx}"
    elif gn == "mcbootstrap":
        assert idx is not None
        return f"weights_bootstrap_mc_{idx}"
    elif gn == "nn-init":
        assert idx is not None
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
        return "weights_muCalID"
    elif gn == "muon-ms":
        return "weights_muCalMS"
    elif gn == "muon-resbias":
        return "weights_muCalResBias"
    elif gn == "muon-scale":
        return "weights_muCalScale"
    elif gn == "muon-effreco":
        return "weights_muEffReco"
    elif gn == "muon-effiso":
        return "weights_muEffIso"
    elif gn == "muon-efftrk":
        return "weights_muEffTrack"
    elif gn == "muon-efftrig":
        return "weights_muEffTrig"
    elif gn == "prw":
        return "weights_pileup"
    elif gn == "theory-qcd":
        return "weights_theoryQCD"
    elif gn == "theory-pdf":
        return "weights_theoryPDF"
    elif gn == "theory-alphas":
        return "weights_theoryAlphaS"
    elif gn == "theory-pssoft":
        return "weights_theoryPSsoft"
    elif gn == "theory-psjet":
        return "weights_theoryPSjet"
    elif gn == "theory-mpi":
        return "weights_theoryMPI"
    elif gn == "theory-psscale":
        return "weights_theoryPSscale"
    elif gn == "top":
        return "weights_topBackground"
    elif gn == "nonstrong-diboson":
        return "weights_nonstrongDiboson"
    elif gn == "nonstrong-ew":
        return "weights_nonstrongEW"
    else:
        raise ValueError(f"Group name {gn} not recognized!")


def get_truth_to_reco_ratio(gn, t_mc, prior_weights, reco_pass, truth_pass):
    """get_truth_to_reco_ratio - This function will calculate the ratio of the truth
    to the reconstructed events for a given run group. Typically this is just the
    sum of the `weight_mc` branch divided by the sum of the `weight` branch, unless
    either of these things are modified by the systematic applied to the run group.

    Note we don't consider the HV systematic here, it is handled separately.
    HVHAD however is handled here.

    Args:
        gn (str): The name of the run group.
        t_mc (uproot.TTree): The tree to get the weights from.
        prior_weights (dict of np.ndarrays):
            The prior weights to use for the systematic
        reco_pass (np.ndarray): The pass190 filter for the reconstructed events.
        truth_pass (np.ndarray): The pass190 filter for the truth events.

    Returns:
        float: The ratio of the truth to the reconstructed events.
    """

    # If systematic shifts the prior, do calculation using the weights from .npz
    if "hv" in gn or "theory" in gn:
        truth_prior_weights = get_prior_weights(gn, prior_weights, use_truth=True)
        reco_prior_weights = get_prior_weights(gn, prior_weights, use_truth=False)
        numerator = np.sum(truth_prior_weights[truth_pass == 1])
        denominator = np.sum(reco_prior_weights[reco_pass == 1])
        factor = numerator / denominator
        print(f"Factor for {gn} is {factor}")
        return factor

    # Non-strong composition uncertainty: scale specific DSIDs on the fly
    elif "nonstrong" in gn:
        mc_channel = ak.to_numpy(t_mc["mcChannelNumber"].array())
        scale = get_nonstrong_scale(gn, mc_channel)
        weight_mc = ak.to_numpy(t_mc["weight_mc"].array())
        weight = ak.to_numpy(t_mc["weight"].array())
        numerator = np.sum((weight_mc * scale)[truth_pass == 1])
        denominator = np.sum((weight * scale)[reco_pass == 1])
        factor = numerator / denominator
        print(f"Factor for {gn} is {factor}")
        return factor

    # Else do the calculation using the weights from the tree
    else:
        nominal_weight_mc = ak.to_numpy(t_mc["weight_mc"].array())
        nominal_weight = ak.to_numpy(t_mc["weight"].array())
        nominal_weight_mc_filtered = nominal_weight_mc[truth_pass == 1]
        nominal_weight_filtered = nominal_weight[reco_pass == 1]
        nominal_numerator = np.sum(nominal_weight_mc_filtered)
        nominal_denominator = np.sum(nominal_weight_filtered)
        nominal_ratio = nominal_numerator / nominal_denominator

        if "muon" in gn:
            if gn == "muon-effreco":
                nominal_sf = ak.to_numpy(t_mc["mu_recoSF"].array())
                varied_sf = ak.to_numpy(t_mc["syst_recoSFDown"].array())
                weight = varied_sf * nominal_weight / nominal_sf
                usepass = reco_pass
            elif gn == "muon-effiso":
                nominal_sf = ak.to_numpy(t_mc["mu_isoSF"].array())
                varied_sf = ak.to_numpy(t_mc["syst_isoSFDown"].array())
                weight = varied_sf * nominal_weight / nominal_sf
                usepass = reco_pass
            elif gn == "muon-efftrk":
                nominal_sf = ak.to_numpy(t_mc["mu_TTVASF"].array())
                varied_sf = ak.to_numpy(t_mc["syst_TTVASFDown"].array())
                weight = varied_sf * nominal_weight / nominal_sf
                usepass = reco_pass
            elif gn == "muon-efftrig":
                nominal_sf = ak.to_numpy(t_mc["singleMuonTrigSF"].array())
                varied_sf = ak.to_numpy(t_mc["syst_trigSFDown"].array())
                weight = varied_sf * nominal_weight / nominal_sf
                usepass = reco_pass
            elif gn == "muon-id":
                weight = nominal_weight
                usepass = du.calc_muon_syst_pass190(
                    t_mc, syst_kw="muon_id", pt_thresh=200
                )
            elif gn == "muon-ms":
                weight = nominal_weight
                usepass = du.calc_muon_syst_pass190(
                    t_mc, syst_kw="muon_ms", pt_thresh=200
                )
            elif gn == "muon-resbias":
                weight = nominal_weight
                usepass = du.calc_muon_syst_pass190(
                    t_mc, syst_kw="muon_resbias", pt_thresh=200
                )
            elif gn == "muon-scale":
                weight = nominal_weight
                usepass = du.calc_muon_syst_pass190(
                    t_mc, syst_kw="muon_scale", pt_thresh=200
                )
            else:
                raise ValueError(f"Systematic {gn} not recognized!")
            weight = weight[usepass == 1]
            denominator = np.sum(weight)
            return nominal_numerator / denominator
        elif gn == "prw":
            nominal_sf = ak.to_numpy(t_mc["prw"].array())
            varied_sf = ak.to_numpy(t_mc["syst_prwDown"].array())
            weight = varied_sf * nominal_weight / nominal_sf
            denominator = np.sum(weight[reco_pass == 1])
            return nominal_numerator / denominator
        else:
            return nominal_ratio


def get_bs_n_data(campaign_path, run_name, truth_pass):
    """get_bs_data_weights - This function will return the weights for a given
    data bootstrap run group. It will re-create the data sample and sum the weights
    to get the number of data events.

    Args:
        campaign_path (str): The path to the campaign directory.
        run_name (str): The full run name (e.g., "dbootstrap_1", "dbootstrap_2").
        truth_pass (np.ndarray): Array of truth pass190 values.

    Returns:
        int: The number of data events.
    """
    # Load the bootstrap file from the run directory
    sample_files = glob.glob(f"{campaign_path}/{run_name}/bootstrap*.npy")
    if not sample_files:
        raise FileNotFoundError(
            f"No bootstrap file found in {campaign_path}/{run_name}/"
        )
    if len(sample_files) > 1:
        raise ValueError(
            f"Multiple bootstrap files found in {campaign_path}/{run_name}/: "
            f"{sample_files}"
        )
    sample = np.load(sample_files[0])
    return np.sum(sample[truth_pass == 1])


def get_nonstrong_scale(gn, mc_channel):
    """get_nonstrong_scale - Compute the per-event weight scale factor for the
    non-strong (Diboson / EW) composition uncertainty. Returns a float array
    with factor > 1 for events belonging to the relevant DSIDs, 1 otherwise.

    Args:
        gn (str): Group name, either "nonstrong-diboson" or "nonstrong-ew".
        mc_channel (np.ndarray): The mcChannelNumber branch values.

    Returns:
        np.ndarray: Per-event scale factors.
    """
    dsids_diboson = [
        363356, 363358, 364250, 364253, 364254, 364255,
        363494, 363355, 363357, 363359, 363360, 363489,
    ]
    dsids_ew = [830007]
    if gn == "nonstrong-diboson":
        dsids, factor = dsids_diboson, 1.3
    elif gn == "nonstrong-ew":
        dsids, factor = dsids_ew, 1.2
    else:
        raise ValueError(f"Unknown nonstrong group name: {gn}")
    return np.where(np.isin(mc_channel, dsids), factor, 1.0).astype(np.float32)


def get_prior_weights(gn, prior_weights, use_truth=True):
    """get_prior_weights - Pull the correct prior weights from the
    weights file.
    """
    suffix = "_truth" if use_truth else "_reco"
    if gn == "theory-qcd":
        return prior_weights[f"w_QCD_dd{suffix}"]
    elif gn == "theory-pdf":
        return prior_weights[f"w_PDF_CT18nnlo{suffix}"]
    elif gn == "theory-alphas":
        return prior_weights[f"w_Alpha_s1{suffix}"]
    elif gn == "theory-pssoft":
        return prior_weights[f"w_Var1Down{suffix}"]
    elif gn == "theory-psjet":
        return prior_weights[f"w_Var2Down{suffix}"]
    elif gn == "theory-mpi":
        return prior_weights[f"w_MPIDown{suffix}"]
    elif gn == "theory-psscale":
        return prior_weights[f"w_RenDown{suffix}"]
    elif "hv" in gn and use_truth:
        return prior_weights["HC_weight_mc"]
    elif "hv" in gn and not use_truth:
        return prior_weights["HC_weight"]
    else:
        raise ValueError(f"Systematic {gn} not recognized!")


def norm_weights(weights, pass190, ratio, n_data, luminosity):
    """norm_weights - This function will normalize a set of weights to restore the
    event yield predicted by the MC given the number of data events.
    """
    pass_weights = weights[pass190 == 1]
    return_weights = weights.copy()
    return_weights[pass190 == 1] = (
        pass_weights * n_data * ratio / (np.sum(pass_weights) * luminosity)
    )
    return_weights[pass190 == 0] = 0
    return return_weights


def calc_pass_200(tree, truth=False, ptll_cut=200):
    """calc_pass_200 - This function will calculate the pass200 filter"""
    tk = "truth_" if truth else ""
    p190 = ak.to_numpy(tree[tk + "pass190"].array())
    ptll = ak.to_numpy(tree[tk + "pT_ll"].array())
    return p190 & (ptll > ptll_cut)


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
    "--iteration",
    type=int,
    required=True,
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
parser.add_argument(
    "--luminosity",
    type=float,
    help="The integrated luminosity of the data sample in units of fb^-1",
    default=140.1,
)
parser.add_argument(
    "--ptll_cut",
    type=float,
    help="The pt_ll threshold cut in GeV. Default is 200.",
    default=200.0,
)
parser.add_argument(
    "--og_order",
    action="store_true",
    help="If set, unshuffle weights to the original order of the MG and Sherpa samples",
    default=False,
)
args = parser.parse_args()

# Set paths to trees
base_path = "/pscratch/sd/k/kgreif/zjets_plot_staging/"
nominal_path = base_path + (
    "ZjetOmnifold_5Jul2025_MGPy8FxFxPlusNonStrong_syst_Test_shuffled.root"
)
hv_path = base_path + (
    "ZjetOmnifold_Mar10_Sherpa2211PlusNonStrong"
    "_LookLike_MgFxFx_HadCompLikeMG_Test_shuffled.root"
)

# Load trees, n_data, and raw MC weights
t = uproot.open(nominal_path)["OmniTree"]
t_hv = uproot.open(hv_path)["OmniTree"]
if args.use_data:
    t_data = uproot.open(
        "/pscratch/sd/k/kgreif/zjets_plot_staging/"
        "ZjetOmnifold_Nov11_data_WithTracks_slim_Systematics_shuffled.root"
    )["OmniTree"]
    pt_ll_data = ak.to_numpy(t_data["pT_ll"].array())
    n_data_nominal = np.sum(pt_ll_data > args.ptll_cut)
else:
    t_data = uproot.open(
        "/pscratch/sd/k/kgreif/zjets_plot_staging/"
        "Pseudodata_SherpaDY_PowhegPythiaTop_June2025_shuffled.root"
    )["OmniTree"]
    pt_ll_data = ak.to_numpy(t_data["pT_ll"].array())
    n_data_nominal = np.sum(pt_ll_data > args.ptll_cut)
nominal_root_weights = ak.to_numpy(t["weight_mc"].array())
hv_root_weights = ak.to_numpy(t_hv["weight_mc"].array())
hv_reco_weights = ak.to_numpy(t_hv["weight"].array())

# Load weights for uncertainties involving prior shifts and data driven
# target
prior_weights = np.load("/pscratch/sd/k/kgreif/data/madgraph_test_prior_weights.npz")
hv_prior_weights = np.load(
    "/pscratch/sd/k/kgreif/data/sherpa_test_prior_weights.npz"
)
dd_target_weights = np.load("/pscratch/sd/k/kgreif/data/target_dd_weights.npz")[
    "target_dd"
]

# Calculate the pass200 filters for the nominal and HV samples at
# both reco and truth level, and the data sample at reco level
pass200 = calc_pass_200(t, ptll_cut=args.ptll_cut)
truth_pass200 = calc_pass_200(t, truth=True, ptll_cut=args.ptll_cut)
data_pass200 = calc_pass_200(t_data, ptll_cut=args.ptll_cut)
hv_pass200 = calc_pass_200(t_hv, ptll_cut=args.ptll_cut)
hv_truth_pass200 = calc_pass_200(t_hv, truth=True, ptll_cut=args.ptll_cut)

# Calculate the nominal fiducial factor
nominal_factor = get_truth_to_reco_ratio(
    "nominal", t, prior_weights, pass200, truth_pass200
)
print(f"Nominal fiducial factor is {nominal_factor}")

# Define the names of the various run groups in a campaign
all_weights = {}

# Loop through the run groups
for gn in args.group_names:

    # Skip the HV group, it is handled separately
    if gn in ["hv", "hv2"]:
        continue

    # Pull the weights for a given group
    if args.use_data and gn != "dd":
        pull_gn = f"{gn}-data"
    else:
        pull_gn = gn
    print(f"Pulling weights for {pull_gn}")
    pulled_weights, run_names = pull_weights(
        args.campaign_path,
        pull_gn,
        args.iteration,
    )
    print(f"Got {len(pulled_weights)} weights for group {pull_gn}")

    # Calculate the central value weights
    if gn not in ["dbootstrap", "mcbootstrap"]:
        central_weights = np.mean(pulled_weights.clip(min=0, max=100), axis=0)
        if gn == "hvhad" or "theory" in gn:
            alt_root_weights = get_prior_weights(gn, prior_weights)
            central_weights *= alt_root_weights
            use_factor = get_truth_to_reco_ratio(
                gn, t, prior_weights, pass200, truth_pass200
            )
        elif "nonstrong" in gn:
            mc_channel = ak.to_numpy(t["mcChannelNumber"].array())
            scale = get_nonstrong_scale(gn, mc_channel)
            alt_root_weights = nominal_root_weights * scale
            central_weights *= alt_root_weights
            use_factor = get_truth_to_reco_ratio(
                gn, t, prior_weights, pass200, truth_pass200
            )
        else:
            central_weights *= nominal_root_weights
            use_factor = nominal_factor
        central_weights = norm_weights(
            central_weights,
            truth_pass200,
            use_factor,
            n_data_nominal,
            args.luminosity,
        )
        write_name = group_name_to_write_name(gn)
        all_weights[write_name] = central_weights

        # Add the luminosity uncertainty weights if this is the nominal group
        if gn == "nominal":
            lumi_weights = norm_weights(
                central_weights,
                truth_pass200,
                nominal_factor,
                n_data_nominal,
                args.luminosity * (1.0 - 0.0083),  # 0.83% luminosity uncertainty
            )
            all_weights["weights_lumi"] = lumi_weights

    # Only save ensemble weights for specific group names
    if gn in ["nominal", "dbootstrap", "mcbootstrap"]:
        # Loop over pulled weights and add each to the all_weights dictionary
        bootstrap_stats = []
        for i, weight in enumerate(pulled_weights):
            weight *= nominal_root_weights
            # If this is the data bootstraps need to get the number of data events
            # for this bootstrap run
            if gn == "dbootstrap":
                # Get the run name for this bootstrap
                n_data = get_bs_n_data(args.campaign_path, run_names[i], data_pass200)
            else:
                n_data = n_data_nominal
            weight = norm_weights(
                weight, truth_pass200, nominal_factor, n_data, args.luminosity
            )
            # Add weights to the all_weights dictionary
            write_name = group_name_to_write_name(gn, i)
            all_weights[write_name] = weight

# Add in dd-target weights, note this are already multiplied by the nominal root weights
# Use the nominal fiducial factor
dd_target_weights = norm_weights(
    dd_target_weights, truth_pass200, nominal_factor, n_data_nominal, args.luminosity
)
all_weights["target_dd"] = dd_target_weights

# Now handle the HV weights
hv_weights = {}
if "hv" in args.group_names:
    pulled_weights, _ = pull_weights(
        args.campaign_path,
        "hv-data" if args.use_data else "hv",
        args.iteration,
    )
    print(f"Got {len(pulled_weights)} weights for group hv")
    # Calculate the central value weights
    central_weights = np.mean(pulled_weights.clip(min=0, max=100), axis=0)
    alt_root_weights = get_prior_weights("hv", hv_prior_weights)
    central_weights *= alt_root_weights
    # Normalize the HV weights
    ratio_hv = get_truth_to_reco_ratio(
        "hv", t_hv, hv_prior_weights, hv_pass200, hv_truth_pass200
    )
    central_weights = norm_weights(
        central_weights, hv_truth_pass200, ratio_hv, n_data_nominal, args.luminosity
    )
    hv_weights["weights_hv"] = central_weights

# If OG order is set, re-order the weights to match the original order
# Note the non-DY events are appended to the end in both cases
if args.og_order:
    og_indices = np.load(
        "/pscratch/sd/k/kgreif/zjets_plot_staging/unshuffle_indices.npy"
    )
    hv_og_indices = np.load(
        "/pscratch/sd/k/kgreif/zjets_plot_staging/unshuffle_indices_hv.npy"
    )
    all_weights = {key: all_weights[key][og_indices] for key in all_weights}
    hv_weights = {key: hv_weights[key][hv_og_indices] for key in hv_weights}

# Create output directory if it doesn't exist
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Save results in HDF5 format compatible with pd.read_hdf()
# Save non-hv weights first (creates the file with mode='w')
df_mc = pd.DataFrame(all_weights)
df_mc.to_hdf(args.output, key="weights", mode="w", format="table")

df_hv = pd.DataFrame(hv_weights)
df_hv.to_hdf(args.output, key="hv_weights", mode="a", format="table")

# Also save results in .npz format (UNCOMPRESSED for compatibility with cnpy)
# Generate .npz filename by replacing HDF5 extension
npz_output = os.path.splitext(args.output)[0] + ".npz"
# Save all weights (MC and HV) to a single .npz file
# Use uncompressed format to avoid issues with cnpy library
# Save to temporary .npz first, then re-save uncompressed
with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
    tmp_npz = tmp.name
    np.savez(tmp_npz, **all_weights, **hv_weights)

# Re-save as uncompressed zip file
with zipfile.ZipFile(tmp_npz, "r") as z_in:
    with zipfile.ZipFile(npz_output, "w", compression=zipfile.ZIP_STORED) as z_out:
        for item in z_in.infolist():
            # Copy each file without compression
            data = z_in.read(item.filename)
            z_out.writestr(item, data, compress_type=zipfile.ZIP_STORED)

# Clean up temporary file
os.remove(tmp_npz)

print(f"Saved weights to {npz_output} (uncompressed format)")
