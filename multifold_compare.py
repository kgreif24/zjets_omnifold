"""multifold_compare.py - This script generates plots of the multifold
measurement for the purpose of comparing to the omnifold result.
It will generate a plot for each of the 24 multifold observables,
in the style of the omnifold plots.

Author: Kevin Greif
Last updated 10/16/2025
python3
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# returns the systematic uncertainty in bins of an observable in percent
def calculate_uncertainty(df, observable, bins, systs):
    total_uncert = np.zeros(len(bins) - 1)
    nom, _ = np.histogram(df[observable], bins=bins, weights=df.weights_nominal)
    for syst_name in systs:
        syst, _ = np.histogram(df[observable], bins=bins, weights=df[syst_name])
        total_uncert += (syst - nom) ** 2
    final_uncert = np.sqrt(total_uncert) / nom
    return final_uncert


# Returns stochastic uncertainty based on boostrap variations ('resampling')
def calculate_stat_uncertainty(df, observable, bins, bs_vars):
    stat = []
    nom, _ = np.histogram(df[observable], bins=bins, weights=df.weights_nominal)
    for bs_name in bs_vars:
        varHist, _ = np.histogram(df[observable], bins=bins, weights=df[bs_name])
        stat.append(varHist)
    final_stat = np.std(stat, axis=0) / nom
    if bs_vars[0].startswith("weights_ensemble"):  # NN initialization
        # Since we take the median, we need the std err on the median,
        # which is 1.253*std err on the mean
        return 1.253 * final_stat
    else:
        return final_stat


parser = argparse.ArgumentParser(description="Run plotting functions")
parser.add_argument("--data", action="store_true", help="If true, will plot data")
args = parser.parse_args()

# Load the multifold data
key = "data" if args.data else "pseudodata"
multifold = pd.read_hdf(
    "/pscratch/sd/k/kgreif/multifold/data/multifold.h5"
    # f"/pscratch/sd/k/kgreif/multifold/{key}/multifold.h5"
)
hv = pd.read_hdf(f"/pscratch/sd/k/kgreif/multifold/{key}/multifold_sherpa.h5")
if not args.data:
    target = pd.read_hdf("/pscratch/sd/k/kgreif/multifold/pseudodata/target.h5")
else:
    mc_preds = np.load(
        "/pscratch/sd/k/kgreif/multifold/data/mc_preds.npy",
        allow_pickle=True,
    )
    # Create a mapping from observable names to mc_preds indices
    mc_preds_mapping = {item["file_label"]: i for i, item in enumerate(mc_preds)}

# Load plotting config
with open("./utils/plots_config.yml", "r") as stream:
    config = yaml.safe_load(stream)
plots = [
    config["plots"][plot]
    for plot in config["plots"]
    if config["plots"][plot]["multifold"]
]

# Load the relevant weights
central_weights = multifold["weights_nominal"]
sherpa_weights = hv["weights_nominal"]
if not args.data:
    target_weights = target["weight_mc"]

# Set the relevant systs
track_systs = ["weights_trackEffMain", "weights_trackEffJet"]
data_stat_systs = [
    col for col in multifold.keys() if col.startswith("weights_bootstrap_data")
]
ensemble_systs = [col for col in multifold.keys() if col.startswith("weights_ensemble")]

# Loop through the observables
for obs_dict in plots:

    key = obs_dict["key"]
    print("Plotting observable", key)

    # Calculate the bins
    bins = np.array(obs_dict["ibubins"])

    # For data mode, check if binning matches mc_preds and use mc_preds bins
    if args.data and key in mc_preds_mapping:
        mc_preds_bins = mc_preds[mc_preds_mapping[key]]["bins"]
        if not np.array_equal(bins, mc_preds_bins):
            print(f"WARNING: Binning mismatch for {key} in data mode!")
            print(f"  Config bins: {bins}")
            print(f"  mc_preds bins: {mc_preds_bins}")
            print("  Using mc_preds bins for data comparison")
            bins = mc_preds_bins

    # Get bin properties
    bin_centers = (bins[1:] + bins[:-1]) / 2
    bin_errors = (bins[1:] - bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    # Calculate the histograms
    source_hist, _ = np.histogram(
        multifold[obs_dict["key"]],
        bins=bins,
        weights=central_weights,
    )
    source_density, _ = np.histogram(
        multifold[obs_dict["key"]], bins=bins, weights=central_weights, density=True
    )
    source_density *= np.sum(central_weights)
    mc_var, _ = np.histogram(
        multifold[obs_dict["key"]], bins=bins, weights=central_weights**2
    )
    hv_hist, _ = np.histogram(hv[obs_dict["key"]], bins=bins, weights=sherpa_weights)
    dd_hist, _ = np.histogram(
        multifold[obs_dict["key"]], bins=bins, weights=multifold["weights_dd"]
    )
    dd_target_hist, _ = np.histogram(
        multifold[obs_dict["key"]], bins=bins, weights=multifold["target_dd"]
    )
    if not args.data:
        target_hist, _ = np.histogram(
            target[obs_dict["key"]], bins=bins, weights=target_weights
        )

    # Calculate the uncertainties
    tracking = calculate_uncertainty(multifold, key, bins, track_systs)
    nn_init = calculate_stat_uncertainty(
        multifold, key, bins, ensemble_systs
    ) / np.sqrt(len(ensemble_systs))
    hidden_variable = np.sqrt((hv_hist - source_hist) ** 2) / source_hist
    data_driven = np.sqrt((dd_hist - dd_target_hist) ** 2) / dd_target_hist
    data_stat = calculate_stat_uncertainty(multifold, key, bins, data_stat_systs)
    mc_stat = np.sqrt(mc_var) / source_hist

    if not args.data:
        # In pseudodata mode: calculate bias between unfolded data and truth
        mbias = np.abs(source_hist - target_hist) / target_hist
    else:
        # In data mode: no bias calculation since we're comparing to generators
        mbias = np.zeros_like(source_hist)

    # Calculate merged uncertainties
    unfolding_uncert = np.sqrt(hidden_variable**2 + data_driven**2)

    # Calculate the total uncertainty
    total_uncert = np.sqrt(
        tracking**2 + nn_init**2 + data_stat**2 + mc_stat**2 + unfolding_uncert**2
    )

    # Calculate the ratio and uncertainty on the ratio
    if not args.data:
        # In pseudodata mode: ratio of data to target (original behavior)
        ratio = source_hist / target_hist
        ratio_uncert = total_uncert * ratio

    # Make density plot
    fig1, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(6, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0},
    )

    if args.data:
        # Define the luminosity for the MG and Sherpa generators
        lumi = 138.96  # fb^-1
        # In data mode: show both generators
        counts_mg = mc_preds[mc_preds_mapping[key]]["mgfxfx_counts"]
        density_mg = counts_mg / lumi / bin_widths
        error_mg = mc_preds[mc_preds_mapping[key]]["mgfxfx_err"] / counts_mg
        ax.errorbar(
            bin_centers,
            density_mg,
            xerr=bin_errors,
            yerr=density_mg * error_mg,
            fmt="+",
            label="MGFxFx",
            color="aqua",
            linewidth=2,
        )
        # Add Sherpa generator
        if key in mc_preds_mapping:
            counts_sherpa = mc_preds[mc_preds_mapping[key]]["sherpa_counts"]
            density_sherpa = counts_sherpa / lumi / bin_widths
            error_sherpa = mc_preds[mc_preds_mapping[key]]["sherpa_err"] / counts_sherpa
            ax.errorbar(
                bin_centers,
                density_sherpa,
                xerr=bin_errors,
                yerr=density_sherpa * error_sherpa,
                fmt="+",
                label="Sherpa",
                color="purple",
                linewidth=2,
            )
    else:
        # In pseudodata mode: show target
        plot_target_hist = np.append(target_hist, target_hist[-1])
        ax.plot(
            bins,
            plot_target_hist,
            "--",
            label="Target",
            color="black",
            drawstyle="steps-post",
        )

    ax.errorbar(
        bin_centers,
        source_density,
        yerr=total_uncert * source_density,
        xerr=bin_errors,
        fmt="o" if args.data else "+",
        label="Multifold",
        color="green",
        linewidth=2,
    )
    if not obs_dict["linear_yscale"]:
        ax.set_yscale("log")
    if obs_dict["log_xscale"]:
        ax.set_xscale("log")
    ax.tick_params(axis="x", direction="in", top=True)
    ax.set_ylabel("Counts")
    ax.legend(frameon=False)
    rax.axhline(1, color="black", linestyle="--")

    if args.data:
        # In data mode: show ratios of both generators to data
        mg_ratio = density_mg / source_density
        mg_ratio_uncert = mg_ratio * error_mg
        rax.errorbar(
            bin_centers,
            mg_ratio,
            xerr=bin_errors,
            yerr=mg_ratio_uncert,
            fmt="+",
            label="MG5aMC@NLO/Data",
            color="aqua",
            linewidth=2,
        )
        # Add Sherpa ratio
        if key in mc_preds_mapping:
            sherpa_ratio = density_sherpa / source_density
            sherpa_ratio_uncert = sherpa_ratio * error_sherpa
            rax.errorbar(
                bin_centers,
                sherpa_ratio,
                yerr=sherpa_ratio_uncert,
                xerr=bin_errors,
                fmt="+",
                label="Sherpa/Data",
                color="purple",
                linewidth=2,
            )
        rax.set_ylabel("Generator/Data")
    else:
        # In pseudodata mode: show ratio of data to target
        rax.errorbar(
            bin_centers,
            ratio,
            xerr=bin_errors,
            yerr=ratio_uncert,
            fmt="+",
            label="Multifold",
            color="green",
            linewidth=2,
        )
        rax.set_ylabel("Ratio to target")

    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.5, 1.0, 1.5])
    rax.set_xlabel(obs_dict["xlabel"])
    rax.tick_params(axis="x", direction="in", top=True)
    fig1.tight_layout()
    fig1.savefig(f"./plot_storage/multifold/{'data' if args.data else 'pd'}/{key}.pdf")
    plt.close()

    # Make uncertainty budget plot
    plot_systs = {
        "nn-init": {
            "name": "NN Init",
            "color": "aqua",
            "vals": np.append(nn_init, nn_init[-1]),
        },
        "tracking": {
            "name": "Tracking",
            "color": "purple",
            "vals": np.append(tracking, tracking[-1]),
        },
        "mc-stat": {
            "name": "MC stat",
            "color": "green",
            "vals": np.append(mc_stat, mc_stat[-1]),
        },
        "unfolding": {
            "name": "Unfolding",
            "color": "red",
            "vals": np.append(unfolding_uncert, unfolding_uncert[-1]),
        },
        "data-stat": {
            "name": "Data stat",
            "color": "blue",
            "vals": np.append(data_stat, data_stat[-1]),
        },
        "mbias": {
            "name": "Bias",
            "color": "gray",
            "vals": np.append(mbias, mbias[-1]),
        },
    }

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(
        bins,
        np.append(total_uncert, total_uncert[-1]),
        "--",
        color="black",
        label="Total unc.",
        drawstyle="steps-post",
        linewidth=2,
    )
    for plot_dict in plot_systs.values():
        if plot_dict["name"] == "Bias":
            ax.fill_between(
                bins,
                0,
                plot_dict["vals"],
                step="post",
                color="gray",
                alpha=0.3,
                label="Method bias",
            )
        else:
            ax.plot(
                bins,
                plot_dict["vals"],
                "-",
                color=plot_dict["color"],
                label=plot_dict["name"],
                drawstyle="steps-post",
            )
    ax.set_ylabel("Uncertainty")
    ax.set_xlabel(obs_dict["xlabel"])
    # ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout()
    fig.savefig(
        f"./plot_storage/multifold/{'data' if args.data else 'pd'}"
        f"/{key}_uncert_budget.pdf"
    )
    plt.close()
