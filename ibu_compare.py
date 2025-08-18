""" ibu_compare.py - This script generates plots of the ibu measurement
for the purpose of comparing to the omnifold result.
It will generate a plot for each of the 24 multifold observables,
plus Ntracks and HT_tracks, in the style of the omnifold plots.

Author: Kevin Greif
Last updated 07/01/2025
python3
"""

import yaml
import numpy as np
import uproot
import matplotlib.pyplot as plt


# Load the IBU data
ibu = uproot.open(
    "/global/cfs/cdirs/m3246/ZjetOmnifold/data/ibu/"
    "unfoldingPlots11Jun2025Pseudodata_UnfoldedHists.root"
)
target = uproot.open(
    "/global/cfs/cdirs/m3246/ZjetOmnifold/data/ibu/"
    "Histograms_TruthPseudodata_Sherpa2211DY_Dibo_EW_SherpaTop_"
    "PosWeights_WithTracks.root"
)

# Load plotting config
with open("./utils/plots_config.yml", "r") as stream:
    config = yaml.safe_load(stream)
plots = [
    config["plots"][plot]
    for plot in config["plots"]
    if config["plots"][plot]["ibu"]
]

# Loop through the observables
for obs_dict in plots:

    key = obs_dict["key"]
    print("Plotting observable", key)

    # Get the unfolded and target histograms
    ibu_prekey = "PseudodataSRewUnfoldWMGPy8FxFxRewPlusNS_"
    target_prekey = "Total_truth_"
    ibu_hist, bins = ibu[ibu_prekey + key].to_numpy()
    target_hist, _ = target[target_prekey + key].to_numpy()

    # Normalize the target histogram to ibu
    target_hist *= ibu_hist.sum() / target_hist.sum()

    # Get the method bias
    rel_mbias = np.abs(ibu_hist / target_hist - 1)

    # Make density plot
    bin_centers = (bins[1:] + bins[:-1]) / 2
    plot_target_hist = np.append(target_hist, target_hist[-1])
    fig1, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(6, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig1.subplots_adjust(hspace=0, top=0.95)
    ax.plot(
        bins,
        plot_target_hist,
        "--",
        label="Target",
        color="black",
        drawstyle="steps-post",
    )
    ax.plot(
        bin_centers,
        ibu_hist,
        "o",
        label="IBU",
        color="red",
    )
    if not obs_dict["linear_yscale"]:
        ax.set_yscale("log")
    if obs_dict["log_xscale"]:
        ax.set_xscale("log")
    ax.tick_params(axis="x", direction="in", top=True)
    ax.set_ylabel("Counts")
    ax.legend()
    rax.axhline(1, color="black", linestyle="--")
    rax.plot(
        bin_centers,
        ibu_hist / target_hist,
        "o",
        label="IBU",
        color="red",
    )
    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.5, 1.0, 1.5])
    rax.set_ylabel("Ratio to target")
    rax.set_xlabel(obs_dict["xlabel"])
    rax.tick_params(axis="x", direction="in", top=True)
    fig1.tight_layout()
    fig1.savefig(f"./plot_storage/ibu/{key}.pdf")
    plt.close()

    # Make uncertainty budget plot
    plot_mbias = np.append(rel_mbias, rel_mbias[-1])
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(
        bins,
        plot_mbias,
        "-",
        color="red",
        label="Method bias",
        drawstyle="steps-post",
        linewidth=2,
    )
    ax.set_ylabel("Fractional method bias")
    ax.set_xlabel(obs_dict["xlabel"])
    # ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout()
    fig.savefig(f"./plot_storage/ibu/{key}_uncert_budget.pdf")
    plt.close()
