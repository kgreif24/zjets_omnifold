"""multifold_compare.py - This script generates plots of the multifold
measurement for the purpose of comparing to the omnifold result.
It will generate a plot for each of the 24 multifold observables,
in the style of the omnifold plots.

Author: Kevin Greif
Last updated 06/26/2025
python3
"""

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


# Load the multifold data
multifold = pd.read_hdf(
    "/global/cfs/cdirs/m3246/ZjetOmnifold/data/multifold/pseudodata/multifold.h5"
)
target = pd.read_hdf(
    "/global/cfs/cdirs/m3246/ZjetOmnifold/data/multifold/pseudodata/target.h5"
)
hv = pd.read_hdf(
    "/global/cfs/cdirs/m3246/ZjetOmnifold/data/multifold/pseudodata/multifold_sherpa.h5"
)

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
target_weights = target["weight_mc"]
sherpa_weights = hv["weights_nominal"]

# Set the relevant systs
track_systs = ["weights_trackEffMain"]
data_stat_systs = [
    col for col in multifold.keys() if col.startswith("weights_bootstrap_data")
]
ensemble_systs = [col for col in multifold.keys() if col.startswith("weights_ensemble")]

# Loop through the observables
for obs_dict in plots:

    key = obs_dict["key"]
    print("Plotting observable", key)

    # Calculate all histograms
    bins = np.array(obs_dict["ibubins"])
    bin_centers = (bins[1:] + bins[:-1]) / 2
    source_hist, _ = np.histogram(
        multifold[obs_dict["key"]], bins=bins, weights=central_weights
    )
    mc_var, _ = np.histogram(
        multifold[obs_dict["key"]], bins=bins, weights=central_weights**2
    )
    target_hist, _ = np.histogram(
        target[obs_dict["key"]], bins=bins, weights=target_weights
    )
    hv_hist, _ = np.histogram(hv[obs_dict["key"]], bins=bins, weights=sherpa_weights)

    # Calculate the uncertainties
    track_eff = calculate_uncertainty(multifold, key, bins, track_systs)
    nn_init = calculate_stat_uncertainty(multifold, key, bins, ensemble_systs)
    hidden_variable = np.sqrt((hv_hist - source_hist) ** 2) / source_hist
    data_stat = calculate_stat_uncertainty(multifold, key, bins, data_stat_systs)
    mc_stat = np.sqrt(mc_var) / source_hist
    mbias = np.abs(source_hist - target_hist) / target_hist

    # Calculate the total uncertainty
    total_uncert = np.sqrt(
        track_eff**2 + nn_init**2 + data_stat**2 + mc_stat**2 + hidden_variable**2
    )

    # Calculate the ratio and uncertainty on the ratio
    ratio = source_hist / target_hist
    rel_total_uncert = total_uncert / target_hist

    # Make density plot
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
    ax.errorbar(
        bin_centers,
        source_hist,
        yerr=total_uncert,
        fmt="o",
        label="Multifold",
        color="green",
    )
    if not obs_dict["linear_yscale"]:
        ax.set_yscale("log")
    if obs_dict["log_xscale"]:
        ax.set_xscale("log")
    ax.tick_params(axis="x", direction="in", top=True)
    ax.set_ylabel("Counts")
    ax.legend()
    rax.axhline(1, color="black", linestyle="--")
    rax.errorbar(
        bin_centers,
        ratio,
        yerr=rel_total_uncert,
        fmt="o",
        label="Multifold",
        color="green",
    )
    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.5, 1.0, 1.5])
    rax.set_ylabel("Ratio to target")
    rax.set_xlabel(obs_dict["xlabel"])
    rax.tick_params(axis="x", direction="in", top=True)
    fig1.tight_layout()
    fig1.savefig(f"./plot_storage/multifold/{key}.pdf")
    plt.close()

    # Make uncertainty budget plot
    plot_total_uncert = np.append(total_uncert, total_uncert[-1])
    plot_systs = {
        "nn-init": {
            "name": "NN Init",
            "color": "blue",
            "vals": np.append(nn_init, nn_init[-1]),
        },
        "track-eff": {
            "name": "Track eff.",
            "color": "purple",
            "vals": np.append(track_eff, track_eff[-1]),
        },
        "mc-stat": {
            "name": "MC stat",
            "color": "green",
            "vals": np.append(mc_stat, mc_stat[-1]),
        },
        "hidden-variable": {
            "name": "Hidden variable",
            "color": "orange",
            "vals": np.append(hidden_variable, hidden_variable[-1]),
        },
        "data-stat": {
            "name": "Data stat",
            "color": "aqua",
            "vals": np.append(data_stat, data_stat[-1]),
        },
        "mbias": {
            "name": "Bias",
            "color": "red",
            "vals": np.append(mbias, mbias[-1]),
        },
    }

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(
        bins,
        plot_total_uncert,
        "--",
        color="black",
        label="Total unc.",
        drawstyle="steps-post",
        linewidth=2,
    )
    for plot_dict in plot_systs.values():
        ax.plot(
            bins,
            plot_dict["vals"],
            "-",
            color=plot_dict["color"],
            label=plot_dict["name"],
            drawstyle="steps-post",
        )
        if plot_dict["name"] == "Bias":
            ax.fill_between(
                bins, 0, plot_dict["vals"], step="post", color="gray", alpha=0.3
            )
    ax.set_ylabel("Uncertainty")
    ax.set_xlabel(obs_dict["xlabel"])
    # ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout()
    fig.savefig(f"./plot_storage/multifold/{key}_uncert_budget.pdf")
    plt.close()
