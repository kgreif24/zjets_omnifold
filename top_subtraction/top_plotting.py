#!/usr/bin/env python3
"""
Top subtraction plotting script.

This script generates plots comparing pseudodata, pseudodata minus top background,
and reweighted pseudodata for various observables.

Usage:
    python top_plotting.py --weight_file <weight_file> --data_file <data_file>

"""

import argparse
import sys
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_data(data_file):
    """Load pseudodata and top background data from ROOT files."""
    print("Loading pseudodata...")
    f_pd = uproot.open(data_file)
    t_pd = f_pd["OmniTree"]

    pass190_pd = ak.to_numpy(t_pd["pass190"].array())
    logit_pd = ak.to_numpy(t_pd["isTop_logit"].array())

    logit_pd = logit_pd[pass190_pd == 1]

    print("Loading top background...")
    f_top = uproot.open(
        "/pscratch/sd/k/kgreif/data/ZjetOmnifold_14May2025_Background"
        "_Sherpa2212_AllTop_WithTracks_slim_Systematics_topLogit.root"
    )
    t_top = f_top["OmniTree"]

    pass190_top = ak.to_numpy(t_top["pass190"].array())
    logit_top = ak.to_numpy(t_top["isTop_logit"].array())
    weight_top = ak.to_numpy(t_top["weight"].array())

    logit_top = logit_top[pass190_top == 1]
    weight_top = weight_top[pass190_top == 1]

    return t_pd, t_top, logit_pd, logit_top, weight_top, pass190_top


def load_weights(weight_file):
    """Load reweighting weights from NPZ file."""
    print(f"Loading weights from {weight_file}...")
    new_weights_file = np.load(weight_file)
    if "weight" in new_weights_file.files:
        new_weights = new_weights_file["weight"]
    else:
        new_weights = new_weights_file["pd_weights"]

    print(f"Mean of new weights: {np.mean(new_weights)}")
    print(f"Std of new weights: {np.std(new_weights)}")

    return new_weights


def plot_logit_comparison(logit_pd, logit_top, pdf):
    """Create logit comparison plot."""
    print("Creating logit comparison plot...")

    max_logit = max(logit_pd.max(), logit_top.max())
    min_logit = min(logit_pd.min(), logit_top.min())
    bins = np.linspace(min_logit, max_logit, 100)
    h_logit_pd, _ = np.histogram(logit_pd, bins=bins, density=True)
    h_logit_top, _ = np.histogram(logit_top, bins=bins, density=True)

    plot_h_logit_pd = np.concatenate([h_logit_pd, h_logit_pd[-1:]])
    plot_h_logit_top = np.concatenate([h_logit_top, h_logit_top[-1:]])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(
        bins, plot_h_logit_pd, drawstyle="steps-post", label=r"$Z$+jets", color="black"
    )
    ax.plot(
        bins,
        plot_h_logit_top,
        drawstyle="steps-post",
        label=r"$t\bar{t}$",
        color="green",
    )
    ax.set_yscale("log")
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax.set_ylabel("Density")
    ax.set_xlabel("Top Classifier Logit")
    ax.legend(frameon=False, loc="lower center")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Saved logit comparison plot")


def plot_logit_subtraction(logit_pd, logit_top, weight_top, new_weights, pdf):
    """Create logit subtraction plot with ratio."""
    print("Creating logit subtraction plot...")

    logit_pd_minus_top = np.concatenate([logit_pd, logit_top])
    weights_pd_minus_top = np.concatenate([np.ones(len(logit_pd)), -1 * weight_top])

    bins = np.linspace(logit_pd.min(), logit_pd.max(), 100)
    h_logit_pd, _ = np.histogram(logit_pd, bins=bins, density=True)
    h_logit_pd_minus_top, _ = np.histogram(
        logit_pd_minus_top, bins=bins, density=True, weights=weights_pd_minus_top
    )
    h_logit_pd_rw, _ = np.histogram(
        logit_pd, bins=bins, density=True, weights=new_weights
    )

    plot_h_logit_pd = np.concatenate([h_logit_pd, h_logit_pd[-1:]])
    plot_h_logit_pd_minus_top = np.concatenate(
        [h_logit_pd_minus_top, h_logit_pd_minus_top[-1:]]
    )
    plot_h_logit_pd_rw = np.concatenate([h_logit_pd_rw, h_logit_pd_rw[-1:]])

    # Create figure with two subplots: main plot and ratio pad
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(6.4, 4.8), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0}
    )

    # Main histogram plot
    ax_main.plot(bins, plot_h_logit_pd, drawstyle="steps-post", label="PD")
    ax_main.plot(
        bins, plot_h_logit_pd_minus_top, drawstyle="steps-post", label="PD - Top"
    )
    ax_main.plot(
        bins, plot_h_logit_pd_rw, drawstyle="steps-post", label="PD (subtracted)"
    )
    ax_main.set_yscale("log")
    ax_main.legend(frameon=False)
    ax_main.set_ylabel("Density")

    # Remove x-axis labels from main plot
    ax_main.set_xticklabels([])

    # Calculate ratio (avoid division by zero)
    ratio_pre = np.divide(
        plot_h_logit_pd,
        plot_h_logit_pd_minus_top,
        out=np.zeros_like(plot_h_logit_pd_minus_top),
        where=plot_h_logit_pd_minus_top != 0,
    )
    ratio_post = np.divide(
        plot_h_logit_pd_rw,
        plot_h_logit_pd_minus_top,
        out=np.zeros_like(plot_h_logit_pd_minus_top),
        where=plot_h_logit_pd_minus_top != 0,
    )

    # Ratio pad
    ax_ratio.plot(
        bins,
        ratio_pre,
        drawstyle="steps-post",
        color="red",
        linewidth=1.5,
        label="Pre-subtraction",
    )
    ax_ratio.plot(
        bins,
        ratio_post,
        drawstyle="steps-post",
        color="blue",
        linewidth=1.5,
        label="Post-subtraction",
    )
    ax_ratio.axhline(y=1, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax_ratio.set_ylabel("Ratio to PD - Top")
    ax_ratio.set_xlabel("Logit")
    ax_ratio.legend(frameon=False)

    # Set reasonable y-limits for ratio (adjust as needed)
    ratio_nonzero_pre = ratio_pre[ratio_pre != 0]
    if len(ratio_nonzero_pre) > 0:
        ratio_min_pre, ratio_max_pre = np.percentile(ratio_nonzero_pre, [5, 95])
        ax_ratio.set_ylim(max(0, ratio_min_pre * 0.95), ratio_max_pre * 1.05)
    ratio_nonzero_post = ratio_post[ratio_post != 0]
    if len(ratio_nonzero_post) > 0:
        ratio_min_post, ratio_max_post = np.percentile(ratio_nonzero_post, [5, 95])
        ax_ratio.set_ylim(max(0, ratio_min_post * 0.95), ratio_max_post * 1.05)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Saved logit subtraction plot")


def plot_weight_histogram(new_weights, pdf):
    """Create histogram of the reweighting weights."""
    print("Creating weight histogram...")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Create histogram
    bins = np.linspace(0.9, 1.0, 100)
    ax.hist(
        new_weights,
        bins=bins,
        density=True,
        alpha=0.7,
        color="green",
    )
    ax.set_yscale("log")

    # Add statistics text
    mean_weight = np.mean(new_weights)
    std_weight = np.std(new_weights)
    min_weight = np.min(new_weights)
    max_weight = np.max(new_weights)

    stats_text = (
        f"Mean: {mean_weight:.4f}\nStd: {std_weight:.4f}\n"
        f"Min: {min_weight:.4f}\nMax: {max_weight:.4f}"
    )
    ax.text(
        0.2,
        0.85,
        stats_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
        fontsize=10,
    )

    ax.set_xlabel("Top subtraction weights")
    ax.set_ylabel("Density")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Saved weight histogram")


def plot_observables(t_pd, t_top, pass190_top, weight_top, new_weights, pdf):
    """Create plots for all observables."""
    plot_obs = [
        "Ntracks",
        "HT_tracks",
        "pT_ll",
        "pT_trackj1",
        "pT_trackj2",
        "y_ll",
        "m_trackj1",
        "m_trackj2",
        "tau1_trackj1",
        "tau2_trackj1",
        "tau1_trackj2",
        "tau2_trackj2",
    ]

    for obs in plot_obs:
        print(f"Creating {obs} plot...")

        obs_pd = ak.to_numpy(t_pd[obs].array())
        obs_top = ak.to_numpy(t_top[obs].array())
        obs_top = obs_top[pass190_top == 1]

        obs_pd_minus_top = np.concatenate([obs_pd, obs_top])
        weights_pd_minus_top = np.concatenate([np.ones(len(obs_pd)), -1 * weight_top])

        # Set bin ranges based on observable
        if obs == "HT_tracks":
            max_obs = 2000
        elif obs == "pT_trackj1":
            min_obs = 0
            max_obs = 1000
        elif obs == "pT_trackj2":
            min_obs = 0
            max_obs = 500
        elif obs == "m_trackj1":
            min_obs = 0
            max_obs = 150
        elif obs == "m_trackj2":
            min_obs = 0
            max_obs = 150
        elif "tau" in obs:
            min_obs = 0
            max_obs = 0.9
        else:
            min_obs = min(obs_pd_minus_top.min(), obs_top.min())
            max_obs = max(obs_pd_minus_top.max(), obs_top.max())

        bins = np.linspace(min_obs, max_obs, 100)
        if obs == "Ntracks":
            bins = np.arange(0, 250, 1)

        h_obs_pd, _ = np.histogram(obs_pd, bins=bins, density=True)
        h_obs_pd_minus_top, _ = np.histogram(
            obs_pd_minus_top, bins=bins, density=True, weights=weights_pd_minus_top
        )
        h_obs_pd_rw, _ = np.histogram(
            obs_pd, bins=bins, density=True, weights=new_weights
        )

        plot_h_obs_pd = np.concatenate([h_obs_pd, h_obs_pd[-1:]])
        plot_h_obs_pd_minus_top = np.concatenate(
            [h_obs_pd_minus_top, h_obs_pd_minus_top[-1:]]
        )
        plot_h_obs_pd_rw = np.concatenate([h_obs_pd_rw, h_obs_pd_rw[-1:]])

        # Make ratio plot
        fig, (ax_main, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(6.4, 4.8),
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0},
        )

        # Main histogram plot
        ax_main.plot(bins, plot_h_obs_pd, drawstyle="steps-post", label="PD")
        ax_main.plot(
            bins, plot_h_obs_pd_minus_top, drawstyle="steps-post", label="PD - Top"
        )
        ax_main.plot(
            bins, plot_h_obs_pd_rw, drawstyle="steps-post", label="PD (subtracted)"
        )
        ax_main.set_yscale("log")
        ax_main.legend(frameon=False)
        ax_main.set_ylabel("Density")

        # Remove x-axis labels from main plot
        ax_main.set_xticklabels([])

        # Calculate ratio (avoid division by zero)
        ratio_pre = np.divide(
            plot_h_obs_pd,
            plot_h_obs_pd_minus_top,
            out=np.zeros_like(plot_h_obs_pd_minus_top),
            where=plot_h_obs_pd_minus_top != 0,
        )
        ratio_post = np.divide(
            plot_h_obs_pd_rw,
            plot_h_obs_pd_minus_top,
            out=np.zeros_like(plot_h_obs_pd_minus_top),
            where=plot_h_obs_pd_minus_top != 0,
        )

        # Ratio pad
        ax_ratio.plot(
            bins,
            ratio_pre,
            drawstyle="steps-post",
            color="red",
            linewidth=1.5,
            label="Pre-subtraction",
        )
        ax_ratio.plot(
            bins,
            ratio_post,
            drawstyle="steps-post",
            color="blue",
            linewidth=1.5,
            label="Post-subtraction",
        )
        ax_ratio.axhline(y=1, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax_ratio.set_ylabel("Ratio to PD - Top")
        ax_ratio.set_xlabel(obs)
        ax_ratio.legend(frameon=False)

        # Set reasonable y-limits for ratio (adjust as needed)
        ratio_nonzero_pre = ratio_pre[ratio_pre != 0]
        if len(ratio_nonzero_pre) > 0:
            ratio_min_pre, ratio_max_pre = np.percentile(ratio_nonzero_pre, [5, 95])
            ax_ratio.set_ylim(max(0, ratio_min_pre * 0.95), ratio_max_pre * 1.05)
        ratio_nonzero_post = ratio_post[ratio_post != 0]
        if len(ratio_nonzero_post) > 0:
            ratio_min_post, ratio_max_post = np.percentile(ratio_nonzero_post, [5, 95])
            ax_ratio.set_ylim(max(0, ratio_min_post * 0.95), ratio_max_post * 1.05)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {obs} plot")


def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Generate top subtraction plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_file",
        help="Path to the ROOT file containing the data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--weight_file",
        help="Path to the NPZ file containing reweighting weights",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        default="top_subtraction_plots.pdf",
        help="Output PDF filename (default: top_subtraction_plots.pdf)",
    )

    args = parser.parse_args()

    try:
        # Load data
        t_pd, t_top, logit_pd, logit_top, weight_top, pass190_top = load_data(
            args.data_file
        )

        # Load weights
        new_weights = load_weights(args.weight_file)

        # Create PDF file
        print(f"Creating PDF: {args.output}")
        with PdfPages(args.output) as pdf:
            # Plot 1: Weight histogram
            plot_weight_histogram(new_weights, pdf)

            # Plot 2: Logit comparison
            plot_logit_comparison(logit_pd, logit_top, pdf)

            # Plot 3: Logit subtraction with ratio
            plot_logit_subtraction(logit_pd, logit_top, weight_top, new_weights, pdf)

            # Plots 4-15: Observable plots
            plot_observables(t_pd, t_top, pass190_top, weight_top, new_weights, pdf)

        print(f"All plots saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
