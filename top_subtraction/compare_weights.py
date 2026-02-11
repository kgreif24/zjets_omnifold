#!/usr/bin/env python3
"""
Weight comparison plotting script.

This script generates plots comparing events from a single ROOT file
with two different sets of event weights applied.

Usage:
    python compare_weights.py --data_file <data_file> --weights1 <weights1.npz>
    --weights2 <weights2.npz> --output <output.pdf>
"""

import argparse
import sys
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_data(data_file, load_target=True):
    """Load data from ROOT file."""
    print(f"Loading data from {data_file}...")
    f = uproot.open(data_file)
    t = f["OmniTree"]

    pass190 = ak.to_numpy(t["pass190"].array())
    logit = ak.to_numpy(t["isTop_logit"].array())

    # Apply pass190 selection
    logit = logit[pass190 == 1]

    print(f"Total events: {len(pass190)}")
    print(f"Events passing pass190: {sum(pass190 == 1)}")

    if load_target:
        isTop = ak.to_numpy(t["isTop"].array())
        isTop = isTop[pass190 == 1]
        print(f"  Signal events (isTop == 0): {sum(isTop == 0)}")
        print(f"  Top events (isTop == 1): {sum(isTop == 1)}")
    else:
        isTop = None
        print("  Target histogram disabled (no isTop branch)")

    return t, logit, pass190, isTop


def load_weights(weight_file):
    """Load weights from NPZ file."""
    print(f"Loading weights from {weight_file}...")
    weights_data = np.load(weight_file)
    if "weight" in weights_data.files:
        weights = weights_data["weight"]
    elif "pd_weights" in weights_data.files:
        weights = weights_data["pd_weights"]
    else:
        raise ValueError(
            f"Could not find weights in {weight_file}. Keys: {weights_data.files}"
        )

    print(f"  Loaded {len(weights)} weights")
    print(f"  Mean: {np.mean(weights):.6f}, Std: {np.std(weights):.6f}")
    print(f"  Min: {np.min(weights):.6f}, Max: {np.max(weights):.6f}")

    return weights


def plot_logit_comparison(logit, isTop, weights1, weights2, label1, label2, pdf):
    """Create logit comparison plot with ratio."""
    print("Creating logit comparison plot...")

    show_target = isTop is not None

    bins = np.linspace(logit.min(), logit.max(), 100)

    # S+B: all events, unweighted (starting point)
    h_sb, _ = np.histogram(logit, bins=bins)

    # Target: signal-only events (isTop == 0), unweighted
    if show_target:
        logit_signal = logit[isTop == 0]
        h_target, _ = np.histogram(logit_signal, bins=bins)

    # Weighted histograms: all events with weights applied
    h_weights1, _ = np.histogram(logit, bins=bins, weights=weights1)
    h_weights2, _ = np.histogram(logit, bins=bins, weights=weights2)

    # Extend for step plotting
    plot_h_sb = np.concatenate([h_sb, h_sb[-1:]])
    if show_target:
        plot_h_target = np.concatenate([h_target, h_target[-1:]])
    plot_h_weights1 = np.concatenate([h_weights1, h_weights1[-1:]])
    plot_h_weights2 = np.concatenate([h_weights2, h_weights2[-1:]])

    # Create figure with two subplots: main plot and ratio pad
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(6.4, 4.8), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0}
    )

    # Main histogram plot
    ax_main.plot(
        bins, plot_h_sb, drawstyle="steps-post", label="S+B", color="gray", alpha=0.5
    )
    if show_target:
        ax_main.plot(
            bins,
            plot_h_target,
            drawstyle="steps-post",
            label="Target (signal only)",
            color="black",
        )
    ax_main.plot(
        bins, plot_h_weights1, drawstyle="steps-post", label=label1, color="blue"
    )
    ax_main.plot(
        bins, plot_h_weights2, drawstyle="steps-post", label=label2, color="red"
    )
    ax_main.set_yscale("log")
    ax_main.tick_params(axis="both", direction="in", top=True, right=True)
    ax_main.legend(frameon=False)
    ax_main.set_ylabel("Events")
    ax_main.set_xticklabels([])

    # Calculate ratios - use float arrays for division
    plot_h_sb_f = plot_h_sb.astype(float)
    if show_target:
        # Ratio to target
        plot_h_target_f = plot_h_target.astype(float)
        ratio_sb = np.divide(
            plot_h_sb_f,
            plot_h_target_f,
            out=np.ones_like(plot_h_target_f),
            where=plot_h_target_f != 0,
        )
        ratio1 = np.divide(
            plot_h_weights1,
            plot_h_target_f,
            out=np.ones_like(plot_h_target_f),
            where=plot_h_target_f != 0,
        )
        ratio2 = np.divide(
            plot_h_weights2,
            plot_h_target_f,
            out=np.ones_like(plot_h_target_f),
            where=plot_h_target_f != 0,
        )
        ratio_ylabel = "Ratio to Target"
    else:
        # Ratio to S+B
        ratio1 = np.divide(
            plot_h_weights1,
            plot_h_sb_f,
            out=np.ones_like(plot_h_sb_f),
            where=plot_h_sb_f != 0,
        )
        ratio2 = np.divide(
            plot_h_weights2,
            plot_h_sb_f,
            out=np.ones_like(plot_h_sb_f),
            where=plot_h_sb_f != 0,
        )
        ratio_ylabel = "Ratio to S+B"

    # Ratio pad
    if show_target:
        ax_ratio.plot(
            bins,
            ratio_sb,
            drawstyle="steps-post",
            color="gray",
            alpha=0.5,
            linewidth=1.5,
            label="S+B",
        )
    ax_ratio.plot(
        bins, ratio1, drawstyle="steps-post", color="blue", linewidth=1.5, label=label1
    )
    ax_ratio.plot(
        bins, ratio2, drawstyle="steps-post", color="red", linewidth=1.5, label=label2
    )
    ax_ratio.axhline(y=1, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax_ratio.set_ylabel(ratio_ylabel)
    ax_ratio.set_xlabel("Top Classifier Logit")
    ax_ratio.tick_params(axis="both", direction="in", top=True, right=True)
    ax_ratio.legend(frameon=False, fontsize=8)

    # Set reasonable y-limits for ratio
    if show_target:
        all_ratios = np.concatenate([ratio_sb, ratio1, ratio2])
    else:
        all_ratios = np.concatenate([ratio1, ratio2])
    ratio_nonzero = all_ratios[(all_ratios != 0) & np.isfinite(all_ratios)]
    if len(ratio_nonzero) > 0:
        ratio_min, ratio_max = np.percentile(ratio_nonzero, [2, 98])
        margin = (ratio_max - ratio_min) * 0.1
        ax_ratio.set_ylim(ratio_min - margin, ratio_max + margin)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Saved logit comparison plot")


def plot_weight_histograms(weights1, weights2, label1, label2, pdf):
    """Create histogram comparing the two weight distributions."""
    print("Creating weight histogram comparison...")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Determine bin range that covers both weight distributions
    bins = np.linspace(0.85, 1.0, 100)

    ax.hist(weights1, bins=bins, density=True, alpha=0.5, color="blue", label=label1)
    ax.hist(weights2, bins=bins, density=True, alpha=0.5, color="red", label=label2)
    ax.set_yscale("log")

    # Add statistics text
    stats_text = (
        f"{label1}:\n"
        f"  Mean: {np.mean(weights1):.4f}\n"
        f"  Std: {np.std(weights1):.4f}\n"
        f"  Min: {np.min(weights1):.4f}\n"
        f"  Max: {np.max(weights1):.4f}\n\n"
        f"{label2}:\n"
        f"  Mean: {np.mean(weights2):.4f}\n"
        f"  Std: {np.std(weights2):.4f}\n"
        f"  Min: {np.min(weights2):.4f}\n"
        f"  Max: {np.max(weights2):.4f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
        fontsize=8,
    )

    ax.set_xlabel("Top subtraction weights")
    ax.set_ylabel("Density")
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Saved weight histogram comparison")


def plot_observable(t, pass190, isTop, weights1, weights2, label1, label2, obs, pdf):
    """Create plot for a single observable."""
    print(f"Creating {obs} plot...")

    show_target = isTop is not None

    obs_data = ak.to_numpy(t[obs].array())
    obs_data = obs_data[pass190 == 1]

    # Target: signal-only events (isTop == 0)
    if show_target:
        obs_signal = obs_data[isTop == 0]

    # Set bin ranges based on observable (matching top_plotting.py)
    if obs == "HT_tracks":
        min_obs = obs_data.min()
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
        min_obs = obs_data.min()
        max_obs = obs_data.max()

    bins = np.linspace(min_obs, max_obs, 100)
    if obs == "Ntracks":
        bins = np.arange(0, 250, 1)

    # S+B histogram: all events, unweighted (starting point)
    h_sb, _ = np.histogram(obs_data, bins=bins)

    # Target histogram: signal-only, unweighted
    if show_target:
        h_target, _ = np.histogram(obs_signal, bins=bins)

    # Weighted histograms: all events with weights applied
    h_weights1, _ = np.histogram(obs_data, bins=bins, weights=weights1)
    h_weights2, _ = np.histogram(obs_data, bins=bins, weights=weights2)

    # Extend for step plotting
    plot_h_sb = np.concatenate([h_sb, h_sb[-1:]])
    if show_target:
        plot_h_target = np.concatenate([h_target, h_target[-1:]])
    plot_h_weights1 = np.concatenate([h_weights1, h_weights1[-1:]])
    plot_h_weights2 = np.concatenate([h_weights2, h_weights2[-1:]])

    # Create figure with two subplots: main plot and ratio pad
    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(6.4, 4.8), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0}
    )

    # Main histogram plot
    ax_main.plot(
        bins, plot_h_sb, drawstyle="steps-post", label="S+B", color="gray", alpha=0.5
    )
    if show_target:
        ax_main.plot(
            bins,
            plot_h_target,
            drawstyle="steps-post",
            label="Target (signal only)",
            color="black",
        )
    ax_main.plot(
        bins, plot_h_weights1, drawstyle="steps-post", label=label1, color="blue"
    )
    ax_main.plot(
        bins, plot_h_weights2, drawstyle="steps-post", label=label2, color="red"
    )
    ax_main.set_yscale("log")
    ax_main.tick_params(axis="both", direction="in", top=True, right=True)
    ax_main.legend(frameon=False)
    ax_main.set_ylabel("Events")
    ax_main.set_xticklabels([])

    # Calculate ratios - use float arrays for division
    plot_h_sb_f = plot_h_sb.astype(float)
    if show_target:
        # Ratio to target
        plot_h_target_f = plot_h_target.astype(float)
        ratio_sb = np.divide(
            plot_h_sb_f,
            plot_h_target_f,
            out=np.ones_like(plot_h_target_f),
            where=plot_h_target_f != 0,
        )
        ratio1 = np.divide(
            plot_h_weights1,
            plot_h_target_f,
            out=np.ones_like(plot_h_target_f),
            where=plot_h_target_f != 0,
        )
        ratio2 = np.divide(
            plot_h_weights2,
            plot_h_target_f,
            out=np.ones_like(plot_h_target_f),
            where=plot_h_target_f != 0,
        )
        ratio_ylabel = "Ratio to Target"
    else:
        # Ratio to S+B
        ratio1 = np.divide(
            plot_h_weights1,
            plot_h_sb_f,
            out=np.ones_like(plot_h_sb_f),
            where=plot_h_sb_f != 0,
        )
        ratio2 = np.divide(
            plot_h_weights2,
            plot_h_sb_f,
            out=np.ones_like(plot_h_sb_f),
            where=plot_h_sb_f != 0,
        )
        ratio_ylabel = "Ratio to S+B"

    # Ratio pad
    if show_target:
        ax_ratio.plot(
            bins,
            ratio_sb,
            drawstyle="steps-post",
            color="gray",
            alpha=0.5,
            linewidth=1.5,
            label="S+B",
        )
    ax_ratio.plot(
        bins, ratio1, drawstyle="steps-post", color="blue", linewidth=1.5, label=label1
    )
    ax_ratio.plot(
        bins, ratio2, drawstyle="steps-post", color="red", linewidth=1.5, label=label2
    )
    ax_ratio.axhline(y=1, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax_ratio.set_ylabel(ratio_ylabel)
    ax_ratio.set_xlabel(obs)
    ax_ratio.tick_params(axis="both", direction="in", top=True, right=True)
    ax_ratio.legend(frameon=False, fontsize=8)

    # Set reasonable y-limits for ratio
    if show_target:
        all_ratios = np.concatenate([ratio_sb, ratio1, ratio2])
    else:
        all_ratios = np.concatenate([ratio1, ratio2])
    ratio_nonzero = all_ratios[(all_ratios != 0) & np.isfinite(all_ratios)]
    if len(ratio_nonzero) > 0:
        ratio_min, ratio_max = np.percentile(ratio_nonzero, [2, 98])
        margin = (ratio_max - ratio_min) * 0.1
        ax_ratio.set_ylim(ratio_min - margin, ratio_max + margin)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {obs} plot")


def plot_observables(t, pass190, isTop, weights1, weights2, label1, label2, pdf):
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
        plot_observable(t, pass190, isTop, weights1, weights2, label1, label2, obs, pdf)


def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Generate weight comparison plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_file",
        help="Path to the ROOT file containing the data",
        type=str,
        default=(
            "/pscratch/sd/k/kgreif/data/"
            "Pseudodata_SherpaDY_PowhegPythiaTop_June2025_shuffled_topLogit.root"
        ),
    )
    parser.add_argument(
        "--weights1",
        help="Path to the first NPZ file containing weights",
        type=str,
        default=(
            "/global/cfs/cdirs/m3246/ZjetOmnifold/weights/"
            "top_subtraction/bsv3_ensemble.npz"
        ),
    )
    parser.add_argument(
        "--weights2",
        help="Path to the second NPZ file containing weights",
        type=str,
        default=(
            "/global/cfs/cdirs/m3246/ZjetOmnifold/weights/"
            "top_subtraction/bs_alt_ensemble.npz"
        ),
    )
    parser.add_argument(
        "--label1",
        help="Label for the first weight set",
        type=str,
        default="BSv3",
    )
    parser.add_argument(
        "--label2",
        help="Label for the second weight set",
        type=str,
        default="BS Alt",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="weight_comparison.pdf",
        help="Output PDF filename (default: weight_comparison.pdf)",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Disable target histogram (for data without isTop branch)",
    )

    args = parser.parse_args()

    try:
        # Load data
        t, logit, pass190, isTop = load_data(
            args.data_file, load_target=not args.no_target
        )

        # Load weights
        weights1 = load_weights(args.weights1)
        weights2 = load_weights(args.weights2)

        # Verify weight lengths match number of events passing selection
        n_events = sum(pass190 == 1)
        if len(weights1) != n_events:
            raise ValueError(
                f"Weight file 1 has {len(weights1)} weights "
                f"but data has {n_events} events"
            )
        if len(weights2) != n_events:
            raise ValueError(
                f"Weight file 2 has {len(weights2)} weights "
                f"but data has {n_events} events"
            )

        # Create PDF file
        print(f"\nCreating PDF: {args.output}")
        with PdfPages(args.output) as pdf:
            # Plot 1: Weight histogram comparison
            plot_weight_histograms(weights1, weights2, args.label1, args.label2, pdf)

            # Plot 2: Logit comparison
            plot_logit_comparison(
                logit, isTop, weights1, weights2, args.label1, args.label2, pdf
            )

            # Plots 3-14: Observable plots
            plot_observables(
                t, pass190, isTop, weights1, weights2, args.label1, args.label2, pdf
            )

        print(f"\nAll plots saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
