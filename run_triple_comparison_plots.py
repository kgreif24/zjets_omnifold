"""run_triple_comparison_plots.py - Standalone script that plots three reweightings
on the same axes for comparison. Uses the same YAML plotting config as plotter.py.

Author: Kevin Greif
python3
"""

import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import plotter as plotter_module


DEFAULT_WEIGHTS = [
    "/global/cfs/cdirs/m3246/ZjetOmnifold/model_repository/zjets-v4/nominal_2/aussie/weights/iteration_1_step_2.npz",
    "/global/cfs/cdirs/m3246/ZjetOmnifold/model_repository/zjets-v4/nominal_2/weights/iteration_1_step_2.npz",
    "/global/cfs/cdirs/m3246/ZjetOmnifold/model_repository/zjets-v4/nominal_2/weights/iteration_7_step_2.npz",
]
DEFAULT_LABELS = ["Aussie", "OmniFold iteration 1", "OmniFold iteration 7"]
REWEIGHTING_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]

parser = argparse.ArgumentParser(
    description="Plot three reweightings on the same axes for comparison"
)
parser.add_argument("--mc", type=str, required=True, help="Path to MC ROOT file")
parser.add_argument("--data", type=str, required=True, help="Path to data ROOT file")
parser.add_argument(
    "--weights",
    type=str,
    nargs=3,
    default=DEFAULT_WEIGHTS,
    metavar=("WEIGHT1", "WEIGHT2", "WEIGHT3"),
    help="Paths to the three weight .npz files",
)
parser.add_argument(
    "--labels",
    type=str,
    nargs=3,
    default=DEFAULT_LABELS,
    metavar=("LABEL1", "LABEL2", "LABEL3"),
    help="Labels for the three reweightings",
)
parser.add_argument(
    "--start_weights",
    type=str,
    default="weight_mc",
    help="Starting (nominal) MC weights: branch name or .npz path",
)
parser.add_argument(
    "--target_weights",
    type=str,
    default="weight",
    help="Target (data) weights: branch name or .npz path",
)
parser.add_argument("--store", type=str, required=True, help="Directory to save plots")
parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level")
parser.add_argument("--pdf", action="store_true", help="Save plots as PDF")
parser.add_argument("--truth", action="store_true", help="Use truth-level observables")
parser.add_argument(
    "--max_events", type=int, default=5000000, help="Max events to use"
)
parser.add_argument("--ibu_bins", action="store_true", help="Use IBU bins from config")
parser.add_argument(
    "--cut_region",
    type=int,
    default=0,
    choices=[-1, 0, 1, 2, 3, 4],
    help="Kinematic region to restrict to",
)
args = parser.parse_args()

store = pathlib.Path(args.store)
store.mkdir(parents=True, exist_ok=True)

# Instantiate Plotter to reuse its data loading, caching, and histogram infrastructure
p = plotter_module.Plotter(
    args.mc,
    args.data,
    args.store,
    use_truth=args.truth,
    labels=("MC", "Data"),
    verbosity=args.verbosity,
    use_pdf=args.pdf,
    max_events=args.max_events,
    ibu_bins=args.ibu_bins,
    kinematic_region=args.cut_region,
    normalize_targets=True,
)

p._ensure_kinematic_cuts_applied()

source_pass190 = p._get_cached_pass190_flags("source")
target_pass190 = p._get_cached_pass190_flags("target")

# Load and filter all weights
start_w = p._get_weights(args.start_weights)[source_pass190 == 1]
target_w = p._get_weights(args.target_weights, is_target=True)[target_pass190 == 1]
end_weights = [
    p._get_weights(wpath)[source_pass190 == 1] for wpath in args.weights
]

for plot in p.plots:
    # Build histograms
    target_hist, _, bins = p._get_histogram(
        plot, weights=target_w, density=False, is_target=True
    )
    start_hist, _, _ = p._get_histogram(plot, weights=start_w, density=False)
    end_hists = [
        p._get_histogram(plot, weights=w, density=False)[0] for w in end_weights
    ]

    # Normalize all distributions to the same integral as the target
    target_sum = np.sum(target_hist)

    def safe_norm(h, target_sum=target_sum):
        s = np.sum(h)
        return h * (target_sum / s) if s > 0 else h

    start_hist_n = safe_norm(start_hist)
    end_hists_n = [safe_norm(h) for h in end_hists]

    # Compute ratios to target before bin-width scaling
    with np.errstate(divide="ignore", invalid="ignore"):
        start_ratio = np.where(target_hist > 0, start_hist_n / target_hist, 1.0)
        end_ratios = [
            np.where(target_hist > 0, h / target_hist, 1.0) for h in end_hists_n
        ]

    # Scale by bin width for display
    def scale_bw(h):
        return p._scale_histogram_by_bin_width(h, None, bins)[0]

    start_hist_s = scale_bw(start_hist_n)
    target_hist_s = scale_bw(target_hist)
    end_hists_s = [scale_bw(h) for h in end_hists_n]

    # Duplicate last bin for step plots
    def dup(arr):
        return np.append(arr, arr[-1])

    start_hist_s = dup(start_hist_s)
    target_hist_s = dup(target_hist_s)
    start_ratio = dup(start_ratio)
    end_hists_s = [dup(h) for h in end_hists_s]
    end_ratios = [dup(r) for r in end_ratios]

    fig = plt.figure()
    ax, axr = p._add_ratios(fig)

    # Unweighted MC (filled)
    ax.plot(
        bins, start_hist_s,
        drawstyle="steps-post",
        label="MC (unweighted)",
        alpha=0.5,
        color="#1f77b4",
    )
    ax.fill_between(bins, 0, start_hist_s, step="post", alpha=0.3, color="#1f77b4")

    # Target (data)
    ax.plot(
        bins, target_hist_s,
        drawstyle="steps-post",
        label="Data",
        color="black",
        linewidth=1.5,
    )

    # Three reweightings
    for label, hs, color in zip(args.labels, end_hists_s, REWEIGHTING_COLORS):
        ax.plot(bins, hs, drawstyle="steps-post", label=label, color=color)

    if plot["ylim"] is not None:
        ax.set_ylim(plot["ylim"])
    ax.set_ylabel(plot["ylabel"])
    if not plot["linear_yscale"]:
        ax.set_yscale("log")
    if plot["log_xscale"]:
        ax.set_xscale("log")
    ax.set_xticks([])
    ax.legend(loc="upper right", fontsize=7)

    # Ratio panel
    axr.hlines(1, bins[0], bins[-1], color="black", linestyle="--", alpha=0.8)
    axr.plot(bins, start_ratio, drawstyle="steps-post", alpha=0.5, color="#1f77b4")
    for ratio, color in zip(end_ratios, REWEIGHTING_COLORS):
        axr.plot(bins, ratio, drawstyle="steps-post", color=color)
    axr.set_xlabel(plot["xlabel"])
    if plot["log_xscale"]:
        axr.set_xscale("log")
    axr.set_ylabel("Ratio to data")
    axr.set_ylim(plot["rlim"])

    fig.tight_layout()

    ext = ".pdf" if args.pdf else ".png"
    fig.savefig(store / (plot["key"] + ext), dpi=300)
    plt.close(fig)
    print(f"Saved {plot['key']}")

print("Done")
