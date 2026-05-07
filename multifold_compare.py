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
import scipy.stats as stats
import pathlib
from analyze.uncertainties import UncertaintyCalculator


parser = argparse.ArgumentParser(description="Run plotting functions")
parser.add_argument("--data", action="store_true", help="If true, will plot data")
parser.add_argument(
    "--omnifold",
    type=str,
    default=None,
    help="Path to omnifold_histograms.npz file to overlay (only with --data)",
)
parser.add_argument(
    "--store",
    type=str,
    default="./plot_storage/multifold/pd",
    help="Path to store the plots (default: ./plot_storage/multifold/pd)",
)
parser.add_argument(
    "--min-pt-trackj1",
    type=float,
    default=5.0,
    help="Minimum pT_trackj1 for track-jet-1 observables (default: 5)",
)
parser.add_argument(
    "--min-pt-trackj2",
    type=float,
    default=5.0,
    help="Minimum pT_trackj2 for track-jet-2 observables (default: 5)",
)
parser.add_argument(
    "--region",
    type=int,
    default=0,
    choices=[0, 1, 2, 3],
    help=(
        "Kinematic region to use for plots: "
        "0=default (pT_ll>200), "
        "1=high pT_Z (pT_trackj2>50, pT_ll>350), "
        "2=EW enhanced (m_jj>200, |dy_jj|>2, pT_ll>200), "
        "3=diboson enhanced (pT_trackj1>32, pT_ll>200)"
    ),
)
parser.add_argument(
    "--png",
    action="store_true",
    help="If true, will save plots as png",
)
args = parser.parse_args()

# Set extension based on whether png is requested
extension = ".png" if args.png else ".pdf"

# Load omnifold data if provided (only used in data mode)
omnifold_data = None
if args.data and args.omnifold is not None:
    omnifold_data = np.load(args.omnifold)
    print(f"Loaded omnifold data from: {args.omnifold}")

# Load the multifold data
key = "data" if args.data else "pseudodata"
multifold = pd.read_hdf(f"/pscratch/sd/k/kgreif/multifold/{key}/multifold.h5")
hv = pd.read_hdf(f"/pscratch/sd/k/kgreif/multifold/{key}/multifold_sherpa.h5")
nonDY = pd.read_hdf(f"/pscratch/sd/k/kgreif/multifold/{key}/multifold_nonDY.h5")
if not args.data:
    target = pd.read_hdf("/pscratch/sd/k/kgreif/multifold/pseudodata/target.h5")

# Load arrays of weights for building histograms
dbs_weights = [
    col for col in multifold.keys() if col.startswith("weights_bootstrap_data")
]
mcbs_weights = [
    col for col in multifold.keys() if col.startswith("weights_bootstrap_mc")
]
ens_weights = [col for col in multifold.keys() if col.startswith("weights_ensemble")]

# Load plotting config
with open("./utils/plots_config.yml", "r") as stream:
    config = yaml.safe_load(stream)
plots = [
    config["plots"][plot]
    for plot in config["plots"]
    if config["plots"][plot]["multifold"]
]

# Initialize the uncertainty calculator
uncertainty_calculator = UncertaintyCalculator(multifold_nn_init=True)
uncertainty_calculator.remove_uncertainty("hvhad")
uncertainty_calculator.remove_uncertainty("nonstrongDiboson")
uncertainty_calculator.remove_uncertainty("nonstrongEW")
uncertainty_calculator.add_uncertainty(
    key="nonDY",
    name="Non-DY",
    color="chocolate",
    stochastic=False,
    prefix=None,
)

# Create output directory for plots
plot_dir = pathlib.Path(args.store)
plot_dir.mkdir(parents=True, exist_ok=True)

# Observables that require a first track-jet, so pT_trackj1 cut is applied
TRACKJ1_KEYS = {
    "pT_trackj1",
    "y_trackj1",
    "phi_trackj1",
    "m_trackj1",
    "tau1_trackj1",
    "tau2_trackj1",
    "tau3_trackj1",
    "Ntracks_trackj1",
}

# Observables that require a second track-jet, so pT_trackj2 cut is applied
TRACKJ2_KEYS = {
    "pT_trackj2",
    "y_trackj2",
    "phi_trackj2",
    "m_trackj2",
    "tau1_trackj2",
    "tau2_trackj2",
    "tau3_trackj2",
    "Ntracks_trackj2",
}


def _compute_jj(df):
    """Return (m_jj, dy_jj) arrays computed from track-jet columns in df."""
    dy_jj = np.abs(df["y_trackj1"].values - df["y_trackj2"].values)
    px1 = df["pT_trackj1"].values * np.cos(df["phi_trackj1"].values)
    py1 = df["pT_trackj1"].values * np.sin(df["phi_trackj1"].values)
    mt1 = np.sqrt(df["pT_trackj1"].values ** 2 + df["m_trackj1"].values ** 2)
    pz1 = mt1 * np.sinh(df["y_trackj1"].values)
    E1 = mt1 * np.cosh(df["y_trackj1"].values)
    px2 = df["pT_trackj2"].values * np.cos(df["phi_trackj2"].values)
    py2 = df["pT_trackj2"].values * np.sin(df["phi_trackj2"].values)
    mt2 = np.sqrt(df["pT_trackj2"].values ** 2 + df["m_trackj2"].values ** 2)
    pz2 = mt2 * np.sinh(df["y_trackj2"].values)
    E2 = mt2 * np.cosh(df["y_trackj2"].values)
    m_jj = np.sqrt(
        (E1 + E2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    )
    return m_jj, dy_jj


def get_kinematic_mask(df, region):
    """Return a boolean mask selecting events in the requested kinematic region.

    Regions:
        0: pT_ll > 200 GeV
        1: pT_trackj2 > 50 GeV AND pT_ll > 350 GeV
        2: m_jj > 200 GeV AND |dy_jj| > 2 AND pT_ll > 200 GeV
        3: pT_trackj1 > 32 GeV AND pT_ll > 200 GeV
    """
    if region == 0:
        return df["pT_ll"].values > 200
    elif region == 1:
        return (df["pT_trackj2"].values > 50) & (df["pT_ll"].values > 350)
    elif region == 2:
        m_jj, dy_jj = _compute_jj(df)
        return (m_jj > 200) & (dy_jj > 2) & (df["pT_ll"].values > 200)
    elif region == 3:
        return (df["pT_trackj1"].values > 32) & (df["pT_ll"].values > 200)


# Pre-compute kinematic region masks and trackj2 pT masks
region_masks = {
    "multifold": get_kinematic_mask(multifold, args.region),
    "hv": get_kinematic_mask(hv, args.region),
    "nonDY": get_kinematic_mask(nonDY, args.region),
}
trackj1_masks = {
    "multifold": multifold["pT_trackj1"].values > args.min_pt_trackj1,
    "hv": hv["pT_trackj1"].values > args.min_pt_trackj1,
    "nonDY": nonDY["pT_trackj1"].values > args.min_pt_trackj1,
}
trackj2_masks = {
    "multifold": multifold["pT_trackj2"].values > args.min_pt_trackj2,
    "hv": hv["pT_trackj2"].values > args.min_pt_trackj2,
    "nonDY": nonDY["pT_trackj2"].values > args.min_pt_trackj2,
}
if not args.data:
    region_masks["target"] = get_kinematic_mask(target, args.region)
    trackj1_masks["target"] = target["pT_trackj1"].values > args.min_pt_trackj1
    trackj2_masks["target"] = target["pT_trackj2"].values > args.min_pt_trackj2

print(
    f"Kinematic region {args.region}: "
    f"{region_masks['multifold'].sum()} / {len(multifold)} multifold events pass"
)

# Dictionary to store chi-squared test results (only used in pseudodata mode)
chi2_results = {}

# Loop through the observables
for obs_dict in plots:

    key = obs_dict["key"]
    print("Plotting observable", key)

    # Determine per-observable masks
    is_trackj1 = key in TRACKJ1_KEYS
    is_trackj2 = key in TRACKJ2_KEYS
    mf_mask = region_masks["multifold"]
    hv_mask = region_masks["hv"]
    nonDY_mask = region_masks["nonDY"]
    if is_trackj1:
        mf_mask = mf_mask & trackj1_masks["multifold"]
        hv_mask = hv_mask & trackj1_masks["hv"]
        nonDY_mask = nonDY_mask & trackj1_masks["nonDY"]
    if is_trackj2:
        mf_mask = mf_mask & trackj2_masks["multifold"]
        hv_mask = hv_mask & trackj2_masks["hv"]
        nonDY_mask = nonDY_mask & trackj2_masks["nonDY"]

    mf = multifold.loc[mf_mask].reset_index(drop=True)
    hv_df = hv.loc[hv_mask].reset_index(drop=True)
    nonDY_df = nonDY.loc[nonDY_mask].reset_index(drop=True)
    if not args.data:
        tgt_mask = region_masks["target"]
        if is_trackj1:
            tgt_mask = tgt_mask & trackj1_masks["target"]
        if is_trackj2:
            tgt_mask = tgt_mask & trackj2_masks["target"]
        tgt = target.loc[tgt_mask].reset_index(drop=True)

    # Calculate the bins, using region-specific binning when defined
    region_key = str(args.region)
    if "region_bins" in obs_dict and region_key in obs_dict["region_bins"]:
        bins = np.array(obs_dict["region_bins"][region_key])
    else:
        bins = np.array(obs_dict["ibubins"])

    # Get bin properties
    bin_centers = (bins[1:] + bins[:-1]) / 2
    bin_errors = (bins[1:] - bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    # Calculate the histograms
    # Nominal histogram
    all_hists = {}
    source_hist, _ = np.histogram(
        mf[obs_dict["key"]],
        bins=bins,
        weights=mf["weights_nominal"],
    )
    mc_var, _ = np.histogram(
        mf[obs_dict["key"]], bins=bins, weights=mf["weights_nominal"] ** 2
    )
    all_hists["nominal"] = (source_hist, mc_var, bins)
    # Data bootstrap histograms (for data-stat uncertainty)
    if "data-stat" in uncertainty_calculator.uncertainty_definitions:
        for weight in dbs_weights:
            hist, _ = np.histogram(
                mf[obs_dict["key"]], bins=bins, weights=mf[weight]
            )
            all_hists[weight.removeprefix("weights_")] = (hist, None, bins)
    # MC bootstrap histograms (for mc-stat uncertainty)
    if "mc-stat" in uncertainty_calculator.uncertainty_definitions:
        for weight in mcbs_weights:
            hist, _ = np.histogram(
                mf[obs_dict["key"]], bins=bins, weights=mf[weight]
            )
            all_hists[weight.removeprefix("weights_")] = (hist, None, bins)
    # Ensemble histograms (for nn-stability uncertainty)
    if "nn-stability" in uncertainty_calculator.uncertainty_definitions:
        for weight in ens_weights:
            hist, _ = np.histogram(
                mf[obs_dict["key"]], bins=bins, weights=mf[weight]
            )
            all_hists[weight.removeprefix("weights_")] = (hist, None, bins)
    # HV histogram
    if "hv" in uncertainty_calculator.uncertainty_definitions:
        hv_hist, _ = np.histogram(
            hv_df[obs_dict["key"]], bins=bins, weights=hv_df["weights_nominal"]
        )
        all_hists["hv"] = (hv_hist, None, bins)
    # Non-DY histogram
    if "nonDY" in uncertainty_calculator.uncertainty_definitions:
        nonDY_hist, _ = np.histogram(
            nonDY_df[obs_dict["key"]], bins=bins, weights=nonDY_df["weights_nominal"]
        )
        all_hists["nonDY"] = (nonDY_hist, None, bins)
    # DD histogram (if dd uncertainty is defined and weights_dd exists)
    if (
        "dd" in uncertainty_calculator.uncertainty_definitions
        and "weights_dd" in mf.columns
    ):
        dd_hist, _ = np.histogram(
            mf[obs_dict["key"]], bins=bins, weights=mf["weights_dd"]
        )
        all_hists["dd"] = (dd_hist, None, bins)
    # DD target histogram (only needed if dd uncertainty is defined)
    if (
        "dd" in uncertainty_calculator.uncertainty_definitions
        and "target_dd" in mf.columns
    ):
        dd_target_hist, _ = np.histogram(
            mf[obs_dict["key"]], bins=bins, weights=mf["target_dd"]
        )
        all_hists["target_dd"] = (dd_target_hist, None, bins)
    # Target histogram
    if not args.data:
        target_hist, _ = np.histogram(
            tgt[obs_dict["key"]], bins=bins, weights=tgt["weight_mc"]
        )
        all_hists["target"] = (target_hist, None, bins)
    # All other histograms
    for (
        uncert_name,
        uncert_def,
    ) in uncertainty_calculator.uncertainty_definitions.items():
        if uncert_def["stochastic"] or uncert_name in ["hv", "dd", "nonDY"]:
            continue
        weight_key = "weights_" + uncert_name
        uncert_hist, _ = np.histogram(
            mf[obs_dict["key"]], bins=bins, weights=mf[weight_key]
        )
        all_hists[uncert_name] = (uncert_hist, None, bins)

    # Calculate the uncertainties and total uncertainty
    ismass = key in ["m_trackj1", "m_trackj2"]
    signed_uncerts, syst_covs_individual, syst_info_individual = (
        uncertainty_calculator.calculate_uncertainties(
            all_hists,
            measured_key="nominal",
            smooth_hv=not ismass,
        )
    )
    syst_uncerts, syst_covs, syst_info = (
        uncertainty_calculator.process_signed_uncertainties(
            signed_uncerts, syst_covs_individual, syst_info_individual
        )
    )

    # If not in data mode, calculate bias between unfolded data and truth
    if not args.data:
        mbias = np.abs(source_hist - target_hist) / target_hist
        rel_mbias = mbias
    else:
        target_hist = None
        rel_mbias = None

    # Calculate total variance and uncertainty from syst_vars
    total_var = np.sum(np.array(list(syst_uncerts.values())) ** 2, axis=0)
    rel_total_uncert = np.sqrt(total_var)
    total_uncert = rel_total_uncert * source_hist

    # Build main uncertainty plot
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(6, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    plt.subplots_adjust(hspace=0, top=0.95)

    # Divide histograms for which we care about density by bin width
    x_source_hist = source_hist / bin_widths
    if not args.data:
        x_target_hist = target_hist / bin_widths
    else:
        x_target_hist = None
    x_total_uncert = total_uncert / bin_widths

    # Main plot: Standard mode (pseudodata vs target)
    if not args.data:

        # Duplicate last bin for step plot
        plot_target_hist = np.append(x_target_hist, x_target_hist[-1])

        # Plot target as dashed line
        ax.plot(
            bins,
            plot_target_hist,
            "--",
            label="Target",
            color="black",
            drawstyle="steps-post",
        )

        # Plot unfolded data with uncertainties
        ax.errorbar(
            bin_centers,
            x_source_hist,
            yerr=x_total_uncert,
            fmt="o",
            label="Multifold",
            color="green",
        )

        # Calculate ratio
        ratio = x_source_hist / x_target_hist
        ratio_uncert = rel_total_uncert
    else:
        # Data mode: just plot the data
        ax.errorbar(
            bin_centers,
            x_source_hist,
            xerr=bin_errors,
            yerr=x_total_uncert,
            fmt="+",
            label="Multifold",
            color="green",
        )

        # Overlay omnifold data if available
        # Note we assume omnifold data is already divided by bin width!
        of_hist = None
        of_uncert = None
        if omnifold_data is not None:
            of_hist_key = key + "_hist"
            of_uncert_key = key + "_uncert"
            if of_hist_key in omnifold_data and of_uncert_key in omnifold_data:
                of_hist = omnifold_data[of_hist_key] / bin_widths
                of_uncert = omnifold_data[of_uncert_key] / bin_widths
                # Plot omnifold as blue points with slight x offset for visibility
                ax.errorbar(
                    bin_centers + bin_errors * 0.1,  # Slight offset to avoid overlap
                    of_hist,
                    xerr=bin_errors,
                    yerr=of_uncert,
                    fmt="x",
                    label="Omnifold",
                    color="blue",
                )
            else:
                print(f"Warning: Observable {key} not found in omnifold data")

        # Calculate ratio of Omnifold to Multifold if both are available
        if of_hist is not None:
            ratio = of_hist / x_source_hist
            # Propagate uncertainties: rel_err = sqrt((dA/A)^2 + (dB/B)^2)
            rel_of_uncert = of_uncert / of_hist
            rel_mf_uncert = x_total_uncert / x_source_hist
            ratio_uncert = ratio * np.sqrt(rel_of_uncert**2 + rel_mf_uncert**2)
        else:
            ratio = None
            ratio_uncert = None

    # Set tick parameters
    ax.tick_params(axis="x", direction="in", top=True)

    # Set y-axis scale
    if not obs_dict.get("linear_yscale", False):
        ax.set_yscale("log")
    if obs_dict.get("log_xscale", False):
        ax.set_xscale("log")

    # Set y-axis label
    if obs_dict.get("ylabel") is not None:
        ax.set_ylabel(obs_dict["ylabel"])
    else:
        ax.set_ylabel("Counts")
    ax.legend()

    # Ratio plot
    if ratio is not None:
        rax.axhline(1, color="black", linestyle="--")
        # Use blue color for Omnifold/Multifold ratio in data mode
        ratio_color = "blue" if args.data else "green"
        rax.errorbar(
            bin_centers,
            ratio,
            xerr=bin_errors,
            yerr=ratio_uncert,
            fmt="+",
            color=ratio_color,
        )
        rax.set_ylim(0.5, 1.5)
        rax.set_yticks([0.5, 1.0, 1.5])
        # Set appropriate y-axis label based on mode
        if args.data:
            rax.set_ylabel("Omnifold / Multifold")
        else:
            rax.set_ylabel("Ratio to target")
    else:
        rax.set_ylabel("Ratio")
        rax.axhline(1, color="black", linestyle="--")

    rax.set_xlabel(obs_dict.get("xlabel", key))
    rax.tick_params(axis="x", direction="in", bottom=True, top=False)

    # Finalize layout
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, top=0.95)

    # Save main plot
    plot_name = plot_dir / (key + extension)
    fig.savefig(plot_name, dpi=300)
    plt.close(fig)

    # Build uncertainty budget plot
    budget_fig, budget_ax = plt.subplots(figsize=(6.4, 4.8))

    # Duplicate last bins for step plots
    plot_total_uncert = np.append(rel_total_uncert, rel_total_uncert[-1])

    # Plot total uncertainty
    budget_ax.plot(
        bins,
        plot_total_uncert,
        "--",
        color="black",
        label="Total unc.",
        drawstyle="steps-post",
        linewidth=2,
    )

    # Plot individual uncertainties
    for uncert_key, uncert in syst_uncerts.items():
        info = syst_info[uncert_key]
        plot_uncert = np.append(uncert, uncert[-1])
        budget_ax.plot(
            bins,
            plot_uncert,
            "-",
            color=info.get("color", "black"),
            label=info.get("name", uncert_key),
            drawstyle="steps-post",
        )

    # Plot method bias if available
    if rel_mbias is not None:
        plot_mbias = np.append(rel_mbias, rel_mbias[-1])
        budget_ax.fill_between(
            bins,
            0,
            plot_mbias,
            step="post",
            color="gray",
            alpha=0.3,
            label="Method bias",
        )

    # Set tick parameters
    budget_ax.tick_params(axis="x", direction="in", top=True)

    # Set other plot properties
    if rel_mbias is not None:
        top_uncert = np.max(np.concatenate([rel_total_uncert, rel_mbias]))
    else:
        top_uncert = np.max(rel_total_uncert)
    if top_uncert > 0.2 or np.isnan(top_uncert):
        budget_ax.set_ylim(bottom=0.0, top=0.2)
    else:
        budget_ax.set_ylim(bottom=0.0, top=top_uncert * 1.1)
    if obs_dict.get("log_xscale", False):
        budget_ax.set_xscale("log")
    budget_ax.set_xlabel(obs_dict.get("xlabel", key))
    budget_ax.set_ylabel("Uncertainty budget")
    budget_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)

    # Finalize layout
    budget_fig.tight_layout()
    budget_fig.subplots_adjust(bottom=0.2)

    # Save uncertainty budget plot
    budget_name = plot_dir / (key + "_uncert_budget" + extension)
    budget_fig.savefig(budget_name, dpi=300)
    plt.close(budget_fig)

    # Calculate total covariance matrix from syst_covs
    total_cov = np.sum(list(syst_covs.values()), axis=0)

    # Create and save correlation matrix plot
    n_bins = total_cov.shape[0]
    std_devs = np.sqrt(np.diag(total_cov))

    # Handle zero standard deviations to avoid division by zero
    std_devs = np.where(std_devs == 0, 1, std_devs)

    # Calculate correlation matrix
    corr_matrix = total_cov / np.outer(std_devs, std_devs)

    # Clip values to [-1, 1] to handle numerical precision issues
    corr_matrix = np.clip(corr_matrix, -1, 1)

    # Create figure
    corr_fig, corr_ax = plt.subplots(figsize=(8, 7))

    # Create the heatmap with origin='lower' (smallest bins at bottom-left)
    im = corr_ax.imshow(
        corr_matrix,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        aspect="equal",
        origin="lower",
    )

    # Add colorbar
    corr_fig.colorbar(im, ax=corr_ax, shrink=0.8)

    # Create bin labels from bin edges (round to hundredths for decimal edges)
    bin_labels = [f"({bins[i]:.2f}, {bins[i+1]:.2f})" for i in range(len(bins) - 1)]

    # Set ticks and labels
    corr_ax.set_xticks(np.arange(n_bins))
    corr_ax.set_yticks(np.arange(n_bins))
    corr_ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    corr_ax.set_yticklabels(bin_labels)

    # Add correlation values as text annotations
    for i in range(n_bins):
        for j in range(n_bins):
            # Choose text color based on background for readability
            corr_val = corr_matrix[i, j]
            text_color = "white" if abs(corr_val) < 0.5 else "black"
            corr_ax.text(
                j,
                i,
                f"{corr_val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    # Set title using the xlabel from plot config
    corr_ax.set_title(f"Correlation Matrix: {obs_dict.get('xlabel', key)}", fontsize=12)

    # Finalize layout
    corr_fig.tight_layout()

    # Save correlation matrix plot
    corr_name = plot_dir / (key + "_corr_matrix" + extension)
    corr_fig.savefig(corr_name, dpi=300)
    plt.close(corr_fig)

    # Chi-squared test (only when comparing to target, i.e., not in data mode)
    if not args.data and target_hist is not None:

        # Calculate covariance matrix for test, exclude muon and track uncertainties
        test_covs = []
        for syst_key in syst_covs.keys():
            if syst_key not in ["Muon", "Tracking", "lumi", "pileup"]:
                test_covs.append(syst_covs[syst_key])
        test_cov = np.sum(test_covs, axis=0)
        dof = len(bins) - 1
        D = source_hist - target_hist
        chi2 = D.dot(np.linalg.inv(test_cov)).dot(D.T)
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        print(f"{key:<20} dof: {dof:<7} χ2: {chi2:.5f} \t p value: {p_value:.4f}")
        # Store results for saving to .npz file
        chi2_results[key + "_chi2"] = chi2
        chi2_results[key + "_p_value"] = p_value
        chi2_results[key + "_dof"] = dof

# Save chi-squared results to .npz file (only in pseudodata mode)
if not args.data and chi2_results:
    chi2_npz_path = plot_dir / "chi2_results.npz"
    np.savez(chi2_npz_path, **chi2_results)
    print(f"\nSaved chi-squared results to: {chi2_npz_path}")
