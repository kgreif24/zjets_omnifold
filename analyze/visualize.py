"""
Visualization functions for jet analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import vector


def plot_jets_eta_phi(
    jets: list[np.ndarray],
    figsize=(10, 10),
    ax=None,
    s_scale=100.0,
    alpha=0.6,
    color=None,
    label=None,
    show_legend=True,
    coords="cartesian",
) -> plt.Figure:
    """Plot jets as scatter plots in the eta-phi plane.

    Each constituent is represented as a circle with radius proportional to its pT.

    Arguments:
    jets - A list of numpy arrays of jet constituents with the form:
        (n_constituents, 4) where the columns depend on the coords parameter:
        - If coords="cartesian": (E, px, py, pz)
        - If coords="ptyphim": (pT, eta, phi, m)
    figsize - Tuple of figure size (width, height) in inches (default: (10, 10)).
    ax - Matplotlib axis to plot on. If None, creates a new figure and axis.
    s_scale - Scale factor for circle sizes (default: 100.0).
        Circle area = s_scale * pT. Increase for larger circles.
    alpha - Transparency of circles (default: 0.6).
    color - Color for the scatter plot. If None, uses default matplotlib color cycle.
    label - Label for the plot (for legend).
    show_legend - If True, show legend when label is provided (default: True).
    coords - Coordinate system: "cartesian" for (E, px, py, pz) or "ptyphim" for
        (pT, eta, phi, m) (default: "cartesian").

    Returns:
    fig - Matplotlib figure object.

    Notes:
    - If multiple jets are provided, all constituents are plotted on the same axis.
    - The circle size is proportional to the constituent's pT.
    - For "cartesian" coordinates, eta and phi are calculated from the four-vectors
      using the vector package.
    - For "ptyphim" coordinates, eta, phi, and pT are used directly from the input.
    """
    # Convert jets to eta-phi coordinates
    all_eta = []
    all_phi = []
    all_pt = []

    for jet in jets:
        if coords == "cartesian":
            # Convert four-vectors to vector objects
            # jet shape: (n_constituents, 4) with (E, px, py, pz)
            vectors = vector.array(
                {"E": jet[:, 0], "px": jet[:, 1], "py": jet[:, 2], "pz": jet[:, 3]}
            )

            # Extract eta, phi, and pT
            all_eta.extend(vectors.eta)
            all_phi.extend(vectors.phi)
            all_pt.extend(vectors.pt)
        elif coords == "ptyphim":
            # jet shape: (n_constituents, 4) with (pT, eta, phi, m)
            all_pt.extend(jet[:, 0])
            all_eta.extend(jet[:, 1])
            all_phi.extend(jet[:, 2])
        else:
            raise ValueError(f'coords must be "cartesian" or "ptyphim", got "{coords}"')

    # Convert to numpy arrays
    all_eta = np.array(all_eta)
    all_phi = np.array(all_phi)
    all_pt = np.array(all_pt)

    # Create figure if axis not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Calculate circle sizes (proportional to pT)
    sizes = s_scale * all_pt

    # Plot scatter plot
    ax.scatter(
        all_eta,
        all_phi,
        s=sizes,
        alpha=alpha,
        color=color,
        label=label,
    )

    # Set labels and title
    ax.set_xlabel(r"$\eta$", fontsize=12)
    ax.set_ylabel(r"$\phi$", fontsize=12)
    ax.set_title(r"Jet constituents in $\eta$-$\phi$ plane", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim(-2.5, 2.5)  # eta range
    ax.set_ylim(-np.pi, np.pi)  # phi range

    # Show legend if label provided
    if label is not None and show_legend:
        ax.legend()

    # Set equal aspect ratio to preserve circular jet shapes
    ax.set_aspect("equal")

    return fig


def compare_to_pd(
    all_hists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    prior_key: str = "prior",
    measured_key: str = "nominal",
    truth_key: str = "truthpd",
    prior_label: str = "Prior",
    measured_label: str = "Reweighted",
    truth_label: str = "Truth Pseudodata",
    figsize=(6.4, 4.8),
    ylabel: str = "A.U.",
    xlabel: str = "Obs",
    ylim=None,
    rlim=(0.8, 1.2),
    log_xscale: bool = True,
    linear_yscale: bool = False,
    fig=None,
    ax=None,
    axr=None,
) -> plt.Figure:
    """Compare prior, measured, and truth correlation dimension distributions.

    This function produces a figure comparing the prior, measured (reweighted),
    and truth correlation dimension distributions, similar to the plot function
    in the Plotter class. It creates a two-panel plot with the main distribution
    plot on top and a ratio plot on the bottom.

    Arguments:
    ----------
    all_hists : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Dictionary mapping histogram names to tuples of (dims, dims_var, midbins)
        where:
        - dims: The correlation dimension values
        - dims_var: The variance of the correlation dimension values
        - midbins: The midpoints of the bins used in the calculation
    prior_key : str, optional
        Key in all_hists for the prior distribution (default: "prior").
    measured_key : str, optional
        Key in all_hists for the measured/reweighted distribution (default: "nominal").
    truth_key : str, optional
        Key in all_hists for the truth pseudodata distribution (default: "truthpd").
    prior_label : str, optional
        Label for the prior distribution in the legend (default: "Prior").
    measured_label : str, optional
        Label for the measured distribution in the legend (default: "Reweighted").
    truth_label : str, optional
        Label for the truth distribution in the legend (default: "Truth Pseudodata").
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10, 8)).
    ylabel : str, optional
        Label for the y-axis of the main plot (default: "Correlation Dimension").
    xlabel : str, optional
        Label for the x-axis of the ratio plot (default: "EMD (GeV)").
    ylim : tuple or None, optional
        Limits for the y-axis of the main plot (default: None).
    rlim : tuple, optional
        Limits for the y-axis of the ratio plot (default: (0.8, 1.2)).
    log_xscale : bool, optional
        If True, use logarithmic scale for x-axis (default: True).
    linear_yscale : bool, optional
        If True, use linear scale for y-axis (default: False, i.e., log scale).
    fig : matplotlib.figure.Figure or None, optional
        Existing figure to plot on. If None, creates a new figure (default: None).
    ax : matplotlib.axes.Axes or None, optional
        Existing main axis to plot on. If None, creates new axes (default: None).
    axr : matplotlib.axes.Axes or None, optional
        Existing ratio axis to plot on. If None, creates new axes (default: None).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """

    # Extract the histograms
    if prior_key not in all_hists:
        available = list(all_hists.keys())
        raise KeyError(
            f"Key '{prior_key}' not found in all_hists. Available keys: {available}"
        )
    if measured_key not in all_hists:
        available = list(all_hists.keys())
        raise KeyError(
            f"Key '{measured_key}' not found in all_hists. Available keys: {available}"
        )
    if truth_key not in all_hists:
        available = list(all_hists.keys())
        raise KeyError(
            f"Key '{truth_key}' not found in all_hists. Available keys: {available}"
        )

    prior_dims, prior_dims_var, prior_midbins = all_hists[prior_key]
    measured_dims, measured_dims_var, measured_midbins = all_hists[measured_key]
    truth_dims, truth_dims_var, truth_midbins = all_hists[truth_key]

    # Use truth midbins as the reference for x-axis (all should be the same)
    midbins = truth_midbins

    # Handle case where dims might be shorter than midbins (by 1)
    # This happens because correlation dimension is calculated from differences
    # Pad dims with the first value to match midbins length for plotting
    if len(prior_dims) == len(midbins) - 1:
        prior_dims = np.concatenate([[prior_dims[0]], prior_dims])
    if len(measured_dims) == len(midbins) - 1:
        measured_dims = np.concatenate([[measured_dims[0]], measured_dims])
    if len(truth_dims) == len(midbins) - 1:
        truth_dims = np.concatenate([[truth_dims[0]], truth_dims])

    # Create bin edges from midbins for plotting
    # Approximate bin edges by assuming uniform spacing in log space
    # This is a reasonable approximation for correlation dimension plots
    if len(midbins) > 1:
        # Calculate bin width (assuming logarithmic spacing)
        log_midbins = np.log(midbins)
        dlog = log_midbins[1] - log_midbins[0]
        # Create bin edges
        bin_edges = np.exp(log_midbins - dlog / 2)
        bin_edges = np.append(bin_edges, np.exp(log_midbins[-1] + dlog / 2))
    else:
        # Fallback if we only have one bin
        bin_edges = np.array([midbins[0] * 0.9, midbins[0] * 1.1])

    # Create figure and axes if not provided
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if ax is None or axr is None:
        # Create two-panel layout (main plot and ratio plot)
        this_grid = gs.GridSpec(2, 1, figure=fig, height_ratios=(7, 2), hspace=0.0)
        axr = fig.add_subplot(this_grid[1, 0])
        ax = fig.add_subplot(this_grid[0, 0])

    # Duplicate last values for steps-post style plotting
    prior_plot = np.append(prior_dims, prior_dims[-1])
    measured_plot = np.append(measured_dims, measured_dims[-1])
    truth_plot = np.append(truth_dims, truth_dims[-1])

    # Plot on main axis
    ax.plot(
        bin_edges,
        prior_plot,
        drawstyle="steps-post",
        label=prior_label,
        alpha=0.5,
    )
    ax.fill_between(
        bin_edges,
        0,
        prior_plot,
        step="post",
        alpha=0.5,
        color="#1f77b4",
    )
    ax.plot(
        bin_edges,
        truth_plot,
        drawstyle="steps-post",
        label=truth_label,
    )
    ax.plot(
        bin_edges,
        measured_plot,
        drawstyle="steps-post",
        label=measured_label,
        color="black",
    )

    # Set main plot properties
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    if not linear_yscale:
        ax.set_yscale("log")
    if log_xscale:
        ax.set_xscale("log")
    ax.set_xticks([])
    ax.legend(loc="upper right", frameon=False)

    # Calculate ratios
    # Avoid division by zero
    truth_plot_safe = np.where(truth_plot > 0, truth_plot, np.nan)
    prior_ratio = prior_plot / truth_plot_safe
    measured_ratio = measured_plot / truth_plot_safe

    # Plot ratios
    axr.hlines(1, bin_edges[0], bin_edges[-1], color="black", linestyle="--", alpha=0.8)
    axr.plot(
        bin_edges,
        prior_ratio,
        drawstyle="steps-post",
        alpha=0.5,
    )
    axr.plot(
        bin_edges,
        measured_ratio,
        drawstyle="steps-post",
        color="black",
    )
    axr.set_xlabel(xlabel)
    if log_xscale:
        axr.set_xscale("log")
    axr.set_ylabel("Ratio to truth")
    axr.set_ylim(rlim)
    axr.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig


def plot_uncertainties(
    all_hists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    measured_key: str = "nominal",
    target_key: str = "truthpd",
    figsize=(6.4, 4.8),
    ylabel: str = "Correlation Dimension",
    xlabel: str = "Q [GeV]",
    log_xscale: bool = True,
    linear_yscale: bool = False,
    color: str = "blue",
) -> tuple[plt.Figure, plt.Figure]:
    """Plot cross-section measurement and uncertainty budget.

    This function produces two figures from correlation dimension histograms:
    1. Cross-section plot comparing measured (unfolded) to target,
       with total uncertainty
    2. Uncertainty budget plot showing individual uncertainty contributions

    The function extracts systematic uncertainties from the all_hists
    dictionary by:
    - Computing differences between systematic variations and the nominal
      measurement
    - Computing NN initialization uncertainty from variance across ensemble
      members
    - Using provided variance information for MC statistical uncertainty
    - Computing data statistical uncertainty if available

    Arguments:
    ----------
    all_hists : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Dictionary mapping histogram names to tuples of (dims, dims_var, midbins)
        where:
        - dims: The correlation dimension values
        - dims_var: The variance of the correlation dimension values
        - midbins: The bin edges (despite the name, these are edges not midpoints)
    measured_key : str, optional
        Key in all_hists for the measured/unfolded distribution (default: "nominal").
    target_key : str, optional
        Key in all_hists for the target/truth distribution (default: "truthpd").
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (6.4, 4.8)).
    ylabel : str, optional
        Label for the y-axis (default: "Correlation Dimension").
    xlabel : str, optional
        Label for the x-axis (default: "Q [GeV]").
    log_xscale : bool, optional
        If True, use logarithmic scale for x-axis (default: True).
    linear_yscale : bool, optional
        If True, use linear scale for y-axis (default: False, i.e., log scale).
    color : str, optional
        Color to use for the measured distribution (default: "blue").

    Returns:
    --------
    fig_cross_section : matplotlib.figure.Figure
        Figure with cross-section measurement plot (main plot + ratio).
    fig_uncertainty_budget : matplotlib.figure.Figure
        Figure with uncertainty budget plot showing individual contributions.
    """
    # Extract measured and target histograms
    if measured_key not in all_hists:
        available = list(all_hists.keys())
        raise KeyError(
            f"Key '{measured_key}' not found in all_hists. Available keys: {available}"
        )
    if target_key not in all_hists:
        available = list(all_hists.keys())
        raise KeyError(
            f"Key '{target_key}' not found in all_hists. Available keys: {available}"
        )

    measured_dims, measured_dims_var, measured_midbins = all_hists[measured_key]
    target_dims, target_dims_var, target_midbins = all_hists[target_key]

    # Use measured midbins as reference (all should be the same)
    # Note: despite the name "midbins", these are actually bin edges
    bin_edges = measured_midbins

    # Define systematic uncertainty names and properties
    syst_definitions = {
        "nn-init": {
            "name": "NN Init",
            "color": "aqua",
            "stochastic": True,
            "prefix": "nn-init-",
        },
        "track-eff": {
            "name": "Track eff.",
            "color": "purple",
            "stochastic": False,
            "prefix": None,
        },
        "jet-track-eff": {
            "name": "Jet track eff.",
            "color": "pink",
            "stochastic": False,
            "prefix": None,
        },
        "track-fake": {
            "name": "Track fake",
            "color": "brown",
            "stochastic": False,
            "prefix": None,
        },
        "track-scale": {
            "name": "Track scale",
            "color": "gray",
            "stochastic": False,
            "prefix": None,
        },
        "muon-id": {
            "name": "Muon ID",
            "color": "lightgreen",
            "stochastic": False,
            "prefix": None,
        },
        "muon-ms": {
            "name": "Muon MS",
            "color": "lightblue",
            "stochastic": False,
            "prefix": None,
        },
        "muon-resbias": {
            "name": "Muon resolution bias",
            "color": "deepskyblue",
            "stochastic": False,
            "prefix": None,
        },
        "muon-scale": {
            "name": "Muon scale",
            "color": "teal",
            "stochastic": False,
            "prefix": None,
        },
        "dd": {
            "name": "Data driven",
            "color": "red",
            "stochastic": False,
            "prefix": None,
        },
        "hv": {
            "name": "Hidden variable",
            "color": "orange",
            "stochastic": False,
            "prefix": None,
        },
        "data-stat": {
            "name": "Data stat",
            "color": "blue",
            "stochastic": True,
            "prefix": None,
        },
    }

    # Calculate systematic uncertainties
    syst_vars = {}
    syst_info = {}

    # MC statistical uncertainty (from variance in measured distribution)
    syst_vars["mc-stat"] = measured_dims_var
    syst_info["mc-stat"] = {
        "name": "MC stat",
        "color": "green",
        "stochastic": True,
    }

    # NN initialization uncertainty (from variance across ensemble members)
    nn_init_keys = [
        key
        for key in all_hists.keys()
        if key.startswith("nn-init-") and key != "nn-init"
    ]
    if nn_init_keys:
        # Extract all ensemble member histograms
        ensemble_dims = []
        for key in sorted(nn_init_keys):
            dims, _, _ = all_hists[key]
            ensemble_dims.append(dims)

        if ensemble_dims:
            ensemble_dims = np.array(ensemble_dims)
            # Calculate variance across ensemble members
            # Normalize each member to match the nominal
            for i in range(len(ensemble_dims)):
                norm_factor = np.sum(measured_dims) / np.sum(ensemble_dims[i])
                ensemble_dims[i] *= norm_factor
            nn_init_var = np.var(ensemble_dims, axis=0) / (len(ensemble_dims) - 1)
            syst_vars["nn-init"] = nn_init_var
            syst_info["nn-init"] = syst_definitions["nn-init"]

    # Other systematic uncertainties (from differences with nominal)
    for syst_key, syst_def in syst_definitions.items():
        if syst_key in ["nn-init", "mc-stat", "data-stat"]:
            continue  # Already handled

        # Check if this systematic exists in all_hists
        if syst_key in all_hists:
            syst_dims, _, _ = all_hists[syst_key]

            # Normalize to match measured distribution
            norm_factor = np.sum(measured_dims) / np.sum(syst_dims)
            syst_dims *= norm_factor

            # Calculate variance as squared difference
            syst_var = np.abs(syst_dims - measured_dims) ** 2
            syst_vars[syst_key] = syst_var
            syst_info[syst_key] = syst_def

    # Data statistical uncertainty (if available)
    if "data-stat" in all_hists:
        _, data_var, _ = all_hists["data-stat"]
        syst_vars["data-stat"] = data_var
        syst_info["data-stat"] = syst_definitions["data-stat"]

    # Apply uncertainty merging (same as uncertainty_plotter.py)
    # Define uncertainty merging groups
    uncertainty_groups = {
        "Tracking": ["track-eff", "jet-track-eff", "track-fake", "track-scale"],
        "Unfolding": ["dd", "hv"],
        "Muon": ["muon-id", "muon-ms", "muon-resbias", "muon-rho", "muon-scale"],
    }
    hide_individual_uncertainties = True

    # Merge uncertainties in quadrature
    merged_uncertainties = {}
    hidden_uncertainties = {}
    for group_name, individual_uncertainties in uncertainty_groups.items():
        # Check which uncertainties are available
        available_uncertainties = []
        for uncert_name in individual_uncertainties:
            if uncert_name in syst_vars:
                available_uncertainties.append(uncert_name)

        if not available_uncertainties:
            continue

        # Merge variances in quadrature
        merged_var = np.sum(
            [syst_vars[uncert] for uncert in available_uncertainties],
            axis=0,
        )

        # Create merged uncertainty entry
        merged_uncertainties[group_name] = {
            "name": group_name.title(),
            "color": syst_info[available_uncertainties[0]]["color"],
            "var": merged_var,
            "merged_from": available_uncertainties,
        }

        # Hide individual uncertainties if enabled
        if hide_individual_uncertainties:
            for uncert_name in available_uncertainties:
                hidden_uncertainties[uncert_name] = {
                    "var": syst_vars[uncert_name],
                    "info": syst_info[uncert_name],
                }
                del syst_vars[uncert_name]
                del syst_info[uncert_name]

    # Add merged uncertainties to syst_vars and syst_info
    for group_name, merged_uncert in merged_uncertainties.items():
        syst_vars[group_name] = merged_uncert["var"]
        syst_info[group_name] = {
            "name": merged_uncert["name"],
            "color": merged_uncert["color"],
            "stochastic": False,
        }

    # Calculate total variance and uncertainty
    total_var = np.sum(list(syst_vars.values()), axis=0)
    total_uncert = np.sqrt(total_var)
    rel_total_uncert = total_uncert / np.where(measured_dims > 0, measured_dims, 1)

    # Normalize target to match measured
    norm_factor = np.sum(measured_dims) / np.sum(target_dims)
    norm_target_dims = norm_factor * target_dims

    # Calculate ratio
    ratio = measured_dims / np.where(norm_target_dims > 0, norm_target_dims, 1)

    # Calculate method bias
    mbias = (measured_dims - norm_target_dims) ** 2
    rel_mbias = np.sqrt(mbias) / np.where(norm_target_dims > 0, norm_target_dims, 1)

    # Duplicate last values for step plots
    measured_plot = np.append(measured_dims, measured_dims[-1])
    norm_target_plot = np.append(norm_target_dims, norm_target_dims[-1])
    rel_total_uncert_plot = np.append(rel_total_uncert, rel_total_uncert[-1])
    if len(rel_mbias) > 0:
        rel_mbias_plot = np.append(rel_mbias, rel_mbias[-1])
    else:
        rel_mbias_plot = rel_mbias

    # Create bin centers and errors for errorbar plots
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_errors = (bin_edges[1:] - bin_edges[:-1]) / 2

    # ===== Figure 1: Cross-section plot =====
    fig_cross_section, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    plt.subplots_adjust(hspace=0, top=0.95)

    # Main plot
    ax.plot(
        bin_edges,
        norm_target_plot,
        "--",
        label="Target",
        color="black",
        drawstyle="steps-post",
    )
    ax.errorbar(
        bin_centers,
        measured_dims,
        yerr=total_uncert,
        fmt="o",
        label="Unfolded",
        color=color,
    )

    # Set plot properties
    if not linear_yscale:
        ax.set_yscale("log")
    if log_xscale:
        ax.set_xscale("log")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.legend()
    ax.tick_params(axis="x", direction="in", top=True)

    # Ratio plot
    rax.axhline(1, color="black", linestyle="--")
    rax.errorbar(
        bin_centers,
        ratio,
        xerr=bin_errors,
        yerr=rel_total_uncert,
        fmt="o",
        color=color,
    )
    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.5, 1.0, 1.5])
    rax.set_ylabel("Ratio to target")
    rax.set_xlabel(xlabel)
    if log_xscale:
        rax.set_xscale("log")
    rax.set_xlim(bin_edges[0], bin_edges[-1])
    rax.tick_params(axis="x", direction="in", bottom=True, top=False)
    rax.grid(True, alpha=0.3)

    fig_cross_section.tight_layout()
    fig_cross_section.subplots_adjust(hspace=0, top=0.95)

    # ===== Figure 2: Uncertainty budget plot =====
    fig_uncertainty_budget, ax = plt.subplots(figsize=figsize)

    # Plot total uncertainty
    ax.plot(
        bin_edges,
        rel_total_uncert_plot,
        "--",
        color="black",
        label="Total unc.",
        drawstyle="steps-post",
        linewidth=2,
    )

    # Plot individual uncertainties
    for syst_key in syst_vars:
        syst_var = syst_vars[syst_key]
        # Pad for step plot
        syst_var_plot = np.append(syst_var, syst_var[-1])
        rel_syst_uncert = np.sqrt(syst_var_plot) / np.where(
            measured_plot > 0, measured_plot, 1
        )
        ax.plot(
            bin_edges,
            rel_syst_uncert,
            "-",
            color=syst_info[syst_key]["color"],
            label=syst_info[syst_key]["name"],
            drawstyle="steps-post",
        )

    # Plot method bias
    if len(rel_mbias_plot) > 0:
        ax.fill_between(
            bin_edges,
            0,
            rel_mbias_plot,
            step="post",
            color="gray",
            alpha=0.3,
            label="Method bias",
        )

    # Set plot properties
    if log_xscale:
        ax.set_xscale("log")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Uncertainty budget")

    # Set y-axis limits
    if len(rel_mbias_plot) > 0:
        top_uncert = np.max(np.concatenate([rel_total_uncert_plot, rel_mbias_plot]))
    else:
        top_uncert = np.max(rel_total_uncert_plot)
    if top_uncert > 0.2 or np.isnan(top_uncert):
        ax.set_ylim(bottom=0.0, top=0.2)
    else:
        ax.set_ylim(bottom=0.0, top=top_uncert * 1.1)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
    ax.tick_params(axis="x", direction="in", top=True)

    fig_uncertainty_budget.tight_layout()
    fig_uncertainty_budget.subplots_adjust(bottom=0.2)

    return fig_cross_section, fig_uncertainty_budget
