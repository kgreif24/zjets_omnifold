"""
Visualization functions for jet analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import vector
import uncertainties


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


def compare_to_target(
    all_hists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    prior_key: str = "prior",
    measured_key: str = "nominal",
    target_key: str = "truthpd",
    prior_label: str = "Prior",
    measured_label: str = "Reweighted",
    target_label: str = "Truth Pseudodata",
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
    if target_key not in all_hists:
        available = list(all_hists.keys())
        raise KeyError(
            f"Key '{target_key}' not found in all_hists. Available keys: {available}"
        )

    prior_hist, _, bin_edges = all_hists[prior_key]
    measured_hist, _, _ = all_hists[measured_key]
    target_hist, _, _ = all_hists[target_key]

    # Create figure and axes if not provided
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if ax is None or axr is None:
        # Create two-panel layout (main plot and ratio plot)
        this_grid = gs.GridSpec(2, 1, figure=fig, height_ratios=(7, 2), hspace=0.0)
        axr = fig.add_subplot(this_grid[1, 0])
        ax = fig.add_subplot(this_grid[0, 0])

    # Duplicate last values for steps-post style plotting
    prior_plot = np.append(prior_hist, prior_hist[-1])
    measured_plot = np.append(measured_hist, measured_hist[-1])
    target_plot = np.append(target_hist, target_hist[-1])

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
        target_plot,
        drawstyle="steps-post",
        label=target_label,
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
    target_plot_safe = np.where(target_plot > 0, target_plot, np.nan)
    prior_ratio = prior_plot / target_plot_safe
    measured_ratio = measured_plot / target_plot_safe

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


def plot_measurement_with_uncertainties(
    all_hists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    measured_key: str = "nominal",
    measured_label: str = "Reweighted",
    target_key: str = "truthpd",
    target_label: str = "Truth Pseudodata",
    target2_key: str = None,
    target2_label: str = "MadGraph",
    data_measurement_mode: bool = False,
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

    The function uses UncertaintyCalculator to extract systematic uncertainties
    from the all_hists dictionary

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
    measured_label : str, optional
        Label for the measured distribution in the legend (default: "Reweighted").
    target_key : str, optional
        Key in all_hists for the target/truth distribution (default: "truthpd").
    target_label : str, optional
        Label for the target distribution in the legend (default: "Truth Pseudodata").
    target2_key : str, optional
        Key in all_hists for the second target distribution. If provided and
        data_measurement_mode is True, both targets will be plotted (default: None).
    target2_label : str, optional
        Label for the second target distribution in the legend (default: "MadGraph").
    data_measurement_mode : bool, optional
        If True, compares data measurement to truth generators
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

    measured_hist, _, bin_edges = all_hists[measured_key]
    target_hist, _, _ = all_hists[target_key]

    # Extract second target if provided
    target2_hist = None
    if target2_key is not None:
        if target2_key not in all_hists:
            available = list(all_hists.keys())
            raise KeyError(
                f"Key '{target2_key}' not found in all_hists. "
                f"Available keys: {available}"
            )
        target2_hist, _, _ = all_hists[target2_key]

    # Calculate method bias (only in standard mode, not data comparison mode)
    if data_measurement_mode:
        mbias = None
        rel_mbias = None
    else:
        mbias = (measured_hist - target_hist) ** 2
        rel_mbias = np.sqrt(mbias) / np.where(target_hist > 0, target_hist, 1)

    # Create bin centers and errors for errorbar plots
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_errors = (bin_edges[1:] - bin_edges[:-1]) / 2

    # Normalize targets to match measured
    norm_factor = np.sum(measured_hist) / np.sum(target_hist)
    norm_target_hist = norm_factor * target_hist
    if target2_hist is not None:
        norm_factor2 = np.sum(measured_hist) / np.sum(target2_hist)
        norm_target2_hist = norm_factor2 * target2_hist

    # Create uncertainty calculator (using default definitions)
    uncertainty_calculator = uncertainties.UncertaintyCalculator()

    # Calculate systematic uncertainties using UncertaintyCalculator
    syst_vars, syst_info = uncertainty_calculator.calculate_uncertainties(
        all_hists, measured_key=measured_key
    )

    # Calculate total variance and uncertainty
    total_uncert = uncertainty_calculator.get_total_uncertainty(
        all_hists, measured_key=measured_key
    )

    # Duplicate last values for step plots (only used in pseudo measurement mode)
    target_plot = np.append(norm_target_hist, norm_target_hist[-1])
    # Calculate relative total uncertainty for uncertainty budget plot
    rel_total_uncert = total_uncert / np.where(measured_hist > 0, measured_hist, 1)
    rel_total_uncert_plot = np.append(rel_total_uncert, rel_total_uncert[-1])

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
    if data_measurement_mode:
        # In data measurement mode: targets as colored points, measured as black points
        ax.errorbar(
            bin_centers,
            norm_target_hist,
            fmt="o",
            label=target_label,
            color="purple",
        )

        # Add second target if provided
        if norm_target2_hist is not None:
            ax.errorbar(
                bin_centers,
                norm_target2_hist,
                fmt="o",
                label=target2_label,
                color="orange",
            )

        ax.errorbar(
            bin_centers,
            measured_hist,
            xerr=bin_errors,
            yerr=total_uncert,
            fmt="+",
            label=measured_label,
            color="black",
        )
    else:
        # Standard mode: target as dashed line, measured as colored points
        ax.plot(
            bin_edges,
            target_plot,
            "--",
            label=target_label,
            color="black",
            drawstyle="steps-post",
        )
        ax.errorbar(
            bin_centers,
            measured_hist,
            yerr=total_uncert,
            fmt="o",
            label=measured_label,
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
    if data_measurement_mode:
        # In data measurement mode: ratio is target/measured
        # Avoid division by zero
        measured_safe = np.where(measured_hist > 0, measured_hist, np.nan)
        ratio = norm_target_hist / measured_safe
        rax.errorbar(
            bin_centers,
            ratio,
            xerr=bin_errors,
            yerr=rel_total_uncert,
            fmt="o",
            color="purple",
        )

        # Add second target ratio if provided
        if norm_target2_hist is not None:
            ratio2 = norm_target2_hist / measured_safe
            rax.errorbar(
                bin_centers,
                ratio2,
                xerr=bin_errors,
                yerr=rel_total_uncert,
                fmt="o",
                color="orange",
            )

        rax.set_ylabel("Ratio")
    else:
        # Standard mode: ratio is measured/target
        rax.errorbar(
            bin_centers,
            measured_hist / target_plot[:-1],  # Remove duplicated last bin
            xerr=bin_errors,
            yerr=rel_total_uncert,
            fmt="o",
            color=color,
        )
        rax.set_ylabel("Ratio to target")

    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.5, 1.0, 1.5])
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
        rel_var_uncert = syst_vars[syst_key] / measured_hist**2
        plot_syst_uncert = np.sqrt(np.append(rel_var_uncert, rel_var_uncert[-1]))
        ax.plot(
            bin_edges,
            plot_syst_uncert,
            "-",
            color=syst_info[syst_key]["color"],
            label=syst_info[syst_key]["name"],
            drawstyle="steps-post",
        )

    # Plot method bias (only in standard mode, not data comparison mode)
    if rel_mbias is not None:
        plot_mbias = np.append(rel_mbias, rel_mbias[-1])
        ax.fill_between(
            bin_edges,
            0,
            plot_mbias,
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
    if rel_mbias is not None:
        top_uncert = np.max(np.concatenate([rel_total_uncert_plot, plot_mbias]))
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
