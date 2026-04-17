"""
Visualization functions for jet analysis.
"""

import re
import io
import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh
import matplotlib.gridspec as gs
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import uncertainties
import scipy.stats as stats
from typing import Optional
from matplotlib.collections import PatchCollection

# Set ATLAS plotting style
mh.style.use("ATLAS")

# DPI used when saving figures to disk (e.g. via pdf_name in draw_plot)
SAVE_DPI = 200


def _strip_latex(s):
    """Return a terminal-friendly version of a LaTeX label string."""
    s = re.sub(r'\\(?:text|mathrm)\{([^}]*)\}', r'\1', s)  # \text{X} -> X
    s = re.sub(r'_\{([^}]*)\}', r'_\1', s)  # _{X} -> _X
    s = re.sub(r'\^\{([^}]*)\}', r'^\1', s)  # ^{X} -> ^X
    s = s.replace('$', '').replace('\\', '')
    return s.strip()


def figs_to_grid(figs, ncols=4):
    """Render a list of figures as rasterised thumbnails arranged in a grid."""
    n = len(figs)
    nrows = (n + ncols - 1) // ncols
    grid_fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes_flat = np.array(axes).flatten()
    for ax, fig in zip(axes_flat, figs):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        ax.imshow(plt.imread(buf))
        ax.axis("off")
        buf.close()
    for ax in axes_flat[n:]:
        ax.axis("off")
    grid_fig.tight_layout(pad=0.3)
    return grid_fig


def draw_textbox(ax, box):
    """Draw a top-left aligned text box on the axes with a rounded border.

    Args:
        ax: matplotlib axes to draw on.
        box: dict with keys:
            - left_align, top_align: position in axes fraction (0-1)
            - text: string to display
            - fontsize: font size
    """
    ax.text(
        box["left_align"],
        box["top_align"],
        box["text"],
        fontsize=box["fontsize"],
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="none", edgecolor="none", alpha=0
        ),
        transform=ax.transAxes,
    )


def get_nnid_uncertainties(
    results_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    index: int,
    measured_key: str = "nominal",
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict]]:
    """Calculate uncertainties and uncertainty budget for NNID results.

    Arguments:
    ----------
    results_dict : dict
        Dictionary mapping weight set names to (nnids, avg_r) tuples.
    index : int
        0 for NNID uncertainties, 1 for avg_r uncertainties.
    measured_key : str
        Key for the nominal results.

    Returns:
    --------
    abs_uncert : np.ndarray
        Absolute total uncertainty array.
    syst_uncerts : dict[str, np.ndarray]
        Dictionary mapping uncertainty names to fractional uncertainty arrays.
    syst_info : dict[str, dict]
        Dictionary mapping uncertainty names to metadata (name, color, etc.).
    """
    # Create UncertaintyCalculator
    calc = uncertainties.UncertaintyCalculator()

    # Convert results to format expected by UncertaintyCalculator
    hists = {}
    for key, val in results_dict.items():
        # val[index] is the array of values (nnids or avg_r)
        # We provide zero for variance initially
        hists[key] = (val[index], np.zeros_like(val[index]), None)

    # Get signed individual uncertainty components, then process to absolute grouped
    signed_uncerts, syst_covs_individual, syst_info_individual = (
        calc.calculate_uncertainties(hists, measured_key=measured_key, smooth_hv=False)
    )
    syst_uncerts, _, syst_info = calc.process_signed_uncertainties(
        signed_uncerts, syst_covs_individual, syst_info_individual
    )

    # Compute total fractional uncertainty as sqrt(sum of squares)
    total_rel_uncert = np.sqrt(
        np.sum(np.array(list(syst_uncerts.values())) ** 2, axis=0)
    )

    # Convert to absolute uncertainty
    nominal_vals = results_dict[measured_key][index]
    abs_uncert = total_rel_uncert * nominal_vals

    return abs_uncert, syst_uncerts, syst_info


def get_theory_nnid_uncertainties(
    results_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    index: int,
    measured_key: str = "nominal",
    is_madgraph: bool = True,
) -> np.ndarray:
    """Calculate absolute theory uncertainties for NNID results.

    Arguments:
    ----------
    results_dict : dict
        Dictionary mapping weight set names to (nnids, avg_r) tuples.
        Should contain the nominal key and theory variation keys like
        "weights_theoryQCD", "weights_theoryPDF", etc.
    index : int
        0 for NNID uncertainties, 1 for avg_r uncertainties.
    measured_key : str
        Key for the nominal results.
    is_madgraph : bool
        If True, use MadGraph theory uncertainties.
        If False, use Sherpa theory uncertainties.

    Returns:
    --------
    abs_uncert : np.ndarray
        Absolute uncertainty array.
    """
    # Create UncertaintyCalculator
    calc = uncertainties.UncertaintyCalculator()

    # Convert results to format expected by UncertaintyCalculator
    # MC stat is neglected since truth generator samples are very large
    hists = {}
    for key, val in results_dict.items():
        # val[index] is the array of values (nnids or avg_r)
        # We provide zero for variance (no MC stat)
        hists[key] = (val[index], np.zeros_like(val[index]), None)

    # Get total fractional theory uncertainty
    rel_uncert = calc.get_total_theory_uncertainty(
        hists, measured_key=measured_key, is_madgraph=is_madgraph
    )

    # Convert to absolute uncertainty
    nominal_vals = results_dict[measured_key][index]
    return rel_uncert * nominal_vals


def plot_nnid_uncert_budget(
    combined_results: dict[str, tuple[np.ndarray, np.ndarray]],
    measured_key: str = "nominal",
    target_results: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    target_key: str = "truthpd",
    low_limit: int = 0,
    high_limit: Optional[int] = None,
    thresholds: Optional[np.ndarray] = None,
    figsize: tuple = (6.4, 4.8),
    llab: str = "Simulation Internal",
    rlab: str = "Anti-kt $R=1.0$ jets\n$p_T \\in [330, 370]$ GeV",
    data: bool = False,
) -> tuple[plt.Figure, plt.Figure]:
    """Plot uncertainty budget for NNID measurement.

    Produces two figures: one for NNID (y-axis) uncertainties and one for
    avg_r (x-axis) uncertainties. Both are plotted as a function of point
    index.

    Arguments:
    ----------
    combined_results : dict
        Dictionary of measurement results including systematic variations.
        Should contain the nominal key and systematic variation keys.
    measured_key : str
        Key for the nominal measurement results.
    target_results : dict, optional
        Dictionary of target results (e.g., truth pseudodata).
        If provided, method bias will be computed and shown.
    target_key : str
        Key for the target results (default: "truthpd").
    low_limit : int
        Low limit for the subset of thresholds to plot.
    high_limit : int, optional
        High limit for the subset of thresholds to plot. If None, plot all thresholds.
    thresholds : np.ndarray, optional
        Array of point indices for x-axis. If None, uses np.arange(n_thresholds).
    figsize : tuple
        Figure size.
    llab : str
        Left label for ATLAS label.
    rlab : str
        Right label for ATLAS label.
    data : bool
        If True, use data labels (default: False).

    Returns:
    --------
    fig_nnid : matplotlib.figure.Figure
        Figure with NNID (y-axis) uncertainty budget.
    fig_avgr : matplotlib.figure.Figure
        Figure with avg_r (x-axis) uncertainty budget.
    """
    # Get uncertainty components for both axes
    # index 0 is NNID (y), index 1 is avg_r (x)
    _, syst_uncerts_nnid, syst_info_nnid = get_nnid_uncertainties(
        combined_results, index=0, measured_key=measured_key
    )
    _, syst_uncerts_avgr, syst_info_avgr = get_nnid_uncertainties(
        combined_results, index=1, measured_key=measured_key
    )

    # Get nominal values
    y_mc, x_mc, _ = combined_results[measured_key]

    # Apply slicing
    slice_indices = slice(low_limit, high_limit)
    n_thresholds = len(y_mc[slice_indices])

    # Use provided point indices or default to 0, 1, 2, ...
    if thresholds is None:
        plot_indices = np.arange(n_thresholds)
    else:
        plot_indices = thresholds[slice_indices]

    # Calculate total fractional uncertainties (sliced)
    total_uncert_nnid = np.sqrt(
        np.sum(
            np.array(list(syst_uncerts_nnid.values()))[:, slice_indices] ** 2, axis=0
        )
    )
    total_uncert_avgr = np.sqrt(
        np.sum(
            np.array(list(syst_uncerts_avgr.values()))[:, slice_indices] ** 2, axis=0
        )
    )

    # Calculate method bias if target is provided
    rel_mbias_nnid = None
    rel_mbias_avgr = None
    if target_results is not None and target_key in target_results:
        y_target, x_target, _ = target_results[target_key]
        # Method bias as absolute difference divided by target (fractional)
        # Avoid division by zero
        y_target_safe = np.where(np.abs(y_target) > 0, y_target, 1)
        x_target_safe = np.where(np.abs(x_target) > 0, x_target, 1)
        rel_mbias_nnid = np.abs(y_mc - y_target) / np.abs(y_target_safe)
        rel_mbias_avgr = np.abs(x_mc - x_target) / np.abs(x_target_safe)
        rel_mbias_nnid = rel_mbias_nnid[slice_indices]
        rel_mbias_avgr = rel_mbias_avgr[slice_indices]

    # ===== Figure 1: NNID (y-axis) uncertainty budget =====
    fig_nnid, ax_nnid = plt.subplots(figsize=figsize)

    # Plot total uncertainty
    ax_nnid.plot(
        plot_indices,
        total_uncert_nnid,
        "--",
        color="black",
        label="Total unc.",
        linewidth=2,
    )

    # Plot individual uncertainties
    for syst_name in syst_info_nnid.keys():
        syst_uncert = syst_uncerts_nnid[syst_name][slice_indices]
        ax_nnid.plot(
            plot_indices,
            syst_uncert,
            "-",
            color=syst_info_nnid[syst_name]["color"],
            label=syst_info_nnid[syst_name]["name"],
        )

    # Plot method bias if available
    if rel_mbias_nnid is not None:
        ax_nnid.fill_between(
            plot_indices,
            0,
            rel_mbias_nnid,
            color="gray",
            alpha=0.3,
            label="Method bias",
        )

    # Set plot properties
    ax_nnid.set_xlabel("$i$")
    ax_nnid.set_ylabel("NNID uncertainty")
    ax_nnid.set_xlim(plot_indices[0], plot_indices[-1])
    ax_nnid.set_xscale("log")

    # Set y-axis limits
    if rel_mbias_nnid is not None:
        top_uncert = np.nanmax(np.concatenate([total_uncert_nnid, rel_mbias_nnid]))
    else:
        top_uncert = np.nanmax(total_uncert_nnid)
    if top_uncert > 0.2 or np.isnan(top_uncert):
        ax_nnid.set_ylim(bottom=0.0, top=0.2)
    else:
        ax_nnid.set_ylim(bottom=0.0, top=top_uncert * 1.4)

    ax_nnid.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, -0.1),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    mh.atlas.label(
        ax=ax_nnid,
        llabel=llab if not data else "Internal",
        data=data,
        rlabel=rlab,
    )

    fig_nnid.tight_layout()
    fig_nnid.subplots_adjust(bottom=0.2)

    # ===== Figure 2: Median EMD uncertainty budget =====
    fig_avgr, ax_avgr = plt.subplots(figsize=figsize)

    # Plot total uncertainty
    ax_avgr.plot(
        plot_indices,
        total_uncert_avgr,
        "--",
        color="black",
        label="Total unc.",
        linewidth=2,
    )

    # Plot individual uncertainties
    for syst_name in syst_info_avgr.keys():
        syst_uncert = syst_uncerts_avgr[syst_name][slice_indices]
        ax_avgr.plot(
            plot_indices,
            syst_uncert,
            "-",
            color=syst_info_avgr[syst_name]["color"],
            label=syst_info_avgr[syst_name]["name"],
        )

    # Plot method bias if available
    if rel_mbias_avgr is not None:
        ax_avgr.fill_between(
            plot_indices,
            0,
            rel_mbias_avgr,
            color="gray",
            alpha=0.3,
            label="Method bias",
        )

    # Set plot properties
    ax_avgr.set_xlabel("$i$")
    ax_avgr.set_ylabel(r"Median EMD uncertainty")
    ax_avgr.set_xlim(plot_indices[0], plot_indices[-1])
    ax_avgr.set_xscale("log")

    # Set y-axis limits
    if rel_mbias_avgr is not None:
        top_uncert = np.nanmax(np.concatenate([total_uncert_avgr, rel_mbias_avgr]))
    else:
        top_uncert = np.nanmax(total_uncert_avgr)
    if top_uncert > 0.2 or np.isnan(top_uncert):
        ax_avgr.set_ylim(bottom=0.0, top=0.2)
    else:
        ax_avgr.set_ylim(bottom=0.0, top=top_uncert * 1.4)

    ax_avgr.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, -0.1),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    mh.atlas.label(
        ax=ax_avgr,
        llabel=llab if not data else "Internal",
        data=data,
        rlabel=rlab,
    )

    fig_avgr.tight_layout()
    fig_avgr.subplots_adjust(bottom=0.2)

    return fig_nnid, fig_avgr


def plot_nnid_pseudodata(
    mc_results: dict[str, tuple[np.ndarray, np.ndarray]],
    pd_results: dict[str, tuple[np.ndarray, np.ndarray]],
    hv_results: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    low_limit: int = 0,
    high_limit: Optional[int] = None,
    thresholds: Optional[np.ndarray] = None,
    mc_label: str = "Measurement",
    pd_label: str = "Truth Pseudodata",
    figsize=(6.4, 4.8),
    xlim: tuple[float, float] = (6, 70),
    ylim: tuple[float, float] = (0, 10),
    xlabel: str = r"Median EMD [GeV]",
    ylabel: str = "NNID",
    llab: str = "Simulation Internal",
    rlab: str = "Anti-kt $R=1.0$ jets\n$p_T \\in [330, 370]$ GeV",
    color: str = "blue",
    plot_uncertainty_budget: bool = False,
    show_connector_lines: bool = False,
) -> plt.Figure | tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Plot NNID results with uncertainties.

    Arguments:
    ----------
    mc_results : dict
        Dictionary of measurement results (e.g. truth_mc_results).
    pd_results : dict
        Dictionary of pseudodata results (e.g. truth_pd_results).
    hv_results : dict, optional
        Dictionary of hidden variable results (e.g. truth_hv_results).
    low_limit : int
        Low limit for the subset of thresholds to plot.
    high_limit : int, optional
        High limit for the subset of thresholds to plot. If None, plot all thresholds.
        Take a subset of the thresholds in the results indexed by these limits
    thresholds : np.ndarray, optional
        Array of point indices for uncertainty budget x-axis. If None, uses
        np.arange(n_thresholds).
    mc_label : str
        Label for the measurement in the legend.
    pd_label : str
        Label for the pseudodata in the legend.
    figsize : tuple
        Figure size.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    color : str
        Color for the measurement crosses.
    plot_uncertainty_budget : bool
        If True, also generate uncertainty budget plots for NNID and avg_r.
    show_connector_lines : bool
        If True, draw faint lines connecting thresholds at the same location:
        prior to truth pseudodata (light gray), measured to truth pseudodata
        (light blue).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The produced main figure. If plot_uncertainty_budget is True, returns
        a tuple of (main_fig, nnid_uncert_fig, avgr_uncert_fig).
    """
    # Merge HV results into MC results for uncertainty calculation if provided
    combined_results = mc_results.copy()
    if hv_results is not None:
        combined_results.update(hv_results)

    # Calculate absolute uncertainties for both axes
    # index 0 is NNID (y), index 1 is avg_r (x)
    dy, _, _ = get_nnid_uncertainties(combined_results, index=0)
    dx, _, _ = get_nnid_uncertainties(combined_results, index=1)

    # Extract nominal measurement and pseudodata
    # Results are stored as (nnids, avg_ri, avg_rj)
    y_prior, x_prior, _ = mc_results["prior"]
    y_mc, x_mc, _ = mc_results["nominal"]
    y_pd, x_pd, _ = pd_results["truthpd"]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot truth pseudodata as black points
    ax.scatter(
        x_prior[low_limit:high_limit],
        y_prior[low_limit:high_limit],
        color="gray",
        label="Prior",
        s=15,
        zorder=1,
    )
    ax.scatter(
        x_pd[low_limit:high_limit],
        y_pd[low_limit:high_limit],
        color="black",
        label=pd_label,
        s=15,
        zorder=2,
    )

    # Plot measurement as blue crosses with error bars in both directions
    ax.errorbar(
        x_mc[low_limit:high_limit],
        y_mc[low_limit:high_limit],
        xerr=dx[low_limit:high_limit],
        yerr=dy[low_limit:high_limit],
        fmt="+",
        color=color,
        label=mc_label,
        markersize=8,
        zorder=3,
        capsize=2,
        linewidth=1,
    )

    if show_connector_lines:
        x_prior_s = x_prior[low_limit:high_limit]
        y_prior_s = y_prior[low_limit:high_limit]
        x_mc_s = x_mc[low_limit:high_limit]
        y_mc_s = y_mc[low_limit:high_limit]
        x_pd_s = x_pd[low_limit:high_limit]
        y_pd_s = y_pd[low_limit:high_limit]
        n_pts = len(x_pd_s)
        for i in range(n_pts):
            ax.plot(
                [x_prior_s[i], x_pd_s[i]],
                [y_prior_s[i], y_pd_s[i]],
                color="lightgray",
                alpha=1.0,
                zorder=0,
                linewidth=0.8,
            )
            ax.plot(
                [x_mc_s[i], x_pd_s[i]],
                [y_mc_s[i], y_pd_s[i]],
                color="lightblue",
                alpha=1.0,
                zorder=0,
                linewidth=0.8,
            )

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

    mh.atlas.label(
        ax=ax,
        llabel=llab,
        rlabel=rlab,
    )

    if plot_uncertainty_budget:
        # Generate uncertainty budget plots with method bias
        fig_nnid, fig_avgr = plot_nnid_uncert_budget(
            combined_results=combined_results,
            measured_key="nominal",
            target_results=pd_results,
            target_key="truthpd",
            low_limit=low_limit,
            high_limit=high_limit,
            thresholds=thresholds,
            figsize=figsize,
            llab=llab,
            rlab=rlab,
            data=False,
        )
        return fig, fig_nnid, fig_avgr

    return fig


def plot_nnid_data(
    mc_results: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    hv_results: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    madgraph_results: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    sherpa_results: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    low_limit: int = 0,
    high_limit: Optional[int] = None,
    thresholds: Optional[np.ndarray] = None,
    mc_label: str = "Data",
    madgraph_label: str = "MadGraph",
    sherpa_label: str = "Sherpa",
    figsize: tuple[float, float] = (6.4, 4.8),
    xlim: tuple[float, float] = (6, 70),
    ylim: tuple[float, float] = (0, 10),
    yscale: str = "linear",
    xlabel: str = r"Median EMD [GeV]",
    ylabel: str = "NNID",
    llab: str = "Internal",
    rlab: str = "Anti-kt $R=1.0$ jets\n$p_T \\in [330, 370]$ GeV",
    plot_uncertainty_budget: bool = False,
) -> plt.Figure | tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Plot NNID data measurement compared to truth generators.

    Arguments:
    ----------
    mc_results : dict, optional
        Dictionary of measurement results (e.g. truth_mc_results). If None,
        no data measurement is plotted. hv_results must also be provided.
    hv_results : dict, optional
        Dictionary of hidden variable results for uncertainty calculation.
        Required if mc_results is provided.
    madgraph_results : dict, optional
        Dictionary of MadGraph truth generator results. If None, not plotted.
    sherpa_results : dict, optional
        Dictionary of Sherpa truth generator results. If None, not plotted.
    low_limit : int
        Low limit for the subset of threshold to plot.
    high_limit : int, optional
        High limit for the subset of threshold to plot. If None, plot all threshold.
    thresholds : np.ndarray, optional
        Array of point indices for uncertainty budget x-axis. If None, uses
        np.arange(n_thresholds).
    mc_label : str
        Label for the measurement in the legend.
    madgraph_label : str
        Label for the MadGraph prediction in the legend.
    sherpa_label : str
        Label for the Sherpa prediction in the legend.
    figsize : tuple
        Figure size.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    yscale : str
        Y-axis scale.
    llab : str
        Left label for ATLAS label.
    rlab : str
        Right label for ATLAS label.
    plot_uncertainty_budget : bool
        If True, also generate uncertainty budget plots for NNID and avg_r.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The produced main figure. If plot_uncertainty_budget is True, returns
        a tuple of (main_fig, nnid_uncert_fig, avgr_uncert_fig).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot MadGraph prediction
    if madgraph_results is not None:
        dy_madgraph = get_theory_nnid_uncertainties(
            madgraph_results, index=0, is_madgraph=True, measured_key="madgraph"
        )
        dx_madgraph = get_theory_nnid_uncertainties(
            madgraph_results, index=1, is_madgraph=True, measured_key="madgraph"
        )
        y_madgraph, x_madgraph, _ = madgraph_results["madgraph"]
        ax.errorbar(
            x_madgraph[low_limit:high_limit],
            y_madgraph[low_limit:high_limit],
            color="purple",
            label=madgraph_label,
            marker="o",
            linestyle="none",
            markersize=4,
            linewidth=1,
        )
        box_height = 2 * dy_madgraph[low_limit:high_limit]
        box_width = 2 * dx_madgraph[low_limit:high_limit]
        mg_areas = box_height * box_width
        mg_area_min, mg_area_max = mg_areas.min(), mg_areas.max()
        mg_alphas = (
            0.1 + 0.4 * (mg_area_max - mg_areas) / (mg_area_max - mg_area_min)
            if mg_area_max > mg_area_min
            else np.full_like(mg_areas, 0.3)
        )
        for x, y, h, w, a in zip(
            x_madgraph[low_limit:high_limit],
            y_madgraph[low_limit:high_limit],
            box_height,
            box_width,
            mg_alphas,
        ):
            ax.add_patch(
                Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    alpha=a,
                    facecolor="purple",
                    edgecolor=None,
                )
            )

    # Plot Sherpa prediction
    if sherpa_results is not None:
        dy_sherpa = get_theory_nnid_uncertainties(
            sherpa_results, index=0, is_madgraph=False, measured_key="sherpa"
        )
        dx_sherpa = get_theory_nnid_uncertainties(
            sherpa_results, index=1, is_madgraph=False, measured_key="sherpa"
        )
        y_sherpa, x_sherpa, _ = sherpa_results["sherpa"]
        ax.plot(
            x_sherpa[low_limit:high_limit],
            y_sherpa[low_limit:high_limit],
            color="orange",
            label=sherpa_label,
            marker="o",
            linestyle="none",
            markersize=4,
            linewidth=1,
        )
        box_height = 2 * dy_sherpa[low_limit:high_limit]
        box_width = 2 * dx_sherpa[low_limit:high_limit]
        sh_areas = box_height * box_width
        sh_area_min, sh_area_max = sh_areas.min(), sh_areas.max()
        sh_alphas = (
            0.1 + 0.2 * (sh_area_max - sh_areas) / (sh_area_max - sh_area_min)
            if sh_area_max > sh_area_min
            else np.full_like(sh_areas, 0.2)
        )
        for x, y, h, w, a in zip(
            x_sherpa[low_limit:high_limit],
            y_sherpa[low_limit:high_limit],
            box_height,
            box_width,
            sh_alphas,
        ):
            ax.add_patch(
                Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    alpha=a,
                    facecolor="orange",
                    edgecolor=None,
                )
            )

    # Plot measurement as black crosses with error bars in both directions
    if mc_results is not None:
        combined_results = mc_results.copy()
        combined_results.update(hv_results)
        dy, _, _ = get_nnid_uncertainties(combined_results, index=0)
        dx, _, _ = get_nnid_uncertainties(combined_results, index=1)
        y_mc, x_mc, _ = mc_results["nominal"]
        ax.errorbar(
            x_mc[low_limit:high_limit],
            y_mc[low_limit:high_limit],
            xerr=dx[low_limit:high_limit],
            yerr=dy[low_limit:high_limit],
            fmt="+",
            color="black",
            label=mc_label,
            markersize=8,
            capsize=2,
            linewidth=1,
        )

    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set legend order: MadGraph (if present), Sherpa (if present), Measurement
    handles, labels = ax.get_legend_handles_labels()
    order = []
    if madgraph_results is not None:
        order.append(labels.index(madgraph_label))
    if sherpa_results is not None:
        order.append(labels.index(sherpa_label))
    if mc_results is not None:
        order.append(labels.index(mc_label))
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
    )

    mh.atlas.label(
        ax=ax,
        llabel=llab,
        data=True,
        rlabel=rlab,
    )

    fig.tight_layout()

    if plot_uncertainty_budget and mc_results is not None:
        # Generate uncertainty budget plots (no method bias for data)
        fig_nnid, fig_avgr = plot_nnid_uncert_budget(
            combined_results=combined_results,
            measured_key="nominal",
            target_results=None,  # No target for data measurement
            low_limit=low_limit,
            high_limit=high_limit,
            thresholds=thresholds,
            figsize=figsize,
            llab=llab,
            rlab=rlab,
            data=True,
        )
        return fig, fig_nnid, fig_avgr

    return fig


def compare_to_target(
    measurement_hists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    target_hists: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    prior_key: str = "prior",
    measured_key: str = "nominal",
    target_key: str = "truthpd",
    prior_label: str = "Prior",
    measured_label: str = "Reweighted",
    target_label: str = "Target",
    llab: str = "Simulation Internal",
    rlab: str = "Z+jets Omnifold",
    normalize: bool = False,
    figsize=(6.4, 4.8),
    ylabel: str = "A.U.",
    xlabel: str = "Obs",
    xlim=None,
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
        Dictionary mapping histogram names to tuples of (dims, dims_var, bins)
        where:
        - dims: The correlation dimension values
        - dims_var: The variance of the correlation dimension values
        - bins: The bin edges used in the calculation
    target_hists : same as measurement_hists but for the target
    prior_key : str, optional
        Key in all_hists for the prior distribution (default: "prior").
    measured_key : str, optional
        Key in all_hists for the measured/reweighted distribution (default: "nominal").
    truth_key : str, optional
        Key in all_hists for the truth pseudodata distribution (default: "truthpd").
    llab : str, optional
        Left label for ATLAS label (default: "Simulation Internal").
    rlab : str, optional
        Right label for ATLAS label (default: "Z+jets Omnifold").
    prior_label : str, optional
        Label for the prior distribution in the legend (default: "Prior").
    measured_label : str, optional
        Label for the measured distribution in the legend (default: "Reweighted").
    truth_label : str, optional
        Label for the truth distribution in the legend (default: "Truth Pseudodata").
    normalize : bool, optional
        If True, normalize the histograms (default: False).
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10, 8)).
    ylabel : str, optional
        Label for the y-axis of the main plot (default: "Correlation Dimension").
    xlabel : str, optional
        Label for the x-axis of the ratio plot (default: "EMD (GeV)").
    xlim : tuple or None, optional
        Limits for the x-axis of the main plot (default: None).
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
    if prior_key not in measurement_hists:
        available = list(measurement_hists.keys())
        raise KeyError(
            f"Key '{prior_key}' not found in all_hists. Available keys: {available}"
        )
    if measured_key not in measurement_hists:
        available = list(measurement_hists.keys())
        raise KeyError(
            f"Key '{measured_key}' not found in all_hists. Available keys: {available}"
        )
    if target_key not in target_hists:
        available = list(target_hists.keys())
        raise KeyError(
            f"Key '{target_key}' not found in all_hists. Available keys: {available}"
        )

    prior_hist, _, bin_edges = measurement_hists[prior_key]
    measured_hist, _, _ = measurement_hists[measured_key]
    target_hist, _, _ = target_hists[target_key]

    # Normalize the histograms if desired
    if normalize:
        prior_hist = prior_hist / np.sum(prior_hist)
        measured_hist = measured_hist / np.sum(measured_hist)
        target_hist = target_hist / np.sum(target_hist)

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
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(bin_edges[0], bin_edges[-1])
    if not linear_yscale:
        ax.set_yscale("log")
    if log_xscale:
        ax.set_xscale("log")
    ax.set_xticks([])
    ax.legend(loc="center right", frameon=False)

    # Calculate ratios
    # Avoid division by zero
    target_plot_safe = np.where(target_plot > 0, target_plot, np.nan)
    prior_ratio = prior_plot / target_plot_safe
    measured_ratio = measured_plot / target_plot_safe

    # Plot ratios
    if xlim is not None:
        axr.set_xlim(xlim)
    else:
        axr.set_xlim(bin_edges[0], bin_edges[-1])
    axr.hlines(
        1,
        axr.get_xlim()[0],
        axr.get_xlim()[1],
        color="black",
        linestyle="--",
        alpha=0.8,
    )
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
    axr.set_ylabel("Ratio")
    axr.set_ylim(rlim)

    mh.atlas.label(
        ax=ax,
        llabel=llab,
        rlabel=rlab,
    )

    return fig


def plot_correlation_matrix(
    total_cov: np.ndarray,
    bins: np.ndarray,
    llab: str = "Simulation Internal",
    figsize: tuple[float, float] = (8, 7),
    simple_labels: bool = False,
) -> plt.Figure:
    """Create a correlation matrix plot from the covariance matrix.

    Arguments:
    ----------
    total_cov : np.ndarray
        Total covariance matrix (n_bins x n_bins).
    bins : np.ndarray
        Array of bin edges for labeling.
    llab : str, optional
        Left label for ATLAS label (default: "Simulation Internal").
    figsize : tuple, optional
        Figure size in inches (width, height). Default: (8, 7).
    simple_labels : bool, optional
        If True, use bin indices instead of bin edge labels and omit
        correlation value annotations. Useful for observables with many
        bins (e.g., EEC). Default: False.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object for the correlation matrix plot.
    """
    # Calculate correlation matrix from covariance matrix
    # correlation[i,j] = covariance[i,j] / sqrt(covariance[i,i] * covariance[j,j])
    n_bins = total_cov.shape[0]
    std_devs = np.sqrt(np.diag(total_cov))

    # Handle zero standard deviations to avoid division by zero
    std_devs = np.where(std_devs == 0, 1, std_devs)

    # Calculate correlation matrix
    corr_matrix = total_cov / np.outer(std_devs, std_devs)

    # Clip values to [-1, 1] to handle numerical precision issues
    corr_matrix = np.clip(corr_matrix, -1, 1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap with origin='lower' to match standard convention
    # (smallest bins at bottom-left)
    im = ax.imshow(
        corr_matrix,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        aspect="equal",
        origin="lower",
    )

    # Add colorbar
    fig.colorbar(im, ax=ax, shrink=0.8)

    if simple_labels:
        # Use bin indices for axes labels (useful for many bins)
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Bin index")
        # Let matplotlib auto-select tick positions for cleaner display
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        # Create bin labels from bin edges (round to hundredths for decimal edges)
        bin_labels = [f"({bins[i]:.2f}, {bins[i+1]:.2f})" for i in range(len(bins) - 1)]

        # Set ticks and labels
        ax.set_xticks(np.arange(n_bins))
        ax.set_yticks(np.arange(n_bins))
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_yticklabels(bin_labels)

        # Add correlation values as text annotations
        for i in range(n_bins):
            for j in range(n_bins):
                # Choose text color based on background for readability
                corr_val = corr_matrix[i, j]
                text_color = "white" if abs(corr_val) < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{corr_val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

    mh.atlas.label(
        loc=0,
        llabel=llab,
        rlabel="",
    )

    return fig


def tprofile(x, y, w, bins):

    nbins = len(bins) - 1

    # compute bin indices
    inds = np.digitize(x, bins) - 1

    # keep only valid bins
    mask = (inds >= 0) & (inds < nbins)
    inds = inds[mask]
    y = y[mask]
    w = w[mask]

    # accumulate sums
    sum_w = np.bincount(inds, weights=w, minlength=nbins)
    sum_wy = np.bincount(inds, weights=w * y, minlength=nbins)
    sum_wy2 = np.bincount(inds, weights=w * y * y, minlength=nbins)

    # compute profile
    mean = sum_wy / sum_w
    variance = (sum_wy2 / sum_w) - mean**2
    error = np.sqrt(variance / sum_w)

    return mean, error, sum_w


def make_error_boxes(
    ax,
    xdata,
    ydata,
    xerror,
    yerror,
    facecolor="r",
    edgecolor="none",
    alpha=0.4,
    label=None,
    marker=".",
    fillstyle="full",
    markersize=5,
    markeredgewidth=2,
):
    """Draw rectangular error boxes with overlaid error bars.

    Visualizes uncertainties using filled boxes plus standard error bars.
    Each box spans the full x and y uncertainty for a point.

    Arguments:
    ----------
    ax : matplotlib.axes.Axes
      Axes to draw on.
    xdata : array-like
      X coordinates.
    ydata : array-like
      Y coordinates.
    xerror : array-like
      X uncertainties (shape: 2 x N).
    yerror : array-like
      Y uncertainties (shape: 2 x N).
    facecolor : str, optional
      Box fill color (default: "r").
    edgecolor : str, optional
      Box edge color (default: "none").
    alpha : float, optional
      Box transparency (default: 0.4).
    label : str or None, optional
      Legend label.
    marker : str, optional
      Marker style (default: ".").
    fillstyle : str, optional
      Marker fill style (default: "full").
    markersize : float, optional
      Marker size (default: 3).
    markeredgewidth : float, optional
      Marker edge width (default: 2).

    Returns:
    --------
    artists : matplotlib.container.ErrorbarContainer
      Errorbar container for the plotted data.
    """
    # Loop over data points; create box from errors at each point
    errorboxes = [
        Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)
    ]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor
    )

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(
        xdata,
        ydata,
        xerr=xerror,
        yerr=yerror,
        linestyle="None",
        linewidth=0,
        label=label,
        marker=marker,
        color=facecolor,
        fillstyle="none",
        markersize=markersize,
        markeredgewidth=markeredgewidth,
    )

    return artists


def make_uncertainty_budget_fig(
    bin_edges,
    uncertainty_details,
    figsize=(6.4, 4.8),
    xlabel="default xlabel",
    log_xscale=False,
    llab: str = "Simulation Internal",
    rlab: str = "Z+jets Omnifold",
    data_measurement_mode=True,
    measured_hist=None,
    target_hist=None,
    do_chi2_test=False,
    simple_corr_labels: bool = False,
    draw_this_group: str = None,
):
    """Create uncertainty budget and correlation matrix plots.

    Produces:
    1. Uncertainty budget plot with total and per-systematic contributions.
    2. Correlation matrix plot from the total covariance.

    Optionally computes method bias and performs a chi-squared test when not in
    data measurement mode.

    Arguments:
    ----------
    bin_edges : array-like
        Bin edges of the observable.
    uncertainty_details : tuple
        (uncertainties, covariance matrices, metadata) per systematic.
    figsize : tuple, optional
        Figure size (default: (6.4, 4.8)).
    xlabel : str, optional
        X-axis label.
    log_xscale : bool, optional
        Use log scale for x-axis (default: False).
    llab : str, optional
        Left ATLAS label.
    rlab : str, optional
        Right ATLAS label.
    data_measurement_mode : bool, optional
        If True, disables method bias and chi2 (default: True).
    measured_hist : array-like, optional
        Measured distribution (required if data_measurement_mode=False).
    target_hist : array-like, optional
        Target distribution (required if data_measurement_mode=False).
    do_chi2_test : bool, optional
        Perform chi-squared test (default: False).
    simple_corr_labels : bool, optional
        Simplify correlation matrix labels (default: False).

    Returns:
    --------
    fig_uncertainty_budget : matplotlib.figure.Figure
        Uncertainty budget plot.
    fig_correlation_matrix : matplotlib.figure.Figure
        Correlation matrix plot.
    """
    if data_measurement_mode:
        mbias = None
        rel_mbias = None
    else:
        if measured_hist is None or target_hist is None:
            raise ValueError(
                "measured_hist and target_hist must be provided "
                "and not None when data_measurement_mode is False"
            )
        mbias = (measured_hist - target_hist) ** 2
        rel_mbias = np.sqrt(mbias) / np.where(target_hist > 0, target_hist, 1)

    total_vars = np.sum(np.array(list(uncertainty_details[0].values())) ** 2, axis=0)
    total_uncert = np.sqrt(total_vars)
    total_uncert_plot = np.append(total_uncert, total_uncert[-1])

    # ===== Figure 2: Uncertainty budget plot =====
    fig_uncertainty_budget, ax = plt.subplots(figsize=figsize)

    # Plot total uncertainty, only for the full uncertainty list
    if draw_this_group is None:
        ax.plot(
            bin_edges,
            total_uncert_plot,
            "--",
            color="black",
            label="Total unc.",
            drawstyle="steps-post",
            linewidth=2,
        )

    # Plot individual uncertainty_details
    for syst_name in uncertainty_details[2].keys():
        if draw_this_group is not None and draw_this_group not in syst_name:
            continue
        syst_uncert = uncertainty_details[0][syst_name]
        plot_syst_uncert = np.append(syst_uncert, syst_uncert[-1])
        ax.plot(
            bin_edges,
            plot_syst_uncert,
            "-",
            color=uncertainty_details[2][syst_name]["color"],
            label=uncertainty_details[2][syst_name]["name"],
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
        top_uncert = np.max(np.concatenate([total_uncert_plot, plot_mbias]))
    else:
        top_uncert = np.max(total_uncert_plot)
    if draw_this_group is not None:  # overrides rel-mbias being not None
        top_uncert = 0
        for syst_name in uncertainty_details[2].keys():
            if draw_this_group in syst_name:
                top_uncert = max(top_uncert, np.max(uncertainty_details[0][syst_name]))
    if top_uncert > 0.2 or np.isnan(top_uncert):
        ax.set_ylim(bottom=0.0, top=0.2)
    else:
        ax.set_ylim(bottom=0.0, top=top_uncert * 1.2)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.4, -0.1),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    mh.atlas.label(
        ax=ax,
        llabel=llab,
        rlabel=rlab,
    )

    fig_uncertainty_budget.tight_layout()
    fig_uncertainty_budget.subplots_adjust(bottom=0.2)

    # Calculate total covariance matrix
    total_cov = np.sum(list(uncertainty_details[1].values()), axis=0)

    # Calculate chi2 covariance matrix (excludes certain systematics)
    chi2_cov = np.sum(
        [
            uncertainty_details[1][key]
            for key in uncertainty_details[1].keys()
            if key not in ["Muon", "Tracking", "lumi", "pileup"]
        ],
        axis=0,
    )
    # Calculate chi-squared test if requested (only when not in data_measurement_mode)
    if do_chi2_test and not data_measurement_mode:
        dof = len(bin_edges) - 1
        D = measured_hist - target_hist
        chi2 = D.dot(np.linalg.inv(chi2_cov)).dot(D.T)
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        feature_label = _strip_latex(xlabel)
        print(f"\n--- χ² test: {feature_label} ---")
        print(f"  dof={dof}, χ²={chi2:.5f}, p-value={p_value:.4f}")

    # ===== Figure 3: Correlation matrix plot =====
    fig_correlation_matrix = None
    if (
        draw_this_group is None
    ):  # only draw correlation matrix for the full covariance
        fig_correlation_matrix = plot_correlation_matrix(
            total_cov=total_cov,
            bins=bin_edges,
            llab=llab,
            simple_labels=simple_corr_labels,
        )

    return fig_uncertainty_budget, fig_correlation_matrix


def nice_midpoint(low, up):
    """
    Compute a 'nice' midpoint between low and up.

    Snaps the midpoint to a "nice" number (multiples of 5, 2.5, or 2 scaled
    to the range size) near the actual center.

    Args:
        low (float): lower bound
        up (float): upper bound

    Returns:
        float: snapped midpoint
    """
    midpoint = 0.5 * (low + up)
    distance = up - low

    # Choose step based on range
    if distance > 5:
        steps = [5, 2.5, 2]
    elif distance > 1:
        steps = [1, 0.75]
    elif distance > 0.5:
        steps = [0.5, 0.25]
    elif distance > 0.2:
        steps = [0.2, 0.1, 0.05]
    else:
        steps = [0.02, 0.01, 0.005]

    # Find the step closest to the range scale
    step = min(steps, key=lambda s: abs(s - distance / 2))

    # Snap midpoint to nearest multiple of step
    snapped = round(midpoint / step) * step
    return snapped


def draw_plot(
    omni_results,
    binning,
    ylabel="default",
    xlabel="default",
    mgfxfx_truth_results=None,
    sherpa_truth_results=None,
    ibu_results=None,
    draw_uncertainty_budget=True,
    is_xSec=True,
    logyScale=True,
    logxScale=False,
    ratio_ylim=[0.2, 1.8],
    text_box=None,
    is_omni_data=False,
    results_list=None,
    formating_dicts=None,
    pdf_name=None,
    do_chi2_test=False,
    llab="Simulation Internal",
    smooth_hv=True,
):
    """Draw measurement with optional theory comparisons and ratio plot.

    Creates a main cross-section (or density) plot with uncertainties,
    optionally including Sherpa and/or MadGraph predictions, and a ratio
    subplot. Can also produce an uncertainty budget figure.

    The OmniFold result (``omni_results``) is always treated as the reference
    in the ratio panel. Additional curves — including truth pseudodata — are
    overlaid via ``results_list`` and ``formating_dicts``.

    Arguments:
    ----------
    omni_results : dict
        Histogram dictionary for the OmniFold result. Must contain a
        ``"nominal"`` key; systematic variation keys are used for the
        uncertainty calculation.
    binning : array-like
        Bin edges of the observable.
    ylabel : str, optional
        Y-axis label.
    xlabel : str, optional
        X-axis label.
    mgfxfx_truth_results : dict, optional
        MadGraph truth histogram dict. Must contain a ``"nominal"`` key
        plus theory variation keys for uncertainty bands.
    sherpa_truth_results : dict, optional
        Sherpa truth histogram dict. Same key requirements as
        ``mgfxfx_truth_results``.
    ibu_results : dict, optional
        IBU result dict with ``"nominal"`` and ``"total_unc"`` keys.
    draw_uncertainty_budget : bool, optional
        If True, also produces an uncertainty budget figure (default: True).
    is_xSec : bool, optional
        If True, converts histograms to cross-section units by dividing by
        bin widths. Set to False if histograms are already normalized or in
        the desired units (default: True).
    logyScale : bool, optional
        If True, use a log y-scale on the main panel (default: True).
    logxScale : bool, optional
        If True, use a log x-scale on both the main and ratio panels
        (default: False).
    ratio_ylim : list, optional
        Y-limits for the ratio subplot (default: [0.2, 1.8]).
    text_box : dict or None, optional
        If provided, draws a text box via ``draw_textbox``. Dict must have
        keys ``left_align``, ``top_align``, ``text``, and ``fontsize``.
    is_omni_data : bool, optional
        If True, labels the ratio y-axis as "MC / Data"; otherwise uses
        "MC / Pseudodata" (default: False).
    results_list : list of dicts, optional
        Additional result dicts to overlay. Each dict must contain a
        ``"nominal"`` key. ``omni_results`` is always the ratio reference.
        Use this to overlay truth pseudodata, e.g.::

            vis.draw_plot(
                omni_results,
                binning,
                results_list=[truth_pd_hists],
                formating_dicts=[{
                    "color": "red",
                    "label": "Truth pseudodata",
                    "marker": "o",
                    "is_omni_data": False,
                    "is_madgraph": True,
                }],
            )

    formating_dicts : list of dicts, optional
        One formatting dict per entry in ``results_list`` (must be the same
        length). Each dict must contain:

        - ``"color"`` — matplotlib color string (e.g. ``"red"``)
        - ``"label"`` — legend label string
        - ``"marker"`` — matplotlib marker string (e.g. ``"o"``)
        - ``"is_omni_data"`` — bool; if True, uses the OmniFold uncertainty
          calculation; if False, uses theory uncertainties
        - ``"is_madgraph"`` — bool; if ``is_omni_data`` is False, determines
          whether MadGraph (True) or Sherpa (False) theory uncertainties are
          applied
    pdf_name : str, optional
        If provided, saves all figures to a single multi-page PDF at this
        path using ``SAVE_DPI``. Page order: main plot, uncertainty budget
        (if ``draw_uncertainty_budget=True``), correlation matrix.
    do_chi2_test : bool, optional
        If True, performs a chi-squared test comparing the OmniFold result
        to the first entry in ``results_list`` and prints the result. Only
        has effect in pseudodata mode (``is_omni_data=False``) and requires
        at least one entry in ``results_list`` to use as the target
        (default: False).
    llab : str, optional
        Left label shown on the uncertainty budget figure (default:
        ``"Simulation Internal"``).
    smooth_hv : bool, optional
        Passed to ``UncertaintyCalculator.calculate_uncertainties``. If True
        (default), applies smoothing to the hadronisation/shower uncertainty
        before combining. Set to False to use the raw variation.

    Returns:
    --------
    tuple[plt.Figure, plt.Figure | None]
        ``(fig, fig_budget)`` where ``fig`` is the main cross-section plot and
        ``fig_budget`` is the uncertainty budget figure (``None`` if
        ``draw_uncertainty_budget=False``).
    """
    additional_results = results_list is not None and formating_dicts is not None
    if additional_results:
        if len(results_list) != len(formating_dicts):
            raise ValueError("results_list and formating_dicts must be the same length")

    hist_params_OmniFold = {
        "color": "black",
        "linewidth": 2,
        "density": True,
        "alpha": 0,
    }
    formating_IBU = {
        "is_madgraph": False,
        "is_omni_data": True,
        "color": "green",
        "marker": "o",
        "label": "IBU Measurement",
    }

    draw_ratioplot = (
        (mgfxfx_truth_results is not None)
        or (sherpa_truth_results is not None)
        or additional_results
    )
    height_ratios = [3, 1] if 1 + int(draw_ratioplot) == 2 else [3]
    fig, axs = plt.subplots(
        1 + int(draw_ratioplot),
        1,
        sharex=True,
        sharey=False,
        gridspec_kw={"height_ratios": height_ratios},
        figsize=(12 * 2 / 3, 8 * 2 / 3),  # is 12 by 8 but smaller
    )
    fig.subplots_adjust(hspace=0.0)
    if not draw_ratioplot:
        axs = [axs]  # Ensure axs is always a list for consistent indexing

    bin_centers = 0.5 * (binning[1:] + binning[:-1])
    bin_widths = np.array(
        [binning[n + 1] - binning[n] for n in range(len(binning) - 1)]
    )

    # Create uncertainty calculator (using default definitions)
    uncertainty_calculator = uncertainties.UncertaintyCalculator()

    # Nominal Omnifold result: calculate then group uncertainties
    signed_uncerts, syst_covs, syst_info = (
        uncertainty_calculator.calculate_uncertainties(
            omni_results, measured_key="nominal", smooth_hv=smooth_hv
        )
    )
    omni_uncert_tuple = uncertainty_calculator.process_signed_uncertainties(
        signed_uncerts, syst_covs, syst_info
    )
    omni_uncert = np.sqrt(
        np.sum(np.array(list(omni_uncert_tuple[0].values())) ** 2, axis=0)
    )
    # Calculate theory uncertainties for MadGraph
    if mgfxfx_truth_results is not None:
        mgfxfx_uncert = uncertainty_calculator.get_total_theory_uncertainty(
            mgfxfx_truth_results, measured_key="nominal", is_madgraph=True
        )
    # Calculate theory uncertainties for Sherpa
    if sherpa_truth_results is not None:
        sherpa_uncert = uncertainty_calculator.get_total_theory_uncertainty(
            sherpa_truth_results, measured_key="nominal", is_madgraph=False
        )

    # Calculate uncertainties for additional results if provided
    if additional_results:
        results_uncerts = []
        for result_dict, fmt in zip(results_list, formating_dicts):
            if fmt["is_omni_data"]:
                result_uncert_tuple = uncertainty_calculator.calculate_uncertainties(
                    result_dict, measured_key="nominal", smooth_hv=smooth_hv
                )
                result_uncert = np.sqrt(
                    np.sum(np.array(list(result_uncert_tuple[0].values())) ** 2, axis=0)
                )
            else:
                result_uncert = uncertainty_calculator.get_total_theory_uncertainty(
                    result_dict, measured_key="nominal", is_madgraph=fmt["is_madgraph"]
                )
            results_uncerts.append(result_uncert)

    fig_budget, fig_cov = None, None
    if draw_uncertainty_budget:
        chi2_target = (
            results_list[0]["nominal"][0]
            if (results_list and not is_omni_data)
            else None
        )
        fig_budget, fig_cov = make_uncertainty_budget_fig(
            binning,
            omni_uncert_tuple,
            figsize=(12 * 2 / 3, 8 * 2 / 3),  # is 12 by 8 but smaller
            xlabel=xlabel,
            log_xscale=logxScale,
            llab=llab,
            rlab="Z+jets Omnifold",
            data_measurement_mode=is_omni_data,
            measured_hist=omni_results["nominal"][0],
            target_hist=chi2_target,
            do_chi2_test=do_chi2_test and chi2_target is not None,
            simple_corr_labels=True,
        )

    # Sherpa
    if sherpa_truth_results is not None:
        sherpa_density = sherpa_truth_results["nominal"][0]
        if is_xSec:
            sherpa_density = sherpa_density / bin_widths
        _ = make_error_boxes(
            axs[0],
            bin_centers,
            sherpa_density,
            np.vstack([bin_widths / 2, bin_widths / 2]),
            np.vstack([sherpa_density * sherpa_uncert, sherpa_density * sherpa_uncert]),
            facecolor="deeppink",
            alpha=0.25,
            marker="s",
            label=r"Drell Yan: Sherpa2.2.11 + X",
        )

    # MGFxFx
    if mgfxfx_truth_results is not None:
        mgfxfx_density = mgfxfx_truth_results["nominal"][0]
        if is_xSec:
            mgfxfx_density = mgfxfx_density / bin_widths
        _ = make_error_boxes(
            axs[0],
            bin_centers,
            mgfxfx_density,
            np.vstack([bin_widths / 2, bin_widths / 2]),
            np.vstack([mgfxfx_density * mgfxfx_uncert, mgfxfx_density * mgfxfx_uncert]),
            facecolor="dodgerblue",
            alpha=0.25,
            marker="^",
            label=("Drell Yan: MG5+Py8 + X"),
        )

    # Additional results
    if additional_results:
        for result_dict, fmt, result_uncert in zip(
            results_list, formating_dicts, results_uncerts
        ):
            result_density = result_dict["nominal"][0]
            if is_xSec:
                result_density = result_density / bin_widths
            _ = make_error_boxes(
                axs[0],
                bin_centers,
                result_density,
                np.vstack([bin_widths / 2, bin_widths / 2]),
                np.vstack(
                    [result_density * result_uncert, result_density * result_uncert]
                ),
                facecolor=fmt["color"],
                alpha=0.25,
                marker=fmt["marker"],
                label=fmt["label"],
            )
    if ibu_results is not None:
        ibu_density = ibu_results["nominal"]
        ibu_uncertainty = ibu_results["total_unc"]
        _ = make_error_boxes(
            axs[0],
            bin_centers,
            ibu_density,
            np.vstack([bin_widths / 2, bin_widths / 2]),
            np.vstack([ibu_uncertainty, ibu_uncertainty]),
            facecolor=formating_IBU["color"],
            alpha=0.25,
            marker=formating_IBU["marker"],
            label=formating_IBU["label"],
        )

    # OmniFold
    omni_density = omni_results["nominal"][0]
    if is_xSec:
        omni_density = omni_density / bin_widths

    axs[0].hist(
        binning[:-1],
        bins=binning,
        weights=omni_density,
        align="mid",
        **hist_params_OmniFold,
    )

    mh.atlas.label(
        ax=axs[0],
        loc=0,
        llabel=llab,
        rlabel="Z+jets Omnifold",
    )

    if text_box is not None:
        draw_textbox(axs[0], text_box)

    axs[0].minorticks_on()
    axs[0].xaxis.set_tick_params(labelsize=16, which="both", direction="in", top=True)
    axs[0].yaxis.set_tick_params(labelsize=16, which="both", direction="in", right=True)
    axs[0].set_ylabel(ylabel, fontsize=16, labelpad=2, loc="top")
    y_min = min(omni_density)
    y_max = max(omni_density)

    # Start from omni (always present)
    y_min = np.min(omni_density)
    y_max = np.max(omni_density)

    # Compare with MG
    if mgfxfx_truth_results is not None:
        y_min = min(y_min, np.min(mgfxfx_density))
        y_max = max(y_max, np.max(mgfxfx_density))

    # Compare with Sherpa
    if sherpa_truth_results is not None:
        y_min = min(y_min, np.min(sherpa_density))
        y_max = max(y_max, np.max(sherpa_density))

    # Compare with additional results
    if additional_results:
        for result_dict, fmt in zip(results_list, formating_dicts):
            y = result_dict["nominal"][0]
            if is_xSec:
                y = y / bin_widths
            y_min = min(y_min, np.min(y))
            y_max = max(y_max, np.max(y))
    if ibu_results is not None:
        y_min = min(y_min, np.min(ibu_density))
        y_max = max(y_max, np.max(ibu_density))

    # Adjust y-axis limits
    axs[0].set_ylim([0.3 * y_min, 1.3 * y_max])

    axs[0].errorbar(
        bin_centers,
        omni_density,
        xerr=bin_widths / 2,
        yerr=omni_uncert * omni_density,
        marker=".",
        linestyle="None",
        color="k",
        alpha=1,
        ecolor="k",
        label="Omnifold Measurement",
        markersize=5,
        linewidth=1,
        markeredgewidth=2,
    )

    # this just adds the X = text under MG or Sherpa
    if mgfxfx_truth_results is not None or sherpa_truth_results is not None:
        handles, labels = axs[0].get_legend_handles_labels()
        targets = ("Drell Yan: Sherpa2.2.11 + X", "Drell Yan: MG5+Py8 + X")
        insert_idx = (
            max(
                (i for i, l in enumerate(labels) if any(t in l for t in targets)),
                default=-1,
            )
            + 1
        )
        handles.insert(insert_idx, plt.Line2D([], [], linestyle="None", color="white"))
        labels.insert(insert_idx, r"X = EW Zjj, VZ$\rightarrow$V$\mu\mu$")
        axs[0].legend(handles, labels, fontsize=12, frameon=False, loc="best")
    else:
        axs[0].legend(fontsize=12, frameon=False, loc="best")

    if logyScale:
        axs[0].set_yscale("log")
    else:
        axs[0].set_yscale("linear")
    if logxScale:
        axs[0].set_xscale("log")
    if draw_ratioplot:
        axs[1].minorticks_on()
        axs[1].errorbar(
            bin_centers,
            np.ones(len(bin_centers)),
            xerr=bin_widths / 2,
            yerr=omni_uncert,
            marker=".",
            linestyle="None",
            color="k",
            alpha=1,
            ecolor="k",
            label="Omnifold Measurement",
            markersize=5,
            linewidth=1,
            markeredgewidth=2,
        )
        if sherpa_truth_results is not None:
            _ = make_error_boxes(
                axs[1],
                bin_centers,
                sherpa_density / omni_density,
                np.vstack([bin_widths / 2, bin_widths / 2]),
                np.vstack(
                    [
                        (sherpa_density / omni_density) * sherpa_uncert,
                        (sherpa_density / omni_density) * sherpa_uncert,
                    ]
                ),
                facecolor="deeppink",
                alpha=0.25,
                marker="s",
                label=r"Sherpa",
            )
        if mgfxfx_truth_results is not None:
            _ = make_error_boxes(
                axs[1],
                bin_centers,
                mgfxfx_density / omni_density,
                np.vstack([bin_widths / 2, bin_widths / 2]),
                np.vstack(
                    [
                        (mgfxfx_density / omni_density) * mgfxfx_uncert,
                        (mgfxfx_density / omni_density) * mgfxfx_uncert,
                    ]
                ),
                facecolor="dodgerblue",
                alpha=0.25,
                marker="^",
                label=r"MGFxFx",
            )
        if additional_results:
            for result_dict, fmt, result_uncert in zip(
                results_list, formating_dicts, results_uncerts
            ):
                result_density = result_dict["nominal"][0]
                if is_xSec:
                    result_density = result_density / bin_widths
                _ = make_error_boxes(
                    axs[1],
                    bin_centers,
                    result_density / omni_density,
                    np.vstack([bin_widths / 2, bin_widths / 2]),
                    np.vstack(
                        [
                            (result_density / omni_density) * result_uncert,
                            (result_density / omni_density) * result_uncert,
                        ]
                    ),
                    facecolor=fmt["color"],
                    alpha=0.25,
                    marker=fmt["marker"],
                    label=fmt["label"],
                )
        if ibu_results is not None:
            _ = make_error_boxes(
                axs[1],
                bin_centers,
                ibu_density / omni_density,
                np.vstack([bin_widths / 2, bin_widths / 2]),
                np.vstack(
                    [
                        (ibu_uncertainty / omni_density),
                        (ibu_uncertainty / omni_density),
                    ]
                ),
                facecolor=formating_IBU["color"],
                alpha=0.25,
                marker=formating_IBU["marker"],
                label=formating_IBU["label"],
            )

        axs[1].set_xlim(binning[0], binning[-1])
        if logxScale:
            axs[1].set_xscale("log")
        axs[1].set_ylim(ratio_ylim)
        axs[1].xaxis.set_tick_params(
            labelsize=16, which="both", direction="in", top=True
        )
        axs[1].yaxis.set_tick_params(
            labelsize=16, which="both", direction="in", right=True
        )
        if is_omni_data:
            ratio_ylabel = "MC / Data"
        else:
            ratio_ylabel = "MC / Pseudodata"
        axs[1].set_ylabel(ratio_ylabel, fontsize=16, labelpad=2, loc="center")
        axs[1].set_xlabel(xlabel, fontsize=16, labelpad=2, loc="right")

        line1 = nice_midpoint(ratio_ylim[0], 1)
        line2 = nice_midpoint(1, ratio_ylim[1])
        axs[1].axhline(line1, color="gray", linestyle="--", linewidth=1)
        axs[1].axhline(line2, color="gray", linestyle="--", linewidth=1)

    if pdf_name is not None:
        with PdfPages(pdf_name) as pdf:
            pdf.savefig(fig, dpi=SAVE_DPI, bbox_inches="tight")
            if fig_budget is not None:
                pdf.savefig(fig_budget, dpi=SAVE_DPI, bbox_inches="tight")
            if fig_cov is not None:
                pdf.savefig(fig_cov, dpi=SAVE_DPI, bbox_inches="tight")
    if fig_cov is not None:
        plt.close(fig_cov)
    return fig, fig_budget


def draw_uncertainty_group(
    omni_results,
    binning,
    group,
    xlabel="default",
):
    # Create uncertainty calculator (using default definitions)
    uncertainty_calculator = uncertainties.UncertaintyCalculator()
    omni_uncert_tuple = uncertainty_calculator.calculate_uncertainties(
        omni_results, measured_key="nominal", ungrouped=True
    )
    make_uncertainty_budget_fig(
        binning,
        omni_uncert_tuple,
        figsize=(7.4, 6),
        xlabel=xlabel,
        log_xscale=False,
        llab="Simulation Internal",
        rlab="Z+jets Omnifold",
        data_measurement_mode=True,
        measured_hist=None,
        target_hist=None,
        do_chi2_test=False,
        simple_corr_labels=True,
        draw_this_group=group,
    )


def draw_mc_syst_uncertainties(results):
    """
    Plots (systematic - nominal)/nominal for all systematics for MC theory predictions.

    Parameters
    ----------
    results : dict
        Keys include 'nominal' and keys of all systematic variations.
        Each value is a tuple: (hist, variance, bin_edges)

    Note this function does not plot stat errors.
    """

    nominal, _, bin_edges = results["nominal"]
    syst_keys = [k for k in results if k != "nominal"]

    # Prepare figure
    plt.figure(figsize=(8, 6))

    colors = plt.cm.tab20.colors  # use 20 distinct colors
    for i, key in enumerate(syst_keys):
        hist, _, _ = results[key]
        rel_variation = (hist - nominal) / nominal  # (var-nom)/nom

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(
            bin_centers,
            rel_variation,
            label=key,
            color=colors[i % len(colors)],
            marker="o",
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("(syst - nominal)/nominal")
    plt.title("systematic variations")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
