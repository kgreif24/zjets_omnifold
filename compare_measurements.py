"""compare_measurements.py - Compare the IBU, Multifold, and Omnifold
measurements of the Z(->mumu)+jets observables.

Two multi-page PDFs are produced, one page per observable:

  - cross_section_comparison.pdf
  - uncertainty_comparison.pdf

For each observable two kinds of plot are produced:

  1. A cross-section comparison plot with two panels.  The upper panel
     (2/3 of the height) shows the three measured differential cross
     sections (Omnifold as black data points with error bars, IBU in
     petroff blue and Multifold in petroff orange with shaded
     uncertainty boxes).  The lower panel (1/3 of the height) shows the
     ratio of Omnifold and Multifold to IBU, using the same styling.

  2. A total-fractional-uncertainty comparison plot.  The total
     uncertainty of each measurement is drawn as a step plot with the
     per-bin centre marked by a measurement-specific marker.

There are 24 observables constrained by all three measurements and two
additional observables (Nch / Ntracks and HT / HT_tracks) that are only
constrained by IBU and Omnifold.

NOTE: y_trackj2 is currently skipped because of a known binning bug in
the IBU result (IBU uses 10 bins while OF/MF use 18).  Put it back in the
COMMON_OBSERVABLES list once that is fixed.

Run with the `eflow` conda environment.

Author: Kevin Greif
python3
"""

import argparse
import pathlib
from json import JSONDecoder

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as mh

mh.style.use("ATLAS")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
OF_COLOR = "black"          # Omnifold
IBU_COLOR = "#3f90da"       # Petroff blue
MF_COLOR = "#ffa90d"        # Petroff orange

OF_MARKER = "o"
IBU_MARKER = "s"
MF_MARKER = "^"

# Observables whose y-axis should be forced to a log scale even though the
# config marks them as linear.
FORCE_LOG_YSCALE = {"HT_tracks"}


# ---------------------------------------------------------------------------
# Observable lists
# ---------------------------------------------------------------------------
# 24 observables measured by all three methods.  y_trackj2 is intentionally
# left out for now due to an IBU binning bug (see module docstring).
COMMON_OBSERVABLES = [
    "Ntracks_trackj1",
    "Ntracks_trackj2",
    "eta_l1",
    "eta_l2",
    "m_trackj1",
    "m_trackj2",
    "pT_l1",
    "pT_l2",
    "pT_ll",
    "pT_trackj1",
    "pT_trackj2",
    "phi_l1",
    "phi_l2",
    "phi_trackj1",
    "phi_trackj2",
    "tau1_trackj1",
    "tau1_trackj2",
    "tau2_trackj1",
    "tau2_trackj2",
    "tau3_trackj1",
    "tau3_trackj2",
    "y_ll",
    "y_trackj1",
    "y_trackj2",
]

# Two observables only constrained by IBU and Omnifold (not Multifold).
IBU_OF_ONLY_OBSERVABLES = ["Ntracks", "HT_tracks"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_ibu(path):
    """Parse the IBU json file.

    The file is a stream of concatenated single-key json objects rather
    than a single json document, so we decode it object by object and
    merge the results into one dictionary.
    """
    text = pathlib.Path(path).read_text()
    decoder = JSONDecoder()
    idx = 0
    n = len(text)
    data = {}
    while idx < n:
        while idx < n and text[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        data.update(obj)
        idx = end
    return data


def get_ibu(ibu, key):
    """Return (differential xsec, fractional total uncert) for an observable.

    The IBU file stores `<key>_xSec` (already differential, i.e. per unit
    of the observable) and `<key>_RelErrors` (fractional total
    uncertainty).  The RelErrors arrays include a leading side bin for
    190 GeV < truth_pT_ll < 200 GeV that is not part of the final fiducial
    measurement, so drop it to match the xSec binning.
    """
    xsec = np.array(ibu[key + "_xSec"], dtype=float)
    rel = np.array(ibu[key + "_RelErrors"], dtype=float)[1:]
    return xsec, rel


def get_npz(store, key, bins):
    """Return (differential xsec, fractional total uncert) for an OF/MF npz.

    The npz histograms are stored as integrated cross section per bin
    (xsec * bin_width), so divide by the bin width to obtain the
    differential cross section.  The `_uncert` array is fractional.
    """
    bin_widths = np.diff(bins)
    xsec = store[key + "_hist"] / bin_widths
    rel = store[key + "_uncert"]
    return xsec, rel


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------
def fix_label(label):
    """Roman-ise mathtext pieces that should not be italic (ATLAS style).

    - the differential 'd' in a ``d\\sigma/d<var>`` cross-section label,
    - the ``T`` in a ``p_T`` / ``H_T`` subscript,
    - the word ``tracks`` in the ``H_{T,tracks}`` subscript,
    - the ``ch`` in a ``n_{ch}`` / ``n_{ch,j}`` subscript.
    """
    if not label:
        return label
    label = label.replace(r"d\sigma/d", r"\mathrm{d}\sigma/\mathrm{d}")
    label = label.replace(r"$d\sigma$", r"$\mathrm{d}\sigma$")
    label = label.replace("_{T,", r"_{\mathrm{T},")
    label = label.replace("tracks", r"\mathrm{tracks}")
    label = label.replace("_{ch,", r"_{\mathrm{ch},")
    label = label.replace("_{ch}", r"_{\mathrm{ch}}")
    return label


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_cross_section(key, info, bins, ibu_vals, of_vals, mf_vals):
    """Build and return the two-panel cross-section comparison figure.

    ibu_vals / of_vals / mf_vals are (xsec, rel_uncert) tuples or None.
    """

    bin_centers = (bins[1:] + bins[:-1]) / 2
    bin_widths = bins[1:] - bins[:-1]

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(7, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    plt.subplots_adjust(hspace=0.0)

    # ---- Upper panel: differential cross sections --------------------------
    # IBU: full-width shaded uncertainty box + central value line
    ibu_xsec, ibu_rel = ibu_vals
    ibu_err = ibu_xsec * ibu_rel
    ax.bar(
        bin_centers,
        2 * ibu_err,
        bottom=ibu_xsec - ibu_err,
        width=bin_widths,
        align="center",
        color=IBU_COLOR,
        alpha=0.30,
        linewidth=0,
        zorder=1,
    )
    ax.hlines(ibu_xsec, bins[:-1], bins[1:], color=IBU_COLOR, linewidth=1.5, zorder=4)

    # Multifold: narrower shaded box + central value line (may be absent)
    if mf_vals is not None:
        mf_xsec, mf_rel = mf_vals
        mf_err = mf_xsec * mf_rel
        ax.bar(
            bin_centers,
            2 * mf_err,
            bottom=mf_xsec - mf_err,
            width=bin_widths * 0.55,
            align="center",
            color=MF_COLOR,
            alpha=0.50,
            linewidth=0,
            zorder=2,
        )
        ax.hlines(
            mf_xsec,
            bin_centers - bin_widths * 0.275,
            bin_centers + bin_widths * 0.275,
            color=MF_COLOR,
            linewidth=1.5,
            zorder=4,
        )

    # Omnifold: black points with error bars (centred)
    of_xsec, of_rel = of_vals
    of_err = of_xsec * of_rel
    ax.errorbar(
        bin_centers,
        of_xsec,
        yerr=of_err,
        fmt=OF_MARKER,
        color=OF_COLOR,
        markersize=4,
        capsize=0,
        linestyle="none",
        zorder=5,
    )

    is_log = key in FORCE_LOG_YSCALE or not info.get("linear_yscale", False)
    if is_log:
        ax.set_yscale("log")
    if info.get("log_xscale", False):
        ax.set_xscale("log")
        rax.set_xscale("log")

    # Set y-axis limits with extra headroom for the three-line label block.
    all_xsec = [ibu_xsec, of_xsec] + ([mf_vals[0]] if mf_vals is not None else [])
    y_min_data = min(np.min(v[v > 0]) for v in all_xsec)
    y_max_data = max(np.max(v) for v in all_xsec)
    if is_log:
        ax.set_ylim(0.3 * y_min_data, 100.0 * y_max_data)
    else:
        ax.set_ylim(0, 2.5 * y_max_data)

    # Span exactly the leftmost to rightmost bin edge.
    ax.set_xlim(bins[0], bins[-1])

    ax.set_ylabel(fix_label(info.get("ylabel", "Cross section")))
    # No x ticks on the inner (top) edge of the upper panel's shared boundary.
    ax.tick_params(axis="x", direction="in", top=True)

    # Custom legend handles (IBU/MF as central lines, Data last as a marker)
    handles = [Line2D([], [], color=IBU_COLOR, linewidth=1.5, label="IBU reference")]
    if mf_vals is not None:
        handles.append(
            Line2D([], [], color=MF_COLOR, linewidth=1.5, label="PRL 133, 261803")
        )
    handles.append(
        Line2D(
            [], [], color=OF_COLOR, marker=OF_MARKER, linestyle="none",
            markersize=4, label="Data",
        )
    )
    ax.legend(handles=handles, loc="best", frameon=False)

    mh.atlas.label(ax=ax, loc=1, llabel="Internal", data=True, rlabel="")
    ax.text(
        0.03, 0.85,
        r"$\sqrt{s} = 13$ TeV, 140.1 fb$^{-1}$",
        fontsize=14,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    ax.text(
        0.03, 0.77,
        r"$Z\rightarrow\mu\mu$, $p_\mathrm{T}^{\mu\mu} > 200$ GeV",
        fontsize=14,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )

    # ---- Lower panel: ratio to IBU -----------------------------------------
    # Reference: IBU uncertainty band around 1
    rax.axhline(1.0, color=IBU_COLOR, linestyle="--", linewidth=1.0, zorder=2)
    rax.bar(
        bin_centers,
        2 * ibu_rel,
        bottom=1 - ibu_rel,
        width=bin_widths,
        align="center",
        color=IBU_COLOR,
        alpha=0.30,
        linewidth=0,
        zorder=1,
    )

    if mf_vals is not None:
        mf_ratio = mf_xsec / ibu_xsec
        mf_ratio_err = mf_ratio * mf_rel
        rax.bar(
            bin_centers,
            2 * mf_ratio_err,
            bottom=mf_ratio - mf_ratio_err,
            width=bin_widths * 0.55,
            align="center",
            color=MF_COLOR,
            alpha=0.50,
            linewidth=0,
            zorder=3,
        )
        rax.hlines(
            mf_ratio,
            bin_centers - bin_widths * 0.275,
            bin_centers + bin_widths * 0.275,
            color=MF_COLOR,
            linewidth=1.5,
            zorder=4,
        )

    of_ratio = of_xsec / ibu_xsec
    of_ratio_err = of_ratio * of_rel
    rax.errorbar(
        bin_centers,
        of_ratio,
        yerr=of_ratio_err,
        fmt=OF_MARKER,
        color=OF_COLOR,
        markersize=4,
        capsize=0,
        linestyle="none",
        zorder=5,
    )
    rax.set_ylim(0.75, 1.25)

    rax.set_ylabel("Ratio to IBU")
    rax.set_xlabel(fix_label(info.get("xlabel", key)))
    # Drop the downward ticks on the top edge of the lower panel.
    rax.tick_params(axis="x", direction="in", top=False, bottom=True)

    # Hide the lowest y-tick label on the upper panel so it does not collide
    # with the topmost label of the ratio panel (the panels share an edge).
    # Works for both linear and log scales by locating the bottom in-range tick.
    fig.canvas.draw()
    ylo, yhi = ax.get_ylim()
    in_range = [
        (pos, lbl)
        for pos, lbl in zip(ax.get_yticks(), ax.get_yticklabels())
        if ylo <= pos <= yhi
    ]
    if in_range:
        min(in_range, key=lambda pl: pl[0])[1].set_visible(False)

    return fig


def plot_uncertainty(key, info, bins, ibu_rel, of_rel, mf_rel):
    """Build and return the total-fractional-uncertainty comparison figure."""
    bin_centers = (bins[1:] + bins[:-1]) / 2

    fig, ax = plt.subplots(figsize=(7, 5))

    # IBU (baseline=None avoids the vertical drop to zero at the panel edges)
    ax.stairs(ibu_rel, bins, baseline=None, color=IBU_COLOR, linewidth=1.5, zorder=2)
    ax.plot(
        bin_centers, ibu_rel, marker=IBU_MARKER, linestyle="none",
        color=IBU_COLOR, markersize=5, label="IBU reference", zorder=3,
    )

    # Multifold (may be absent)
    if mf_rel is not None:
        ax.stairs(mf_rel, bins, baseline=None, color=MF_COLOR, linewidth=1.5, zorder=2)
        ax.plot(
            bin_centers, mf_rel, marker=MF_MARKER, linestyle="none",
            color=MF_COLOR, markersize=5, label="PRL 133.261803", zorder=3,
        )

    # Omnifold
    ax.stairs(of_rel, bins, baseline=None, color=OF_COLOR, linewidth=1.5, zorder=2)
    ax.plot(
        bin_centers, of_rel, marker=OF_MARKER, linestyle="none",
        color=OF_COLOR, markersize=5, label="Data", zorder=3,
    )

    if info.get("log_xscale", False):
        ax.set_xscale("log")

    # Span exactly the leftmost to rightmost bin edge.
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0.0, 0.25)
    # Use plain (non-scientific) tick labels so no "x10^-2" offset is drawn,
    # which would otherwise collide with the ATLAS label.
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel(fix_label(info.get("xlabel", key)))
    ax.set_ylabel("Total fractional uncertainty")
    ax.legend(loc="best", frameon=False)

    mh.atlas.label(ax=ax, loc=1, llabel="Internal", data=True, rlabel="")
    ax.text(
        0.03, 0.85,
        r"$\sqrt{s} = 13$ TeV, 140.1 fb$^{-1}$",
        fontsize=14,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    ax.text(
        0.03, 0.77,
        r"$Z\rightarrow\mu\mu$, $p_\mathrm{T}^{\mu\mu} > 200$ GeV",
        fontsize=14,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare IBU, Multifold and Omnifold measurements."
    )
    parser.add_argument(
        "--ibu",
        type=str,
        default=(
            "/global/cfs/cdirs/m3246/ZjetOmnifold/json_outfiles/"
            "IBU_data_pass200andSideBin_nomVars_16Mar26_TestSamples.json"
        ),
        help="Path to the IBU json file.",
    )
    parser.add_argument(
        "--multifold",
        type=str,
        default=(
            "/global/u2/k/kgreif/zjets/work_dirs/zjets_omnifold1/"
            "plot_storage/multifold/data/multifold_histograms.npz"
        ),
        help="Path to the Multifold histograms .npz file.",
    )
    parser.add_argument(
        "--omnifold",
        type=str,
        default=(
            "/global/u2/k/kgreif/zjets/work_dirs/zjets_omnifold1/"
            "plot_storage/zjets-v4-data/omnifold_histograms.npz"
        ),
        help="Path to the Omnifold histograms .npz file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./utils/plots_config.yml",
        help="Path to the plotting config yaml (for axis labels / scales).",
    )
    parser.add_argument(
        "--store",
        type=str,
        default="./plot_storage/measurement_comparison",
        help="Directory in which to store the comparison plots.",
    )
    args = parser.parse_args()

    # Load inputs
    ibu = load_ibu(args.ibu)
    mf = np.load(args.multifold, allow_pickle=True)
    of = np.load(args.omnifold)

    # Build a key -> plot-info map from the config
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    info_by_key = {
        p["key"]: p for p in config["plots"].values() if "key" in p
    }

    # Output directory
    plot_dir = pathlib.Path(args.store)
    plot_dir.mkdir(parents=True, exist_ok=True)

    observables = COMMON_OBSERVABLES + IBU_OF_ONLY_OBSERVABLES

    # Two multi-page PDFs: one for the cross-section comparisons, one for the
    # uncertainty comparisons (one observable per page).
    xsec_path = plot_dir / "cross_section_comparison.pdf"
    uncert_path = plot_dir / "uncertainty_comparison.pdf"
    n_pages = 0

    with PdfPages(xsec_path) as xsec_pdf, PdfPages(uncert_path) as uncert_pdf:
        for key in observables:
            has_mf = key not in IBU_OF_ONLY_OBSERVABLES

            # Sanity checks
            missing = [
                label
                for label, present in (
                    ("IBU", key + "_xSec" in ibu),
                    ("Omnifold", key + "_hist" in of),
                )
                if not present
            ]
            if has_mf and key + "_hist" not in mf:
                missing.append("Multifold")
            if missing:
                print(f"Skipping {key}: missing {', '.join(missing)}")
                continue

            info = info_by_key.get(key, {})

            # Use the Omnifold binning as the common reference
            bins = of[key + "_bins"]

            ibu_xsec, ibu_rel = get_ibu(ibu, key)
            of_xsec, of_rel = get_npz(of, key, bins)
            if has_mf:
                mf_xsec, mf_rel = get_npz(mf, key, bins)
                mf_vals = (mf_xsec, mf_rel)
            else:
                mf_vals = None

            # Bin-count consistency check
            n = len(bins) - 1
            if len(ibu_xsec) != n or len(of_xsec) != n:
                print(
                    f"Skipping {key}: bin-count mismatch "
                    f"(IBU={len(ibu_xsec)}, OF={len(of_xsec)}, bins={n})"
                )
                continue

            print(f"Plotting {key}")

            fig_x = plot_cross_section(
                key, info, bins, (ibu_xsec, ibu_rel), (of_xsec, of_rel), mf_vals
            )
            xsec_pdf.savefig(fig_x, bbox_inches="tight")
            plt.close(fig_x)

            fig_u = plot_uncertainty(
                key,
                info,
                bins,
                ibu_rel,
                of_rel,
                mf_vals[1] if mf_vals is not None else None,
            )
            uncert_pdf.savefig(fig_u, bbox_inches="tight")
            plt.close(fig_u)

            n_pages += 1

    print(f"\nWrote {n_pages}-page PDFs:\n  {xsec_path}\n  {uncert_path}")


if __name__ == "__main__":
    main()
