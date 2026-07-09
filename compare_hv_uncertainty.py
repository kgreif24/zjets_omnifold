"""compare_hv_uncertainty.py - Compare the signed hidden-variable (HV)
uncertainty of the IBU, Multifold, and Omnifold measurements of the
Z(->mumu)+jets observables.

A single multi-page PDF is produced, one page per observable:

  - hv_uncertainty_comparison.pdf

Each page shows the signed percent uncertainty coming from the hidden
variable systematic alone, drawn as a step plot with the per-bin centre
marked by a measurement-specific marker (same styling as
compare_measurements.py's total-fractional-uncertainty plot).

There are 24 observables constrained by all three measurements and two
additional observables (Nch / Ntracks and HT / HT_tracks) that are only
constrained by IBU and Omnifold.

NOTE: y_trackj2 is currently skipped because of a known binning bug in
the IBU result (IBU uses 10 bins while OF/MF use 18).  This is handled
automatically by the bin-count consistency check below.

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


# ---------------------------------------------------------------------------
# Observable lists
# ---------------------------------------------------------------------------
# 24 observables measured by all three methods.  y_trackj2 is intentionally
# left in the list; it is dropped automatically by the bin-count
# consistency check due to a known IBU binning bug (see module docstring).
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


def get_ibu_hv(ibu, key):
    """Return the signed fractional HV uncertainty for an observable.

    The IBU file stores `<key>_HVOther` with a leading side bin for
    190 GeV < truth_pT_ll < 200 GeV that is not part of the final fiducial
    measurement (mirroring `<key>_RelErrors`), so drop it to match the
    xSec binning.
    """
    return np.array(ibu[key + "_HVOther"], dtype=float)[1:]


def get_npz_hv(store, key):
    """Return the signed fractional HV uncertainty for an OF/MF npz."""
    return np.array(store[key + "_hv_signed_uncert"], dtype=float)


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
def plot_hv_uncertainty(key, info, bins, ibu_hv, of_hv, mf_hv):
    """Build and return the signed HV-uncertainty comparison figure."""
    bin_centers = (bins[1:] + bins[:-1]) / 2

    ibu_pct = ibu_hv * 100.0
    of_pct = of_hv * 100.0
    mf_pct = mf_hv * 100.0 if mf_hv is not None else None

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1.0, zorder=1)

    # IBU (baseline=None avoids the vertical drop to zero at the panel edges)
    ax.stairs(ibu_pct, bins, baseline=None, color=IBU_COLOR, linewidth=1.5, zorder=2)
    ax.plot(
        bin_centers, ibu_pct, marker=IBU_MARKER, linestyle="none",
        color=IBU_COLOR, markersize=5, label="IBU reference", zorder=3,
    )

    # Multifold (may be absent)
    if mf_pct is not None:
        ax.stairs(mf_pct, bins, baseline=None, color=MF_COLOR, linewidth=1.5, zorder=2)
        ax.plot(
            bin_centers, mf_pct, marker=MF_MARKER, linestyle="none",
            color=MF_COLOR, markersize=5, label="PRL 133.261803", zorder=3,
        )

    # Omnifold
    ax.stairs(of_pct, bins, baseline=None, color=OF_COLOR, linewidth=1.5, zorder=2)
    ax.plot(
        bin_centers, of_pct, marker=OF_MARKER, linestyle="none",
        color=OF_COLOR, markersize=5, label="Data", zorder=3,
    )

    if info.get("log_xscale", False):
        ax.set_xscale("log")

    # Span exactly the leftmost to rightmost bin edge.
    ax.set_xlim(bins[0], bins[-1])

    # Fixed symmetric y-axis, shared across all observables.
    ax.set_ylim(-20.0, 20.0)

    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel(fix_label(info.get("xlabel", key)))
    ax.set_ylabel("HV uncertainty [%]")
    ax.legend(loc="best", frameon=False)

    mh.atlas.label(ax=ax, loc=1, llabel="Internal", data=True, rlabel="")
    ax.text(
        0.03, 0.86,
        r"$\sqrt{s} = 13$ TeV, 140.1 fb$^{-1}$",
        fontsize=14,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    ax.text(
        0.03, 0.78,
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
        description=(
            "Compare the signed hidden-variable uncertainty of the IBU, "
            "Multifold and Omnifold measurements."
        )
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
            "plot_storage/multifold/signed/multifold_histograms.npz"
        ),
        help="Path to the Multifold histograms .npz file.",
    )
    parser.add_argument(
        "--omnifold",
        type=str,
        default=(
            "/global/u2/k/kgreif/zjets/work_dirs/zjets_omnifold1/"
            "plot_storage/zjets-v4-data-signed/omnifold_histograms.npz"
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

    hv_path = plot_dir / "hv_uncertainty_comparison.pdf"
    n_pages = 0

    with PdfPages(hv_path) as hv_pdf:
        for key in observables:
            has_mf = key not in IBU_OF_ONLY_OBSERVABLES

            # Sanity checks
            missing = [
                label
                for label, present in (
                    ("IBU", key + "_HVOther" in ibu),
                    ("Omnifold", key + "_hv_signed_uncert" in of),
                )
                if not present
            ]
            if has_mf and key + "_hv_signed_uncert" not in mf:
                missing.append("Multifold")
            if missing:
                print(f"Skipping {key}: missing {', '.join(missing)}")
                continue

            info = info_by_key.get(key, {})

            # Use the Omnifold binning as the common reference
            bins = of[key + "_bins"]

            ibu_hv = get_ibu_hv(ibu, key)
            of_hv = get_npz_hv(of, key)
            mf_hv = get_npz_hv(mf, key) if has_mf else None

            # Bin-count consistency check
            n = len(bins) - 1
            if len(ibu_hv) != n or len(of_hv) != n:
                print(
                    f"Skipping {key}: bin-count mismatch "
                    f"(IBU={len(ibu_hv)}, OF={len(of_hv)}, bins={n})"
                )
                continue

            print(f"Plotting {key}")

            fig = plot_hv_uncertainty(key, info, bins, ibu_hv, of_hv, mf_hv)
            hv_pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            n_pages += 1

    print(f"\nWrote {n_pages}-page PDF:\n  {hv_path}")


if __name__ == "__main__":
    main()
