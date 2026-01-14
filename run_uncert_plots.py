"""run_uncert_plots.py - This script will use the UncertaintyPlotter class
to generate uncertainty plots for the final result of Omnifold.

Supports both standard pseudodata vs target comparison and new data comparison
mode for comparing data measurements to truth-level generators. Also supports
dual target mode for comparing against two truth-level generators.

Author: Kevin Greif
Last updated 10/10/2025
python3
"""

import argparse
import uncertainty_plotter

parser = argparse.ArgumentParser(description="Run plotting functions")
parser.add_argument("--mc", type=str, help="The path to MC root file")
parser.add_argument(
    "--target",
    type=str,
    help="The path to target file, either truth pseudodata or truth gen. "
    "Must be a single ROOT file.",
)
parser.add_argument("--hv", type=str, help="The path to the sherpa MC file")
parser.add_argument(
    "--root_files",
    type=str,
    nargs="+",
    default=None,
    help=(
        "If plotting observables that must be computed with fastjet"
        ", provide the path to the root files in order (mc, target, hv)"
        ". If using --target2, add target2 as the 4th file."
    ),
)
parser.add_argument(
    "--weights",
    type=str,
    help=(
        "The path to the weight file outputted by the ensemble_weights.py script. "
        "This file should contain all of the omnifold weights for a given iteration "
        "of a campaign."
    ),
)
parser.add_argument("--store", type=str, help="The path to store the plots")
parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level")
parser.add_argument(
    "--pdf", action="store_true", help="If true, will save plots as pdf"
)
parser.add_argument(
    "--max_events",
    type=int,
    default=-1,
    help=(
        "The maximum number of events to use for plotting. "
        "If set, recommended to run with --normalize_targets."
    ),
)
parser.add_argument(
    "--color",
    type=str,
    default="blue",
    help="The color to use for the plots (default: blue)",
)
parser.add_argument(
    "--cut_region",
    type=int,
    default=0,
    choices=[-1, 0, 1, 2, 3],
    help="Select a kinematic region to restrict to",
)
parser.add_argument(
    "--target2",
    type=str,
    default=None,
    help=(
        "Path(s) to second target file(s) for dual truth-level generator comparison. "
    ),
)
parser.add_argument(
    "--data_comparison_mode",
    action="store_true",
    help=(
        "Enable data comparison mode (compares data to truth generators, "
        "removes method bias)"
    ),
)
parser.add_argument(
    "--normalize_targets",
    action="store_true",
    help="Normalize the target histograms to match the source histograms",
)
parser.add_argument(
    "--do_chi2_test",
    action="store_true",
    help="Perform a chi^2 test and print the results",
)
parser.add_argument(
    "--smooth_hv",
    action="store_true",
    help="If true, will smooth the hidden variable uncertainty",
)
parser.add_argument(
    "--smooth_all",
    action="store_true",
    help="If true, will smooth all Hessian uncertainties",
)
args = parser.parse_args()

# Validate arguments
if args.target2 is not None and args.root_files is not None:
    if len(args.root_files) != 4:
        print("Warning: When using --target2, you should provide 4 root files:")
        print("  [mc, target, hv, target2]")
        print(f"  You provided {len(args.root_files)} files.")

# Build the plotter and run
plotter = uncertainty_plotter.UncertaintyPlotter(
    args.mc,
    args.target,
    args.hv,
    args.store,
    root_files=args.root_files,
    target2_path=args.target2,
    data_comparison_mode=args.data_comparison_mode,
    normalize_targets=args.normalize_targets,
    do_chi2_test=args.do_chi2_test,
    verbosity=args.verbosity,
    use_pdf=args.pdf,
    max_events=args.max_events,
    ibu_bins=True,
    kinematic_region=args.cut_region,
)
plotter.plot(
    args.weights,
    color=args.color,
)
print("Plotting complete")
