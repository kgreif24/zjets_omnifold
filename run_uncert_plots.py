"""run_uncert_plots.py - This script will use the UncertaintyPlotter class
to generate uncertainty plots for the final result of Omnifold.

Author: Kevin Greif
Last updated 06/06/2025
python3
"""

import argparse
import uncertainty_plotter

parser = argparse.ArgumentParser(description="Run plotting functions")
parser.add_argument("--f1", type=str, help="The path to file 1")
parser.add_argument("--f2", type=str, help="The path to file 2")
parser.add_argument("--hv", type=str, help="The path to the sherpa file")
parser.add_argument("--data", type=str, help="The path to the data file")
parser.add_argument(
    "--root_files",
    type=str,
    nargs="+",
    default=None,
    help=(
        "If plotting observables that must be computed with fastjet"
        ", provide the path to the root files in order (end, target)"
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
    default=5000000,
    help="The maximum number of events to use for plotting",
)
parser.add_argument(
    "--color",
    type=str,
    default="blue",
    help="The color to use for the plots (default: blue)",
)
parser.add_argument(
    "--recalculate", action="store_true",
    help=(
        "If true, recalculate all fastjet observables, regardless of whether root files"
        "already exist."
    ),
)
parser.add_argument(
    "--cut_region",
    type=int,
    default=0,
    choices=[0, 1, 2, 3],
    help="Select a kinematic region to restrict to",
)
args = parser.parse_args()

# Build the plotter and run
plotter = uncertainty_plotter.UncertaintyPlotter(
    args.f1,
    args.f2,
    args.hv,
    args.data,
    args.store,
    verbosity=args.verbosity,
    use_pdf=args.pdf,
    max_events=args.max_events,
    root_files=args.root_files,
    ibu_bins=True,
    kinematic_region=args.cut_region,
)
plot_dict = plotter.plot(
    args.weights,
    color=args.color,
    recalculate=args.recalculate,
)
print("Plotting complete")
