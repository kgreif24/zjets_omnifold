"""run_standalone_plots.py - This script will use the Plotter class to generate
basic reweighting plots for any iteration of Omnifold, or final comparison
to the truth pseudodata distributions.

It will also calculate both the starting and final Wasserstein distances
to the target data.

Author: Kevin Greif
Last updated 03/24/2025
python3
"""

import argparse
import plotter

parser = argparse.ArgumentParser(description="Run plotting functions")
parser.add_argument("--f1", type=str, help="The path to file 1")
parser.add_argument("--f2", type=str, help="The path to file 2")
parser.add_argument(
    "--root_files",
    type=str,
    nargs="+",
    default=None,
    help=(
        "If plotting observables that must be computed with fastjet"
        ", provide the path to the root files in order (start, end target)"
    ),
)
parser.add_argument("--name1", type=str, help="The name of file 1")
parser.add_argument("--name2", type=str, help="The name of file 2")
parser.add_argument(
    "--truth", action="store_true", help="If true, will use truth level data"
)
parser.add_argument(
    "--start_weights",
    type=str,
    help=(
        "The path to the start weight file, or if string does not end in '.npz',"
        "will look for a branch with this name in the tree"
    ),
)
parser.add_argument(
    "--end_weights",
    type=str,
    help=(
        "The path to the end weight file, or if string does not end in '.npz',"
        "will look for a branch with this name in the tree"
    ),
)
parser.add_argument(
    "--target_weights",
    type=str,
    help=(
        "The path to the target weight file, or if string does not end in '.npz',"
        "will look for a branch with this name in the tree"
    ),
)
parser.add_argument("--store", type=str, help="The path to store the plots")
parser.add_argument(
    "--train", action="store_true", help="If true, plot using training weights"
)
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
    "--calc_w1", action="store_true",
    help="If true, will calculate the wasserstein distances",
)
parser.add_argument(
    "--recalculate", action="store_true",
    help=(
        "If true, recalculate all fastjet observables, regardless of whether root files"
        "already exist."
    ),
)
parser.add_argument(
    "--cut_region", type=int, default=0,
    choices=[0, 1],
    help="Select a kinematic region to restrict to",
)
args = parser.parse_args()

# Build the plotter and run
plotter = plotter.Plotter(
    args.f1,
    args.f2,
    args.store,
    use_truth=args.truth,
    labels=(args.name1, args.name2),
    verbosity=args.verbosity,
    use_pdf=args.pdf,
    max_events=args.max_events,
    root_files=args.root_files,
)

# If requested apply kinematic cuts
if args.cut_region == 1:
    plotter.apply_kinematic_cuts(
        {
            "pT_ll": lambda x: x > 200,
        }
    )

# Make plots
plot_dict = plotter.plot(
    args.start_weights,
    args.end_weights,
    args.target_weights,
    recalculate=args.recalculate,
    use_train=args.train,
)

print("Plotting complete")

# If requested calculate the wasserstein distances as well
if args.calc_w1:
    start_dist, end_dist = plotter.wasserstein_distance(
        args.start_weights,
        args.end_weights,
        args.target_weights,
        use_train=args.train,
    )
    print(f"Start Wasserstein distance: {start_dist}")
    print(f"End Wasserstein distance: {end_dist}")
