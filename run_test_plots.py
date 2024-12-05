""" run_plotting.py - This function will make "test" plots for comparing the source and
target distributions for a given reweighting.

Author: Kevin Greif
Last updated 09/11/2024
python3
"""

import argparse
import numpy as np
import uproot
import awkward as ak

import utils.plotting_utils as pu
import utils.data_utils as du


parser = argparse.ArgumentParser(description="Run plotting functions")
parser.add_argument("--f1", type=str, help="The path to file 1")
parser.add_argument("--f2", type=str, help="The path to file 2")
parser.add_argument("--name1", type=str, help="The name of file 1")
parser.add_argument("--name2", type=str, help="The name of file 2")
parser.add_argument(
    "--step", type=int, choices=[1, 2], help="The step of the reweighting"
)
parser.add_argument(
    "--passBoth",
    action="store_true",
    help="If true, will require that the MC events pass both reco and truth selection",
)
parser.add_argument(
    "--start_weights",
    type=str,
    default=None,
    help="The path to the start weight file, if left at none use source",
)
parser.add_argument("--end_weights", type=str, help="The path to the end weight file")
parser.add_argument("--store", type=str, help="The path to store the plots")
parser.add_argument(
    "--train", action="store_true", help="If true, plot using training weights"
)
args = parser.parse_args()

# Set max tracks
max_events = 1000000
max_tracks = 150

# Get the trees
file1 = uproot.open(args.f1)
tree1 = file1["OmniTree"]
file2 = uproot.open(args.f2)
tree2 = file2["OmniTree"]

# Get filters
if args.passBoth:
    filter1 = np.logical_and(
        ak.to_numpy(tree1["truth_pass190"].array(entry_stop=max_events)),
        ak.to_numpy(tree1["pass190"].array(entry_stop=max_events)),
    )
    filter2 = np.logical_and(
        ak.to_numpy(tree2["truth_pass190"].array(entry_stop=max_events)),
        ak.to_numpy(tree2["pass190"].array(entry_stop=max_events)),
    )
elif args.step == 1:
    filter1 = ak.to_numpy(tree1["pass190"].array(entry_stop=max_events))
    filter2 = ak.to_numpy(tree2["pass190"].array(entry_stop=max_events))
elif args.step == 2:
    filter1 = ak.to_numpy(tree1["truth_pass190"].array(entry_stop=max_events))
    filter2 = ak.to_numpy(tree2["truth_pass190"].array(entry_stop=max_events))

# Truncate filters
if len(filter1) > max_events:
    filter1 = filter1[:max_events]
if len(filter2) > max_events:
    filter2 = filter2[:max_events]

# Get start / end weights, then truncate and filter weights
if args.start_weights is None:
    start_weights = ak.to_numpy(tree1["weight"].array(entry_stop=max_events))
    start_weights = start_weights[filter1 == 1]
else:
    start_weight_file = np.load(args.start_weights)
    start_weights = start_weight_file["train" if args.train else "test"]
    if len(start_weights) > max_events:
        start_weights = start_weights[:max_events]
    start_weights = start_weights[filter1 == 1]

end_weight_file = np.load(args.end_weights)
end_weights = end_weight_file["train" if args.train else "test"]
if len(end_weights) > max_events:
    end_weights = end_weights[:max_events]
end_weights = end_weights[filter1 == 1]

# Get target weights
targ_weights = ak.to_numpy(tree2["weight"].array(entry_stop=max_events))
targ_weights = targ_weights[filter2 == 1]

# Concatenate the weights
start_weights = np.concatenate([start_weights, targ_weights], axis=0)
end_weights = np.concatenate([end_weights, targ_weights], axis=0)

# Get the plot data
plotting_variables = pu.default_settings.keys()
plotting1 = du.get_plotting(
    tree1,
    vars=plotting_variables,
    get_truth=True if args.step == 2 else False,
    stop=max_events,
    passBoth=args.passBoth,
)
plotting2 = du.get_plotting(
    tree2,
    vars=plotting_variables,
    get_truth=True if args.step == 2 else False,
    stop=max_events,
    passBoth=args.passBoth,
)
plotting = np.concatenate([plotting1, plotting2], axis=0)

# Get the track kinematics
kinematics1, _ = du.get_kinematics(
    tree1,
    get_truth=True if args.step == 2 else False,
    stop=max_events,
    passBoth=args.passBoth,
)
kinematics2, _ = du.get_kinematics(
    tree2,
    get_truth=True if args.step == 2 else False,
    stop=max_events,
    passBoth=args.passBoth,
)

# Make labels
labels1 = np.zeros(len(plotting1))
labels2 = np.ones(len(plotting2))
labels = np.concatenate([labels1, labels2], axis=0)

# Zero pad and concatenate kinematics
kinematics1 = du.pad_kinematics(kinematics1, max_tracks)
kinematics2 = du.pad_kinematics(kinematics2, max_tracks)
kinematics = np.concatenate([kinematics1, kinematics2], axis=0)

# Drop the muons
kinematics = kinematics[:, :, 2:]

# Modify the legends if we have truth level data
new_settings = pu.default_settings.copy()
new_track_settings = pu.track_hists.copy()
if args.step == 2:
    for key, val in new_settings.items():
        val.update({"truth_mc": True})
    for key, val in new_settings.items():
        val.update({"truth_data": True})
    for key, val in new_track_settings.items():
        val.update({"truth_mc": True})
    for key, val in new_track_settings.items():
        val.update({"truth_data": True})

# Make names argument
names = (args.name1, args.name2)

# Make the logged plots
pu.make_logged_plots(
    plotting,
    labels,
    start_weights,
    end_weights,
    definitions=new_settings,
    save_location=args.store,
    names=names,
)
pu.make_inclusive_track_plots(
    kinematics,
    labels,
    start_weights,
    end_weights,
    definitions=new_track_settings,
    save_location=args.store,
    names=names,
)
