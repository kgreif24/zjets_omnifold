""" run_plotting.py - This function will run the "make_logged_plots" and the 
"make_inclusive_track_plots" functions from plotting_utils.py. Command line arguments tell the 
code whether to plot as reco or truth level.

At all points we'll assume we aren't plotting a reweighting. This is just to look at raw data.

Author: Kevin Greif
Last updated 03/18/2024
python3
"""

import argparse
import numpy as np
import uproot
import awkward as ak

import plotting_utils as pu
import data_utils as du


parser = argparse.ArgumentParser(description="Run plotting functions")
parser.add_argument("--f1", type=str, help="The path to file 1")
parser.add_argument("--f2", type=str, help="The path to file 2")
parser.add_argument("--name1", type=str, help="The name of file 1")
parser.add_argument("--name2", type=str, help="The name of file 2")
parser.add_argument("--truth1", action="store_true", help="If true, will plot truth level data for file 1")
parser.add_argument("--truth2", action="store_true", help="If true, will plot truth level data for file 2")
parser.add_argument("--store", type=str, help="The path to store the plots")
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
if args.truth1:
    filter1 = ak.to_numpy(tree1["truth_pass190"].array())
else:
    filter1 = ak.to_numpy(tree1["pass190"].array())

if args.truth2:
    filter2 = ak.to_numpy(tree2["truth_pass190"].array())
else:
    filter2 = ak.to_numpy(tree2["pass190"].array())

# Get weights and filter weights
weights1 = ak.to_numpy(tree1["weight"].array())
weights1 = weights1[filter1 == 1]
weights1 = weights1[:max_events]
weights2 = ak.to_numpy(tree2["weight"].array())
weights2 = weights2[filter2 == 1]
weights2 = weights2[:max_events]
weights = np.concatenate([weights1, weights2], axis=0)

# Make labels
labels1 = np.zeros_like(weights1)
labels2 = np.ones_like(weights2)
labels = np.concatenate([labels1, labels2], axis=0)

# Get the plot data
plotting_variables = pu.default_settings.keys()
plotting1 = du.get_plotting(tree1, vars=plotting_variables, filter=filter1, get_truth=args.truth1, max_events=max_events)
plotting2 = du.get_plotting(tree2, vars=plotting_variables, filter=filter2, get_truth=args.truth2, max_events=max_events)
plotting = np.concatenate([plotting1, plotting2], axis=0)

# Get the track kinematics
kinematics1 = du.get_kinematics(tree1, filter=filter1, get_mask=False, one_hot=False, get_truth=args.truth1, max_tracks=max_tracks, max_events=max_events)
kinematics2 = du.get_kinematics(tree2, filter=filter2, get_mask=False, one_hot=False, get_truth=args.truth2, max_tracks=max_tracks, max_events=max_events)
kinematics = np.concatenate([kinematics1, kinematics2], axis=0)

# Drop the muons
kinematics = kinematics[:,:,2:]

# Modify the legends if we have truth level data
new_settings = pu.default_settings.copy()
if args.truth1:
    for key, val in new_settings.items():
        val.update({'truth_mc': True})
if args.truth2:
    for key, val in new_settings.items():
        val.update({'truth_data': True})

new_track_settings = pu.track_hists.copy()
if args.truth1:
    for key, val in new_track_settings.items():
        val.update({'truth_mc': True})
if args.truth2:
    for key, val in new_track_settings.items():
        val.update({'truth_data': True})

# Make names argument
names = (args.name1, args.name2)

# Make the logged plots
pu.make_logged_plots(plotting, labels, weights, definitions=new_settings, save_location=args.store, names=names)
pu.make_inclusive_track_plots(kinematics, labels, weights, definitions=new_track_settings, save_location=args.store, names=names)