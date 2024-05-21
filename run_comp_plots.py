""" run_comp_plots.py - This function will generate "comp" plots which compare the reweighted
MC truth distributions produced by Omnifold step 2 with the truth pseudodata distributions.

It requires only the paths to the MC and truth pseudodata files, and a path to the file storing
the final weights that should be applied to the MC truth distribution.

Author: Kevin Greif
Last updated 05/21/2024
python3
"""

import argparse
import numpy as np
import uproot
import awkward as ak

import utils.plotting_utils as pu
import utils.data_utils as du


parser = argparse.ArgumentParser(description="Generate comp plots for reweighted MC and truth pseudodata comparison.")
parser.add_argument("--mc", type=str, help="The path to MC file")
parser.add_argument("--weights", type=str, help="The path to the weights file")
parser.add_argument("--pd", type=str, help="The path to PD file")
parser.add_argument("--store", type=str, help="The path to store the plots")
parser.add_argument("--use_test", action="store_true", help="If true, will use the weights for the testing MC rather than training")
args = parser.parse_args()

# Set max events and max tracks to use for plotting
max_events = 1000000
max_tracks = 150

# Get the trees
mc_file = uproot.open(args.mc)
mc_tree = mc_file["OmniTree"]
pd_file = uproot.open(args.pd)
pd_tree = pd_file["OmniTree"]

# Get filters, since we are using truth level data, we need to use the truth_pass190 filter
mc_filter = ak.to_numpy(mc_tree["truth_pass190"].array())
pd_filter = ak.to_numpy(pd_tree["truth_pass190"].array())

# Get original MC weights
mc_start_weights = ak.to_numpy(mc_tree["weight"].array())
mc_start_weights = mc_start_weights[mc_filter == 1]
if len(mc_start_weights) > max_events:
    mc_start_weights = mc_start_weights[:max_events]

# Get new MC weights
mc_wgt_file = np.load(args.weights)
mc_end_weights = mc_wgt_file["train"]
if args.use_test:
    mc_end_weights = mc_wgt_file["test"]
mc_end_weights = mc_end_weights[mc_filter == 1]
if len(mc_end_weights) > max_events:
    mc_end_weights = mc_end_weights[:max_events]

# Get pseudodata weights
pd_weights = ak.to_numpy(pd_tree["weight"].array())
pd_weights = pd_weights[pd_filter == 1]
if len(pd_weights) > max_events:
    pd_weights = pd_weights[:max_events]

# Concatenate the weights
start_weights = np.concatenate([mc_start_weights, pd_weights], axis=0)
end_weights = np.concatenate([mc_end_weights, pd_weights], axis=0)

# Make labels
mc_labels = np.zeros_like(mc_start_weights)
pd_labels = np.ones_like(pd_weights)
labels = np.concatenate([mc_labels, pd_labels], axis=0)

# Get the plot data, always want the truth level for this script
plotting_variables = pu.default_settings.keys()
mc_plotting = du.get_plotting(mc_tree, vars=plotting_variables, filter=mc_filter, get_truth=True, max_events=max_events)
pd_plotting = du.get_plotting(pd_tree, vars=plotting_variables, filter=pd_filter, get_truth=True, max_events=max_events)
plotting = np.concatenate([mc_plotting, pd_plotting], axis=0)

# Get the track kinematics
mc_kinematics = du.get_kinematics(mc_tree, filter=mc_filter, get_mask=False, one_hot=False, get_truth=True, max_tracks=max_tracks, max_events=max_events)
pd_kinematics = du.get_kinematics(pd_tree, filter=pd_filter, get_mask=False, one_hot=False, get_truth=True, max_tracks=max_tracks, max_events=max_events)
kinematics = np.concatenate([mc_kinematics, pd_kinematics], axis=0)

# Drop the muons
kinematics = kinematics[:,:,2:]

# Modify the legends to show the truth level data
new_settings = pu.default_settings.copy()
for key, val in new_settings.items():
    val.update({'truth_mc': True})
    val.update({'truth_data': True})

new_track_settings = pu.track_hists.copy()
for key, val in new_track_settings.items():
    val.update({'truth_mc': True})
    val.update({'truth_data': True})

# Make names argument
names = ("TruthMC", "TruthPD")

# Make the logged plots
pu.make_logged_plots(plotting, labels, start_weights, end_weights=end_weights, definitions=new_settings, save_location=args.store, names=names)
pu.make_inclusive_track_plots(kinematics, labels, start_weights, end_weights=end_weights, definitions=new_track_settings, save_location=args.store, names=names)