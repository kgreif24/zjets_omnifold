""" run_comp_plots.py - This function will generate "comp" plots which compare the reweighted
MC truth distributions produced by Omnifold step 2 with the truth pseudodata distributions.

It requires only the paths to the MC and truth pseudodata files, and a path to the file storing
the final weights that should be applied to the MC truth distribution.

Author: Kevin Greif
Last updated 08/30/2024
python3
"""

import argparse
import numpy as np
import uproot
import awkward as ak

import utils.plotting_utils as pu
import utils.data_utils as du
from wasserstein_metric import WassersteinOne

parser = argparse.ArgumentParser(description="Generate comp plots for reweighted MC and truth pseudodata comparison.")
parser.add_argument("--mc", type=str, help="The path to MC file")
parser.add_argument("--weights", type=str, help="The path to the weights file")
parser.add_argument("--pd", type=str, help="The path to PD file")
parser.add_argument("--store", type=str, help="The path to store the plots")
parser.add_argument("--use_test", action="store_true", help="If true, will use the weights for the testing MC rather than training")
parser.add_argument("--passBoth", action="store_true", help="If true, will require that the MC events pass the reco filter in addition to truth")
args = parser.parse_args()

# Set max events and max tracks to use for plotting
max_events = 5000000
max_track_events = 800000
max_tracks = 120

# Get the trees
mc_file = uproot.open(args.mc)
mc_tree = mc_file["OmniTree"]
pd_file = uproot.open(args.pd)
pd_tree = pd_file["OmniTree"]

# Get filters, since we are using truth level data, we need to use the truth_pass190 filter
mc_filter = ak.to_numpy(mc_tree["truth_pass190"].array(entry_stop=max_events))
pd_filter = ak.to_numpy(pd_tree["truth_pass190"].array(entry_stop=max_events))

# If we only want to use events that pass both reco and truth filters, take and here
if args.passBoth:
    mc_filter = np.logical_and(mc_filter, ak.to_numpy(mc_tree["pass190"].array(entry_stop=max_events)))
    pd_filter = np.logical_and(pd_filter, ak.to_numpy(pd_tree["pass190"].array(entry_stop=max_events)))

# Get original MC weights
mc_start_weights = ak.to_numpy(mc_tree["weight"].array(entry_stop=max_events))

# Truncate and filter (already truncated MC start weights for event observables, so just filter)
mc_start_weights_event = mc_start_weights[mc_filter == 1]

if len(mc_start_weights) > max_track_events:
    mc_start_weights_track = mc_start_weights[:max_track_events]
    mc_track_filter = mc_filter[:max_track_events]
else:
    mc_start_weights_track = mc_start_weights
    mc_track_filter = mc_filter
mc_start_weights_track = mc_start_weights_track[mc_track_filter == 1]

# Get new MC weights
mc_wgt_file = np.load(args.weights)
if args.use_test:
    mc_end_weights = mc_wgt_file["test"]
else:
    mc_end_weights = mc_wgt_file["train"]

# Truncate and filter for both event and track observables
if len(mc_end_weights) > max_events:
    mc_end_weights_event = mc_end_weights[:max_events]
else:
    mc_end_weights_event = mc_end_weights
mc_end_weights_event = mc_end_weights_event[mc_filter == 1]

if len(mc_end_weights) > max_track_events:
    mc_end_weights_track = mc_end_weights[:max_track_events]
else:
    mc_end_weights_track = mc_end_weights
mc_end_weights_track = mc_end_weights_track[mc_track_filter == 1]

# Get pseudodata weights (dropping 3 events that have negative weights)
pd_weights = ak.to_numpy(pd_tree["weight"].array(entry_stop=max_events))

# Truncate and filter weights
if len(pd_weights) > max_events:
    pd_weights_event = pd_weights[:max_events]
else:
    pd_weights_event = pd_weights
pd_weights_event = pd_weights_event[pd_filter == 1]

if len(pd_weights) > max_track_events:
    pd_weights_track = pd_weights[:max_track_events]
    pd_track_filter = pd_filter[:max_track_events]
else:
    pd_weights_track = pd_weights
    pd_track_filter = pd_filter
pd_weights_track = pd_weights_track[pd_track_filter == 1]

# Concatenate the weights
start_weights = np.concatenate([mc_start_weights_event, pd_weights_event], axis=0)
end_weights = np.concatenate([mc_end_weights_event, pd_weights_event], axis=0)
start_weights_track = np.concatenate([mc_start_weights_track, pd_weights_track], axis=0)
end_weights_track = np.concatenate([mc_end_weights_track, pd_weights_track], axis=0)

# Make labels
mc_labels = np.zeros_like(mc_start_weights_event)
pd_labels = np.ones_like(pd_weights_event)
labels = np.concatenate([mc_labels, pd_labels], axis=0)

mc_labels_track = np.zeros_like(mc_start_weights_track)
pd_labels_track = np.ones_like(pd_weights_track)
labels_track = np.concatenate([mc_labels_track, pd_labels_track], axis=0)

# Get the plot data, always want the truth level for this script. Take all events for event level observables
plotting_variables = pu.default_settings.keys()
mc_plotting = du.get_plotting(mc_tree, vars=plotting_variables, get_truth=True, stop=max_events, passBoth=args.passBoth)
pd_plotting = du.get_plotting(pd_tree, vars=plotting_variables, get_truth=True, stop=max_events, passBoth=args.passBoth)
plotting = ak.concatenate([mc_plotting, pd_plotting], axis=0)

# Get the track kinematics
mc_kinematics, _ = du.get_kinematics(mc_tree, get_truth=True, stop=max_track_events, passBoth=args.passBoth)
pd_kinematics, _ = du.get_kinematics(pd_tree, get_truth=True, stop=max_track_events, passBoth=args.passBoth)
kinematics = ak.concatenate([mc_kinematics, pd_kinematics], axis=0)

# Drop the muons
kinematics = kinematics[:,:,2:]

# Zero pad the kinematics, sends kinematics to a numpy array
kinematics = du.pad_kinematics(kinematics)

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

# Need to drop any events with negative weights for input to W1 calculation (annoying)
pos_weights = np.logical_and(start_weights > 0, end_weights > 0)
start_weights = start_weights[pos_weights]
end_weights = end_weights[pos_weights]
labels = labels[pos_weights]
plotting = plotting[pos_weights]
# Track plots can handle negative weights, so leave them

# Make the logged plots, using wasserstein metric class for the pre-computed overservables
wass = WassersteinOne(hist_info=new_settings, draw_plots=True, save_location=args.store)
wass.update(plotting, start_weights, end_weights, labels)
w1, plot_dict = wass.compute(from_torch=False, names=names, is_comp=True)
print(f"Computed Wasserstein One: {w1}")
pu.make_inclusive_track_plots(kinematics, labels_track, start_weights_track, end_weights=end_weights_track, definitions=new_track_settings, save_location=args.store, names=names)