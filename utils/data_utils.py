""" data_utils.py - This file contains functions for preprocessing and handling 
data for the training of Omnifold discriminators.

Author: Kevin Greif
12/18/2023
python3
"""

import numpy as np
import awkward as ak
from pytorch_lightning.utilities.rank_zero import *


def pad_kinematics(input_array, max_tracks=200, fill=0) -> np.ndarray:
    """ pad_kinematics - This function will take an awkward array of track kinematic
    information, and returns a zero-padded array with the length of padding given by
    the max_tracks argument. Events with more tracks than max_tracks will be truncated.
    If max_track is None, the maximum number of tracks in the input array will be used.

    Arguments:
    input_array - awkward array of track kinematic information
    max_tracks - maximum number of tracks to pad to
    fill - value to fill the padding with, optional
    
    Returns:
    padded_array - zero-padded numpy array of track kinematic information
    """

    # Create zero-padded array
    input_array = ak.pad_none(input_array, max_tracks, axis=2, clip=True)
    input_array = ak.to_numpy(ak.fill_none(input_array, fill, axis=2))

    return input_array


def get_one_hot(kinematics, track_jet_indeces, n_jets=5):
    """ get_one_hot - This function produces a one-hot encoding, which says whether
    each object contained in the kinematics array is a muon, belongs to any of the 
    track jets from 1 to n_jets, or belongs to any other jet. All arrays initialized
    here will be of type np.int8. This will be cast to torch tensor with type float32.

    Arguments:
    kinematics - numpy array of muon and track kinematics
    track_jet_indeces - numpy array of track jet indeces, one for each track
    n_jets - number of track jets to consider, optional

    Returns:
    one_hot - one-hot encoding as a numpy array
    """

    # List for accepting one-hot encodings
    one_hots = []

    # Assume muons are the first two objects in the kinematics array
    muon_one_hot = np.concatenate([np.ones((kinematics.shape[0], 2), dtype=np.int8), np.zeros((kinematics.shape[0], kinematics.shape[2]-2), dtype=np.int8)], axis=1)
    one_hots.append(muon_one_hot)

    # Loop through track jet indeces
    # Remove the singleton 1st dimension from the track_jet_indeces array
    track_jet_indeces = np.squeeze(track_jet_indeces, axis=1)
    for i in range(n_jets):

        # Find tracks that belong to the i-th track jet
        tj_one_hot = np.zeros((kinematics.shape[0], kinematics.shape[2]), dtype=np.int8)
        tj_one_hot[track_jet_indeces == i] = 1

        # Append to one-hot list
        one_hots.append(tj_one_hot)

    # For any other tracks, do the same as above
    other_one_hot = np.zeros((kinematics.shape[0], kinematics.shape[2]), dtype=np.int8)
    other_one_hot[track_jet_indeces >= n_jets] = 1
    one_hots.append(other_one_hot)

    # Stack one-hot encodings and return
    return np.stack(one_hots, axis=1)


def get_kinematics(tree, filter=None, muon_only=False, get_truth=False, max_events=None):
    """ get_kinematics - This function will accept an uproot TTree object, and return the
    muon and track kinematics concatenated as a single awkward.
    
    The function will also return a set of indeces which describe which AK4 track jet
    in the event a given track corresponds to.

    An optional "filter" argument will allow the user to filter events with a boolean array.

    Arguments:
    tree - uproot TTree object
    filter - boolean array to filter events, optional
    muon_only - boolean to return only muon kinematics, optional
    get_truth - If true, get the truth level data instead of reco, optional
    max_events - maximum number of events to process, optional

    Returns:
    (ak.Array) - awkward array of the concatenated muon and track kinematics
    (ak.Array) - awkward array of the track jet indeces
    """

    # Muon information, take logarithm of pT values immediately
    prekey = ""
    if get_truth:
        prekey = "truth_"
    m1_pt = np.log(ak.unflatten(tree[prekey+'pT_l1'].array(), 1, axis=0))
    m1_eta = ak.unflatten(tree[prekey+'eta_l1'].array(), 1, axis=0)
    m1_phi = ak.unflatten(tree[prekey+'phi_l1'].array(), 1, axis=0)
    m2_pt = np.log(ak.unflatten(tree[prekey+'pT_l2'].array(), 1, axis=0))
    m2_eta = ak.unflatten(tree[prekey+'eta_l2'].array(), 1, axis=0)
    m2_phi = ak.unflatten(tree[prekey+'phi_l2'].array(), 1, axis=0)

    m1_kinematics = ak.concatenate([m1_pt, m1_eta, m1_phi], axis=1)
    m1_kinematics = ak.unflatten(m1_kinematics, 1, axis=1)
    m2_kinematics = ak.concatenate([m2_pt, m2_eta, m2_phi], axis=1)
    m2_kinematics = ak.unflatten(m2_kinematics, 1, axis=1)
    kinematics = ak.concatenate([m1_kinematics, m2_kinematics], axis=2)

    # Check if filter and kinematics have the same size in 0th dimension
    # They often don't, so longer one needs to be truncated
    if filter is not None:
        if len(filter) != len(kinematics):
            rank_zero_info("Warning! Filter shape does not match muon kinematics shape!")
            rank_zero_info(f"Filter shape: {len(filter)}")
            rank_zero_info(f"Muon kinematics shape: {len(kinematics)}")
            lesser_shape = min(len(filter), len(kinematics))
            kinematics = kinematics[:lesser_shape,...]
            filter = filter[:lesser_shape]

    # Apply filter then truncate if necessary
    if filter is not None:
        kinematics = kinematics[filter == 1,...]
    if max_events is not None:
        kinematics = kinematics[:int(max_events),...]

    # Track information if requested
    indeces = None
    if not muon_only:

        # Pull info, note taking log of track pT values here
        track_pt = np.log(ak.unflatten(tree[prekey+'pT_tracks'].array(), 1, axis=0))
        track_eta = ak.unflatten(tree[prekey+'eta_tracks'].array(), 1, axis=0)
        track_phi = ak.unflatten(tree[prekey+'phi_tracks'].array(), 1, axis=0)
        track_kinematics = ak.concatenate([track_pt, track_eta, track_phi], axis=1)
        indeces = ak.unflatten(tree[prekey+'trackJetIndex_tracks'].array(), 1, axis=0)

        # Check if filter and track kinematics have the same size in 0th dimension
        # They often don't, so longer one needs to be truncated
        if filter is not None:
            if len(filter) != len(track_kinematics):
                rank_zero_info("Warning! Filter shape does not match track kinematics shape!")
                rank_zero_info(f"Filter shape: {len(filter)}")
                rank_zero_info(f"Track kinematics shape: {len(track_kinematics)}")
                lesser_shape = min(len(filter), len(track_kinematics))
                track_kinematics = track_kinematics[:lesser_shape,...]
                indeces = indeces[:lesser_shape,...]
                filter = filter[:lesser_shape]

        # Apply filter then truncate if necessary
        if filter is not None:
            track_kinematics = track_kinematics[filter == 1,...]
            indeces = indeces[filter == 1,...]
        if max_events is not None:
            track_kinematics = track_kinematics[:max_events,...]
            indeces = indeces[:max_events,...]

        # Concatenate muon and track kinematics + indeces
        kinematics = ak.concatenate([kinematics, track_kinematics], axis=2)
        muon_indeces = -1 * ak.from_numpy(np.ones((len(indeces), 1, 2), dtype=np.int8))
        indeces = ak.concatenate([muon_indeces, indeces], axis=2)

    # Return kinematics and track indeces
    return kinematics, indeces
    

def get_plotting(tree, vars=[], filter=None, muon_only=False, get_truth=False, max_events=None, **kwargs):
    """ get_plotting - This function will accept an uproot TTree object, and return the
    requested branches as numpy arrays. Branches are passed in as a list of strings to the
    "vars" keyword argument.

    Arguments:
    tree - uproot TTree object
    vars - list of strings of branches to return
    filter - boolean array to filter events, optional
    muon_only - boolean to return only muon kinematics, in practice just does not
    truncate away the 11k events with missing track information
    get_truth - If true, get the truth level data instead of reco
    max_events - maximum number of events to process, optional

    Returns:
    plotting - numpy array of requested branches, stacked along the second axis
    """

    # Initialize empty list to hold requested branches
    plotting = []

    # Compare the number of events with tracks to the number of events with muons
    # If they are not the same we need to truncate
    if not muon_only:
        track_pt_key = 'pT_tracks'
        if get_truth:
            track_pt_key = 'truth_pT_tracks'
        track_pts = tree[track_pt_key].array()
        num_track_events = len(track_pts)
        muon_pts = tree['pT_l1'].array()
        num_muon_events = len(muon_pts)
        num_good_events = min(num_track_events, num_muon_events)
    # Else look at the leading muon pT to find the # of good events
    else:
        muon_pts = tree['pT_l1'].array()
        num_good_events = len(muon_pts)

    # Take the minimum of the number of good events and the max events
    if max_events is not None:
        take_events = min(num_good_events, max_events)
    else:
        take_events = num_good_events

    # Loop over requested branches
    for var in vars:
        if get_truth:
            var = "truth_" + var
        plotting.append(ak.to_numpy(tree[var].array()))

    # Stack requested branches
    plotting = np.stack(plotting, axis=1)

    # Apply filter if passed
    if filter is not None:
        plotting = plotting[filter == True,...]

    # Take the correct number of events that pass filter
    if max_events is not None:
        plotting = plotting[:take_events,...]

    return plotting


def custom_collate(batch):
    """ custom_collate - This is a custom collate function for passing to the Pytorch
    DataLoader objects. Since the OfDataset is already designed to use batched data,
    this is just the identity function.

    Arguments:
    batch - a batch of data from the DataLoader

    Returns:
    batch - the same batch of data
    """
    return batch