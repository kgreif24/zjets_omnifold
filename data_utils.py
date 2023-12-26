""" data_utils.py - This file contains functions for preprocessing and handling 
data for the training of Omnifold discriminators.

Author: Kevin Greif
12/18/2023
python3
"""

import numpy as np
import awkward as ak


def pad_kinematics(input_array, max_tracks=None) -> np.ndarray:
    """ pad_kinematics - This function will take an awkward array of track kinematic
    information, and returns a zero-padded array with the length of padding given by
    the max_tracks argument. Events with more tracks than max_tracks will be truncated.
    If max_track is None, the maximum number of tracks in the input array will be used.

    Arguments:
    input_array - awkward array of track kinematic information
    max_tracks - maximum number of tracks to pad to
    
    Returns:
    padded_array - zero-padded numpy array of track kinematic information
    """

    # If max_tracks is None, get the maximum number of tracks in the input array
    if max_tracks is None:
        max_tracks = int(np.max(ak.count(input_array, axis=1)))

    # Create zero-padded array
    input_array = ak.pad_none(input_array, max_tracks, axis=1, clip=True)
    input_array = ak.to_numpy(ak.fill_none(input_array, 0, axis=1))

    return input_array


def get_kinematics(tree, filter=None, get_mask=True, **kwargs):
    """ get_kinematics - This function will accept an uproot TTree object, and return the
    muon and track kinematics concatenatd as a single numpy array. An optional "filter"
    argument will allow the user to filter events with a boolean array.

    Note: Should add option to include one-hot encoding eventually

    Arguments:
    tree - uproot TTree object
    filter - boolean array to filter events, optional
    get_mask - boolean to return mask of padded tracks, optional
    **kwargs - keyword arguments to pass to the "pad_kinematics" function

    Returns:
    kinematics - numpy array of muon and track kinematics
    """

    # Muon information
    m1_pt = ak.to_numpy(tree['pT_l1'].array())
    m1_eta = ak.to_numpy(tree['eta_l1'].array())
    m1_phi = ak.to_numpy(tree['phi_l1'].array())
    m2_pt = ak.to_numpy(tree['pT_l2'].array())
    m2_eta = ak.to_numpy(tree['eta_l2'].array())
    m2_phi = ak.to_numpy(tree['phi_l2'].array())

    m1_kinematics = np.stack([m1_pt, m1_eta, m1_phi], axis=1)
    m2_kinematics = np.stack([m2_pt, m2_eta, m2_phi], axis=1)
    muon_kinematics = np.stack([m1_kinematics, m2_kinematics], axis=2)

    # Track information
    track_pt = tree['pT_tracks'].array()
    track_eta = tree['eta_tracks'].array()
    track_phi = tree['phi_tracks'].array()

    # Run padding function
    pad_pt = pad_kinematics(track_pt, **kwargs)
    pad_eta = pad_kinematics(track_eta, **kwargs)
    pad_phi = pad_kinematics(track_phi, **kwargs)

    # Stack tracks
    track_kinematics = np.stack([pad_pt, pad_eta, pad_phi], axis=1)
    # Hack to fix missing 11k events, should be removed!
    if track_kinematics.shape[0] != muon_kinematics.shape[0]:
        print("Warning! Track kinematics shape does not match muon kinematics shape!")
        muon_kinematics = muon_kinematics[:track_kinematics.shape[0],...]
        filter = filter[:track_kinematics.shape[0]]

    # Concatenate muon and track kinematics
    kinematics = np.concatenate([muon_kinematics, track_kinematics], axis=2)

    # Filter kinematics by pass 190 flag
    if filter is None:
        filter = np.ones(kinematics.shape[0], dtype=bool)
    kinematics = kinematics[filter == True,...]

    # Build padded track mask if necessary
    if get_mask:
        track_mask = np.zeros_like(kinematics[:,0,:], dtype=bool)
        # Note assumes pT is the 0th element along axis 1
        track_mask[kinematics[:,0,:] != 0] = True
        track_mask = np.expand_dims(track_mask, axis=1)
        return kinematics, track_mask

    # Else just return the kinematics array
    else:
        return kinematics
