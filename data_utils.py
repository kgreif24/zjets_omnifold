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
    for i in range(n_jets):

        # Find tracks that belong to the i-th track jet
        tj_one_hot = np.zeros((kinematics.shape[0], kinematics.shape[2]-2), dtype=np.int8)
        tj_one_hot[track_jet_indeces == i] = 1

        # Add on more zeros for muon one-hot encodings
        tj_one_hot = np.concatenate([np.zeros((kinematics.shape[0], 2), dtype=np.int8), tj_one_hot], axis=1)

        # Append to one-hot list
        one_hots.append(tj_one_hot)

    # For any other tracks, do the same as above
    other_one_hot = np.zeros((kinematics.shape[0], kinematics.shape[2]-2), dtype=np.int8)
    other_one_hot[track_jet_indeces >= n_jets] = 1
    other_one_hot = np.concatenate([np.zeros((kinematics.shape[0], 2), dtype=np.int8), other_one_hot], axis=1)
    one_hots.append(other_one_hot)

    # Stack one-hot encodings and return
    return np.stack(one_hots, axis=1)


def get_kinematics(tree, filter=None, get_mask=True, muon_only=False, one_hot=True, **kwargs):
    """ get_kinematics - This function will accept an uproot TTree object, and return the
    muon and track kinematics concatenatd as a single numpy array. An optional "filter"
    argument will allow the user to filter events with a boolean array.

    Note: Should add option to include one-hot encoding eventually

    Arguments:
    tree - uproot TTree object
    filter - boolean array to filter events, optional
    get_mask - boolean to return mask of padded tracks, optional
    muon_only - boolean to return only muon kinematics, optional
    one_hot - boolean to include one-hot encoded labels of muon, track jet 1-5,
        and everything else, optional
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
    kinematics = np.stack([m1_kinematics, m2_kinematics], axis=2)

    # Track information if requested
    if not muon_only:

        # Pull info, note taking log of track pT values here
        track_pt = np.log(tree['pT_tracks'].array())
        track_eta = tree['eta_tracks'].array()
        track_phi = tree['phi_tracks'].array()

        # Run padding function
        pad_pt = pad_kinematics(track_pt, **kwargs)
        pad_eta = pad_kinematics(track_eta, **kwargs)
        pad_phi = pad_kinematics(track_phi, **kwargs)

        # Stack tracks
        track_kinematics = np.stack([pad_pt, pad_eta, pad_phi], axis=1)

        # Hack to fix missing 11k events, should be removed!
        if track_kinematics.shape[0] != kinematics.shape[0]:
            print("Warning! Track kinematics shape does not match muon kinematics shape!")
            kinematics = kinematics[:track_kinematics.shape[0],...]
            filter = filter[:track_kinematics.shape[0]]

        # Concatenate muon and track kinematics
        kinematics = np.concatenate([kinematics, track_kinematics], axis=2)

        # Make one hot encodings if requested
        if one_hot:
            track_jet_indeces = tree['trackJetIndex_tracks'].array()[:track_kinematics.shape[0],...]
            pad_indeces = pad_kinematics(track_jet_indeces, **kwargs)
            one_hot_inputs = get_one_hot(kinematics, pad_indeces)
            kinematics = np.concatenate([kinematics, one_hot_inputs], axis=1)

    # Apply filter
    if filter is not None:
        kinematics = kinematics[filter == True,...]

    # Take log of muon pT values
    kinematics[:,0,0:2] = np.log(kinematics[:,0,0:2])

    # Build padded mask if necessary
    if get_mask:
        mask = np.zeros_like(kinematics[:,0,:], dtype=bool)
        # Note assumes pT is the 0th element along axis 1
        mask[kinematics[:,0,:] != 0] = True
        mask = np.expand_dims(mask, axis=1)
        return kinematics, mask

    # Else just return the kinematics array
    else:
        return kinematics


def get_plotting(tree, vars=[], filter=None, muon_only=False, **kwargs):
    """ get_plotting - This function will accept an uproot TTree object, and return the
    requested branches as numpy arrays. Branches are passed in as a list of strings to the
    "vars" keyword argument.

    Arguments:
    tree - uproot TTree object
    vars - list of strings of branches to return
    filter - boolean array to filter events, optional
    muon_only - boolean to return only muon kinematics, in practice just does not
    truncate away the 11k events with missing track information

    Returns:
    plotting - numpy array of requested branches, stacked along the second axis
    """

    # Initialize empty list to hold requested branches
    plotting = []

    # Look at track pT information if we need to truncate away 11k events with missing tracks
    if not muon_only:
        track_pts = tree['pT_tracks'].array()
        num_good_events = len(track_pts)
    else:
        muon_pts = tree['pT_l1'].array()
        num_good_events = len(muon_pts)

    # Loop over requested branches
    for var in vars:
        plotting.append(ak.to_numpy(tree[var].array())[:num_good_events])

    # Stack requested branches
    plotting = np.stack(plotting, axis=1)

    # Apply filter if passed
    if filter is not None:
        filter = filter[:num_good_events]
        plotting = plotting[filter == True,...]

    return plotting
