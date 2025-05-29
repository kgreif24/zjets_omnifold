"""data_utils.py - This file contains functions for preprocessing and handling
data for the training of Omnifold discriminators.

Author: Kevin Greif
12/18/2023
python3
"""

import yaml
import numpy as np
import awkward as ak


def pad_kinematics(input_array, max_tracks=200, fill=0) -> np.ndarray:
    """pad_kinematics - This function will take an awkward array of track kinematic
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
    """get_one_hot - This function produces a one-hot encoding, which says whether
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
    muon_one_hot = np.concatenate(
        [
            np.ones((kinematics.shape[0], 2), dtype=np.int8),
            np.zeros((kinematics.shape[0], kinematics.shape[2] - 2), dtype=np.int8),
        ],
        axis=1,
    )
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


def get_kinematics(
    tree,
    muon_only=False,
    get_truth=False,
    start=None,
    stop=None,
    passBoth=False,
    **kwargs,
):
    """get_kinematics - This function will accept an uproot TTree object, and return the
    muon and track kinematics concatenated as a single awkward array.

    The function will also return a set of indeces which describe which AK4 track jet
    in the event a given track corresponds to.

    Note this function filters events by the appropriate pass190 branch, so do not
    expect to see exactly the number of events requested by the start and stop
    arguments.

    Arguments:
    tree - uproot TTree object
    muon_only - boolean to return only muon kinematics, optional
    get_truth - If true, get the truth level data instead of reco, optional
    start - starting event index, optional
    stop - stopping event index, optional
    passBoth - If true, require events pass both reco and truth selection
    **kwargs - keyword arguments for systematics to apply to track kinematics

    Returns:
    (ak.Array) - awkward array of the concatenated muon and track kinematics
    (ak.Array) - awkward array of the track jet indeces
    """

    # Set prekey
    prekey = ""
    if get_truth:
        prekey = "truth_"

    # Filter information
    evt_filter = ak.to_numpy(
        tree[prekey + "pass190"].array(entry_start=start, entry_stop=stop)
    )
    if passBoth:
        p190 = ak.to_numpy(tree["pass190"].array(entry_start=start, entry_stop=stop))
        truth_p190 = ak.to_numpy(
            tree["truth_pass190"].array(entry_start=start, entry_stop=stop)
        )
        evt_filter = np.local_and(p190, truth_p190)

    # Muon information, take logarithm of pT values immediately
    m1_pt = np.log(
        ak.unflatten(
            tree[prekey + "pT_l1"].array(entry_start=start, entry_stop=stop), 1, axis=0
        )
    )
    m1_eta = ak.unflatten(
        tree[prekey + "eta_l1"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )
    m1_phi = ak.unflatten(
        tree[prekey + "phi_l1"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )
    m2_pt = np.log(
        ak.unflatten(
            tree[prekey + "pT_l2"].array(entry_start=start, entry_stop=stop), 1, axis=0
        )
    )
    m2_eta = ak.unflatten(
        tree[prekey + "eta_l2"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )
    m2_phi = ak.unflatten(
        tree[prekey + "phi_l2"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )

    m1_kinematics = ak.concatenate([m1_pt, m1_eta, m1_phi], axis=1)
    m1_kinematics = ak.unflatten(m1_kinematics, 1, axis=1)
    m2_kinematics = ak.concatenate([m2_pt, m2_eta, m2_phi], axis=1)
    m2_kinematics = ak.unflatten(m2_kinematics, 1, axis=1)
    kinematics = ak.concatenate([m1_kinematics, m2_kinematics], axis=2)

    # Apply filter
    kinematics = kinematics[evt_filter == 1, ...]

    # Track information if requested
    indeces = None
    if not muon_only:

        # Pull info, note taking log of track pT values here
        track_pt = np.log(
            ak.unflatten(
                tree[prekey + "pT_tracks"].array(entry_start=start, entry_stop=stop),
                1,
                axis=0,
            )
        )
        track_eta = ak.unflatten(
            tree[prekey + "eta_tracks"].array(entry_start=start, entry_stop=stop),
            1,
            axis=0,
        )
        track_phi = ak.unflatten(
            tree[prekey + "phi_tracks"].array(entry_start=start, entry_stop=stop),
            1,
            axis=0,
        )
        track_kinematics = ak.concatenate([track_pt, track_eta, track_phi], axis=1)

        # Run possible systematics on track kinematics, and get indices
        indeces, track_kinematics = run_systematics(
            tree,
            track_kinematics,
            start=start,
            stop=stop,
            get_truth=get_truth,
            **kwargs,
        )

        # Apply filter then truncate if necessary
        track_kinematics = track_kinematics[evt_filter == 1, ...]
        indeces = indeces[evt_filter == 1, ...]

        # Concatenate muon and track kinematics + indeces
        kinematics = ak.concatenate([kinematics, track_kinematics], axis=2)
        muon_indeces = -1 * ak.from_numpy(np.ones((len(indeces), 1, 2), dtype=np.int8))
        indeces = ak.concatenate([muon_indeces, indeces], axis=2)

    # Return kinematics and track indeces
    return kinematics, indeces


def get_observables(
    tree,
    key_list,
    get_truth=False,
    start=None,
    stop=None,
    passBoth=False,
):
    """get_observables - This function will accept an uproot TTree object, and
    return the requested branches as numpy arrays.
    Branches are passed in as a list of strings to the "vars" keyword argument.

    Arguments:
    tree - uproot TTree object
    vars - list of strings of branches to return
    get_truth - If true, get the truth level data instead of reco
    start - starting event index, optional
    stop - stopping event index, optional
    passBoth - If true, require events pass both reco and truth selections

    Returns:
    plotting - numpy array of requested branches, stacked along the second axis
    """

    # Initialize empty list to hold requested branches
    observables = []

    # Get filter, need to know whether we want truth or reco level data
    prekey = ""
    if get_truth:
        prekey = "truth_"
    evt_filter = ak.to_numpy(
        tree[prekey + "pass190"].array(entry_start=start, entry_stop=stop)
    )
    if passBoth:
        p190 = ak.to_numpy(tree["pass190"].array(entry_start=start, entry_stop=stop))
        truth_p190 = ak.to_numpy(
            tree["truth_pass190"].array(entry_start=start, entry_stop=stop)
        )
        evt_filter = np.logical_and(p190, truth_p190)

    # Loop over requested branches
    for key in key_list:
        this_var = ak.to_numpy(
            tree[key].array(entry_start=start, entry_stop=stop)
        )
        observables.append(this_var)

    # Stack requested branches
    observables = np.stack(observables, axis=1)

    # Apply filter if passed
    observables = observables[evt_filter == 1, ...]

    return observables


def run_systematics(
    tree, track_kinematics, start=None, stop=None, get_truth=False, syst_kw=None
):
    """run_systematics - This function will run the systematics on the track
    kinematics. The systematics are passed in as keyword arguments, and are
    applied to the track kinematics.

    Arguments:
    tree - uproot TTree object
    track_kinematics - awkward array of track kinematics with shape
        (n_events, 3, n_tracks), where the 3 is (log(pT), eta, phi)
    start - starting event index, optional
    stop - stopping event index, optional
    get_truth - If true, get the truth level data instead of reco
    syst_kw - keyword argument for the systematic to apply. options:
        None: nominal
        "track_eff": apply track efficiency systematic
        "jet_track_eff": apply jet track efficiency systematic

    Returns:
    indices - numpy array of indices for the track jets
    track_kinematics - numpy array of track kinematics with systematics applied
    """

    # Set prekey
    prekey = ""
    if get_truth:
        assert syst_kw is None, "Cannot run systematics on truth level data"
        prekey = "truth_"

    # If syst_kw is None, just load the track jet indices and return
    if syst_kw is None:
        indices = tree[prekey + "trackJetIndex_tracks"].array(
            entry_start=start,
            entry_stop=stop,
        )
        indices = ak.unflatten(indices, 1, axis=0)
        return indices, track_kinematics
    # Else load the correct track jet indices
    elif syst_kw == "track_eff":
        indices = tree["syst_TrackFilter_trackJetIndex"].array(
            entry_start=start, entry_stop=stop
        )
    elif syst_kw == "jet_track_eff":
        indices = tree["syst_JetTrackFilter_trackJetIndex"].array(
            entry_start=start, entry_stop=stop
        )
    else:
        raise ValueError(f"Systematic {syst_kw} not recognized!")
    indices = ak.unflatten(indices, 1, axis=0)

    # Else we are running some systematic.
    # To start we will only implement track efficiency
    # Load track filter for efficiency systematics
    if syst_kw == "track_eff":
        key = "syst_passTrackTruthFilter_tracks"
    elif syst_kw == "jet_track_eff":
        key = "syst_passJetTrackFilter_tracks"
    else:
        raise ValueError(f"Systematic {syst_kw} not recognized!")
    track_filter = tree[key].array(entry_start=start, entry_stop=stop)

    # Broadcast the filter to the shape of the track kinematics
    track_filter = ak.unflatten(track_filter, 1, axis=0)
    bc_track_filter, _ = ak.broadcast_arrays(track_filter, track_kinematics)

    # Drop tracks and indices that fail the filter
    track_kinematics = track_kinematics[bc_track_filter == 1]
    indices = indices[track_filter == 1]

    # Return the indices and track kinematics
    return indices, track_kinematics


def get_w1_obs(get_truth=False, syst_kw=None):
    """ get_w1_obs - This function implements the logic for determining which
    keys to use for pulling observables for the wasserstein distance calculation.
    The list will depend on the "plots_config.yml" file, as well as whether we want
    truth or reco level data, and whether a systematic is applied.

    Arguments:
    get_truth - If true, get the truth level data instead of reco
    syst_kw - keyword argument for the systematic to apply. options:
        None: nominal
        "track_eff": apply track efficiency systematic
        "jet_track_eff": apply jet track efficiency systematic
        "track_fake": apply track fake systematic
        "track_scale": apply track scale systematic

    Returns:
    w1_keys - list of keys to use for pulling observables for the wasserstein
        distance calculation
    """

    # Mapping dictionary for syst_kw to prekey in trees
    syst_map = {
        None: "",
        "track_eff": "syst_TrackFilter_",
        "jet_track_eff": "syst_JetTrackFilter_",
        "track_fake": "syst_Fake_",
        "track_scale": "syst_pTScale_",
    }

    # Set prekey
    if get_truth:
        assert syst_kw is None, "Cannot run systematics on truth level data"
        prekey = "truth_"
    elif syst_kw is not None:
        assert not get_truth, "Cannot run systematics on truth level data"
        prekey = syst_map[syst_kw]
    else:
        prekey = ""

    # Loop through plotting config and get keys
    with open("./utils/plots_config.yml", "r") as stream:
        plots_config = yaml.safe_load(stream)
    w1_keys = []
    for plot in plots_config["plots"]:
        # Skip the observables not used for w1 calculation
        if not plots_config["plots"][plot]["w1_eval"]:
            continue
        # Check if systematic is active
        if syst_kw is not None:
            # Add prekey if systematic effects this observabel
            if syst_kw in plots_config["plots"][plot]["modified"]:
                key = prekey + plots_config["plots"][plot]["key"]
            # Else use the key as is
            else:
                key = plots_config["plots"][plot]["key"]
        # If systematic is not active, always use the prekey
        else:
            # Else awlays use
            key = prekey + plots_config["plots"][plot]["key"]
        w1_keys.append(key)

    return w1_keys


def null_collate(batch):
    """null_collate - This is a custom collate function for passing to the Pytorch
    DataLoader objects. Since the OfDataset is already designed to use batched data,
    this is just the identity function.

    Arguments:
    batch - a batch of data from the DataLoader

    Returns:
    batch - the same batch of data
    """
    return batch
