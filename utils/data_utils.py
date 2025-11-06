"""data_utils.py - This file contains functions for preprocessing and handling
data for the training of Omnifold discriminators.

Author: Kevin Greif
Last updated 8/2/2025
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
    """get_one_hot - This function produces a one-hot encoding, which encodes
    which of the track jets in a given event the object belongs to.
    If the object is a muon, all of the one-hot encodings will be 0.
    There will be n_jets + 1 one-hot encodings, where the last one-hot encoding
    encodes whether the object belongs to any of the n_jets+1th jets or higher.

    Arguments:
    kinematics - numpy array of muon and track kinematics
    track_jet_indeces - numpy array of track jet indeces, one for each track
    n_jets - number of track jets to consider, optional

    Returns:
    one_hot - one-hot encoding as a numpy array
    """

    # List for accepting one-hot encodings
    tj_one_hots = []

    # Create the track jet one-hot encodings
    # Remove the singleton 1st dimension from the track_jet_indeces array
    track_jet_indeces = np.squeeze(track_jet_indeces, axis=1)
    for i in range(n_jets):

        # Find tracks that belong to the i-th track jet
        tj_one_hot = np.zeros((kinematics.shape[0], kinematics.shape[2]), dtype=np.int8)
        tj_one_hot[track_jet_indeces == i] = 1

        # Append to one-hot list
        tj_one_hots.append(tj_one_hot)

    # For any tracks not yet assigned a track jet, add one extra one-hot encoding
    # for the underlying event
    other_one_hot = np.zeros((kinematics.shape[0], kinematics.shape[2]), dtype=np.int8)
    other_one_hot[track_jet_indeces >= n_jets] = 1
    tj_one_hots.append(other_one_hot)

    # Stack one-hot encodings and return
    return np.stack(tj_one_hots, axis=1)


def get_masses(pdgids: ak.Array | np.ndarray) -> ak.Array | np.ndarray:
    """get_masses - This function will return the mass of the particle given the
    pdgid.

    Arguments:
        pdgids {np.ndarray or ak.Array} -- numpy or awkward array of pdgids,
            shape (n_events, 1, n_tracks)

    Returns:
        masses {np.ndarray or ak.Array} -- numpy or awkward array of masses,
            same type and shape as input pdgids
    """

    # Take absolute value of pdgids
    pdgids = np.abs(pdgids)

    # Check if input is awkward array
    is_awkward = isinstance(pdgids, ak.Array)

    if is_awkward:
        # Use awkward array operations
        # Create zeros array with same structure as pdgids
        # Use same pattern as ak.ones_like used elsewhere in this file
        masses = ak.ones_like(pdgids, dtype=np.float32) * 0.0
        masses = ak.where(pdgids == 13, 0.105658, masses)
        masses = ak.where(pdgids == 211, 0.13957, masses)
        masses = ak.where(pdgids == 321, 0.493677, masses)
        masses = ak.where(pdgids == 2212, 0.938272, masses)
        masses = ak.where(pdgids == 11, 0.000511, masses)
        masses = ak.where(pdgids == 3222, 1.189, masses)
        masses = ak.where(pdgids == 3112, 1.197, masses)
        masses = ak.where(pdgids == 3312, 1.321, masses)
        masses = ak.where(pdgids == 3334, 1.672, masses)
    else:
        # Use numpy array operations
        # Validate PDG IDs for numpy arrays
        assert np.all(
            np.isin(pdgids, [13, 211, 321, 2212, 11, 3222, 3112, 3312, 3334, -999])
        )
        masses = np.zeros_like(pdgids, dtype=np.float32)
        masses[pdgids == 13] = 0.105658
        masses[pdgids == 211] = 0.13957
        masses[pdgids == 321] = 0.493677
        masses[pdgids == 2212] = 0.938272
        masses[pdgids == 11] = 0.000511
        masses[pdgids == 3222] = 1.189
        masses[pdgids == 3112] = 1.197
        masses[pdgids == 3312] = 1.321
        masses[pdgids == 3334] = 1.672

    return masses


def get_kinematics(
    tree,
    evt_filter,
    muon_only=False,
    get_truth=False,
    get_truth_pdgids=False,
    start=None,
    stop=None,
    syst_kw=None,
    take_log_pt=True,
    **kwargs,
):
    """get_kinematics - This function will accept an uproot TTree object, and return the
    muon and track kinematics concatenated as a single awkward array.

    The function will also return a set of indeces which describe which AK4 track jet
    in the event a given track corresponds to, and pdgids for the particles.
    For reco level data, the pdgids will be either 13 (muon) or 211 (charged pion).
    For truth level data, the they can have the pdgid for common charged hadrons.
    Note the absolute value of the pdgids is used.

    Filtering can be applied to the events by passing a boolean array to the
    "evt_filter" argument.

    Arguments:
    tree - uproot TTree object
    evt_filter - boolean array of events to keep
    muon_only - boolean to return only muon kinematics, optional
    get_truth - If true, get the truth level data instead of reco, optional
    get_truth_pdgids - If true, get the truth level pdgids instead of fixing all tracks
        to 211 (charged pion), optional
    start - starting event index, optional
    stop - stopping event index, optional
    syst_kw - keyword argument for the systematic to apply, optional
    take_log_pt - boolean to take the logarithm of the pT values, optional
    **kwargs - keyword arguments for systematics to apply to track kinematics, optional

    Returns:
    (ak.Array) - awkward array of the concatenated muon and track kinematics
    (ak.Array) - awkward array of the track jet indeces
    (ak.Array) - awkward array of the pdgids
    """

    # Apply start / stop to filter
    if start is not None or stop is not None:
        evt_filter = evt_filter[start:stop]

    # Set prekeys and postkeys
    prekey = ""
    muon_prekey, muon_postkey = "", ""
    if get_truth:
        assert syst_kw is None, "Cannot run systematics on truth level data"
        prekey = "truth_"
        muon_prekey = "truth_"
    # Note we only adjust the muon keys if we are running muon systematics
    # track systematics do not effect the muon kinematics
    elif syst_kw is not None and "muon" in syst_kw:
        muon_prekey, muon_postkey = get_syst_pre_and_post_keys(syst_kw)

    # Muon information, take logarithm of pT values immediately
    # Note also that muon systs only effect the pT values, so only need to add special
    # pre and post keys for these
    m1_pt = ak.unflatten(
        tree[muon_prekey + "pT_l1" + muon_postkey].array(
            entry_start=start, entry_stop=stop
        ),
        1,
        axis=0,
    )
    m1_eta = ak.unflatten(
        tree[prekey + "eta_l1"].array(entry_start=start, entry_stop=stop),
        1,
        axis=0,
    )
    m1_phi = ak.unflatten(
        tree[prekey + "phi_l1"].array(entry_start=start, entry_stop=stop),
        1,
        axis=0,
    )
    m2_pt = ak.unflatten(
        tree[muon_prekey + "pT_l2" + muon_postkey].array(
            entry_start=start, entry_stop=stop
        ),
        1,
        axis=0,
    )
    m2_eta = ak.unflatten(
        tree[prekey + "eta_l2"].array(entry_start=start, entry_stop=stop),
        1,
        axis=0,
    )
    m2_phi = ak.unflatten(
        tree[prekey + "phi_l2"].array(entry_start=start, entry_stop=stop),
        1,
        axis=0,
    )
    if take_log_pt:
        m1_pt = np.log(m1_pt)
        m2_pt = np.log(m2_pt)
    m1_kinematics = ak.concatenate([m1_pt, m1_eta, m1_phi], axis=1)
    m1_kinematics = ak.unflatten(m1_kinematics, 1, axis=1)
    m2_kinematics = ak.concatenate([m2_pt, m2_eta, m2_phi], axis=1)
    m2_kinematics = ak.unflatten(m2_kinematics, 1, axis=1)
    kinematics = ak.concatenate([m1_kinematics, m2_kinematics], axis=2)

    # Set muon pdgids
    m1_pdgids = 13 * ak.ones_like(m1_pt)
    m1_pdgids = ak.unflatten(m1_pdgids, 1, axis=1)
    m2_pdgids = 13 * ak.ones_like(m2_pt)
    m2_pdgids = ak.unflatten(m2_pdgids, 1, axis=1)
    pdgids = ak.concatenate([m1_pdgids, m2_pdgids], axis=2)

    # Apply filter
    kinematics = kinematics[evt_filter == 1, ...]
    assert ak.all(kinematics > -98), "Kinematics contains -99s"
    pdgids = pdgids[evt_filter == 1, ...]

    # Track information if requested
    indeces = None
    if not muon_only:

        # Pull info, note taking log of track pT values here
        if syst_kw == "track_scale":
            assert not get_truth, "Cannot run track scale systematic on truth data"
            track_pt = ak.unflatten(
                tree["syst_correctedpT_tracks"].array(
                    entry_start=start, entry_stop=stop
                ),
                1,
                axis=0,
            )
        else:
            track_pt = ak.unflatten(
                tree[prekey + "pT_tracks"].array(entry_start=start, entry_stop=stop),
                1,
                axis=0,
            )
        if take_log_pt:
            track_pt = np.log(track_pt)
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
        indeces, track_kinematics = run_track_systematics(
            tree,
            track_kinematics,
            start=start,
            stop=stop,
            get_truth=get_truth,
            **kwargs,
        )

        # Track pdgids
        if get_truth_pdgids:
            assert get_truth, "Cannot get truth level pdgids without truth level data"
            track_pdgids = ak.unflatten(tree["truth_pdgId_tracks"].array(
                entry_start=start, entry_stop=stop
            ), 1, axis=0)
        else:
            track_pdgids = 211 * ak.ones_like(track_pt)

        # Apply filter then truncate if necessary
        track_kinematics = track_kinematics[evt_filter == 1, ...]
        indeces = indeces[evt_filter == 1, ...]
        track_pdgids = track_pdgids[evt_filter == 1, ...]

        # Concatenate muon and track kinematics, indeces, and pdgids
        kinematics = ak.concatenate([kinematics, track_kinematics], axis=2)
        muon_indeces = -1 * ak.from_numpy(np.ones((len(indeces), 1, 2), dtype=np.int32))
        indeces = ak.concatenate([muon_indeces, indeces], axis=2)
        pdgids = ak.concatenate([pdgids, track_pdgids], axis=2)

    # Return kinematics and track indeces
    return kinematics, indeces, pdgids


def get_observables(
    tree,
    key_list,
    evt_filter,
    start=None,
    stop=None,
):
    """get_observables - This function will accept an uproot TTree object, and
    return the requested branches as numpy arrays.
    Branches are passed in as a list of strings to the "key_list" argument.
    A filter can be passed in as a boolean array to the "evt_filter" argument.

    Arguments:
    tree - uproot TTree object
    key_list - list of strings of branches to return
    evt_filter - boolean array of events to keep, optional
    start - starting event index, optional
    stop - stopping event index, optional

    Returns:
    observables - numpy array of requested branches, stacked along the second axis
    """

    # Apply start / stop to filter
    if start is not None or stop is not None:
        evt_filter = evt_filter[start:stop]

    # Initialize empty list to hold requested branches
    observables = []

    # Loop over requested branches
    for key in key_list:
        this_var = ak.to_numpy(tree[key].array(entry_start=start, entry_stop=stop))
        observables.append(this_var)

    # Stack requested branches
    observables = np.stack(observables, axis=1)

    # Apply filter if passed
    if evt_filter is not None:
        observables = observables[evt_filter == 1, ...]

    return observables


def run_track_systematics(
    tree, track_kinematics, start=None, stop=None, get_truth=False, syst_kw=None
):
    """run_track_systematics - This function will run the systematics on the track
    kinematics.

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
    elif syst_kw == "track_fake":
        indices = tree["syst_Fake_trackJetIndex"].array(
            entry_start=start, entry_stop=stop
        )
    elif syst_kw == "track_scale":
        indices = tree["syst_correctedpT_trackJetIndex"].array(
            entry_start=start, entry_stop=stop
        )
    else:
        raise ValueError(f"Systematic {syst_kw} not recognized!")
    indices = ak.unflatten(indices, 1, axis=0)

    # If we are running track scale systematic, just return the indices and track
    # kinematics since we should have loaded the corrected pT values already
    if syst_kw == "track_scale":
        return indices, track_kinematics
    # Else we are running some other systematic that requires a track filter
    elif syst_kw == "track_eff":
        key = "syst_passTrackTruthFilter_tracks"
    elif syst_kw == "jet_track_eff":
        key = "syst_passJetTrackFilter_tracks"
    elif syst_kw == "track_fake":
        key = "syst_passTrackFake_tracks"
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
    """get_w1_obs - This function implements the logic for determining which
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

    # Get pre and post keys
    if get_truth:
        assert syst_kw is None
        prekey, postkey = "truth_", ""
    else:
        prekey, postkey = get_syst_pre_and_post_keys(syst_kw)

    # Loop through plotting config and get keys
    with open("./utils/plots_config.yml", "r") as stream:
        plots_config = yaml.safe_load(stream)
    w1_keys = []
    for plot in plots_config["plots"]:
        # Skip the observables not used for w1 calculation
        if not plots_config["plots"][plot]["w1_eval"]:
            continue
        # Pull the correct key
        key = plots_config["plots"][plot]["key"]
        # Check if a systematic is activate
        if syst_kw is not None:
            # Adjust key if systematic effects this observable
            if syst_kw in plots_config["plots"][plot]["modified"]:
                key = prekey + key + postkey
        # If systematic is not active, always use the prekey
        # to possibly get the truth level data
        else:
            key = prekey + key
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


def get_jj_info(tree, use_truth=False, start=None, stop=None):
    """get_jj_info - This function will return information about the system
    of the two leading track jets in the event. Specifically it calculates
    the invariant mass, and the rapidity difference. Information is
    returned as a tuple of numpy arrays.

    Arguments:
    tree - uproot TTree object
    use_truth - If true, get the truth level data instead of reco
    start - starting event index, optional
    stop - stopping event index, optional

    Returns:
    m_jj - numpy array of the invariant mass of the two leading track jets
    dy_jj - numpy array of the rapidity difference of the two leading track jets
    """

    prekey = "truth_" if use_truth else ""

    jkeys = ["pT_trackj", "y_trackj", "phi_trackj", "m_trackj"]
    j1keys = [prekey + key + "1" for key in jkeys]
    j2keys = [prekey + key + "2" for key in jkeys]

    j1 = tree.arrays(j1keys, entry_start=start, entry_stop=stop)
    j2 = tree.arrays(j2keys, entry_start=start, entry_stop=stop)

    dy_jj = np.abs(j1[prekey + "y_trackj1"] - j2[prekey + "y_trackj2"])

    px1 = j1[prekey + "pT_trackj1"] * np.cos(j1[prekey + "phi_trackj1"])
    py1 = j1[prekey + "pT_trackj1"] * np.sin(j1[prekey + "phi_trackj1"])
    mt1 = np.sqrt(j1[prekey + "pT_trackj1"] ** 2 + j1[prekey + "m_trackj1"] ** 2)
    pz1 = mt1 * np.sinh(j1[prekey + "y_trackj1"])
    E1 = mt1 * np.cosh(j1[prekey + "y_trackj1"])

    px2 = j2[prekey + "pT_trackj2"] * np.cos(j2[prekey + "phi_trackj2"])
    py2 = j2[prekey + "pT_trackj2"] * np.sin(j2[prekey + "phi_trackj2"])
    mt2 = np.sqrt(j2[prekey + "pT_trackj2"] ** 2 + j2[prekey + "m_trackj2"] ** 2)
    pz2 = mt2 * np.sinh(j2[prekey + "y_trackj2"])
    E2 = mt2 * np.cosh(j2[prekey + "y_trackj2"])

    m_jj = np.sqrt(
        (E1 + E2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    )

    return m_jj, dy_jj


def get_syst_pre_and_post_keys(syst_kw):
    """get_syst_pre_and_post_keys - This function returns the pre and post keys
    to apply to the key used to extract data from the trees for a given syst_kw.
    This function will be used by the data module and plotter classes.

    Arguments:
        syst_kw (str): Systematic to apply. options:
            None: nominal
            "track_eff": apply track efficiency systematic
            "jet_track_eff": apply jet track efficiency systematic
            "track_fake": apply track fake systematic
            "track_scale": apply track scale systematic
            "muon_id": apply muon ID track resolution up systematic
            "muon_ms": apply muon MS track resolution up systematic
            "muon_resbias": apply muon resolution bias up systematic
            "muon_rho": apply muon efficiency (?) up sytematic
            "muon_scale": apply muon scale up systematic

    Returns:
        pre_key (str): Pre key for the systematic
        post_key (str): Post key for the systematic
    """
    if syst_kw is None:
        return "", ""
    elif syst_kw == "muon_id":
        return "syst_", "_ID_Up"
    elif syst_kw == "muon_ms":
        return "syst_", "_MS_Up"
    elif syst_kw == "muon_resbias":
        return "syst_", "_MSResbias_Up"
    elif syst_kw == "muon_rho":
        return "syst_", "_MSRho_Up"
    elif syst_kw == "muon_scale":
        return "syst_", "_Scale_Up"
    elif syst_kw == "track_eff":
        return "syst_TrackFilter_", ""
    elif syst_kw == "jet_track_eff":
        return "syst_JetTrackFilter_", ""
    elif syst_kw == "track_fake":
        return "syst_Fake_", ""
    elif syst_kw == "track_scale":
        return "syst_pTScale_", ""
    else:
        raise ValueError(f"Systematic {syst_kw} not recognized!")


def calc_muon_syst_pass190(tree, stop=None, syst_kw=None):
    """calc_muon_syst_pass190 - This function will calculate the pass190 filter for
    the source data when running muon systematics.

    Arguments:
    tree - uproot TTree object
    stop - stopping event index
    syst_kw - keyword argument for the systematic to apply

    Returns:
    pass190_filter - boolean array of events that pass the pass190 filter
    """

    # Get the pre and post keys
    prekey, postkey = get_syst_pre_and_post_keys(syst_kw)

    # Load the needed branches
    ptll = ak.to_numpy(tree[prekey + "pT_ll" + postkey].array(entry_stop=stop))
    mll = ak.to_numpy(tree[prekey + "m_ll" + postkey].array(entry_stop=stop))
    yll = ak.to_numpy(tree["y_ll"].array(entry_stop=stop))

    # Return the filter
    return (ptll > 190) & (mll > 81) & (mll < 101) & (yll > -98)
