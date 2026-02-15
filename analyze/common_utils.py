"""
Common utility functions for use in analyzing the Z+jets Omnifold data.
"""

from __future__ import annotations

import numpy as np
import awkward as ak


def extract_kinematics(
    tree,
    pass_flags,
    get_truth: bool = True,
    start: int = None,
    stop: int = None,
    filter_presliced: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract kinematics and masses from a ROOT file TTree.

    Arguments:
    ----------
    tree : uproot.TTree
        The TTree object from uproot containing event data.
    pass_flags : np.ndarray
        Boolean array of pass flags for event filtering.
    get_truth : bool, optional
        If True, get truth-level data. If False, get reco-level data.
    start : int, optional
        Starting event index. If None, start at the beginning of the array.
    stop : int, optional
        Stopping event index. If None, stop at the end of the array.
    filter_presliced : bool, optional
        If True, pass_flags is already sliced for the [start:stop] range and
        should not be sliced again. Default False.

    Returns:
    --------
    pt : np.ndarray
        Per-event pT arrays, shape (n_events, n_particles).
    eta : np.ndarray
        Per-event eta arrays, shape (n_events, n_particles).
    phi : np.ndarray
        Per-event phi arrays, shape (n_events, n_particles).
    masses : np.ndarray
        Per-event mass arrays, shape (n_events, n_particles).
    """

    kinematics, pdgids = get_kinematics(
        tree,
        pass_flags,
        get_truth=get_truth,
        get_truth_pdgids=get_truth,
        start=start,
        stop=stop,
        filter_presliced=filter_presliced,
    )

    # Extract pT, eta, phi from kinematics array
    # kinematics shape is (n_events, 3, n_particles) where axis 1 is [pT, eta, phi]
    pt = kinematics[:, 0, :]  # Extract pT
    eta = kinematics[:, 1, :]  # Extract eta
    phi = kinematics[:, 2, :]  # Extract phi

    # Get masses from pdgids
    masses = get_masses(pdgids)[:, 0, :]

    return pt, eta, phi, masses


def get_kinematics(
    tree,
    evt_filter,
    get_truth=False,
    get_truth_pdgids=False,
    start=None,
    stop=None,
    filter_presliced=False,
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
    get_truth - If true, get the truth level data instead of reco, optional
    get_truth_pdgids - If true, get the truth level pdgids instead of fixing all tracks
        to 211 (charged pion), optional
    start - starting event index, optional
    stop - stopping event index, optional
    filter_presliced - If true, evt_filter is already sliced for [start:stop] range
        and should not be sliced again, optional

    Returns:
    (ak.Array) - awkward array of the concatenated kinematics
    (ak.Array) - awkward array of the pdgids
    """

    # Apply start / stop to filter (unless already pre-sliced)
    if not filter_presliced and (start is not None or stop is not None):
        evt_filter = evt_filter[start:stop]

    # Set prekey based on get_truth
    prekey = ""
    if get_truth:
        prekey = "truth_"

    # Get kinematics
    m1_pt = ak.unflatten(
        tree[prekey + "pT_l1"].array(entry_start=start, entry_stop=stop),
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
        tree[prekey + "pT_l2"].array(entry_start=start, entry_stop=stop),
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

    # Track information
    track_pt = ak.unflatten(
        tree[prekey + "pT_tracks"].array(entry_start=start, entry_stop=stop),
        1,
        axis=0,
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

    # Track pdgids
    track_pdgids = 211 * ak.ones_like(track_pt)
    if get_truth_pdgids:
        assert get_truth, "Cannot get truth level pdgids without truth level data"
        track_pdgids = ak.unflatten(
            tree[prekey + "pdgId_tracks"].array(entry_start=start, entry_stop=stop),
            1,
            axis=0,
        )

    # Apply filter then truncate if necessary
    track_kinematics = track_kinematics[evt_filter == 1, ...]
    track_pdgids = track_pdgids[evt_filter == 1, ...]

    # Concatenate muon and track information
    kinematics = ak.concatenate([kinematics, track_kinematics], axis=2)
    pdgids = ak.concatenate([pdgids, track_pdgids], axis=2)

    # Return kinematics and track indeces
    return kinematics, pdgids


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
            np.isin(pdgids, [13, 211, 321, 2212, 11, 3222, 3112, 3312, 3334, 999])
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
