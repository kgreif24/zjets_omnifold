"""
Common utility functions for use in analyzing the Z+jets Omnifold data.
"""

from __future__ import annotations

import numpy as np
import awkward as ak
import jet_clusterer
import energyflow as ef
import psutil
import sys


def check_memory(limit_gb=20):
    """Check current process memory and exit if over limit"""
    mem_gb = psutil.Process().memory_info().rss / 1e9
    print(f"[Memory] Current usage: {mem_gb:.2f} GB")
    if mem_gb > limit_gb:
        print(f"[Memory] Exceeded {limit_gb} GB, exiting safely!")
        sys.exit(1)


def extract_kinematics(
    tree,
    get_truth: bool = True,
    start: int = None,
    stop: int = None,
    min_track_pt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract kinematics and masses from a ROOT file TTree.

    Arguments:
    ----------
    tree : uproot.TTree
        The TTree object from uproot containing event data.
    get_truth : bool, optional
        If True, get truth-level data. If False, get reco-level data.
    start : int, optional
        Starting event index. If None, start at the beginning of the array.
    stop : int, optional
        Stopping event index. If None, stop at the end of the array.
    min_track_pt : float, optional
        Minimum pT threshold (in GeV) applied to tracks only. Tracks below this
        threshold have their kinematics zeroed out; muons are unaffected.
        If None, no cut is applied (default: None).

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
        get_truth=get_truth,
        get_truth_pdgids=get_truth,
        start=start,
        stop=stop,
    )

    # Extract pT, eta, phi from kinematics array
    # kinematics shape is (n_events, 3, n_particles) where axis 1 is [pT, eta, phi]
    pt = kinematics[:, 0, :]  # Extract pT
    eta = kinematics[:, 1, :]  # Extract eta
    phi = kinematics[:, 2, :]  # Extract phi

    # Get masses from pdgids
    masses = get_masses(pdgids)[:, 0, :]

    if min_track_pt is not None:
        # Particles 0 and 1 are muons; indices 2+ are tracks.
        # Build a per-particle keep mask: always True for muons, pT-filtered for tracks.
        muon_mask = ak.ones_like(pt[:, :2], dtype=bool)
        track_keep = pt[:, 2:] >= min_track_pt
        keep_mask = ak.concatenate([muon_mask, track_keep], axis=1)
        pt = ak.where(keep_mask, pt, 0.0)
        eta = ak.where(keep_mask, eta, 0.0)
        phi = ak.where(keep_mask, phi, 0.0)
        masses = ak.where(keep_mask, masses, 0.0)

    return pt, eta, phi, masses


def get_kinematics(
    tree,
    get_truth=False,
    get_truth_pdgids=False,
    start=None,
    stop=None,
):
    """get_kinematics - This function will accept an uproot TTree object, and return the
    muon and track kinematics concatenated as a single awkward array.

    The function will also return a set of indeces which describe which AK4 track jet
    in the event a given track corresponds to, and pdgids for the particles.
    For reco level data, the pdgids will be either 13 (muon) or 211 (charged pion).
    For truth level data, the they can have the pdgid for common charged hadrons.
    Note the absolute value of the pdgids is used.

    Arguments:
    tree - uproot TTree object
    get_truth - If true, get the truth level data instead of reco, optional
    get_truth_pdgids - If true, get the truth level pdgids instead of fixing all tracks
        to 211 (charged pion), optional
    start - starting event index, optional
    stop - stopping event index, optional

    Returns:
    (ak.Array) - awkward array of the concatenated kinematics
    (ak.Array) - awkward array of the pdgids
    """

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

    # Concatenate muon and track information
    kinematics = ak.concatenate([kinematics, track_kinematics], axis=2)
    pdgids = ak.concatenate([pdgids, track_pdgids], axis=2)

    # Return kinematics and track indeces
    return kinematics, pdgids


def get_hadron_kinematics(
    tree,
    get_truth=False,
    start=None,
    stop=None,
):
    """get_kinematics - This function will accept an uproot TTree object, and return the
    track kinematics concatenated as a single awkward array.

    The function will also return a set of indeces which describe which AK4 track jet
    in the event a given track corresponds to, and pdgids for the particles.
    For reco level datam, 211 (charged pion).
    For truth level data, the they can have the pdgid for common charged hadrons.
    Note the absolute value of the pdgids is used.

    Arguments:
    tree - uproot TTree object
    get_truth - If true, get the truth level data instead of reco, optional
    get_truth_pdgids - If true, get the truth level pdgids instead of fixing all tracks
        to 211 (charged pion), optional
    start - starting event index, optional
    stop - stopping event index, optional

    Returns:
    (ak.Array) - awkward array of the concatenated kinematics
    (ak.Array) - awkward array of the pdgids
    """

    # Set prekey based on get_truth
    prekey = ""
    if get_truth:
        prekey = "truth_"

    counts = tree["n" + prekey + "pT_tracks"].array(entry_start=start, entry_stop=stop)
    # Track information
    track_pt = tree[prekey + "pT_tracks"].array(entry_start=start, entry_stop=stop)
    track_eta = tree[prekey + "eta_tracks"].array(entry_start=start, entry_stop=stop)
    track_phi = tree[prekey + "phi_tracks"].array(entry_start=start, entry_stop=stop)
    # Track pdgids
    track_pdgids = 211 * ak.ones_like(track_pt)
    if get_truth:
        assert get_truth, "Cannot get truth level pdgids without truth level data"
        track_pdgids = tree[prekey + "pdgId_tracks"].array(
            entry_start=start, entry_stop=stop
        )
    masses = get_masses(track_pdgids)

    # Return kinematics and track indeces
    return track_pt, track_eta, track_phi, masses


def get_muon_kinematics(
    tree,
    get_truth=False,
    start=None,
    stop=None,
):
    """get_muon_kinematics - Extracts the kinematics of the two muons in each event.

    Arguments:
    tree - uproot TTree object
    get_truth - If True, get the truth-level muon data instead of reco, optional
    start - starting event index, optional
    stop - stopping event index, optional

    Returns:
    (ak.Array) - awkward array of shape (n_events, 2, 4) with [pt, eta, phi, mass] for each muon
    """

    # Set prekey based on get_truth
    prekey = "truth_" if get_truth else ""

    # Get muon kinematics
    m1_pt = ak.unflatten(
        tree[prekey + "pT_l1"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )
    m1_eta = ak.unflatten(
        tree[prekey + "eta_l1"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )
    m1_phi = ak.unflatten(
        tree[prekey + "phi_l1"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )

    m2_pt = ak.unflatten(
        tree[prekey + "pT_l2"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )
    m2_eta = ak.unflatten(
        tree[prekey + "eta_l2"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )
    m2_phi = ak.unflatten(
        tree[prekey + "phi_l2"].array(entry_start=start, entry_stop=stop), 1, axis=0
    )

    # Combine pt, eta, phi for each muon
    masses = ak.ones_like(m1_pt) * 0.105658

    m1_kinematics = ak.unflatten(
        ak.concatenate([m1_pt, m1_eta, m1_phi, masses], axis=1), 1, axis=1
    )
    m2_kinematics = ak.unflatten(
        ak.concatenate([m2_pt, m2_eta, m2_phi, masses], axis=1), 1, axis=1
    )

    # Concatenate the two muons
    kinematics = ak.concatenate([m1_kinematics, m2_kinematics], axis=2)

    return kinematics


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
        masses = ak.where(pdgids == 3222, 1.18937, masses)
        masses = ak.where(pdgids == 3112, 1.19745, masses)
        masses = ak.where(pdgids == 3312, 1.32171, masses)
        masses = ak.where(pdgids == 3334, 1.67245, masses)
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
        masses[pdgids == 3222] = 1.18937
        masses[pdgids == 3112] = 1.19745
        masses[pdgids == 3312] = 1.32171
        masses[pdgids == 3334] = 1.67245

    return masses


def build_jets(
    tree,
    pass190_flags,
    algorithm,
    R,
    pt,
    eta,
    phi,
    masses,
    ptmin=None,
    ptmax=None,
    etamax=None,
    max_jets=None,
    get_truth=True,
    n_jobs=-1,
    random_seed: int | None = None,
):
    """
    Cluster jets and return track-to-jet assignment indices for each event.

    Each track receives the index of the jet it belongs to, where jets are
    ordered by descending transverse momentum (pT). Tracks that do not belong
    to any jet above the ptmin threshold are assigned a value of -1.

    Arguments:
    ----------
    tree : uproot.TTree or list[uproot.TTree]
        The TTree object(s) containing event data.
    pass190_flags : np.ndarray
        Boolean array for event filtering (currently unused).
    algorithm : fastjet.JetAlgorithm
        Jet clustering algorithm (e.g., fj.antikt_algorithm).
    R : float
        Jet clustering radius parameter.
    ptmin : float, optional
        Minimum jet transverse momentum in GeV. Default is 0.5.
    ptmax : float, optional
        Maximum jet transverse momentum. (currently unused)
    etamax : float, optional
        Maximum |rapidity| for jets. (currently unused)
    max_jets : int, optional
        Maximum number of jets to keep per event. (currently unused)
    get_truth : bool, optional
        If True, use truth-level particles; otherwise reco-level.
    n_jobs : int, optional
        Number of parallel jobs for jet clustering.
        -1 uses all available CPUs.
    random_seed : int | None, optional
        Random seed used when limiting jets. (currently unused)

    Returns:
    --------
    jets : ak.Array of arrays
        Awkward array of shape [n_events, n_jets, 4]. 4 corresponds to (pt, y, phi, m) for each jet.
    event_jet_indices : ak.Array
        Awkward array of shape [n_events, n_tracks].

        Each element contains the jet index assigned to the track:

        0  -> track belongs to the highest-pT jet
        1  -> track belongs to the second highest-pT jet
        2  -> track belongs to the third highest-pT jet
        -1 -> track does not belong to any jet passing ptmin

    Notes:
    ------
    - Tracks are clustered into jets using FastJet.
    - Jets are ordered by descending pT within each event.
    - The returned structure mirrors the event → track structure
      of the input particle arrays.
    """

    if ptmin is None:
        ptmin = 0.5

    print(
        "Sample kinematics for first event: "
        f"pt {[f'{x:.6f}' for x in pt[0][:5]]}, "
        f"eta {[f'{x:.6f}' for x in eta[0][:5]]}, "
        f"phi {[f'{x:.6f}' for x in phi[0][:5]]}, "
        f"masses {[f'{x:.6f}' for x in masses[0][:5]]}"
    )

    # Cluster jets
    cluster_n_jobs = None if n_jobs == -1 else n_jobs

    clusterer = jet_clusterer.JetClusterer(pt, eta, phi, masses)

    jets, event_jet_indices = clusterer.cluster_tracks_to_jets(
        algorithm=algorithm,
        R=R,
        ptmin=ptmin,
        n_jobs=cluster_n_jobs,
    )

    return ak.Array(jets), ak.Array(event_jet_indices)


def build_jets_from_indices(
    event_jet_indices,
    tree,
    pass190_flags,
    algorithm,
    R,
    ptmin=None,
    ptmax=None,
    etamax=None,
    max_jets=None,
    get_truth=True,
    n_jobs=-1,
    nEvents=-1,  # (-1 = all)
    random_seed: int | None = None,
):
    """
    Build jet kinematics from precomputed track-to-jet indices.

    Jets are reconstructed by summing the four-momenta of tracks assigned
    to each jet. Jets are returned as (pt, y, phi, m) and sorted by
    descending pt.

    Returns
    -------
    jets : ak.Array
        Shape: [n_events, n_jets, 4]
        Jet kinematics: [pt, y, phi, m]
    """

    # # -----------------------------------------
    # # Get track kinematics
    # # -----------------------------------------
    if isinstance(tree, list):
        all_kinematics = get_hadron_kinematics(tree, get_truth=get_truth)

        pt = ak.concatenate([k[0] for k in all_kinematics], axis=0)
        eta = ak.concatenate([k[1] for k in all_kinematics], axis=0)
        phi = ak.concatenate([k[2] for k in all_kinematics], axis=0)
        masses = ak.concatenate([k[3] for k in all_kinematics], axis=0)
    else:
        pt, eta, phi, masses = get_hadron_kinematics(tree, get_truth=get_truth)

    # # -----------------------------------------
    # # Get jet assignments
    # # -----------------------------------------
    # event_jet_indices = get_jet_indices(
    #     tree=tree,
    #     pass190_flags=pass190_flags,
    #     algorithm=algorithm,
    #     R=R,
    #     pt=pt,
    #     eta=eta,
    #     phi=phi,
    #     masses=masses,
    #     ptmin=ptmin,
    #     ptmax=ptmax,
    #     etamax=etamax,
    #     max_jets=max_jets,
    #     get_truth=get_truth,
    #     n_jobs=n_jobs,
    #     random_seed=random_seed,
    # )

    # -----------------------------------------
    # Build jets
    # -----------------------------------------
    all_event_jets = []
    n_events = (
        len(event_jet_indices)
        if nEvents == -1
        else min(nEvents, len(event_jet_indices))
    )

    progress_interval = max(1, n_events // 10)

    for event_idx, indices in enumerate(event_jet_indices):
        if event_idx >= n_events:
            break
        # Print progress
        if (event_idx + 1) % progress_interval == 0 or event_idx == n_events - 1:
            print(
                f"Processing event {event_idx + 1}/{n_events} "
                f"({100*(event_idx+1)/n_events:.0f}%)"
            )

        # -----------------------------------------
        # Extract per-event track kinematics
        # -----------------------------------------
        pt_event = ak.to_numpy(pt[event_idx])
        eta_event = ak.to_numpy(eta[event_idx])
        phi_event = ak.to_numpy(phi[event_idx])
        mass_event = ak.to_numpy(masses[event_idx])

        # Convert tracks → 4-vectors for this event only
        E = np.sqrt(pt_event**2 * np.cosh(eta_event) ** 2 + mass_event**2)
        pz = pt_event * np.sinh(eta_event)
        y = 0.5 * np.log((E + pz) / (E - pz))
        event_p4s = ef.p4s_from_ptyphims(
            np.stack([pt_event, y, phi_event, mass_event], axis=-1)
        )
        indices = np.asarray(indices)
        event_jets = []

        # Find jets in this event
        jet_ids = np.unique(indices)
        jet_ids = jet_ids[jet_ids >= 0]  # ignore tracks not in any jet

        for jet_id in jet_ids:

            mask = indices == jet_id
            jet_p4 = np.sum(event_p4s[mask], axis=0)

            jet_pt, jet_y, jet_phi, jet_m = ef.ptyphims_from_p4s(
                jet_p4, phi_ref=0, mass=True
            )

            # Apply optional cuts
            if (ptmax is not None and jet_pt >= ptmax) or (
                etamax is not None and abs(jet_y) >= etamax
            ):
                continue

            event_jets.append([jet_pt, jet_y, jet_phi, jet_m])

        # Sort by descending pT
        event_jets.sort(key=lambda x: x[0], reverse=True)

        # Limit max jets if requested
        if max_jets is not None and len(event_jets) > max_jets:
            if random_seed is not None:
                np.random.seed(random_seed)
            event_jets = event_jets[:max_jets]

        all_event_jets.append(event_jets)

    jets = ak.Array(all_event_jets)

    return jets
