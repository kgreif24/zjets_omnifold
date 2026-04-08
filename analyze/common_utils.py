"""
Common utility functions for use in analyzing the Z+jets Omnifold data.
"""

from __future__ import annotations

import numpy as np
import awkward as ak
import jet_clusterer
import psutil
import sys
import re
import ast
import operator
import time
import numba as nb
import json


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


def extract_hadron_kinematics(
    tree,
    selection=None,
    get_truth: bool = True,
    start: int = None,
    stop: int = None,
    min_track_pt: float | None = None,
):
    """Extract hadron (track-only) kinematics and masses from a ROOT TTree.

    Arguments:
    ----------
    tree : uproot.TTree
        The TTree object from uproot containing event data.
    selection : array-like, optional
        Event-level selection mask.
    get_truth : bool, optional
        If True, get truth-level data. If False, get reco-level data.
    start : int, optional
        Starting event index.
    stop : int, optional
        Stopping event index.
    min_track_pt : float, optional
        Minimum pT threshold (in GeV). Tracks below this are zeroed.

    Returns:
    --------
    pt : ak.Array
    eta : ak.Array
    phi : ak.Array
    masses : ak.Array
    """

    pt, eta, phi, masses = get_hadron_kinematics(
        tree,
        selection=selection,
        get_truth=get_truth,
        start=start,
        stop=stop,
    )

    # Apply pT cut if requested
    if min_track_pt is not None:
        keep_mask = pt >= min_track_pt

        pt = ak.where(keep_mask, pt, 0.0)
        eta = ak.where(keep_mask, eta, 0.0)
        phi = ak.where(keep_mask, phi, 0.0)
        masses = ak.where(keep_mask, masses, 0.0)

    return pt, eta, phi, masses


def get_hadron_kinematics(
    tree,
    selection=None,
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

    if selection is not None:
        track_pt = track_pt[selection]
        track_eta = track_eta[selection]
        track_phi = track_phi[selection]
        masses = masses[selection]
    # Return kinematics and track indeces
    return track_pt, track_eta, track_phi, masses


def get_muon_kinematics(
    tree,
    selection=None,
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
    )[selection]
    m2_kinematics = ak.unflatten(
        ak.concatenate([m2_pt, m2_eta, m2_phi, masses], axis=1), 1, axis=1
    )[selection]

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


def eval_jet_expressions(jets, expressions, nevents=-1):
    """eval_jet_expressions - Evaluates multiple arithmetic expressions involving jet info fields,
    propagating -99 for missing jets, with progress output every 10%.

    Arguments:
    jets - list/array of per-event jet info arrays, shape (n_events, n_jets, 4)
    expressions - list of str, arithmetic expressions using tokens like INFO_trackj#
    nevents - int, maximum number of events to process (-1 for all)

    Returns:
    (dict) - mapping {expression: array of evaluated values per event}, shape (nevents,)
    """
    import re
    import ast
    import operator
    import numpy as np

    print(
        "Processing expressions:"
        + ", ".join(expressions)
        + f" for {nevents if nevents > 0 else len(jets)} events"
    )
    field_map = {"pT": 0, "y": 1, "phi": 2, "m": 3}
    nevents = len(jets) if nevents < 0 else min(len(jets), nevents)

    # Gather all unique jet tokens across all expressions
    token_pattern = r"\b(\w+_trackj\d+)\b"
    tokens = set()
    for expr in expressions:
        tokens.update(re.findall(token_pattern, expr))

    # Build token arrays in a single loop
    print(f"Extracting jet info for tokens: {', '.join(tokens)}")
    token_arrays = {tok: np.full(nevents, -99.0, dtype=float) for tok in tokens}
    progress_step = max(nevents // 10, 1)

    for i, event in enumerate(jets[:nevents]):
        for tok in tokens:
            match = re.match(r"(\w+)_trackj(\d+)", tok)
            field_name, jet_num = match.groups()
            idx = field_map[field_name]
            jet_idx = int(jet_num) - 1
            if len(event) > jet_idx:
                token_arrays[tok][i] = event[jet_idx][idx]
        if (i + 1) % progress_step == 0 or i == nevents - 1:
            print(
                f"Processed {i+1}/{nevents} events ({100*(i+1)/nevents:.0f}%)", end="\r"
            )
    print("\nFinished extracting jet info for all tokens.")

    # Allowed operators
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_vectorized(node):
        """Recursively evaluate AST node for all events at once"""
        if isinstance(node, ast.Expression):
            return _eval_vectorized(node.body)
        elif isinstance(node, ast.BinOp):
            left, left_mask = _eval_vectorized(node.left)
            right, right_mask = _eval_vectorized(node.right)
            combined_mask = left_mask | right_mask
            result = np.where(
                combined_mask, -99.0, allowed_ops[type(node.op)](left, right)
            )
            return result, combined_mask
        elif isinstance(node, ast.UnaryOp):
            val, mask = _eval_vectorized(node.operand)
            result = np.where(mask, -99.0, allowed_ops[type(node.op)](val))
            return result, mask
        elif isinstance(node, ast.Name):
            if node.id not in token_arrays:
                raise ValueError(f"Unknown token: {node.id}")
            arr = token_arrays[node.id]
            mask = arr == -99
            return arr, mask
        elif isinstance(node, ast.Constant):
            return np.full(nevents, node.value), np.zeros(nevents, dtype=bool)
        elif hasattr(ast, "Num") and isinstance(node, ast.Num):  # for Python <3.8
            return np.full(nevents, node.n), np.zeros(nevents, dtype=bool)
        else:
            raise ValueError(f"Unsupported element: {node}")

    # Evaluate all expressions **vectorized**
    results = {}
    print(f"Evaluating {len(expressions)} expressions for {nevents} events...")
    for j, expr in enumerate(expressions):
        node = ast.parse(expr, mode="eval")
        val, _ = _eval_vectorized(node)
        results[expr] = val
        if (j + 1) % max(1, len(expressions) // 10) == 0 or j == len(expressions) - 1:
            print(
                f"Processed {j+1}/{len(expressions)} expressions ({100*(j+1)/len(expressions):.0f}%)"
            )

    print("Finished evaluating all expressions.")
    return results


def count_jets(jets, weights, individual_jet_selection, leading_jet_only=False):
    """Count jets passing a selection with optional weighting.

    Applies a per-jet selection and counts the number of jets passing it,
    either per event or for leading jets only. Returns weighted jet counts,
    supporting both single and multiple weight sets.

    Arguments:
    ----------
    jets : array-like
        Per-event jet collection with shape (n_events, n_jets, 4).
    weights : array-like
        Event weights (1D or 2D for multiple weight sets).
    individual_jet_selection : callable
        Function taking (pt, y, phi, m) and returning a boolean mask.
    leading_jet_only : bool, optional
        If True, only consider the leading jet per event (default: False).

    Returns:
    --------
    count : float or np.ndarray
        Weighted jet count. Returns a float for 1D weights, or array for
        multiple weight sets.
    """
    if leading_jet_only:
        leading = ak.firsts(jets)  # None if event has no jets

        pt = leading[:, 0]
        y = leading[:, 1]
        phi = leading[:, 2]
        m = leading[:, 3]

        mask = individual_jet_selection(pt, y, phi, m)

        # None → False
        njets_pass = ak.fill_none(mask, False)

    else:
        pt = jets[:, :, 0]
        y = jets[:, :, 1]
        phi = jets[:, :, 2]
        m = jets[:, :, 3]

        mask = individual_jet_selection(pt, y, phi, m)

        # count jets per event
        njets_pass = ak.sum(mask, axis=1)

    weights = np.asarray(weights)

    if weights.ndim == 1:
        return float(ak.sum(njets_pass * weights))

    return np.array([float(ak.sum(njets_pass * w)) for w in weights])


def count_jets_vectorized(
    jets, weights, individual_jet_selection, leading_jet_only=False
):
    # --- Compute njets_pass (already vectorized via awkward) ---
    if leading_jet_only:
        leading = ak.firsts(jets)

        pt = leading[:, 0]
        y = leading[:, 1]
        phi = leading[:, 2]
        m = leading[:, 3]

        mask = individual_jet_selection(pt, y, phi, m)
        njets_pass = ak.fill_none(mask, False)

    else:
        pt = jets[:, :, 0]
        y = jets[:, :, 1]
        phi = jets[:, :, 2]
        m = jets[:, :, 3]

        mask = individual_jet_selection(pt, y, phi, m)
        njets_pass = ak.sum(mask, axis=1)

    # convert once
    njets_pass = ak.to_numpy(njets_pass)
    weights = np.asarray(weights)

    # --- 1D weights ---
    if weights.ndim == 1:
        weighted_sum = np.sum(njets_pass * weights)
        weighted_var = np.sum(njets_pass * weights**2)
        return weighted_sum, weighted_var

    # --- 2D weights: (n_weights, n_events) ---
    # Broadcast njets_pass to (n_weights, n_events)
    weighted_sum = np.sum(weights * njets_pass, axis=1)
    weighted_var = np.sum(weights**2 * njets_pass, axis=1)
    return weighted_sum, weighted_var


def make_jet_count_histograms(
    jets_list,
    weights_dict,
    individual_jet_selection,
    bin_edges,
    observable_name="jet_count",
    leading_jet_only=False,
):
    """
    Build histogram of jet counts across bins using count_jets.

    Parameters
    ----------
    jets_list : list
        List of jet collections, one per bin.
    weights_dict : dict
        Same structure as before (nominal, systematics, bootstrap, etc.).
    individual_jet_selection : callable
        Per-jet selection function.
    bin_edges : array-like
        Histogram bin edges (len = n_bins + 1).
    observable_name : str
        Name of observable for output dict.
    leading_jet_only : bool
        Passed to count_jets.

    Returns
    -------
    hist_dict : dict
      dictionary: hist_dict[weight_name] = (hist, variance, bin_edges)
    """

    n_bins = len(jets_list)

    # --- Identify weight groups ---
    ensemble = [k for k in weights_dict if k.startswith("ensemble_")]
    bootstrap_mc = [k for k in weights_dict if k.startswith("bootstrap_mc_")]
    bootstrap_data = [k for k in weights_dict if k.startswith("bootstrap_data_")]
    special = set(["nominal"] + ensemble + bootstrap_mc + bootstrap_data)
    syst_weights = [k for k in weights_dict if k not in special]

    hist_dict = {observable_name: {}}

    # --- Nominal ---
    nominal_w = np.asarray(weights_dict["nominal"])
    hist_nom = []
    hist_var = []

    print("Calculating nominal jet counts...")
    for jets in jets_list:
        val, var = count_jets_vectorized(
            jets,
            nominal_w,
            individual_jet_selection,
            leading_jet_only,
        )
        hist_nom.append(val)
        hist_var.append(var)

    hist_nom = np.asarray(hist_nom)
    hist_var = np.asarray(hist_var)

    hist_dict[observable_name]["nominal"] = (
        hist_nom.copy(),
        hist_var.copy(),
        np.asarray(bin_edges),
    )

    # --- Systematics ---
    for name in syst_weights:
        print("Calculating jet counts for systematic:", name)
        w = np.asarray(weights_dict[name])
        w_eff = nominal_w * w if name.startswith("*") else w

        hist_val = []
        hist_var = []
        for jets in jets_list:
            val, var = count_jets_vectorized(
                jets,
                w_eff,
                individual_jet_selection,
                leading_jet_only,
            )
            hist_val.append(val)
            hist_var.append(var)

        key = name[1:] if name.startswith("*") else name
        hist_val = np.asarray(hist_val)
        hist_var = np.asarray(hist_var)

        hist_dict[observable_name][key] = (
            hist_val.copy(),
            hist_var.copy(),
            np.asarray(bin_edges),
        )

    # --- Replicas ---
    for key in ensemble + bootstrap_mc + bootstrap_data:
        print("Calculating jet counts for systematic:", key)
        w = np.asarray(weights_dict[key])

        hist_val = []
        hist_var = []
        for jets in jets_list:
            val, var = count_jets_vectorized(
                jets,
                w,
                individual_jet_selection,
                leading_jet_only,
            )
            hist_val.append(val)
            hist_var.append(var)

        hist_val = np.asarray(hist_val)
        hist_var = np.asarray(hist_var)

        hist_dict[observable_name][key] = (
            hist_val.copy(),
            hist_var.copy(),
            np.asarray(bin_edges),
        )

    return hist_dict


def make_hists_with_uncertainty(data, bins, observables, weights_dict):
    """Compute histograms with weights

    Generates nominal, systematic, and replica histograms per observable,
    returning a structure compatible with `plot_measurement_with_uncertainties`.
    All variances are set to zero for now.

    Arguments:
    ----------
    data : dict
        Dictionary of arrays per observable.
    bins : dict
        Dictionary of bin edges per observable.
    observables : list
        List of observable names to histogram.
    weights_dict : dict
        Dictionary of event weights: nominal, systematics, and replicas.

    Returns:
    --------
    hist_dict : dict
        Nested dictionary: hist_dict[observable][weight_name] = (hist, variance, bin_edges)
    """
    hist_dict = {}

    for obs in observables:
        data_obs = np.asarray(data[obs])
        bin_edges = np.asarray(bins[obs])
        n_bins = len(bin_edges) - 1

        # --- Identify weight groups ---
        nominal_w = np.asarray(weights_dict["nominal"])
        ensemble = [k for k in weights_dict if k.startswith("ensemble_")]
        bootstrap_mc = [k for k in weights_dict if k.startswith("bootstrap_mc_")]
        bootstrap_data = [k for k in weights_dict if k.startswith("bootstrap_data_")]
        special = set(["nominal"] + ensemble + bootstrap_mc + bootstrap_data)
        syst_weights = [k for k in weights_dict if k not in special]

        # --- Nominal histogram ---
        hist_nom, _ = np.histogram(data_obs, bins=bin_edges, weights=nominal_w)
        hist_nom_v, _ = np.histogram(
            data_obs, bins=bin_edges, weights=nominal_w * nominal_w
        )

        # --- Systematic histograms ---
        hist_syst = {}
        hist_syst_v = {}
        for name in syst_weights:
            w = np.asarray(weights_dict[name])
            w_eff = nominal_w * w if name.startswith("*") else w
            hist_var, _ = np.histogram(data_obs, bins=bin_edges, weights=w_eff)
            hist_var_v, _ = np.histogram(
                data_obs, bins=bin_edges, weights=w_eff * w_eff
            )
            key = name[1:] if name.startswith("*") else name
            hist_syst[key] = hist_var
            hist_syst_v[key] = hist_var_v

        # --- Replica histograms ---
        hist_replica = {}
        hist_replica_v = {}
        for key in ensemble + bootstrap_mc + bootstrap_data:
            w = np.asarray(weights_dict[key])
            hist, _ = np.histogram(data_obs, bins=bin_edges, weights=w)
            hist_v, _ = np.histogram(data_obs, bins=bin_edges, weights=w * w)
            hist_replica[key] = hist
            hist_replica_v[key] = hist_v

        # --- Build output dict for this observable ---
        hist_dict[obs] = {}
        hist_dict[obs]["nominal"] = (
            hist_nom.copy(),
            hist_nom_v.copy(),
            bin_edges,
        )
        for key, hist in hist_syst.items():
            hist_dict[obs][key] = (hist.copy(), hist_syst_v[key].copy(), bin_edges)
        for key, hist in hist_replica.items():
            hist_dict[obs][key] = (hist.copy(), hist_replica_v[key].copy(), bin_edges)

    return hist_dict


def append_ns_weights_with_theory(
    weights, t_mc_truth, t_ns_truth, selection_mc, selection_ns
):
    """
    Append NS weights to MC weights and create NS theory branches.

    Parameters
    ----------
    weights : dict
        Existing MG weight branches.
    t_mc_truth : structured array or dataframe
        MG truth tree.
    t_ns_truth : structured array or dataframe
        NS truth tree.
    selection_mc : array-like
        Boolean mask for MG events.
    selection_ns : array-like
        Boolean mask for NS events.
    """

    # --- Load NS nominal weights ---
    ns_nominal = np.array(t_ns_truth["weight_mc"])[selection_ns]

    # --- Append NS weights to MG branches ---
    for key in weights:
        if key == "nominal":
            weights[key] = np.concatenate([weights[key], ns_nominal])
        elif key.startswith("*"):
            weights[key] = np.concatenate([weights[key], np.ones_like(ns_nominal)])
        else:
            weights[key] = np.concatenate([weights[key], ns_nominal])

    # --- Define DSID/factor maps ---
    DSIDs_Diboson = [
        363356,
        363358,
        364250,
        364253,
        364254,
        364255,
        363494,
        363355,
        363357,
        363359,
        363360,
        363489,
    ]
    DSIDs_Herwig7 = [830007]

    factors = {"Diboson": 0.3, "EW_Zjj": 0.2}
    DSID_map = {"Diboson": DSIDs_Diboson, "EW_Zjj": DSIDs_Herwig7}

    # --- Function to compute theory scale ---
    def get_theory_scale(mc_channel_numbers, DSIDs, fraction, is_up):
        scale = np.ones_like(mc_channel_numbers, dtype=float)
        mask = np.isin(mc_channel_numbers, DSIDs)
        scale[mask] = 1 + fraction if is_up else 1 - fraction
        return scale

    # --- Concatenate MG + NS mcChannelNumbers ---
    mcChannelNumbers_mc = np.array(t_mc_truth["mcChannelNumber"])[selection_mc]
    mcChannelNumbers_ns = np.array(t_ns_truth["mcChannelNumber"])[selection_ns]
    mcChannelNumbers_all = np.concatenate([mcChannelNumbers_mc, mcChannelNumbers_ns])

    # --- Create NS theory branches ---
    for non_strong_smpl in ["Diboson", "EW_Zjj"]:
        DSIDs = DSID_map[non_strong_smpl]
        fraction = factors[non_strong_smpl]

        for updown in ["_Up"]:
            is_up = updown == "_Up"
            # Start with a copy of the nominal weights (MG+NS)
            weight_branch = weights["nominal"].copy()
            # Apply theory scaling to all events (MG stays 1 automatically)
            theory_scale = get_theory_scale(
                mcChannelNumbers_all, DSIDs, fraction, is_up
            )
            weight_branch *= theory_scale
            # Store in weights
            branch_name = f"weights_ns_theory_{non_strong_smpl.lower()}{updown.lower()}"
            weights[branch_name] = weight_branch

    return weights


def json_to_hist(
    file_name,
    bins,
    var,
    unfold_pT200=True,
    add_stat=True,
    only_stat=False,
):
    """
    Convert JSON blocks to a binned histogram using NumPy.

    Returns a dictionary:
      'nominal'   : bin contents
      'total_unc' : bin errors
      'bins'      : bin edges
    """
    print(f"reading json for {var}")

    xSec = []
    relErr = []
    statErr = []

    # --- Read JSON blocks ---
    with open(file_name, "r") as f:
        json_block = ""
        for line in f:
            json_block += line
            if "}" in line:
                try:
                    j = json.loads(json_block)
                    if f"{var}_xSec" in j:
                        xSec = j[f"{var}_xSec"]
                    elif f"{var}_RelErrors" in j:
                        relErr = j[f"{var}_RelErrors"]
                    elif f"{var}_xSec_AbsStatErr" in j:
                        statErr = j[f"{var}_xSec_AbsStatErr"]
                except Exception as e:
                    print(f"Error parsing JSON block: {e}")
                json_block = ""

    xSec = np.array(xSec)
    relErr = np.array(relErr)
    statErr = np.array(statErr)

    # --- Fill arrays ---
    n_bins = len(bins) - 1
    nominal = np.zeros(n_bins)
    total_unc = np.zeros(n_bins)

    start = 1 if unfold_pT200 else 0

    for i in range(n_bins):
        content = xSec[i] if (i) < len(xSec) else 0.0
        # syst = (relErr[i + start] * content) if (i + start) < len(relErr) else 0.0
        syst = (relErr[i] * content) if (i) < len(relErr) else 0.0
        stat = statErr[i + start] if (i + start) < len(statErr) else 0.0

        nominal[i] = content

        if only_stat:
            total_unc[i] = stat
        elif add_stat:
            total_unc[i] = np.sqrt(stat**2 + syst**2)
        else:
            total_unc[i] = syst

    return {"nominal": nominal, "total_unc": total_unc, "bins": np.array(bins)}


def validate_expression(result, *inputs):
    """
    Enforce -99 masking on a computed array.

    Arguments:
    result : np.ndarray
        Computed result from numpy operations

    *inputs : np.ndarray
        Input arrays used to compute the result (used to propagate -99)

    Returns:
    np.ndarray
        Cleaned array with invalid entries set to -99
    """
    import numpy as np

    result = np.asarray(result).copy()

    # --- Build mask from inputs ---
    mask = np.zeros_like(result, dtype=bool)

    for arr in inputs:
        if arr is not None:
            mask |= arr == -99

    # --- Include invalid numeric results ---
    mask |= ~np.isfinite(result)  # catches nan, inf

    # --- Apply mask ---
    result[mask] = -99.0

    return result
