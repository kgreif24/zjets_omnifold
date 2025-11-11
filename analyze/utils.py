"""
Utility functions for jet analysis calculations.
"""

import numpy as np
import awkward as ak
import vector
import energyflow as ef
import jet_clusterer
import multiprocessing

import sys

sys.path.append("../utils")
import data_utils as du  # noqa: E402


def calculate_jet_4vectors(jets: list[np.ndarray]) -> list:
    """Calculate jet four-vectors from a list of jet constituents.

    Arguments:
    jets - A list of numpy arrays of jet constituents with the form:
        (n_constituents, 4) where the columns are (E, px, py, pz).

    Returns:
    jet_4vectors - A list of vector.obj objects representing the jet 4-vectors.
    """
    jet_4vectors = []
    for jet in jets:
        summed = jet.sum(axis=0)
        jet_vec = vector.obj(E=summed[0], px=summed[1], py=summed[2], pz=summed[3])
        jet_4vectors.append(jet_vec)
    return jet_4vectors


def preprocess_jets(jets: list[np.ndarray], drop_mass: bool = True) -> list[np.ndarray]:
    """Apply pre-processing to jets: convert to pT-y-phi, center, and rotate.

    Arguments:
    jets - A list of numpy arrays of jet constituents with the form:
        (n_constituents, 4) where the columns are (E, px, py, pz).
    drop_mass - If True, drop the mass column from the output (default: True).
        If False, keep pT, y, phi, and mass columns.

    Returns:
    preprocessed_jets - A list of numpy arrays of preprocessed jets in pT-y-phi format
        (or pT-y-phi-mass if drop_mass=False), centered and rotated.
    """
    preprocessed_jets = []
    for jet in jets:
        # Convert to pT, y, phi, mass
        ptyphims = ef.ptyphims_from_p4s(jet, phi_ref=0, mass=True)
        # Optionally drop the mass
        if drop_mass:
            ptyphi = ptyphims[:, :3]
        else:
            ptyphi = ptyphims
        # Center and rotate the jets
        centered = ef.center_ptyphims(ptyphi, copy=True, center="escheme")
        centered[:, 2] = (centered[:, 2] + np.pi) % (2 * np.pi) - np.pi
        rotated = ef.rotate_ptyphims(centered, copy=True, rotate="ptscheme")
        preprocessed_jets.append(rotated)
    return preprocessed_jets


def calculate_emds(
    jets: list[np.ndarray],
    jet_4vectors: list[vector.obj],
    R=1.0,
    n_jobs=1,
) -> np.ndarray:
    """Calculate Earth Mover's Distances (EMDs) between jets.

    Arguments:
    jets - A list of numpy arrays of jet constituents with the form:
        (n_constituents, 3) where the columns are (pT, eta, phi).
    jet_4vectors - A list of vector.obj objects representing the jet 4-vectors.
    R - The R parameter for the EMD calculation, this is typically set to the
        jet clustering radius parameter.
    n_jobs - The number of jobs to use for the calculation.
        If n_jobs > 1, the calculation will be parallelized.

    Returns:
    emds - A numpy array of EMD values with shape (n_jets, n_jets).
    """
    # Compute the mean pT of the jets
    jet_pts = np.array([vec.pt for vec in jet_4vectors])
    mean_pt = np.mean(jet_pts)
    print(f"Got mean pT of {mean_pt} GeV")
    print(f"Got min pT of {np.min(jet_pts)} GeV")
    print(f"Got max pT of {np.max(jet_pts)} GeV")

    # Calculate the EMDs using energyflow
    emds = mean_pt * ef.emd.emds(
        jets,
        R=R,
        norm=True,
        gdim=2,
        mask=False,
        verbose=1,
        print_every=5 * 10**7,
        n_jobs=n_jobs,
    )
    print(f"Got EMDs with shape of {emds.shape}")

    return emds


def calculate_emds_from_file(
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
    save_path=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Earth Mover's Distances (EMDs) from a ROOT file TTree.

    This function performs the complete analysis pipeline:
    1. Extracts kinematics from the TTree
    2. Clusters jets using the specified algorithm
    3. Filters jets based on pT and eta cuts
    4. Preprocesses jets (centers and rotates)
    5. Calculates EMDs between all jet pairs

    Arguments:
    ----------
    tree : uproot.TTree
        The TTree object from uproot containing event data.
    pass190_flags : np.ndarray
        Boolean array of pass190 flags for event filtering.
        If all True (data case), truth data will not be used.
    algorithm : fastjet.JetAlgorithm
        Jet clustering algorithm (e.g., fj.antikt_algorithm).
    R : float
        Jet clustering radius parameter.
    ptmin : float, optional
        Minimum pT threshold for clustered jets in GeV (default: None, uses 500.0).
        Also used as a filter after clustering if ptmax is specified.
    ptmax : float, optional
        Maximum pT threshold for filtering jets after clustering
        (default: None, no max).
    etamax : float, optional
        Maximum |eta| threshold for filtering jets after clustering
        (default: None, no max).
    max_jets : int, optional
        Maximum number of jets to keep after filtering. If specified,
        jets beyond this limit will be dropped (default: None, no limit).
    get_truth : bool, optional
        If True, get truth level data. If False, get reco level data.
    n_jobs : int, optional
        Number of parallel jobs for jet clustering and EMD calculation.
        If -1, uses all available CPUs (default: -1).
    save_path : str, optional
        Optional path to save the EMDs and event indices to a .npz file.
        If provided, the EMDs and event indices will be saved with keys 'emds'
        and 'event_indices'. If None, the data will not be saved (default: None).

    Returns:
    --------
    emds : np.ndarray
        A numpy array of EMD values with shape (n_jets, n_jets).
    event_indices : np.ndarray
        A numpy array of integer event indices with shape (n_jets,),
        indicating which event each jet originated from.
    """

    # Set default ptmin if not provided
    if ptmin is None:
        ptmin = 500.0

    # Get kinematics from the tree
    print(f"Getting kinematics from tree (get_truth={get_truth})")
    if isinstance(tree, list):
        kinematics, _, pdgids = du.get_kinematics_multiple(
            tree,
            pass190_flags,
            get_truth=get_truth,
            get_truth_pdgids=get_truth,
            take_log_pt=False,
        )
    else:
        kinematics, _, pdgids = du.get_kinematics(
            tree,
            pass190_flags,
            get_truth=get_truth,
            get_truth_pdgids=get_truth,
            take_log_pt=False,
            stop=len(pass190_flags),
        )

    # Extract pT, eta, phi from kinematics array
    # kinematics shape is (n_events, 3, n_particles) where axis 1 is [pT, eta, phi]
    pt = kinematics[:, 0, :]  # Extract pT
    eta = kinematics[:, 1, :]  # Extract eta
    phi = kinematics[:, 2, :]  # Extract phi

    # Get masses from pdgids
    masses = du.get_masses(pdgids)[:, 0, :]

    print(f"Extracted kinematics: {len(pt)} events")
    print(f"pT shape: {ak.type(pt)}")
    print(f"eta shape: {ak.type(eta)}")
    print(f"phi shape: {ak.type(phi)}")
    print(f"masses shape: {ak.type(masses)}")

    # Cluster jets
    # Convert n_jobs=-1 to None for clusterer (which uses all CPUs)
    cluster_n_jobs = None if n_jobs == -1 else n_jobs
    clusterer = jet_clusterer.JetClusterer(pt, eta, phi, masses)
    event_jet_constituents = clusterer.cluster_events(
        algorithm=algorithm, R=R, ptmin=ptmin, n_jobs=cluster_n_jobs
    )

    # Flatten jets across all events and track event indices
    flat_jets = []
    flat_event_indices = []
    for event_idx, event_jets in enumerate(event_jet_constituents):
        for jet in event_jets:
            flat_jets.append(jet)
            flat_event_indices.append(event_idx)

    print(f"Clustered {len(flat_jets)} jets from {len(event_jet_constituents)} events")

    # Calculate jet 4-vectors for filtering
    jet_4vectors = calculate_jet_4vectors(flat_jets)

    # Filter jets based on pT and eta cuts
    # Note: ptmin is already applied during clustering,
    # so we only need to check ptmax here
    filter_mask = []
    for jet_vec in jet_4vectors:
        keep = True
        if ptmax is not None and jet_vec.pt >= ptmax:
            keep = False
        if etamax is not None and abs(jet_vec.eta) >= etamax:
            keep = False
        filter_mask.append(keep)

    filtered_jets = [jet for jet, keep in zip(flat_jets, filter_mask) if keep]
    filtered_jet_4vectors = [
        jet_vec for jet_vec, keep in zip(jet_4vectors, filter_mask) if keep
    ]
    filtered_event_indices = np.array(
        [idx for idx, keep in zip(flat_event_indices, filter_mask) if keep]
    )

    print(
        f"Filtered to {len(filtered_jets)} jets "
        f"(ptmin={ptmin}, ptmax={ptmax}, etamax={etamax})"
    )

    # Limit to max_jets if specified (randomly sample to avoid bias)
    if max_jets is not None and len(filtered_jets) > max_jets:
        n_dropped = len(filtered_jets) - max_jets
        # Randomly sample indices
        random_indices = np.random.choice(
            len(filtered_jets), size=max_jets, replace=False
        )
        random_indices = np.sort(random_indices)  # Sort to maintain order
        # Apply random sampling
        filtered_jets = [filtered_jets[i] for i in random_indices]
        filtered_jet_4vectors = [filtered_jet_4vectors[i] for i in random_indices]
        filtered_event_indices = filtered_event_indices[random_indices]
        print(f"Randomly sampled {max_jets} jets (dropped {n_dropped} jets)")

    # Preprocess the filtered jets
    # (drop_mass=False to keep mass for EMD calculation)
    # Note: preprocessing doesn't change which event each jet belongs to
    preprocessed_jets = preprocess_jets(filtered_jets, drop_mass=False)

    # Calculate EMDs
    # Note: calculate_emds uses n_jobs directly with energyflow
    # which accepts -1 for all CPUs
    print("Calculating EMDs...")
    emds = calculate_emds(
        preprocessed_jets,
        filtered_jet_4vectors,
        R=R,
        n_jobs=n_jobs,
    )

    # Drop lower triangle of emds since the matrix is symmetric
    emds = np.triu(emds)

    # Save to file if path is provided
    if save_path is not None:
        np.savez(save_path, emds=emds, event_indices=filtered_event_indices)
        print(f"Saved EMDs and event indices to {save_path}")

    return emds, filtered_event_indices


def calculate_correlation_dimension_from_emds(
    emds: np.ndarray,
    emd_weights: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate correlation dimension from Earth Mover's Distances (EMDs).

    Arguments:
    emds - A numpy array of EMD values (typically from calculate_emds).
    emd_weights - A numpy array of weights corresponding to each EMD value.
    bins - A numpy array of bins to use in the histogram.

    Returns:
    dims - The correlation dimension values.
    dims_var - The variance of the correlation dimension values.
    midbins - The midpoints of the bins used in the calculation.
    """

    # Take upper triangle and mask invalid emds
    emds = np.triu(emds)
    mask = emds > 0 & np.isfinite(emds)
    emds = emds[mask].flatten()
    emd_weights = emd_weights[mask].flatten()

    # Calculate the correlation dimension
    midbins = (bins[1:] + bins[:-1]) / 2
    dmidbins = np.log(midbins[1:]) - np.log(midbins[:-1])
    hist, _ = np.histogram(emds, bins=bins, weights=emd_weights)
    var, _ = np.histogram(emds, bins=bins, weights=emd_weights**2)

    # Calculate the CDF
    counts = np.cumsum(hist) + np.finfo(float).eps
    counts_err = np.sqrt(np.cumsum(var))

    # Calculate the correlation dimension
    dims = (np.log(counts[1:]) - np.log(counts[:-1])) / dmidbins
    dims_var = (
        (counts_err[1:] / counts[1:]) ** 2 + (counts_err[:-1] / counts[:-1]) ** 2
    ) / dmidbins

    return dims, dims_var, midbins


def _process_weight_set(args):
    """Helper function to process a single weight set for parallel execution.

    Arguments:
    ----------
    args : tuple
        Tuple containing (hist_name, weights, emds, event_indices, bins)

    Returns:
    --------
    tuple : (hist_name, dims, dims_var, midbins)
    """
    hist_name, weights, emds, event_indices, bins = args

    # Ensure event_indices is a 1D array
    event_indices = np.asarray(event_indices).flatten()

    weights = np.asarray(weights)

    # Get the weights of the jets (fine if some events have multiple jets)
    jet_weights = weights[event_indices]

    # Create EMD weights matrix: weight for EMD(i,j) = weight(i) * weight(j)
    # This represents the product of weights for the two events
    emd_weights = jet_weights[:, None] * jet_weights[None, :]

    print(f"Calculating correlation dimension for '{hist_name}'...")
    dims, dims_var, midbins = calculate_correlation_dimension_from_emds(
        emds, emd_weights, bins
    )

    return hist_name, dims, dims_var, midbins


def calculate_correlation_dimension_from_file(
    npz_path: str,
    weights_dict: dict[str, np.ndarray],
    bins: np.ndarray,
    n_jobs=1,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculate correlation dimension from EMDs stored in a .npz file.

    This function loads EMDs and event indices from a .npz file (produced by
    calculate_emds_from_file), broadcasts event-level weights to EMD pairs,
    and calculates the correlation dimension for each set of weights.

    Since there are huge numbers of EMDs for a reasonable number of jets,
    binning the EMDs will be done in parallel when possible.

    Arguments:
    ----------
    npz_path : str
        Path to the .npz file containing 'emds' and 'event_indices' arrays.
    weights_dict : dict[str, np.ndarray]
        Dictionary mapping histogram names to event-level weights.
        Each weights array should have shape (n_events,) where n_events is
        the number of unique events in the event_indices array.
    bins : np.ndarray
        Array of bin edges to use for the histogram in the correlation
        dimension calculation.
    n_jobs : int, optional
        Number of parallel jobs for the calculation.
        If -1, parallelize to the number of histograms in weights_dict
         or the number of CPUs available. Whichever is smaller.

    Returns:
    --------
    results : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Dictionary mapping histogram names to tuples of (dims, dims_var, midbins)
        where:
        - dims: The correlation dimension values
        - dims_var: The variance of the correlation dimension values
        - midbins: The midpoints of the bins used in the calculation
    """

    # Load EMDs and event indices once from disk
    print(f"Loading EMDs from {npz_path}...")
    data = np.load(npz_path)
    emds = data["emds"]
    event_indices = data["event_indices"]
    data.close()

    print(f"Loaded EMDs with shape {emds.shape}")
    print(f"Loaded event indices with shape {event_indices.shape}")

    # Ensure event_indices is a 1D array
    event_indices = np.asarray(event_indices).flatten()

    # Check that EMDs shape matches event_indices
    n_jets = len(event_indices)
    if emds.shape != (n_jets, n_jets):
        raise ValueError(
            f"EMD shape {emds.shape} does not match expected shape "
            f"({n_jets}, {n_jets}) based on event_indices length"
        )

    # Determine number of parallel jobs
    n_histograms = len(weights_dict)
    if n_jobs == -1:
        # Use all available CPUs or number of histograms, whichever is smaller
        n_workers = min(multiprocessing.cpu_count(), n_histograms)
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, n_histograms)

    results = {}

    # Prepare arguments for parallel processing
    # Pass arrays directly (will be pickled and sent to worker processes)
    args_list = [
        (hist_name, weights, emds, event_indices, bins)
        for hist_name, weights in weights_dict.items()
    ]

    # Process each set of weights in parallel
    if n_workers > 1:
        print(
            f"Processing {n_histograms} weight sets with "
            f"{n_workers} parallel workers..."
        )
        with multiprocessing.Pool(processes=n_workers) as pool:
            processed_results = pool.map(_process_weight_set, args_list)

        # Collect results into dictionary
        for hist_name, dims, dims_var, midbins in processed_results:
            results[hist_name] = (dims, dims_var, midbins)
    else:
        # Sequential processing (n_jobs=1 or only one histogram)
        for args in args_list:
            hist_name, dims, dims_var, midbins = _process_weight_set(args)
            results[hist_name] = (dims, dims_var, midbins)

    return results
