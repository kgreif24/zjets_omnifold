"""
Utility functions for intrinsic dimension calculations.
"""

import scipy.optimize as opt
import numpy as np
import vector
import awkward as ak
import energyflow as ef
import jet_clusterer
import multiprocessing
import dask
import dask.array as da
from tqdm import tqdm
from numba import njit, prange

from common_utils import extract_kinematics


@njit(parallel=True, cache=True)
def parallel_argsort_rows(matrix: np.ndarray) -> np.ndarray:
    """Sort each row of a matrix in parallel using numba.

    Arguments:
    matrix - A 2D numpy array to sort along axis 1.

    Returns:
    sort_indices - A 2D array of sorting indices with the same shape as matrix.
    """
    n_rows, n_cols = matrix.shape
    result = np.empty((n_rows, n_cols), dtype=np.int64)
    for i in prange(n_rows):
        result[i] = np.argsort(matrix[i])
    return result


@njit(parallel=True, cache=True)
def parallel_take_along_axis(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Apply sorting indices to reorder each row in parallel.

    Equivalent to np.take_along_axis(arr, indices, axis=1) but parallelized.

    Arguments:
    arr - A 2D numpy array to reorder.
    indices - A 2D array of indices (same shape as arr).

    Returns:
    result - The reordered array.
    """
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows, n_cols), dtype=arr.dtype)
    for i in prange(n_rows):
        for j in range(n_cols):
            result[i, j] = arr[i, indices[i, j]]
    return result


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
    algorithm,
    R,
    ptmin=None,
    ptmax=None,
    etamax=None,
    max_jets=None,
    get_truth=True,
    n_jobs=-1,
    random_seed: int | None = None,
    save_jet_info: bool = False,
    save_path=None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Earth Mover's Distances (EMDs) from a ROOT file TTree.

    This function performs the complete analysis pipeline:
    1. Extracts kinematics from the TTree (or list of TTrees)
    2. Clusters jets using the specified algorithm
    3. Filters jets based on pT and eta cuts
    4. Preprocesses jets (centers and rotates)
    5. Calculates EMDs between all jet pairs

    Arguments:
    ----------
    tree : uproot.TTree or list of uproot.TTree
        The TTree object(s) from uproot containing event data. If a list is
        provided, kinematics are loaded from each tree and concatenated along
        the event dimension. Returned event indices address the concatenated
        events (i.e. tree 0 events first, then tree 1, etc.).
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
    random_seed : int, optional
        Random seed used when subsampling jets (default: None).
    save_jet_info : bool, optional
        If True and save_path is provided, save jet pT / mass values to the .npz
        file under the key 'jet_pts' / 'jet_ms' (default: False).
    save_path : str, optional
        Optional path to save the EMDs and event indices to a .npz file.
        If provided, the EMDs and event indices will be saved with keys 'emds'
        and 'event_indices'. If save_jet_pts is True, 'jet_pts' will also be
        saved. If None, the data will not be saved (default: None).
    **kwargs
        Additional keyword arguments forwarded to :func:`extract_kinematics`
        (e.g. ``min_track_pt`` to apply a minimum pT cut on tracks).

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

    # Get kinematics from the tree(s)
    if isinstance(tree, list):
        all_kinematics = [
            extract_kinematics(t, get_truth=get_truth, **kwargs) for t in tree
        ]
        pt = ak.concatenate([k[0] for k in all_kinematics], axis=0)
        eta = ak.concatenate([k[1] for k in all_kinematics], axis=0)
        phi = ak.concatenate([k[2] for k in all_kinematics], axis=0)
        masses = ak.concatenate([k[3] for k in all_kinematics], axis=0)
        print(
            f"Concatenated kinematics from {len(tree)} trees "
            f"({len(pt)} total events)"
        )
    else:
        pt, eta, phi, masses = extract_kinematics(
            tree,
            get_truth=get_truth,
            **kwargs,
        )

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
        if len(event_jets) == 0:
            continue
        # Also require jets have at least 3 constituents so EMD is well defined
        if len(event_jets[0]) < 3:
            continue
        flat_jets.append(event_jets[0])
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
        if random_seed is not None:
            np.random.seed(random_seed)
        random_indices = np.random.choice(
            len(filtered_jets), size=max_jets, replace=False
        )
        random_indices = np.sort(random_indices)  # Sort to maintain order
        # Apply random sampling
        filtered_jets = [filtered_jets[i] for i in random_indices]
        filtered_jet_4vectors = [filtered_jet_4vectors[i] for i in random_indices]
        filtered_event_indices = filtered_event_indices[random_indices]
        print(f"Randomly sampled {max_jets} jets (dropped {n_dropped} jets)")

    jet_pts = np.array([jet_vec.pt for jet_vec in filtered_jet_4vectors])
    jet_ms = np.array([jet_vec.mass for jet_vec in filtered_jet_4vectors])
    jet_nconstits = np.array([len(jet) for jet in filtered_jets])

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
        if save_jet_info:
            np.savez(
                save_path,
                emds=emds,
                event_indices=filtered_event_indices,
                jet_pts=jet_pts,
                jet_ms=jet_ms,
                jet_nconstits=jet_nconstits,
            )
        else:
            np.savez(save_path, emds=emds, event_indices=filtered_event_indices)
        print(f"Saved EMDs and event indices to {save_path}")

    return emds, filtered_event_indices


def calculate_correlation_dimension_from_emds(
    emds: np.ndarray | da.Array,
    emd_weights: np.ndarray | da.Array,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate correlation dimension from Earth Mover's Distances (EMDs).

    This function now supports both numpy arrays and dask arrays. When using
    dask arrays, the histogramming will be performed in parallel across chunks.

    Arguments:
    emds - A numpy or dask array of EMD values (typically from calculate_emds).
    emd_weights - A numpy or dask array of weights corresponding to each EMD value.
    bins - A numpy array of bins to use in the histogram.

    Returns:
    dims - The correlation dimension values.
    dims_var - The variance of the correlation dimension values.
    midbins - The midpoints of the bins used in the calculation.
    """
    # Check if we're working with dask arrays
    is_dask = isinstance(emds, da.Array) or isinstance(emd_weights, da.Array)

    if is_dask:
        # Ensure both are dask arrays
        if not isinstance(emds, da.Array):
            emds = da.from_array(emds, chunks=emd_weights.chunks)
        if not isinstance(emd_weights, da.Array):
            emd_weights = da.from_array(emd_weights, chunks=emds.chunks)

        # Take upper triangle and mask invalid emds
        emds = da.triu(emds)
        mask = (emds > 0) & da.isfinite(emds)
        emds = emds[mask].flatten()
        emd_weights = emd_weights[mask].flatten()

        # Calculate the correlation dimension
        midbins = (bins[1:] + bins[:-1]) / 2
        dmidbins = np.log(midbins[1:]) - np.log(midbins[:-1])

        # Use dask histogram for parallel computation
        hist, _ = da.histogram(emds, bins=bins, weights=emd_weights)
        var, _ = da.histogram(emds, bins=bins, weights=emd_weights**2)

        # Compute the histograms (trigger computation)
        hist = hist.compute()
        var = var.compute()

    else:
        # Original numpy implementation
        # Take upper triangle and mask invalid emds
        emds = np.triu(emds)
        mask = (emds > 0) & np.isfinite(emds)
        emds = emds[mask].flatten()
        emd_weights = emd_weights[mask].flatten()

        # Calculate the correlation dimension
        midbins = (bins[1:] + bins[:-1]) / 2
        dmidbins = np.log(midbins[1:]) - np.log(midbins[:-1])
        hist, _ = np.histogram(emds, bins=bins, weights=emd_weights)
        var, _ = np.histogram(emds, bins=bins, weights=emd_weights**2)

    # Calculate the CDF (same for both numpy and dask after computation)
    counts = np.cumsum(hist) + np.finfo(float).eps
    counts_err = np.sqrt(np.cumsum(var))

    # Calculate the correlation dimension
    dims = (np.log(counts[1:]) - np.log(counts[:-1])) / dmidbins
    dims_var = (
        (counts_err[1:] / counts[1:]) ** 2 + (counts_err[:-1] / counts[:-1]) ** 2
    ) / dmidbins

    return dims, dims_var, midbins


def calculate_correlation_dimension_from_file(
    npz_path: str,
    weights_dict: dict[str, np.ndarray],
    bins: np.ndarray,
    n_jobs=1,
    chunk_size: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculate correlation dimension from EMDs stored in a .npz file.

    This function loads EMDs and event indices from a .npz file (produced by
    calculate_emds_from_file), broadcasts event-level weights to EMD pairs,
    and calculates the correlation dimension for each set of weights.

    The histogramming is performed using dask arrays for efficient parallel
    computation, avoiding the pickling overhead of multiprocessing.

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
        Number of parallel workers for dask computation.
        If -1, uses all available CPUs (default: 1).
    chunk_size : int, optional
        Size of chunks for dask array. If None, automatically determines
        a reasonable chunk size based on array size and available memory.
        Smaller chunks use less memory but have more overhead.

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

    # Convert EMDs to dask array with appropriate chunking
    # For large arrays, we want chunks that are large enough to be efficient
    # but small enough to fit in memory. A good default is ~100MB per chunk.
    if chunk_size is None:
        # Estimate chunk size: aim for ~100MB chunks (assuming float64)
        bytes_per_element = 8  # float64
        target_chunk_bytes = 100 * 1024 * 1024  # 100 MB
        # For a 2D array, chunk_size^2 * 8 bytes = target_chunk_bytes
        chunk_size = int(np.sqrt(target_chunk_bytes / bytes_per_element))
        # Round down to a reasonable size (e.g., multiple of 1000)
        chunk_size = (chunk_size // 1000) * 1000
        if chunk_size < 1000:
            chunk_size = 1000
        if chunk_size > n_jets:
            chunk_size = n_jets

    print(f"Converting EMDs to dask array with chunk size {chunk_size}...")
    # Create dask array with square chunks
    emds_da = da.from_array(emds, chunks=(chunk_size, chunk_size))

    # Configure dask threading/processes based on n_jobs
    if n_jobs == -1:
        n_workers = multiprocessing.cpu_count()
    else:
        n_workers = n_jobs

    # Set dask threading configuration
    # Use threads for better memory efficiency with numpy operations
    with dask.config.set(scheduler="threads", num_workers=n_workers):
        n_histograms = len(weights_dict)
        print(
            f"Processing {n_histograms} weight sets using dask "
            f"with {n_workers} workers..."
        )
        results = {}
        for hist_name, weights in tqdm(
            weights_dict.items(),
            total=n_histograms,
            desc="Processing weight sets",
            unit="set",
        ):
            # Ensure event_indices is a 1D array
            event_indices_arr = np.asarray(event_indices).flatten()
            weights_arr = np.asarray(weights)

            # Get the weights of the jets (fine if some events have multiple jets)
            jet_weights = weights_arr[event_indices_arr]

            # Create EMD weights matrix: weight for EMD(i,j) = weight(i) * weight(j)
            # This represents the product of weights for the two events
            # Convert to dask array with same chunking as EMDs
            jet_weights_da = da.from_array(jet_weights, chunks=emds_da.chunks[0])
            emd_weights_da = jet_weights_da[:, None] * jet_weights_da[None, :]

            dims, dims_var, midbins = calculate_correlation_dimension_from_emds(
                emds_da, emd_weights_da, bins
            )
            results[hist_name] = (dims, dims_var, midbins)

    return results


def sort_emd_matrix(
    emds: np.ndarray,
) -> np.ndarray:
    """Sort EMD matrix rows by distance.

    This function performs the expensive sorting operation once, allowing the
    sorted matrix to be reused for multiple NNID calculations with different
    weight sets.

    Arguments:
    emds - A numpy array of EMD values with shape (n_jets, n_jets).
        Can be upper triangular (will be symmetrized).

    Returns:
    sorted_emds - A 2D array where each row i contains distances from jet i
        to all other jets, sorted in ascending order. Shape is (n_jets, n_jets).
        The last column contains inf (self-distance).
    """
    # Calculate symmetric matrix of EMDs
    full_emds = emds + emds.T

    # Handle zeros - set to small value to avoid division issues
    full_emds = np.where(full_emds == 0, 1e-10, full_emds)

    # Set diagonal to inf so self-distances sort to the end
    np.fill_diagonal(full_emds, np.inf)

    # Parallel sort: get sorting indices for all rows using numba parallelization.
    # First call compiles the function (~1-2s overhead), subsequent calls are fast.
    print("Sorting EMD matrix (parallel with numba)...")
    sort_indices = parallel_argsort_rows(full_emds)

    # Apply sorting to EMDs using parallel advanced indexing
    sorted_emds = parallel_take_along_axis(full_emds, sort_indices)

    print(f"Sorted EMD matrix with shape {sorted_emds.shape}")
    return sorted_emds


def calculate_nnids_from_sorted_emds(
    sorted_emds: np.ndarray,
    weights_dict: dict[str, np.ndarray],
    points: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculate nearest neighbor intrinsic dimension from pre-sorted EMDs.

    This function takes pre-sorted EMD matrices (from sort_emd_matrix) and
    calculates NNIDs for each point and each weight set.

    The jet weights are used to weight each jet's contribution to the
    likelihood function.

    Arguments:
    sorted_emds - A 2D array of sorted EMD values from sort_emd_matrix.
    weights_dict - Dictionary mapping weight set names to 1D weight arrays.
    points - A numpy array of points to use in the calculation.

    Returns:
    results - Dictionary mapping weight set names to (nnids, avg_ris, avg_rjs) tuples.
    """
    n_weight_sets = len(weights_dict)
    print(f"Processing {n_weight_sets} weight sets for {len(points)} points...")

    # Pre-compute EMD ratios for each point (these don't depend on weights)
    emd_ratios_by_point = {}
    for point in points:
        emd_ratios_by_point[point] = sorted_emds[:, 2 * point] / sorted_emds[:, point]

    results = {}
    for weight_name, jet_weights in tqdm(
        weights_dict.items(),
        total=n_weight_sets,
        desc="Processing weight sets",
        unit="set",
    ):
        jet_weights = np.asarray(jet_weights).flatten()

        # Prepare output arrays for this weight set
        nnids = np.zeros(len(points))
        avg_ris = np.zeros(len(points))
        avg_rjs = np.zeros(len(points))

        # Loop through each point
        for pidx, point in enumerate(points):
            emd_ratios = emd_ratios_by_point[point]

            # Weighted average distance
            avg_ris[pidx] = np.average(sorted_emds[:, point], weights=jet_weights)
            avg_rjs[pidx] = np.average(sorted_emds[:, 2 * point], weights=jet_weights)

            # Minimize the likelihood function
            nnid = opt.minimize(
                nnid_nll,
                x0=2.0,
                args=(emd_ratios, jet_weights, point, 2 * point),
                method="L-BFGS-B",
                bounds=[(0.01, None)],
                options={"disp": False, "iprint": -1},
            )
            nnids[pidx] = nnid.x

        results[weight_name] = (nnids, avg_ris, avg_rjs)

    return results


def nnid_nll(
    d: float,
    emd_ratios: np.ndarray,
    jet_weights: np.ndarray,
    i: int,
    j: int,
) -> float:
    """Calculate the weighted negative log-likelihood for intrinsic dimension.

    This computes the NLL for the NNID estimator, where each jet's contribution
    to the likelihood is weighted by its jet_weight. The EMD ratios μ = r_j/r_i
    must satisfy μ >= 1 (since j > i means r_j >= r_i for sorted distances).

    For the unweighted case, the likelihood is:
        L(d) = d^n * prod_k [ μ_k^(-(1+i*d)) * (1 - μ_k^(-d))^(j-i-1) ]

    For weighted samples, we weight each jet's log-likelihood contribution:
        -log L(d) = -(Σw) log(d) + (1+i*d) Σ(w * log(μ)) - (j-i-1) Σ(w * log(1-μ^(-d)))

    Arguments:
    d - The intrinsic dimension to evaluate.
    emd_ratios - Array of EMD ratios μ = r_j/r_i. Must be >= 1.
    jet_weights - Array of per-jet weights (same length as emd_ratios).
    i - Index of first nearest neighbor (e.g., i=1 for 1st NN).
    j - Index of second nearest neighbor (e.g., j=2 for 2nd NN).

    Returns:
    nll - The weighted negative log-likelihood.
    """
    # Validate that ratios are >= 1 (required for the likelihood to be valid)
    if np.any(emd_ratios < 1):
        n_invalid = np.sum(emd_ratios < 1)
        min_ratio = np.min(emd_ratios)
        # For ratios very close to 1, this is fine numerically
        # Only warn/error for significantly < 1
        if min_ratio < 0.99:
            print(f"Warning: {n_invalid} EMD ratios < 1 (min={min_ratio:.6f})")

    # Compute weighted sums
    sum_w = np.sum(jet_weights)

    # Term 1: -( Σ w_i ) * log(d)
    t1 = -sum_w * np.log(d)

    # Term 2: (1 + i*d) * Σ( w_i * log(μ_i) )
    t2 = (1 + i * d) * np.sum(jet_weights * np.log(emd_ratios))

    # Term 3: -(j - i - 1) * Σ( w_i * log(1 - μ_i^(-d)) )
    # Note: μ >= 1 implies μ^(-d) <= 1, so 1 - μ^(-d) >= 0
    powers = np.power(emd_ratios, -d)
    # Clip to avoid log(0) when μ = 1 exactly
    log_arg = np.clip(1 - powers, 1e-300, None)
    t3 = -(j - i - 1) * np.sum(jet_weights * np.log(log_arg))

    return t1 + t2 + t3
