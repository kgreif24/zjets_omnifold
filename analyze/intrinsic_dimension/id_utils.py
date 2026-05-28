"""
Utility functions for intrinsic dimension calculations.
"""

import sys
import os
import scipy.optimize as opt
import numpy as np
import vector
import awkward as ak
import energyflow as ef
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import uproot
import weightedstats as ws
from tqdm import tqdm
from numba import njit, prange

sys.path.insert(0, os.path.abspath(".."))
from utils.common_utils import extract_kinematics  # noqa: E402
import utils.jet_clusterer as jet_clusterer  # noqa: E402


def _nnid_worker(args: tuple) -> tuple:
    """Worker function for parallel NNID calculation.

    Must be defined at module level so it is picklable (required by
    multiprocessing). Attaches to pre-existing SharedMemory blocks by name
    to access sorted_emds and sort_indices without copying.

    Arguments:
    args - Tuple of (weight_name, jet_weights_arr, thresholds,
                     shm_emds_name, emds_shape, emds_dtype,
                     shm_idx_name, idx_shape, idx_dtype).

    Returns:
    Tuple of (weight_name, nnids, median_ris, median_rjs).
    """
    (
        weight_name,
        jet_weights_arr,
        thresholds,
        shm_emds_name,
        emds_shape,
        emds_dtype,
        shm_idx_name,
        idx_shape,
        idx_dtype,
    ) = args

    # Attach to shared memory — zero per-worker copy, safe with spawn.
    shm_emds = SharedMemory(name=shm_emds_name)
    shm_idx = SharedMemory(name=shm_idx_name)
    sorted_emds = np.ndarray(emds_shape, dtype=emds_dtype, buffer=shm_emds.buf)
    sort_indices = np.ndarray(idx_shape, dtype=idx_dtype, buffer=shm_idx.buf)

    n_jets = sorted_emds.shape[0]

    jet_weights = np.asarray(jet_weights_arr).flatten()
    jet_weights = jet_weights * n_jets / np.sum(jet_weights)

    thresholds_arr = np.asarray(thresholds, dtype=np.float64)

    # Find threshold crossing columns without materializing O(n²) arrays.
    # find_threshold_cols computes per-row cumsum on-the-fly with numba
    # parallelism and early termination once all 2k thresholds are satisfied.
    cols_k, cols_2k = find_threshold_cols(jet_weights, sort_indices, thresholds_arr)

    if np.any(cols_k < 0) or np.any(cols_2k < 0):
        raise RuntimeError(
            "Some thresholds were not reached for all jets; check weight normalization."
        )

    # Index sorted EMDs for all thresholds at once: (n_jets, n_thres)
    jet_range = np.arange(n_jets)
    r_k = sorted_emds[jet_range[:, None], cols_k]
    r_2k = sorted_emds[jet_range[:, None], cols_2k]
    emd_ratios = r_2k / r_k

    nnids = np.zeros(len(thresholds))
    median_ris = np.zeros(len(thresholds))
    median_rjs = np.zeros(len(thresholds))

    for thres_idx, thres in enumerate(thresholds):
        median_ris[thres_idx] = ws.numpy_weighted_median(
            r_k[:, thres_idx], weights=jet_weights
        )
        median_rjs[thres_idx] = ws.numpy_weighted_median(
            r_2k[:, thres_idx], weights=jet_weights
        )
        nnid = opt.minimize(
            nnid_nll,
            x0=2.0,
            args=(emd_ratios[:, thres_idx], jet_weights, thres, 2 * thres),
            method="L-BFGS-B",
            bounds=[(0.01, None)],
            options={"disp": False, "iprint": -1},
        )
        nnids[thres_idx] = nnid.x

    shm_emds.close()
    shm_idx.close()

    return weight_name, nnids, median_ris, median_rjs


@njit(parallel=True, cache=True)
def find_threshold_cols(
    jet_weights: np.ndarray, sort_indices: np.ndarray, thresholds: np.ndarray
) -> tuple:
    """Find column indices where cumulative sorted weight first reaches each threshold.

    Computes the per-row cumulative sum on-the-fly without materializing the
    full sorted_weights or cumsum arrays. Exits each row as soon as all
    2*threshold levels are satisfied (early termination).

    Arguments:
    jet_weights  - 1D array of normalized jet weights, shape (n_jets,).
    sort_indices - 2D array of sort indices, shape (n_jets, n_jets).
                   sort_indices[i, j] is the original index of the j-th nearest
                   neighbor of jet i.
    thresholds   - 1D array of threshold values, shape (n_thres,).
                   Should be sorted in ascending order.

    Returns:
    cols_k  - 2D int64 array, shape (n_jets, n_thres). cols_k[i, t] is the
              column where the cumsum for jet i first reaches thresholds[t].
    cols_2k - 2D int64 array, shape (n_jets, n_thres). Same but for
              2 * thresholds[t].
    """
    n_jets, n_neighbors = sort_indices.shape
    n_thres = len(thresholds)
    cols_k = np.full((n_jets, n_thres), -1, dtype=np.int64)
    cols_2k = np.full((n_jets, n_thres), -1, dtype=np.int64)
    for i in prange(n_jets):
        cumsum = 0.0
        found_2k = 0
        for j in range(n_neighbors):
            cumsum += jet_weights[sort_indices[i, j]]
            for t in range(n_thres):
                if cols_k[i, t] == -1 and cumsum >= thresholds[t]:
                    cols_k[i, t] = j
                if cols_2k[i, t] == -1 and cumsum >= 2.0 * thresholds[t]:
                    cols_2k[i, t] = j
                    found_2k += 1
            if found_2k == n_thres:
                break  # all thresholds satisfied for this jet
    return cols_k, cols_2k


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
    file_path,
    tree_name,
    algorithm,
    R,
    ptmin=None,
    ptmax=None,
    etamax=None,
    max_jets=None,
    n_jobs=-1,
    random_seed: int | None = None,
    save_jet_info: bool = False,
    save_path=None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Earth Mover's Distances (EMDs) from a ROOT file.

    This function performs the complete analysis pipeline:
    1. Opens the ROOT file(s) and reads the TTree
    2. Extracts kinematics from the TTree (or list of TTrees)
    3. Clusters jets using the specified algorithm
    4. Filters jets based on pT and eta cuts
    5. Preprocesses jets (centers and rotates)
    6. Calculates EMDs between all jet pairs

    Arguments:
    ----------
    file_path : str or list of str
        Path(s) to the ROOT file(s). If a list is provided, kinematics are
        loaded from each file and concatenated along the event dimension.
        Returned event indices address the concatenated events
        (i.e. file 0 events first, then file 1, etc.).
    tree_name : str
        Name of the TTree within the ROOT file(s).
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

    # Get kinematics from the file(s)
    if isinstance(file_path, list):
        all_kinematics = []
        for path in file_path:
            with uproot.open(path) as f:
                tree = f[tree_name]
                all_kinematics.append(extract_kinematics(tree, **kwargs))
        pt = ak.concatenate([k[0] for k in all_kinematics], axis=0)
        eta = ak.concatenate([k[1] for k in all_kinematics], axis=0)
        phi = ak.concatenate([k[2] for k in all_kinematics], axis=0)
        masses = ak.concatenate([k[3] for k in all_kinematics], axis=0)
        print(
            f"Concatenated kinematics from {len(file_path)} files "
            f"({len(pt)} total events)"
        )
    else:
        with uproot.open(file_path) as f:
            tree = f[tree_name]
            pt, eta, phi, masses = extract_kinematics(
                tree,
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


def sort_emd_matrix(
    emds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    sort_indices - A 2D array of the original jet indices corresponding to the
        sorted positions in sorted_emds. sort_indices[i, k] is the original
        index of the k-th nearest neighbor of jet i.
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
    return sorted_emds, sort_indices


def calculate_nnids_from_sorted_emds(
    sorted_emds: np.ndarray,
    sort_indices: np.ndarray,
    weights_dict: dict[str, np.ndarray],
    thresholds: np.ndarray,
    n_jobs: int = 1,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculate nearest neighbor intrinsic dimension from pre-sorted EMDs.

    This function takes pre-sorted EMD matrices (from sort_emd_matrix) and
    calculates NNIDs for each point and each weight set.

    For each weight set the weighted k-th nearest neighbor of jet i is defined
    as the neighbor at which the cumulative (normalized) weight of all closer
    jets first reaches k.  This replaces the unweighted column index k used in
    the raw sorted matrix, so the EMD ratio μ = r_{2k}/r_k is computed from
    weight-dependent column positions rather than fixed ones.

    Jet weights are normalized to sum to N (the number of jets) independently
    for each weight set before the weighted NN positions are determined.

    Arguments:
    sorted_emds - A 2D array of sorted EMD values from sort_emd_matrix,
        shape (n_jets, n_jets).
    sort_indices - A 2D array of original jet indices from sort_emd_matrix,
        shape (n_jets, n_jets). sort_indices[i, k] is the original index of
        the k-th nearest neighbor of jet i.
    weights_dict - Dictionary mapping weight set names to 1D weight arrays.
    thresholds - A numpy array of weighted rank thresholds to use in the
        calculation.
    n_jobs - Number of parallel workers. Use -1 for all CPUs, 1 for sequential
        (default: 1). Values > 1 use spawn-based multiprocessing; the large
        read-only arrays (sorted_emds, sort_indices) are placed in shared
        memory once and attached by each worker without per-worker copies.

    Returns:
    results - Dictionary mapping weight set names to (nnids, median_ris, median_rjs)
    """
    n_weight_sets = len(weights_dict)
    print(f"Processing {n_weight_sets} weight sets for {len(thresholds)} thresholds...")

    # Place the large read-only arrays into shared memory once.
    # Workers attach by name — no per-worker copies, and safe with spawn
    # (unlike fork, which conflicts with OpenMP/numba in Jupyter).
    shm_emds = SharedMemory(create=True, size=sorted_emds.nbytes)
    shm_idx = SharedMemory(create=True, size=sort_indices.nbytes)
    try:
        np.copyto(
            np.ndarray(sorted_emds.shape, dtype=sorted_emds.dtype, buffer=shm_emds.buf),
            sorted_emds,
        )
        np.copyto(
            np.ndarray(
                sort_indices.shape, dtype=sort_indices.dtype, buffer=shm_idx.buf
            ),
            sort_indices,
        )

        task_args = [
            (
                name,
                np.asarray(weights).flatten(),
                thresholds,
                shm_emds.name,
                sorted_emds.shape,
                sorted_emds.dtype,
                shm_idx.name,
                sort_indices.shape,
                sort_indices.dtype,
            )
            for name, weights in weights_dict.items()
        ]

        if n_jobs == 1:
            worker_results = [
                _nnid_worker(args)
                for args in tqdm(task_args, desc="Processing weight sets", unit="set")
            ]
        else:
            n_workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
            print(f"Processing {len(task_args)} weight, {n_workers} workers...")
            with multiprocessing.get_context("spawn").Pool(n_workers) as pool:
                worker_results = list(
                    tqdm(
                        pool.imap_unordered(_nnid_worker, task_args),
                        total=len(task_args),
                        desc="Processing weight sets",
                        unit="set",
                    )
                )
    finally:
        shm_emds.close()
        shm_emds.unlink()
        shm_idx.close()
        shm_idx.unlink()

    return {
        name: (nnids, median_ris, median_rjs)
        for name, nnids, median_ris, median_rjs in worker_results
    }


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


def poisson_bootstrap_weights(
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Fluctuate jet weights within their Poisson uncertainty.

    Samples Poisson(1) multipliers independently for each jet and multiplies
    them by the original weights. This is the standard Poisson bootstrap
    used to estimate statistical uncertainties on weighted distributions.

    Arguments:
    weights - 1D array of jet weights.
    rng     - Optional numpy random Generator. If None, uses
              np.random.default_rng().

    Returns:
    bootstrapped - weights multiplied by Poisson(1) samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = np.asarray(weights).flatten()
    poisson_multipliers = rng.poisson(lam=1.0, size=len(weights)).astype(float)
    return poisson_multipliers * weights
