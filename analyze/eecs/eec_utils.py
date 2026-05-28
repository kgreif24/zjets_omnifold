"""
Utility functions for energy-energy correlator calculations.

This module provides both serial and parallel (Dask-based) implementations
for computing EECs and building weighted histograms.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import dask
import awkward as ak
import vector
import uproot

sys.path.insert(0, os.path.abspath(".."))
import utils.common_utils as cu  # noqa: E402


def _calculate_eec_chunk(
    file_path: str,
    tree_name: str,
    start: int,
    stop: int,
    invert_z: bool = False,
) -> tuple[ak.Array, ak.Array, np.ndarray]:
    """Calculate EEC for a chunk of events.

    This is an internal function that processes a subset of events from a TTree.
    Each worker opens its own file handle to avoid serialization issues with
    uproot TTree objects.

    Only off-diagonal pairs (i < j) are formed; the factor of 2 on the energy
    products accounts for both orderings (i,j) and (j,i). The normalization
    denominator is (sum E_i)^2, so each per-event histogram sums to less than 1
    (the missing fraction equals the diagonal self-pair contributions).

    Arguments:
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str
        Name of the TTree in the ROOT file.
    start : int
        Starting event index (for tree reading).
    stop : int
        Stopping event index (for tree reading).
    invert_z : bool, optional
        If True, invert the z values to look at the back-to-back region.
        Default is False.

    Returns:
    --------
    energy_sums : ak.Array
        Energy sums for each event, shape [n_events_chunk]
    energy_products : ak.Array
        Energy products for each pair, shape [n_events_chunk, var]
    zs : ak.Array
        Angular z values for each pair, shape [n_events_chunk, var]
    """

    # Open file handle within the worker to avoid serialization issues
    with uproot.open(file_path) as f:
        tree = f[tree_name]

        pt, eta, phi, masses = cu.extract_kinematics(
            tree,
            start=start,
            stop=stop,
        )

    # Build four-vectors
    vector.register_awkward()
    four_vectors = ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": masses},
        with_name="Momentum4D",
    )

    # Calculate Et and p3 unit vectors
    Et = four_vectors.Et
    p3_unit = four_vectors.to_Vector3D().unit()

    # Make pairs of energy weights and unit vectors (i < j only; each pair
    # is multiplied by 2 to account for both orderings (i,j) and (j,i))
    energy_weight_pairs = ak.combinations(
        Et, 2, axis=1, fields=["a", "b"], replacement=False
    )
    p3_unit_pairs = ak.combinations(
        p3_unit, 2, axis=1, fields=["a", "b"], replacement=False
    )

    # Normalize by total Et squared (includes self-pairs), so each per-event
    # histogram sums to less than 1 (missing the diagonal self-pair contributions).
    energy_sums = ak.to_numpy(ak.sum(Et, axis=1)) ** 2
    energy_products = 2 * energy_weight_pairs.a * energy_weight_pairs.b
    cos_thetas = p3_unit_pairs.a.dot(p3_unit_pairs.b)
    zs = (1 - cos_thetas) / 2

    # Invert z values if we want to look at the back-to-back region
    if invert_z:
        zs = 1 - zs

    return energy_sums, energy_products, zs


def _histogram_chunk(
    eec_result: tuple[ak.Array, ak.Array, np.ndarray],
    event_weights: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Build weighted EEC accumulator arrays from a single chunk.

    Arguments:
    ----------
    eec_result : tuple
        Tuple of (energy_sums, energy_products, zs, counts) from _calculate_eec_chunk.
    event_weights : np.ndarray
        Event-level weights to apply, shape [n_events_chunk].
    bins : np.ndarray
        Bin edges for the histogram.

    Returns:
    --------
    H : np.ndarray
        Weighted sum of per-event EEC histograms, shape [len(bins) - 1].
    HH : np.ndarray
        Weighted outer-product sum, shape [len(bins) - 1, len(bins) - 1].
    W : float
        Sum of event weights.
    W2 : float
        Sum of squared event weights.
    bins : np.ndarray
        Bin edges (passed through for aggregation).
    """

    energy_sums, energy_products, zs = eec_result
    n_events = len(energy_sums)
    n_bins = len(bins) - 1

    # Flatten pair-level arrays and build event index array
    flat_zs = ak.to_numpy(ak.flatten(zs))
    flat_ep = ak.to_numpy(ak.flatten(energy_products))
    pair_counts = ak.to_numpy(ak.count(energy_products, axis=1))
    event_indices = np.repeat(np.arange(n_events), pair_counts)

    # Normalize each pair contribution by its event's energy sum^2
    pair_contributions = flat_ep / np.repeat(energy_sums, pair_counts)

    # Digitize z values; pairs outside [bins[0], bins[-1]] are discarded
    bin_indices = np.searchsorted(bins, flat_zs, side='right') - 1
    valid = (bin_indices >= 0) & (bin_indices < n_bins)

    # Build V matrix [n_events, n_bins] via a single vectorized bincount
    flat_idx = event_indices[valid] * n_bins + bin_indices[valid]
    V = np.bincount(
        flat_idx, weights=pair_contributions[valid], minlength=n_events * n_bins
    ).reshape(n_events, n_bins)

    # Accumulate weighted sums
    Vw = V * event_weights[:, None]
    H = Vw.sum(axis=0)
    HH = Vw.T @ Vw
    W = float(event_weights.sum())
    W2 = float((event_weights**2).sum())

    return H, HH, W, W2, bins


def _sum_histograms(
    hist_list: list[tuple[np.ndarray, np.ndarray, float, float, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate chunk accumulators and compute the final EEC histogram and covariance.
    """
    H_total = np.sum([h[0] for h in hist_list], axis=0)
    HH_total = np.sum([h[1] for h in hist_list], axis=0)
    W_total = sum(h[2] for h in hist_list)
    W2_total = sum(h[3] for h in hist_list)
    bins = hist_list[0][4]

    h_bar = H_total / W_total
    cov = HH_total / W_total**2 - np.outer(h_bar, h_bar) * W2_total / W_total**2

    return h_bar, cov, bins


def calculate_eec_parallel(
    file_path: str,
    tree_name: str,
    chunk_size: int = 100_000,
    max_events: int | None = None,
    invert_z: bool = False,
) -> list:
    """Calculate EEC in parallel using Dask delayed tasks.

    This function creates delayed tasks for each chunk but does NOT compute them.
    The delayed results can be passed to build_histograms_parallel for efficient
    pipelining where histogram computation begins as soon as chunks complete.

    Arguments:
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str
        Name of the TTree in the ROOT file.
    chunk_size : int, optional
        Number of events per chunk. Default 100,000.
    max_events : int, optional
        Maximum number of events to process. If None, all events are processed.

    Returns:
    --------
    chunk_results : list
        List of dask.delayed objects, each representing the EEC result for a chunk.
        Each delayed object will resolve to (energy_products, zs, counts).
    chunk_ranges : list
        List of (start, stop) tuples for each chunk, useful for weight slicing.
    """

    n_events = uproot.open(file_path)[tree_name].num_entries
    if max_events is not None:
        n_events = min(n_events, max_events)
    n_chunks = (n_events + chunk_size - 1) // chunk_size

    chunk_results = []
    chunk_ranges = []

    for i in range(n_chunks):
        start = i * chunk_size
        stop = min((i + 1) * chunk_size, n_events)

        # Create delayed task for this chunk with pre-sliced flags
        # Pass file_path and tree_name instead of tree object
        chunk_result = dask.delayed(_calculate_eec_chunk)(
            file_path, tree_name, start, stop, invert_z=invert_z
        )
        chunk_results.append(chunk_result)
        chunk_ranges.append((start, stop))

    return chunk_results, chunk_ranges


def build_histograms_parallel(
    chunk_results: list,
    chunk_ranges: list[tuple[int, int]],
    weights_dict: dict[str, np.ndarray],
    bins: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build histograms in parallel for multiple weight sets using Dask.

    This function takes delayed EEC chunk results and builds histograms for
    each weight set in parallel. Histograms are aggregated across chunks by
    summation (histograms are additive).

    Arguments:
    ----------
    chunk_results : list
        List of dask.delayed objects from calculate_eec_parallel.
    chunk_ranges : list
        List of (start, stop) tuples for each chunk.
    weights_dict : dict
        Dictionary mapping weight names to event-level weight arrays.
        Each array should have shape [n_passing_events].
    bins : np.ndarray
        Bin edges for the histogram.

    Returns:
    --------
    histograms : dict
        Dictionary mapping weight names to histogram arrays.
    """

    # Build delayed histogram computations for each chunk and weight set
    all_delayed = {}
    for name in weights_dict.keys():
        all_delayed[name] = []

    for i, chunk_result in enumerate(chunk_results):
        wgt_start, wgt_stop = chunk_ranges[i]

        for name, event_weights in weights_dict.items():
            # Create delayed histogram task
            hist_tuple = dask.delayed(_histogram_chunk)(
                chunk_result, event_weights[wgt_start:wgt_stop], bins
            )
            all_delayed[name].append(hist_tuple)

    # Create delayed aggregation tasks
    final_delayed = {}
    for name, chunk_hists in all_delayed.items():
        final_delayed[name] = dask.delayed(_sum_histograms)(chunk_hists)

    # Compute all histograms in parallel
    results = dask.compute(final_delayed)[0]
    return results


def run_eec_workflow_parallel(
    file_path: str,
    tree_name: str,
    weights_dict: dict[str, np.ndarray],
    bins: np.ndarray,
    chunk_size: int = 100_000,
    max_events: int | None = None,
    invert_z: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run the full EEC workflow in parallel: compute EECs and build histograms.

    This is a convenience function that combines calculate_eec_parallel and
    build_histograms_parallel into a single call.

    Arguments:
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str
        Name of the TTree in the ROOT file.
    weights_dict : dict
        Dictionary mapping weight names to event-level weight arrays.
        Each array should have shape [n_passing_events].
    bins : np.ndarray
        Bin edges for the histogram.
    chunk_size : int, optional
        Number of events per chunk. Default 100,000.
    max_events : int, optional
        Maximum number of events to process. If None or greater than the number
        of events in the tree, all events are processed. If specified, only the
        first max_events events are used. The weights_dict arrays are also
        truncated to match the number of passing events in the limited range.
    invert_z : bool, optional
        If True, invert the z values (z → 1 - z) to look at the back-to-back
        region. Default is False.

    Returns:
    --------
    histograms : dict
        Dictionary mapping weight names to tuples of
        (h_bar, cov_matrix, bin_edges), where h_bar is the normalized EEC
        histogram of shape [n_bins], cov_matrix is the [n_bins, n_bins]
        statistical covariance of the weighted mean, and bin_edges has shape
        [n_bins + 1]. Per-bin variances are np.diag(cov_matrix).
    """

    # Apply max_events limit if specified
    nevents = uproot.open(file_path)[tree_name].num_entries
    if max_events is not None and max_events < nevents:
        weights_dict = {
            name: weights[:max_events] for name, weights in weights_dict.items()
        }

    # Create delayed EEC tasks
    chunk_results, chunk_ranges = calculate_eec_parallel(
        file_path,
        tree_name,
        chunk_size=chunk_size,
        max_events=max_events,
        invert_z=invert_z,
    )

    # Build histograms in parallel
    histograms = build_histograms_parallel(
        chunk_results, chunk_ranges, weights_dict, bins
    )

    return histograms
