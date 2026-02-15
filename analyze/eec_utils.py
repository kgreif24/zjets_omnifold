"""
Utility functions for energy-energy correlator calculations.

This module provides both serial and parallel (Dask-based) implementations
for computing EECs and building weighted histograms.
"""

from __future__ import annotations


import numpy as np
import dask
import awkward as ak
import vector
import uproot

import common_utils as cu


def _calculate_eec_chunk(
    file_path: str,
    tree_name: str,
    pass_flags_chunk: np.ndarray,
    start: int,
    stop: int,
    get_truth: bool = True,
) -> tuple[ak.Array, ak.Array, np.ndarray]:
    """Calculate EEC for a chunk of events.

    This is an internal function that processes a subset of events from a TTree.
    Each worker opens its own file handle to avoid serialization issues with
    uproot TTree objects.

    Arguments:
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str
        Name of the TTree in the ROOT file.
    pass_flags_chunk : np.ndarray
        Boolean array of pass flags for event filtering, pre-sliced for this chunk.
    start : int
        Starting event index (for tree reading).
    stop : int
        Stopping event index (for tree reading).
    get_truth : bool, optional
        If True, get truth-level data.

    Returns:
    --------
    energy_sums : ak.Array
        Energy sums for each event, shape [n_events_chunk]
    energy_products : ak.Array
        Energy products for each pair, shape [n_events_chunk, var]
    zs : ak.Array
        Angular z values for each pair, shape [n_events_chunk, var]
    counts : np.ndarray
        Number of pairs per event, shape [n_events_chunk]
    """

    # Open file handle within the worker to avoid serialization issues
    with uproot.open(file_path) as f:
        tree = f[tree_name]

        # Extract kinematics for this chunk and get counts of tracks per event
        # pass_flags_chunk is already sliced for this chunk
        pt, eta, phi, masses = cu.extract_kinematics(
            tree,
            pass_flags_chunk,
            get_truth=get_truth,
            start=start,
            stop=stop,
            filter_presliced=True,
        )
    counts = ak.to_numpy(ak.count(pt, axis=1))

    # Build four-vectors
    vector.register_awkward()
    four_vectors = ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": masses},
        with_name="Momentum4D",
    )

    # Calculate Et and p3 unit vectors
    Et = four_vectors.Et
    p3_unit = four_vectors.to_Vector3D().unit()

    # Make pairs of energy weights and unit vectors
    energy_weight_pairs = ak.combinations(
        Et, 2, axis=1, fields=["a", "b"], replacement=True
    )
    p3_unit_pairs = ak.combinations(
        p3_unit, 2, axis=1, fields=["a", "b"], replacement=True
    )

    # Calculate energy sums, products, and z values
    energy_sums = ak.to_numpy(ak.sum(Et, axis=1)) ** 2
    energy_products = energy_weight_pairs.a * energy_weight_pairs.b
    cos_thetas = p3_unit_pairs.a.dot(p3_unit_pairs.b)
    zs = (1 - cos_thetas) / 2

    return energy_sums, energy_products, zs, counts


def _histogram_chunk(
    eec_result: tuple[ak.Array, ak.Array, np.ndarray],
    event_weights: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a weighted histogram from a single EEC chunk.

    Arguments:
    ----------
    eec_result : tuple
        Tuple of (energy_products, zs, counts) from _calculate_eec_chunk.
    event_weights : np.ndarray
        Event-level weights to apply, shape [n_events_chunk].
    bins : np.ndarray
        Bin edges for the histogram.

    Returns:
    --------
    hist : np.ndarray
        Histogram counts, shape [len(bins) - 1].
    hist_var : np.ndarray
        Histogram variances, shape [len(bins) - 1].
    bins : np.ndarray
        Bin edges for the histogram.
    """

    # Extract EEC results
    energy_sums, energy_products, zs, counts = eec_result

    # Broadcast event weights to pair level
    pair_weights = ak.broadcast_arrays(
        event_weights / energy_sums,
        energy_products,
    )[0]

    # Flatten arrays for histogramming
    flat_zs = ak.to_numpy(ak.flatten(zs))
    flat_weights = ak.to_numpy(ak.flatten(pair_weights) * ak.flatten(energy_products))

    # Build histograms
    hist, _ = np.histogram(flat_zs, bins=bins, weights=flat_weights)
    hist_var, _ = np.histogram(flat_zs, bins=bins, weights=flat_weights**2)
    return hist, hist_var, bins


def _sum_histograms(
    hist_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum a list of histograms."""
    hist = np.sum([h[0] for h in hist_list], axis=0)
    hist_var = np.sum([h[1] for h in hist_list], axis=0)
    bins = hist_list[0][2]
    return hist, hist_var, bins


def calculate_eec_parallel(
    file_path: str,
    tree_name: str,
    pass_flags: np.ndarray,
    get_truth: bool = True,
    chunk_size: int = 100_000,
) -> list:
    """Calculate EEC in parallel using Dask delayed tasks.

    This function creates delayed tasks for each chunk but does NOT compute them.
    The delayed results can be passed to build_histograms_parallel for efficient
    pipelining where histogram computation begins as soon as chunks complete.

    Note: pass_flags is pre-sliced per chunk before being attached to the task
    graph, avoiding memory issues with large arrays being serialized in full.
    File path and tree name are passed instead of TTree objects to allow each
    worker to open its own file handle (TTree objects don't serialize properly).

    Arguments:
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str
        Name of the TTree in the ROOT file.
    pass_flags : np.ndarray
        Boolean array of pass flags for event filtering.
    get_truth : bool, optional
        If True, get truth-level data. Default True.
    chunk_size : int, optional
        Number of events per chunk. Default 100,000.

    Returns:
    --------
    chunk_results : list
        List of dask.delayed objects, each representing the EEC result for a chunk.
        Each delayed object will resolve to (energy_products, zs, counts).
    chunk_ranges : list
        List of (start, stop) tuples for each chunk, useful for weight slicing.
    """

    n_events = len(pass_flags)
    n_chunks = (n_events + chunk_size - 1) // chunk_size

    chunk_results = []
    chunk_ranges = []

    for i in range(n_chunks):
        start = i * chunk_size
        stop = min((i + 1) * chunk_size, n_events)

        # Pre-slice pass_flags for this chunk to avoid serializing the full array
        # with each task (which causes memory issues for large arrays)
        pass_flags_chunk = pass_flags[start:stop]

        # Create delayed task for this chunk with pre-sliced flags
        # Pass file_path and tree_name instead of tree object
        chunk_result = dask.delayed(_calculate_eec_chunk)(
            file_path, tree_name, pass_flags_chunk, start, stop, get_truth
        )
        chunk_results.append(chunk_result)
        chunk_ranges.append((start, stop))

    return chunk_results, chunk_ranges


def build_histograms_parallel(
    chunk_results: list,
    chunk_ranges: list[tuple[int, int]],
    weights_dict: dict[str, np.ndarray],
    bins: np.ndarray,
    pass_flags: np.ndarray,
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
    pass_flags : np.ndarray
        Boolean array of pass flags, used to slice weights correctly.

    Returns:
    --------
    histograms : dict
        Dictionary mapping weight names to histogram arrays.
    """

    # Pre-compute cumulative counts of passing events for each chunk
    # This is needed to correctly slice the weights (which only include passing events)
    passing_counts = []
    cumsum = 0
    for start, stop in chunk_ranges:
        chunk_flags = pass_flags[start:stop]
        n_passing = int(np.sum(chunk_flags == 1))
        passing_counts.append((cumsum, cumsum + n_passing))
        cumsum += n_passing

    # Build delayed histogram computations for each chunk and weight set
    all_delayed = {}
    for name in weights_dict.keys():
        all_delayed[name] = []

    for i, chunk_result in enumerate(chunk_results):
        wgt_start, wgt_stop = passing_counts[i]

        for name, event_weights in weights_dict.items():
            # Slice weights for this chunk (only passing events)
            chunk_weights = event_weights[wgt_start:wgt_stop]

            # Create delayed histogram task
            hist_tuple = dask.delayed(_histogram_chunk)(
                chunk_result, chunk_weights, bins
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
    pass_flags: np.ndarray,
    weights_dict: dict[str, np.ndarray],
    bins: np.ndarray,
    get_truth: bool = True,
    chunk_size: int = 100_000,
    max_events: int | None = None,
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
    pass_flags : np.ndarray
        Boolean array of pass flags for event filtering.
    weights_dict : dict
        Dictionary mapping weight names to event-level weight arrays.
        Each array should have shape [n_passing_events].
    bins : np.ndarray
        Bin edges for the histogram.
    get_truth : bool, optional
        If True, get truth-level data. Default True.
    chunk_size : int, optional
        Number of events per chunk. Default 100,000.
    max_events : int, optional
        Maximum number of events to process. If None or greater than the number
        of events in the tree, all events are processed. If specified, only the
        first max_events events are used. The weights_dict arrays are also
        truncated to match the number of passing events in the limited range.

    Returns:
    --------
    histograms : dict
        Dictionary mapping weight names to tuples of
        (histogram, histogram_variance, bin_edges).
    """

    # Apply max_events limit if specified
    if max_events is not None and max_events < len(pass_flags):
        # Truncate pass_flags to first max_events
        pass_flags = pass_flags[:max_events]

        # Count passing events in the limited range to truncate weights
        n_passing_limited = int(np.sum(pass_flags == 1))

        # Truncate all weight arrays to match
        weights_dict = {
            name: weights[:n_passing_limited] for name, weights in weights_dict.items()
        }

    # Create delayed EEC tasks
    chunk_results, chunk_ranges = calculate_eec_parallel(
        file_path, tree_name, pass_flags, get_truth=get_truth, chunk_size=chunk_size
    )

    # Build histograms in parallel
    histograms = build_histograms_parallel(
        chunk_results, chunk_ranges, weights_dict, bins, pass_flags
    )

    return histograms
