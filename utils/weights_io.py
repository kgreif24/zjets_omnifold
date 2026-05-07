"""weights_io.py - Shared helpers for writing the per-iteration weight .npz files
produced by the Omnifold and AUSSIE training loops.

The output schema is identical for both methods so that downstream tooling
(ensemble_weights.py, plotting) does not need to distinguish between them.
"""

import numpy as np


def write_weights_npz(
    path,
    raw_train,
    raw_test,
    network_train,
    network_test,
    train,
    test,
):
    """Write the standard Omnifold / AUSSIE weight npz file.

    Arguments:
        path (str) - Full path to the output .npz file
        raw_train (np.ndarray) - Raw network output (log density ratio) on train split
        raw_test (np.ndarray) - Raw network output on test split
        network_train (np.ndarray) - exp(raw_train), per-passed-event ratios
        network_test (np.ndarray) - exp(raw_test), per-passed-event ratios
        train (np.ndarray) - Final weights on train split (start * network)
        test (np.ndarray) - Final weights on test split

    Returns:
        None
    """
    np.savez(
        path,
        raw_train_output=raw_train,
        raw_test_output=raw_test,
        network_train=network_train,
        network_test=network_test,
        train=train,
        test=test,
    )
