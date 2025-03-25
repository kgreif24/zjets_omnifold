"""
test_plotter.py - Test suite for the plotter class
"""

import os
import numpy as np
from plotter import Plotter


def test_plots(tmp_path):

    plotter = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=2,
    )
    _ = plotter.plot(
        "weight",
        "./assets/wgts.npz",
        np.ones(100),
    )

    png_files = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert len(png_files) == 6


def test_w1(tmp_path):

    plotter = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=2,
    )
    start, end = plotter.wasserstein_distance(
        "weight",
        "./assets/wgts.npz",
        np.ones(100),
    )
    assert np.isclose(start, 146.237, atol=0.1)
    assert np.isclose(end, 83.916, atol=0.1)