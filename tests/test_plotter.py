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
    assert len(png_files) == 26


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
    assert np.isclose(start, 352.057, atol=0.1)
    assert np.isclose(end, 364.483, atol=0.1)


def test_apply_kinematic_cuts(tmp_path):

    # Initialize the Plotter object
    plotter = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=2,
        kinematic_region=1,
    )

    # Get pT_ll data
    source_pT_ll = plotter._get_data("pT_ll", is_target=False)
    source_pT_trackj2 = plotter._get_data("pT_trackj2", is_target=False)
    target_pT_ll = plotter._get_data("pT_ll", is_target=True)
    target_pT_trackj2 = plotter._get_data("pT_trackj2", is_target=True)

    # Check the cut worked
    assert np.all(source_pT_ll > 350)
    assert np.all(source_pT_trackj2 > 50)
    assert np.all(target_pT_ll > 350)
    assert np.all(target_pT_trackj2 > 50)
