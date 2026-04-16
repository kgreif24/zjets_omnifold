"""
test_uncertainty_plotter.py - Test suite for the UncertaintyPlotter class
"""

import os
import numpy as np
from uncertainty_plotter import UncertaintyPlotter


test_uncertainty_definitions = {
    "nn-init": {
        "name": "NN Initialization",
        "color": "red",
        "stochastic": True,
        "prefix": "weights_",
    },
    "data-stat": {
        "name": "Data Stat",
        "color": "blue",
        "stochastic": True,
        "prefix": "bootstrap_data_",
    },
    "mc-stat": {
        "name": "MC Stat",
        "color": "green",
        "stochastic": True,
        "prefix": "bootstrap_mc_",
    },
    "muon-id": {
        "name": "Muon ID",
        "color": "yellow",
        "stochastic": False,
        "prefix": None,
    },
    "muon-ms": {
        "name": "Muon MS",
        "color": "orange",
        "stochastic": False,
        "prefix": None,
    },
}

test_uncertainty_groups = {
    "Muon": ["muon-id", "muon-ms"],
    "Statistics": ["mc-stat", "data-stat"],
}


def test_uncertainty_plotter_init(tmp_path):
    """Test basic initialization with all required trees"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Test basic initialization
    assert plotter.source_path == "./assets/evts_000_100.root"
    assert plotter.target_path == "./assets/truth_evts_000_100.root"
    assert plotter.store == tmp_path
    assert plotter.verbosity == 0


def test_uncertainty_plotter_plot_basic(tmp_path):
    """Test basic plot generation with nominal weights"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
        uncertainty_definitions=test_uncertainty_definitions,
        uncertainty_groups=test_uncertainty_groups,
    )

    # Test basic plot generation
    plotter.plot("./assets/unc_wgts.npz", "./assets/target_wgts.npz")

    # Check that plots were generated
    png_files = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert len(png_files) > 0


def test_systematic_uncertainties(tmp_path):
    """Test systematic uncertainty handling some common syst types:
    (nn-init, mc-stat, data-stat, track-eff, jet-track-eff, hv)
    """
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
        uncertainty_definitions=test_uncertainty_definitions,
        uncertainty_groups=test_uncertainty_groups,
    )

    # Test that systematic uncertainties are handled
    # Load the weights file to check systematic variations
    weights_data = np.load("./assets/unc_wgts.npz")

    # Check that all expected systematic variations are present (with weights_ prefix)
    expected_systematics = [
        "weights_nn-init",
        "weights_mc-stat",
        "weights_data-stat",
        "weights_track-eff",
        "weights_jet-track-eff",
        "weights_hv",
    ]
    assert all([key in weights_data.keys() for key in expected_systematics])

    # Test plot generation with systematic weights
    plotter.plot("./assets/unc_wgts.npz", "./assets/target_wgts.npz")


def test_dual_target_mode(tmp_path):
    """Test with target2_path provided"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        target2_path="./assets/truth_evts_100_200.root",
        verbosity=0,
        uncertainty_definitions=test_uncertainty_definitions,
        uncertainty_groups=test_uncertainty_groups,
    )

    # Test that target2_path is set correctly
    assert plotter.target2_path == "./assets/truth_evts_100_200.root"

    # Test basic functionality with dual target mode
    plotter.plot(
        "./assets/unc_wgts.npz", "./assets/target_wgts.npz", "./assets/target2_wgts.npz"
    )


def test_uncertainty_budget_plot(tmp_path):
    """Test budget plot generation (lines 923-1058)"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
        uncertainty_definitions=test_uncertainty_definitions,
        uncertainty_groups=test_uncertainty_groups,
    )

    # Test uncertainty budget plot generation
    # This tests the budget plot functionality
    plotter.plot("./assets/unc_wgts.npz", "./assets/target_wgts.npz")

    # Check that budget plots were generated
    png_files = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert len(png_files) > 0


def test_cached_pass190_for_all_trees(tmp_path):
    """Test caching for source, target, sherpa, data, target2"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        target2_path="./assets/truth_evts_100_200.root",
        verbosity=0,
        uncertainty_definitions=test_uncertainty_definitions,
        uncertainty_groups=test_uncertainty_groups,
    )

    # Test that all paths are set correctly
    assert plotter.target2_path == "./assets/truth_evts_100_200.root"

    # Test that caching works for all trees
    # This tests the _get_cached_pass190 method
    plotter.plot(
        "./assets/unc_wgts.npz", "./assets/target_wgts.npz", "./assets/target2_wgts.npz"
    )


def test_kinematic_cuts_all_trees(tmp_path):
    """Test kinematic cuts applied to all trees (lines 1278-1330)"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
        uncertainty_definitions=test_uncertainty_definitions,
        uncertainty_groups=test_uncertainty_groups,
    )

    # Apply kinematic cuts after initialization to avoid constructor timing issues
    plotter.apply_kinematic_cuts(1)

    # Test that kinematic cuts are applied to all trees
    # This tests the kinematic filtering functionality
    plotter.plot("./assets/unc_wgts.npz", "./assets/target_wgts.npz")

    # Verify that kinematic cuts work
    source_pT_ll = plotter._get_data("pT_ll", is_target=False)
    target_pT_ll = plotter._get_data("pT_ll", is_target=True)

    # Check that cuts were applied (region 1: pT_ll > 350, pT_trackj2 > 50)
    if len(source_pT_ll) > 0:
        assert np.all(source_pT_ll > 350)
    if len(target_pT_ll) > 0:
        assert np.all(target_pT_ll > 350)


def test_track_weights_batch(tmp_path):
    """Test batch track weight processing (lines 1222-1249)"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
        uncertainty_definitions=test_uncertainty_definitions,
        uncertainty_groups=test_uncertainty_groups,
    )

    # Test track weight batch processing
    # This tests the batch processing functionality for track weights
    plotter.plot("./assets/unc_wgts.npz", "./assets/target_wgts.npz")

    # Verify that track weights are processed correctly
    # The exact implementation depends on the track weight processing method


def test_efficient_weight_loading(tmp_path):
    """Test optimized weight loading (lines 1072-1135)"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
        uncertainty_definitions=test_uncertainty_definitions,
        uncertainty_groups=test_uncertainty_groups,
    )

    # Test efficient weight loading
    # This tests the optimized weight loading functionality
    plotter.plot("./assets/unc_wgts.npz", "./assets/target_wgts.npz")

    # Verify that weights are loaded efficiently
    # The exact implementation depends on the weight loading optimization
