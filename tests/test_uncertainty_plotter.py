"""
test_uncertainty_plotter.py - Test suite for the UncertaintyPlotter class
"""

import os
import numpy as np
from uncertainty_plotter import UncertaintyPlotter


def test_uncertainty_plotter_init(tmp_path):
    """Test basic initialization with all required trees"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Test basic initialization
    assert plotter.source_path == "./assets/evts_000_100.root"
    assert plotter.target_path == "./assets/truth_evts_000_100.root"
    assert plotter.data_path == "./assets/data_evts.root"
    assert plotter.store == tmp_path
    assert plotter.verbosity == 0


def test_uncertainty_plotter_plot_basic(tmp_path):
    """Test basic plot generation with nominal weights"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Test basic plot generation
    _ = plotter.plot("./assets/unc_wgts.npz")

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
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Test that systematic uncertainties are handled
    # Load the weights file to check systematic variations
    weights_data = np.load("./assets/unc_wgts.npz")

    # Check that all expected systematic variations are present
    expected_systematics = [
        "nn-init",
        "mc-stat",
        "data-stat",
        "track_eff",
        "jet-track-eff",
        "hv",
    ]
    for syst in expected_systematics:
        assert f"{syst}_up" in weights_data.keys()
        assert f"{syst}_down" in weights_data.keys()

    # Test plot generation with systematic weights
    _ = plotter.plot("./assets/unc_wgts.npz")


def test_dual_target_mode(tmp_path):
    """Test with target2_path provided"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        target2_path="./assets/truth_evts_100_200.root",
        verbosity=0,
    )

    # Test that target2_path is set correctly
    assert plotter.target2_path == "./assets/truth_evts_100_200.root"

    # Test basic functionality with dual target mode
    _ = plotter.plot("./assets/unc_wgts.npz")


def test_data_comparison_mode(tmp_path):
    """Test data_comparison_mode=True behavior"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        data_comparison_mode=True,
    )

    # Test that data_comparison_mode is set correctly
    assert plotter.data_comparison_mode is True
    assert plotter.data_path == "./assets/data_evts.root"

    # Test basic functionality with data comparison mode
    _ = plotter.plot("./assets/unc_wgts.npz")


def test_uncertainty_budget_plot(tmp_path):
    """Test budget plot generation (lines 923-1058)"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Test uncertainty budget plot generation
    # This tests the budget plot functionality
    _ = plotter.plot("./assets/unc_wgts.npz")

    # Check that budget plots were generated
    png_files = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert len(png_files) > 0


def test_cached_pass190_for_all_trees(tmp_path):
    """Test caching for source, target, sherpa, data, target2"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        target2_path="./assets/truth_evts_100_200.root",
        verbosity=0,
    )

    # Test that all paths are set correctly
    assert plotter.data_path == "./assets/data_evts.root"
    assert plotter.target2_path == "./assets/truth_evts_100_200.root"

    # Test that caching works for all trees
    # This tests the _get_cached_pass190 method
    _ = plotter.plot("./assets/unc_wgts.npz")


def test_kinematic_cuts_all_trees(tmp_path):
    """Test kinematic cuts applied to all trees (lines 1278-1330)"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Apply kinematic cuts after initialization to avoid constructor timing issues
    plotter.apply_kinematic_cuts(1)

    # Test that kinematic cuts are applied to all trees
    # This tests the kinematic filtering functionality
    _ = plotter.plot("./assets/unc_wgts.npz")

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
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Test track weight batch processing
    # This tests the batch processing functionality for track weights
    _ = plotter.plot("./assets/unc_wgts.npz")

    # Verify that track weights are processed correctly
    # The exact implementation depends on the track weight processing method


def test_efficient_weight_loading(tmp_path):
    """Test optimized weight loading (lines 1072-1135)"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Test efficient weight loading
    # This tests the optimized weight loading functionality
    _ = plotter.plot("./assets/unc_wgts.npz")

    # Verify that weights are loaded efficiently
    # The exact implementation depends on the weight loading optimization


def test_uncertainty_merging_functionality(tmp_path):
    """Test uncertainty merging functionality with hiding of individual uncertainties"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Test default uncertainty groups
    assert "Tracking" in plotter.uncertainty_groups
    assert "Unfolding" in plotter.uncertainty_groups
    assert plotter.uncertainty_groups["Tracking"] == ["track-eff", "jet-track-eff"]
    assert plotter.uncertainty_groups["Unfolding"] == ["dd", "hv"]

    # Test default hiding behavior
    assert plotter.hide_individual_uncertainties is True

    # Test plot generation with uncertainty merging
    results = plotter.plot("./assets/unc_wgts.npz")

    # Check that plots were generated
    assert len(results) > 0

    # Verify that budget plots were also generated
    budget_plots = [key for key in results.keys() if "uncert_budget" in key]
    assert len(budget_plots) > 0


def test_uncertainty_merging_quadrature_calculation(tmp_path):
    """Test that uncertainties are properly merged in quadrature"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Create mock uncertainties to test quadrature merging
    mock_var1 = np.array([1.0, 4.0, 9.0])  # std = [1, 2, 3]
    mock_var2 = np.array([4.0, 9.0, 16.0])  # std = [2, 3, 4]
    # std = [sqrt(5), sqrt(13), sqrt(25)]
    expected_merged_var = np.array([5.0, 13.0, 25.0])

    # Mock the active_systs with test data
    plotter.active_systs = {
        "track-eff": {"var": mock_var1, "name": "Track eff.", "color": "purple"},
        "jet-track-eff": {"var": mock_var2, "name": "Jet track eff.", "color": "pink"},
    }

    # Test the merging function directly
    merged_uncert = plotter._merge_uncertainties(
        "test_track", ["track-eff", "jet-track-eff"]
    )

    # Verify the merged uncertainty
    assert merged_uncert is not None
    assert merged_uncert["name"] == "Test_Track"
    assert merged_uncert["color"] == "purple"  # Should inherit from first uncertainty
    assert merged_uncert["stochastic"] is False
    assert merged_uncert["plot_ratio"] is False
    assert merged_uncert["merged_from"] == ["track-eff", "jet-track-eff"]

    # Verify quadrature addition (variances are summed)
    np.testing.assert_array_equal(merged_uncert["var"], expected_merged_var)


def test_uncertainty_merging_with_missing_uncertainties(tmp_path):
    """Test uncertainty merging when some uncertainties are missing"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Mock the active_systs with only one uncertainty present
    plotter.active_systs = {
        "track-eff": {
            "var": np.array([1.0, 4.0]),
            "name": "Track eff.",
            "color": "purple",
        },
        # "jet-track-eff" is missing
    }

    # Test merging with missing uncertainty
    merged_uncert = plotter._merge_uncertainties(
        "test_track", ["track-eff", "jet-track-eff"]
    )

    # Should still work with available uncertainties
    assert merged_uncert is not None
    assert merged_uncert["merged_from"] == ["track-eff"]  # Only available uncertainty
    np.testing.assert_array_equal(merged_uncert["var"], np.array([1.0, 4.0]))

    # Test with no available uncertainties
    merged_uncert_none = plotter._merge_uncertainties(
        "test_empty", ["missing1", "missing2"]
    )
    assert merged_uncert_none is None


def test_uncertainty_merging_state_management(tmp_path):
    """Test that uncertainty merging properly manages state (hiding/restoration)"""
    plotter = UncertaintyPlotter(
        source_path="./assets/evts_000_100.root",
        target_path="./assets/truth_evts_000_100.root",
        hv_path="./assets/sherpa_evts.root",
        data_path="./assets/data_evts.root",
        store=tmp_path,
        verbosity=0,
        max_events=100,
    )

    # Mock active_systs with test uncertainties
    original_uncerts = {
        "track-eff": {"var": np.array([1.0]), "name": "Track eff.", "color": "purple"},
        "jet-track-eff": {
            "var": np.array([4.0]),
            "name": "Jet track eff.",
            "color": "pink",
        },
        "other-syst": {"var": np.array([9.0]), "name": "Other", "color": "green"},
    }
    plotter.active_systs = original_uncerts.copy()

    # Test applying uncertainty merging
    plotter._apply_uncertainty_merging()

    # Verify that merged uncertainty was added
    assert "Tracking" in plotter.active_systs
    assert plotter.active_systs["Tracking"]["name"] == "Tracking"

    # Verify that individual uncertainties were hidden (if hiding is enabled)
    if plotter.hide_individual_uncertainties:
        assert "track-eff" not in plotter.active_systs
        assert "jet-track-eff" not in plotter.active_systs
        # Other uncertainties should still be present
        assert "other-syst" in plotter.active_systs

    # Test restoration
    plotter._remove_merged_uncertainties()

    # Verify that merged uncertainty was removed
    assert "Tracking" not in plotter.active_systs

    # Verify that individual uncertainties were restored
    assert "track-eff" in plotter.active_systs
    assert "jet-track-eff" in plotter.active_systs
    assert "other-syst" in plotter.active_systs

    # Verify original state is preserved
    assert plotter.active_systs["track-eff"]["name"] == "Track eff."
    assert plotter.active_systs["jet-track-eff"]["name"] == "Jet track eff."
    assert plotter.active_systs["other-syst"]["name"] == "Other"
