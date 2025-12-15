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
        normalize_targets=False,
    )
    start, end = plotter.wasserstein_distance(
        "weight",
        "./assets/wgts.npz",
        np.ones(100),
    )
    assert np.isclose(start, 375.650, atol=0.1)
    assert np.isclose(end, 402.671, atol=0.1)


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

    # Call plot method to trigger kinematic cuts application
    _ = plotter.plot(
        "weight",
        "./assets/wgts.npz",
        np.ones(100),
    )

    # Now test that kinematic cuts were applied
    # Get pT_ll data after cuts have been applied
    source_pT_ll = plotter._get_data("pT_ll", is_target=False)
    source_pT_trackj2 = plotter._get_data("pT_trackj2", is_target=False)
    target_pT_ll = plotter._get_data("pT_ll", is_target=True)
    target_pT_trackj2 = plotter._get_data("pT_trackj2", is_target=True)

    # Check the cut worked (region 1: pT_ll > 350, pT_trackj2 > 50)
    if len(source_pT_ll) > 0:
        assert np.all(source_pT_ll > 350)
    if len(source_pT_trackj2) > 0:
        assert np.all(source_pT_trackj2 > 50)
    if len(target_pT_ll) > 0:
        assert np.all(target_pT_ll > 350)
    if len(target_pT_trackj2) > 0:
        assert np.all(target_pT_trackj2 > 50)


def test_ibu_bins(tmp_path):
    """Test IBU binning mode (lines 498-511 in plotter.py)"""
    plotter = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=0,
        ibu_bins=True,
    )

    # Test that IBU binning is enabled
    assert plotter.ibu_bins is True

    # Test basic functionality with IBU bins
    _ = plotter.plot(
        "weight",
        "./assets/wgts.npz",
        np.ones(100),
    )


def test_use_truth(tmp_path):
    """Test truth-level data access with use_truth=True"""
    # Test with use_truth=True
    plotter_truth = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=0,
        use_truth=True,
    )

    # Test with use_truth=False
    plotter_reco = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=0,
        use_truth=False,
    )

    # Test accessing truth-level data
    truth_pT_ll = plotter_truth._get_data("pT_ll", is_target=False)
    truth_pT_trackj2 = plotter_truth._get_data("pT_trackj2", is_target=False)

    # Verify we get reco data
    reco_pT_ll = plotter_reco._get_data("pT_ll", is_target=False)
    reco_pT_trackj2 = plotter_reco._get_data("pT_trackj2", is_target=False)

    # Truth and reco may have different lengths due to different filtering
    # Just verify that we get data from both
    assert len(truth_pT_ll) > 0
    assert len(truth_pT_trackj2) > 0
    assert len(reco_pT_ll) > 0
    assert len(reco_pT_trackj2) > 0


def test_pdf_output(tmp_path):
    """Test PDF file generation with use_pdf=True"""
    plotter = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=0,
        use_pdf=True,
    )

    # Generate plots with PDF output
    _ = plotter.plot(
        "weight",
        "./assets/wgts.npz",
        np.ones(100),
    )

    # Check that PDF files were created
    pdf_files = [f for f in os.listdir(tmp_path) if f.endswith(".pdf")]
    assert len(pdf_files) > 0


def test_max_events(tmp_path):
    """Test event truncation with max_events parameter"""
    plotter = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=0,
        max_events=50,
    )

    # Test that max_events is set correctly
    assert plotter.max_events == 50

    # Test basic functionality with limited events
    _ = plotter.plot(
        "weight",
        "./assets/wgts.npz",
        np.ones(50),  # Match the max_events
    )


def test_histogram_normalization(tmp_path):
    """Test _normalize_to method (lines 436-448)"""
    plotter = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=0,
    )

    # Test histogram normalization
    # Create some mock histogram data
    hist_data = np.random.uniform(0, 100, 50)
    target_sum = 1000.0

    # Test normalization
    normalized_data = plotter._normalize_to(hist_data, target_sum)

    # Check that normalization worked
    assert np.isclose(np.sum(normalized_data), target_sum, rtol=1e-6)


def test_get_bins_for_plot(tmp_path):
    """Test bin edge calculation (lines 487-511)"""
    plotter = Plotter(
        "./assets/evts_000_100.root",
        "./assets/evts_100_200.root",
        tmp_path,
        labels=("Test1", "Test2"),
        verbosity=0,
    )

    # Test bin calculation for different variables
    # The method expects a plot dictionary, not a string
    plot_dict_pT_ll = {"binlow": 0, "binhigh": 1000, "nbins": 50}
    plot_dict_pT_trackj2 = {"binlow": 0, "binhigh": 200, "nbins": 30}

    bins_pT_ll = plotter._get_bins_for_plot(plot_dict_pT_ll)
    bins_pT_trackj2 = plotter._get_bins_for_plot(plot_dict_pT_trackj2)

    # Check that bins are reasonable
    assert len(bins_pT_ll) > 0
    assert len(bins_pT_trackj2) > 0
    assert np.all(np.diff(bins_pT_ll) > 0)  # Bins should be increasing
    assert np.all(np.diff(bins_pT_trackj2) > 0)
