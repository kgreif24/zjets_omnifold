"""Pytest configuration and fixtures for top_subtraction tests."""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock
import uproot
import awkward as ak


@pytest.fixture
def sample_kinematics():
    """Create sample kinematics data for testing."""
    # Create sample data with shape (n_events, n_objects, n_features)
    n_events = 100
    n_objects = 10
    n_features = 10

    kinematics = np.random.randn(n_events, n_objects, n_features).astype(np.float32)
    return kinematics


@pytest.fixture
def sample_indices():
    """Create sample object indices for testing."""
    n_events = 100
    n_objects = 10

    indices = np.random.randint(0, 5, size=(n_events, n_objects)).astype(np.int32)
    return indices


@pytest.fixture
def sample_observables():
    """Create sample observables data for testing."""
    n_events = 100
    n_obs = 5

    observables = np.random.randn(n_events, n_obs).astype(np.float32)
    return observables


@pytest.fixture
def sample_pdgids():
    """Create sample pdgids data for testing."""
    n_events = 100
    n_pdgids = 10
    pdgids = np.random.choice([13, 211, -999], size=(n_events, n_pdgids)).astype(
        np.int32
    )
    return pdgids


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    n_events = 100
    labels = np.random.randint(0, 2, size=(n_events, 1)).astype(np.float32)
    return labels


@pytest.fixture
def sample_weights():
    """Create sample weights for testing."""
    n_events = 100
    weights = np.random.uniform(0.5, 2.0, size=(n_events, 1)).astype(np.float32)
    return weights


@pytest.fixture
def mock_root_file():
    """Create a mock ROOT file for testing."""
    mock_file = Mock()
    mock_tree = Mock()

    # Mock tree data
    n_events = 1000
    mock_tree.num_entries = n_events
    mock_tree.keys.return_value = [
        "pass190",
        "isTop",
        "isTop_logit",
        "weight",
        "Ntracks",
        "HT_tracks",
        "pT_ll",
        "pT_trackj1",
        "pT_trackj2",
        "y_ll",
        "m_trackj1",
        "m_trackj2",
        "tau1_trackj1",
        "tau2_trackj1",
        "tau1_trackj2",
        "tau2_trackj2",
    ]

    # Mock branch data
    mock_tree["pass190"].array.return_value = ak.Array(
        np.random.randint(0, 2, n_events)
    )
    mock_tree["isTop"].array.return_value = ak.Array(np.random.randint(0, 2, n_events))
    mock_tree["isTop_logit"].array.return_value = ak.Array(np.random.randn(n_events))
    mock_tree["weight"].array.return_value = ak.Array(
        np.random.uniform(0.5, 2.0, n_events)
    )
    mock_tree["Ntracks"].array.return_value = ak.Array(
        np.random.randint(0, 200, n_events)
    )
    mock_tree["HT_tracks"].array.return_value = ak.Array(
        np.random.uniform(0, 2000, n_events)
    )
    mock_tree["pT_ll"].array.return_value = ak.Array(
        np.random.uniform(0, 500, n_events)
    )
    mock_tree["pT_trackj1"].array.return_value = ak.Array(
        np.random.uniform(0, 1000, n_events)
    )
    mock_tree["pT_trackj2"].array.return_value = ak.Array(
        np.random.uniform(0, 500, n_events)
    )
    mock_tree["y_ll"].array.return_value = ak.Array(np.random.uniform(-3, 3, n_events))
    mock_tree["m_trackj1"].array.return_value = ak.Array(
        np.random.uniform(0, 150, n_events)
    )
    mock_tree["m_trackj2"].array.return_value = ak.Array(
        np.random.uniform(0, 150, n_events)
    )
    mock_tree["tau1_trackj1"].array.return_value = ak.Array(
        np.random.uniform(0, 0.9, n_events)
    )
    mock_tree["tau2_trackj1"].array.return_value = ak.Array(
        np.random.uniform(0, 0.9, n_events)
    )
    mock_tree["tau1_trackj2"].array.return_value = ak.Array(
        np.random.uniform(0, 0.9, n_events)
    )
    mock_tree["tau2_trackj2"].array.return_value = ak.Array(
        np.random.uniform(0, 0.9, n_events)
    )

    mock_file.__getitem__.return_value = mock_tree
    mock_file.__enter__ = Mock(return_value=mock_file)
    mock_file.__exit__ = Mock(return_value=None)

    return mock_file, mock_tree


@pytest.fixture
def temp_root_file():
    """Create a temporary ROOT file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    # Create a simple ROOT file with test data
    n_events = 1000
    data = {
        "pass190": np.random.randint(0, 2, n_events),
        "isTop": np.random.randint(0, 2, n_events),
        "isTop_logit": np.random.randn(n_events),
        "weight": np.random.uniform(0.5, 2.0, n_events),
        "Ntracks": np.random.randint(0, 200, n_events),
        "HT_tracks": np.random.uniform(0, 2000, n_events),
        "pT_ll": np.random.uniform(0, 500, n_events),
        "pT_trackj1": np.random.uniform(0, 1000, n_events),
        "pT_trackj2": np.random.uniform(0, 500, n_events),
        "y_ll": np.random.uniform(-3, 3, n_events),
        "m_trackj1": np.random.uniform(0, 150, n_events),
        "m_trackj2": np.random.uniform(0, 150, n_events),
        "tau1_trackj1": np.random.uniform(0, 0.9, n_events),
        "tau2_trackj1": np.random.uniform(0, 0.9, n_events),
        "tau1_trackj2": np.random.uniform(0, 0.9, n_events),
        "tau2_trackj2": np.random.uniform(0, 0.9, n_events),
    }

    with uproot.recreate(tmp_path) as f:
        f["OmniTree"] = data

    yield tmp_path

    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def temp_npz_file():
    """Create a temporary NPZ file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    # Create test weights
    n_events = 1000
    weights = np.random.uniform(0.8, 1.2, n_events)

    np.savez(tmp_path, pd_weights=weights)

    yield tmp_path

    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.load_from_checkpoint.return_value = model
    model.eval.return_value = None
    model.to.return_value = model
    model.device = torch.device("cpu")
    return model


@pytest.fixture
def mock_trainer():
    """Create a mock trainer for testing."""
    trainer = Mock()
    trainer.predict.return_value = [torch.randn(100, 1) for _ in range(5)]
    trainer.fit.return_value = None
    trainer.checkpoint_callback = Mock()
    trainer.checkpoint_callback.best_model_path = "/fake/path/checkpoint.ckpt"
    return trainer


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    dataloader = Mock()
    # Create mock batches
    batches = []
    for i in range(5):
        batch = (
            torch.randn(100, 10),  # features
            torch.randint(0, 2, (100, 1)).float(),  # labels
            torch.ones(100, 1),  # weights
        )
        batches.append(batch)

    dataloader.__iter__ = Mock(return_value=iter(batches))
    return dataloader
