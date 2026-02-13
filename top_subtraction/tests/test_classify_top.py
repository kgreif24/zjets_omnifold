"""Unit tests for classify_top.py module."""

import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify_top import (  # noqa: E402
    ModelConfig,
    TrainingConfig,
    load_and_filter_data,
    create_datasets,
    create_data_loaders,
    create_model,
    setup_trainer,
    main,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        """Test that ModelConfig has correct default values."""
        config = ModelConfig()

        assert config.input_dim == 10
        assert config.seed == 42
        assert config.pair_input_dim == 4
        assert config.num_classes == 1
        assert config.num_heads == 8
        assert config.num_layers == 6
        assert config.num_cls_layers == 2
        assert config.fc_dropout == 0.0
        assert config.activation == "gelu"
        assert config.remove_self_pair is False
        assert config.max_lr == 1e-5
        assert config.min_lr == 1e-6

    def test_post_init_defaults(self):
        """Test that __post_init__ sets correct default values."""
        config = ModelConfig()

        assert config.embed_dims == [128, 512, 128]
        assert config.pair_embed_dims == [64, 64, 64]
        assert config.fc_nodes == [256, 256]

    def test_custom_values(self):
        """Test that custom values are preserved."""
        config = ModelConfig(
            input_dim=20,
            seed=123,
            embed_dims=[256, 1024, 256],
            fc_nodes=[512, 512, 256],
        )

        assert config.input_dim == 20
        assert config.seed == 123
        assert config.embed_dims == [256, 1024, 256]
        assert config.fc_nodes == [512, 512, 256]


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test that TrainingConfig has correct default values."""
        config = TrainingConfig()

        assert config.batch_size == 256
        assert config.max_epochs == 70
        assert config.num_nodes == 1
        assert config.devices == 4
        assert config.test_split == 0.2
        assert config.patience == 20
        assert config.n_jets == 5
        assert config.max_tracks == 264


class TestLoadAndFilterData:
    """Test load_and_filter_data function."""

    @patch("classify_top.uproot.open")
    @patch("classify_top.du.get_kinematics")
    @patch("classify_top.du.get_observables")
    def test_load_pseudodata(
        self, mock_get_observables, mock_get_kinematics, mock_open
    ):
        """Test loading pseudodata with filtering."""
        # Setup mocks
        mock_file = Mock()
        mock_tree = Mock()
        mock_open.return_value = mock_file
        mock_file.__getitem__ = Mock(return_value=mock_tree)

        # Mock tree data
        n_events = 1000
        # Create consistent pass190 array with some events passing
        pass190_array = np.zeros(n_events).astype(bool)
        pass190_array[:500] = 1  # First 500 events pass
        is_top_array = np.zeros(n_events).astype(bool)
        is_top_array[:250] = 1  # First 250 events are top
        expected_pass_array = np.logical_and(pass190_array, ~is_top_array)

        mock_pass190 = Mock()
        mock_isTop = Mock()
        mock_pass190.array.return_value = pass190_array
        mock_isTop.array.return_value = is_top_array
        mock_tree.__getitem__ = Mock(
            side_effect=lambda key: {"pass190": mock_pass190, "isTop": mock_isTop}[key]
        )

        # Mock kinematics and observables - these should match the filtered data size
        # After filtering, we'll have 500 - 250 = 250 events (pass190=1 and isTop=0)
        # The kinematics and observables should be the size of events that pass pass190
        n_passing_events = 500  # events where pass190 == 1
        mock_kinematics = np.random.randn(n_passing_events, 10, 10)
        mock_indices = np.random.randint(0, 5, (n_passing_events, 10))
        mock_pdgids = 211 * np.ones_like(mock_indices)
        # With the fix: np.stack(axis=1) creates (n_passing_events, 3) shape
        # This matches what the filtering expects
        mock_observables = np.random.randn(n_passing_events, 3)

        mock_get_kinematics.return_value = (mock_kinematics, mock_indices, mock_pdgids)
        # get_observables returns a stacked numpy array of shape
        # (n_passing_events, n_observables)
        mock_get_observables.return_value = mock_observables

        # Test function
        result = load_and_filter_data("test_file.root", is_pseudodata=True)

        # Verify calls
        mock_open.assert_called_once_with("test_file.root")

        # Verify get_kinematics call - use call_args to handle numpy array comparison
        mock_get_kinematics.assert_called_once()
        kin_call_args, kin_call_kwargs = mock_get_kinematics.call_args
        assert kin_call_args[0] is mock_tree
        np.testing.assert_array_equal(kin_call_args[1], expected_pass_array)
        assert kin_call_kwargs == {"muon_only": False}

        # Verify get_observables call - use call_args to handle numpy array comparison
        mock_get_observables.assert_called_once()
        obs_call_args, obs_call_kwargs = mock_get_observables.call_args
        assert obs_call_args[0] is mock_tree
        assert obs_call_args[1] == ["Ntracks", "HT_tracks", "pT_ll"]
        np.testing.assert_array_equal(obs_call_args[2], expected_pass_array)

        # Verify return values - function returns 4 values
        kinematics, indices, observables, pdgids = result
        assert isinstance(kinematics, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert isinstance(observables, np.ndarray)
        assert isinstance(pdgids, np.ndarray)

    @patch("classify_top.uproot.open")
    @patch("classify_top.du.get_kinematics")
    @patch("classify_top.du.get_observables")
    def test_load_top_data(self, mock_get_observables, mock_get_kinematics, mock_open):
        """Test loading top MC data (not pseudodata)."""
        # Setup mocks
        mock_file = Mock()
        mock_tree = Mock()
        mock_open.return_value = mock_file
        mock_file.__getitem__ = Mock(return_value=mock_tree)

        # Mock tree data
        n_events = 1000
        # Create consistent pass190 array with some events passing
        pass190_array = np.zeros(n_events).astype(bool)
        pass190_array[:600] = 1  # First 600 events pass

        mock_pass190 = Mock()
        mock_pass190.array.return_value = pass190_array
        mock_tree.__getitem__ = Mock(
            side_effect=lambda key: {"pass190": mock_pass190}[key]
        )

        # Mock kinematics and observables
        n_passing_events = 600  # events where pass190 == 1
        mock_kinematics = np.random.randn(n_passing_events, 10, 10)
        mock_indices = np.random.randint(0, 5, (n_passing_events, 10))
        mock_pdgids = 211 * np.ones_like(mock_indices)
        # get_observables returns a stacked numpy array of shape
        # (n_passing_events, n_observables)
        mock_observables = np.random.randn(n_passing_events, 3)

        mock_get_kinematics.return_value = (mock_kinematics, mock_indices, mock_pdgids)
        mock_get_observables.return_value = mock_observables

        # Test function
        result = load_and_filter_data("test_file.root", is_pseudodata=False)

        # Verify calls
        mock_open.assert_called_once_with("test_file.root")

        # Verify get_kinematics call - use call_args to handle numpy array comparison
        mock_get_kinematics.assert_called_once()
        kin_call_args, kin_call_kwargs = mock_get_kinematics.call_args
        assert kin_call_args[0] is mock_tree
        np.testing.assert_array_equal(kin_call_args[1], pass190_array)
        assert kin_call_kwargs == {"muon_only": False}

        # Verify get_observables call - use call_args to handle numpy array comparison
        mock_get_observables.assert_called_once()
        obs_call_args, obs_call_kwargs = mock_get_observables.call_args
        assert obs_call_args[0] is mock_tree
        assert obs_call_args[1] == ["Ntracks", "HT_tracks", "pT_ll"]
        np.testing.assert_array_equal(obs_call_args[2], pass190_array)

        # Verify return values - function returns 4 values
        kinematics, indices, observables, pdgids = result
        assert isinstance(kinematics, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert isinstance(observables, np.ndarray)
        assert isinstance(pdgids, np.ndarray)


class TestCreateDatasets:
    """Test create_datasets function."""

    def test_create_datasets(
        self, sample_kinematics, sample_indices, sample_observables, sample_pdgids
    ):
        """Test dataset creation with balanced weights."""
        # Create test data
        pd_data = (
            sample_kinematics[:50],
            sample_indices[:50],
            sample_observables[:50],
            sample_pdgids[:50],
        )
        top_data = (
            sample_kinematics[50:],
            sample_indices[50:],
            sample_observables[50:],
            sample_pdgids[50:],
        )

        config = TrainingConfig()

        # Test function
        pd_dataset, top_dataset = create_datasets(pd_data, top_data, config)

        # Verify datasets are created
        assert pd_dataset is not None
        assert top_dataset is not None

        # Verify dataset properties
        assert len(pd_dataset) == 50
        assert len(top_dataset) == 50

        # Verify labels
        assert np.all(pd_dataset.labels.numpy() == 0)  # Pseudodata should be 0
        assert np.all(top_dataset.labels.numpy() == 1)  # Top should be 1


class TestCreateDataLoaders:
    """Test create_data_loaders function."""

    def test_create_data_loaders(self):
        """Test data loader creation with train/test split."""
        # Create a mock combined dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        # Mock the random_split
        with patch("classify_top.random_split") as mock_split:
            train_dataset = Mock()
            test_dataset = Mock()
            train_dataset.__len__ = Mock(return_value=80)
            test_dataset.__len__ = Mock(return_value=20)
            mock_split.return_value = (train_dataset, test_dataset)

            config = TrainingConfig()

            # Test function
            train_loader, test_loader = create_data_loaders(mock_dataset, config)

            # Verify loaders are created
            assert train_loader is not None
            assert test_loader is not None

            # Verify split was called correctly
            mock_split.assert_called_once()
            call_args = mock_split.call_args
            assert call_args[0][0] == mock_dataset
            assert call_args[0][1] == [80, 20]  # train_size, test_size


class TestCreateModel:
    """Test create_model function."""

    @patch("classify_top.LOfTransformer")
    def test_create_model(self, mock_transformer):
        """Test model creation with correct parameters."""
        config = ModelConfig()

        # Test function
        create_model(config)

        # Verify LOfTransformer was called with correct parameters
        mock_transformer.assert_called_once()
        call_args = mock_transformer.call_args

        # Check key parameters
        assert call_args[1]["input_dim"] == config.input_dim
        assert call_args[1]["seed"] == config.seed
        assert call_args[1]["pair_input_dim"] == config.pair_input_dim
        assert call_args[1]["embed_dims"] == config.embed_dims
        assert call_args[1]["num_classes"] == config.num_classes
        assert call_args[1]["num_heads"] == config.num_heads
        assert call_args[1]["num_layers"] == config.num_layers
        assert call_args[1]["no_w1"] is True  # Should be True for classification


class TestSetupTrainer:
    """Test setup_trainer function."""

    @patch("classify_top.L.Trainer")
    @patch("classify_top.L.pytorch.callbacks.ModelCheckpoint")
    @patch("classify_top.L.pytorch.callbacks.EarlyStopping")
    @patch("classify_top.WandbLogger")
    def test_setup_trainer(
        self, mock_logger, mock_early_stopping, mock_checkpoint, mock_trainer
    ):
        """Test trainer setup with correct callbacks and logger."""
        run_name = "test_run"
        config = TrainingConfig()

        # Test function
        setup_trainer(run_name, config)

        # Verify callbacks were created
        mock_checkpoint.assert_called_once()
        mock_early_stopping.assert_called_once()

        # Verify logger was created
        mock_logger.assert_called_once()
        logger_call_args = mock_logger.call_args
        assert logger_call_args[1]["project"] == "top-subtraction"
        assert logger_call_args[1]["group"] == "classifier"
        assert logger_call_args[1]["name"] == run_name

        # Verify trainer was created
        mock_trainer.assert_called_once()
        trainer_call_args = mock_trainer.call_args
        assert trainer_call_args[1]["accelerator"] == "auto"
        assert trainer_call_args[1]["num_nodes"] == config.num_nodes
        assert trainer_call_args[1]["devices"] == config.devices
        assert trainer_call_args[1]["max_epochs"] == config.max_epochs


class TestMain:
    """Test main function."""

    @patch("classify_top.setup_trainer")
    @patch("classify_top.create_model")
    @patch("classify_top.create_data_loaders")
    @patch("classify_top.create_datasets")
    @patch("classify_top.load_and_filter_data")
    @patch("classify_top.rank_zero_info")
    @patch(
        "sys.argv",
        [
            "classify_top.py",
            "--data_path",
            "test_pd.root",
            "--top_path",
            "test_top.root",
            "--run_name",
            "test_run",
        ],
    )
    def test_main_function(
        self,
        mock_info,
        mock_load_data,
        mock_create_datasets,
        mock_create_loaders,
        mock_create_model,
        mock_setup_trainer,
    ):
        """Test main function execution."""
        # Setup mocks
        mock_pd_data = (
            np.random.randn(100, 10, 10),
            np.random.randint(0, 5, (100, 10)),
            np.random.randn(100, 3),
        )
        mock_top_data = (
            np.random.randn(100, 10, 10),
            np.random.randint(0, 5, (100, 10)),
            np.random.randn(100, 3),
        )
        mock_load_data.side_effect = [mock_pd_data, mock_top_data]

        mock_pd_dataset = Mock()
        mock_top_dataset = Mock()
        mock_create_datasets.return_value = (mock_pd_dataset, mock_top_dataset)

        mock_pd_dataset.concatenate = Mock()

        mock_train_loader = Mock()
        mock_test_loader = Mock()
        mock_create_loaders.return_value = (mock_train_loader, mock_test_loader)

        mock_model = Mock()
        mock_create_model.return_value = mock_model

        mock_trainer = Mock()
        mock_setup_trainer.return_value = mock_trainer

        # Test function
        main()

        # Verify function calls
        assert mock_load_data.call_count == 2
        mock_create_datasets.assert_called_once()
        mock_create_loaders.assert_called_once()
        mock_create_model.assert_called_once()
        mock_setup_trainer.assert_called_once()
        mock_trainer.fit.assert_called_once_with(
            mock_model, mock_train_loader, mock_test_loader
        )
