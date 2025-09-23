"""Unit tests for eval_classify_top.py module."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_classify_top import parse_args, copy_root_with_predictions, main  # noqa: E402


class TestParseArgs:
    """Test parse_args function."""

    def test_parse_args_required(self):
        """Test parsing required arguments."""
        test_args = [
            "eval_classify_top.py",
            "--checkpoint_path",
            "test_checkpoint.ckpt",
            "--data_path",
            "test_data.root",
        ]

        with patch("sys.argv", test_args):
            args = parse_args()

            assert args.checkpoint_path == "test_checkpoint.ckpt"
            assert args.data_path == "test_data.root"
            assert args.is_pseudodata is False
            assert args.append_filename is None

    def test_parse_args_with_optional(self):
        """Test parsing with optional arguments."""
        test_args = [
            "eval_classify_top.py",
            "--checkpoint_path",
            "test_checkpoint.ckpt",
            "--data_path",
            "test_data.root",
            "--is_pseudodata",
            "--append_filename",
            "output_with_predictions.root",
        ]

        with patch("sys.argv", test_args):
            args = parse_args()

            assert args.checkpoint_path == "test_checkpoint.ckpt"
            assert args.data_path == "test_data.root"
            assert args.is_pseudodata is True
            assert args.append_filename == "output_with_predictions.root"


class TestCopyRootWithPredictions:
    """Test copy_root_with_predictions function."""

    @patch("eval_classify_top.uproot.recreate")
    @patch("eval_classify_top.uproot.open")
    def test_copy_root_with_predictions(self, mock_open, mock_recreate):
        """Test copying ROOT file with predictions."""
        # Setup mocks
        mock_input_file = Mock()
        mock_tree = Mock()
        mock_output_file = Mock()
        mock_output_file.__setitem__ = Mock()

        mock_open.return_value.__enter__ = Mock(return_value=mock_input_file)
        mock_input_file.__getitem__ = Mock(return_value=mock_tree)
        mock_recreate.return_value.__enter__ = Mock(return_value=mock_output_file)

        # Mock tree data
        n_entries = 1000
        mock_tree.num_entries = n_entries
        mock_tree.keys.return_value = ["pass190", "other_branch"]

        # Mock branch data
        pass190_data = np.random.randint(0, 2, n_entries)
        other_data = np.random.randn(n_entries)

        # Create mock branches
        mock_pass190_branch = Mock()
        mock_pass190_branch.array.return_value = pass190_data
        mock_other_branch = Mock()
        mock_other_branch.array.return_value = other_data

        # Mock tree subscripting
        def mock_tree_getitem(self, key):
            if key == "pass190":
                return mock_pass190_branch
            elif key == "other_branch":
                return mock_other_branch
            else:
                return Mock()

        mock_tree.__getitem__ = mock_tree_getitem

        # Test data
        input_path = "input.root"
        output_path = "output.root"
        predictions = np.random.randn(np.sum(pass190_data == 1))

        # Test function
        copy_root_with_predictions(input_path, output_path, predictions)

        # Verify file operations
        mock_open.assert_called_once_with(input_path)
        mock_recreate.assert_called_once_with(output_path)

        # Verify tree operations
        mock_tree.keys.assert_called_once()
        assert mock_pass190_branch.array.call_count == 1
        assert mock_other_branch.array.call_count == 1

        # Verify output file operations
        mock_output_file.__setitem__.assert_called_once()
        call_args = mock_output_file.__setitem__.call_args
        assert call_args[0][0] == "OmniTree"

        # Verify the data structure contains the new branch
        output_data = call_args[0][1]
        assert "isTop_logit" in output_data
        assert "pass190" in output_data
        assert "other_branch" in output_data

        # Verify isTop_logit branch structure
        isTop_logit = output_data["isTop_logit"]
        assert len(isTop_logit) == n_entries
        assert np.all(isTop_logit[pass190_data == 0] == np.mean(predictions))
        assert np.all(isTop_logit[pass190_data == 1] == predictions)


class TestMain:
    """Test main function."""

    @patch("eval_classify_top.copy_root_with_predictions")
    @patch("eval_classify_top.L.Trainer")
    @patch("eval_classify_top.LOfTransformer.load_from_checkpoint")
    @patch("eval_classify_top.uproot.open")
    @patch("eval_classify_top.du.get_kinematics")
    @patch("eval_classify_top.OfDataset")
    @patch("eval_classify_top.torch.utils.data.DataLoader")
    @patch(
        "sys.argv",
        [
            "eval_classify_top.py",
            "--checkpoint_path",
            "test.ckpt",
            "--data_path",
            "test.root",
        ],
    )
    def test_main_with_data(
        self,
        mock_dataloader,
        mock_dataset,
        mock_get_kinematics,
        mock_open,
        mock_load_checkpoint,
        mock_trainer,
        mock_copy_root,
    ):
        """Test main function with data (not pseudodata)."""
        # Setup mocks
        mock_model = Mock()
        mock_load_checkpoint.return_value = mock_model

        mock_file = Mock()
        mock_tree = Mock()
        mock_open.return_value = mock_file
        mock_file.__getitem__ = Mock(return_value=mock_tree)

        # Mock kinematics data
        mock_kinematics = np.random.randn(100, 10, 10)
        mock_indices = np.random.randint(0, 5, (100, 10))
        mock_get_kinematics.return_value = (mock_kinematics, mock_indices)

        # Mock dataset
        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance

        # Mock dataloader
        mock_dataloader_instance = Mock()
        mock_dataloader.return_value = mock_dataloader_instance

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.predict.return_value = [
            torch.randn(50, 1),
            torch.randn(50, 1),
        ]

        # Test function
        main()

        # Verify model loading
        mock_load_checkpoint.assert_called_once_with("test.ckpt")

        # Verify data loading
        mock_open.assert_called_once_with("test.root")
        mock_get_kinematics.assert_called_once_with(mock_tree, muon_only=False)

        # Verify dataset creation
        mock_dataset.assert_called_once()

        # Verify trainer setup and prediction
        mock_trainer.assert_called_once()
        mock_trainer_instance.predict.assert_called_once_with(
            mock_model, mock_dataloader_instance
        )

        # Verify no file copying (no append_filename)
        mock_copy_root.assert_not_called()

    @patch("eval_classify_top.plt.savefig")
    @patch("eval_classify_top.copy_root_with_predictions")
    @patch("eval_classify_top.L.Trainer")
    @patch("eval_classify_top.LOfTransformer.load_from_checkpoint")
    @patch("eval_classify_top.uproot.open")
    @patch("eval_classify_top.du.get_kinematics")
    @patch("eval_classify_top.OfDataset")
    @patch("eval_classify_top.torch.utils.data.DataLoader")
    @patch(
        "sys.argv",
        [
            "eval_classify_top.py",
            "--checkpoint_path",
            "test.ckpt",
            "--data_path",
            "test.root",
            "--is_pseudodata",
            "--append_filename",
            "output.root",
        ],
    )
    def test_main_with_pseudodata_and_append(
        self,
        mock_dataloader,
        mock_dataset,
        mock_get_kinematics,
        mock_open,
        mock_load_checkpoint,
        mock_trainer,
        mock_copy_root,
        mock_savefig,
    ):
        """Test main function with pseudodata and file appending."""
        # Setup mocks
        mock_model = Mock()
        mock_load_checkpoint.return_value = mock_model

        mock_file = Mock()
        mock_tree = Mock()
        mock_open.return_value = mock_file
        mock_file.__getitem__ = Mock(return_value=mock_tree)

        # Mock tree data for pseudodata
        mock_isTop_branch = Mock()
        mock_isTop_branch.array.return_value = np.random.randint(0, 2, 100)

        # Mock tree subscripting
        def mock_tree_getitem(self, key):
            if key == "isTop":
                return mock_isTop_branch
            else:
                return Mock()

        mock_tree.__getitem__ = mock_tree_getitem

        # Mock kinematics data
        mock_kinematics = np.random.randn(100, 10, 10)
        mock_indices = np.random.randint(0, 5, (100, 10))
        mock_get_kinematics.return_value = (mock_kinematics, mock_indices)

        # Mock dataset
        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance

        # Mock dataloader
        mock_dataloader_instance = Mock()
        mock_dataloader.return_value = mock_dataloader_instance

        # Mock trainer
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.predict.return_value = [
            torch.randn(50, 1),
            torch.randn(50, 1),
        ]

        # Test function
        main()

        # Verify model loading
        mock_load_checkpoint.assert_called_once_with("test.ckpt")

        # Verify data loading (multiple files when is_pseudodata=True)
        assert mock_open.call_count >= 1  # At least one call to open
        assert (
            mock_get_kinematics.call_count >= 1
        )  # Called at least once (twice when is_pseudodata=True)

        # Verify isTop data loading for pseudodata
        mock_isTop_branch.array.assert_called_once()

        # Verify dataset creation
        assert mock_dataset.call_count >= 1

        # Verify trainer setup and prediction
        mock_trainer.assert_called_once()
        assert (
            mock_trainer_instance.predict.call_count >= 1
        )  # Called at least once (twice when is_pseudodata=True)

        # Verify file copying
        mock_copy_root.assert_called_once()
        call_args = mock_copy_root.call_args
        assert call_args[0][0] == "test.root"
        assert call_args[0][1] == "output.root"
        # The third argument should be a numpy array (processed predictions)
        assert isinstance(call_args[0][2], np.ndarray)


class TestErrorHandling:
    """Test error handling in eval_classify_top module."""

    def test_missing_checkpoint_file(self):
        """Test behavior when checkpoint file doesn't exist."""
        with patch(
            "eval_classify_top.LOfTransformer.load_from_checkpoint"
        ) as mock_load:
            mock_load.side_effect = FileNotFoundError("Checkpoint not found")

            with patch(
                "sys.argv",
                [
                    "eval_classify_top.py",
                    "--checkpoint_path",
                    "nonexistent.ckpt",
                    "--data_path",
                    "test.root",
                ],
            ):
                with pytest.raises(FileNotFoundError):
                    main()

    def test_missing_data_file(self):
        """Test behavior when data file doesn't exist."""
        with patch(
            "eval_classify_top.LOfTransformer.load_from_checkpoint"
        ) as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            with patch("eval_classify_top.uproot.open") as mock_open:
                mock_open.side_effect = FileNotFoundError("Data file not found")

                with patch(
                    "sys.argv",
                    [
                        "eval_classify_top.py",
                        "--checkpoint_path",
                        "test.ckpt",
                        "--data_path",
                        "nonexistent.root",
                    ],
                ):
                    with pytest.raises(FileNotFoundError):
                        main()
