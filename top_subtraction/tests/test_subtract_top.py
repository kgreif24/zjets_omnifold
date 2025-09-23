"""Unit tests for subtract_top.py module."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subtract_top import (  # noqa: E402
    SimpleNN,
    SubtractTop,
    predict_weights,
    replace_exits,
    parse_args,
)


class TestSimpleNN:
    """Test SimpleNN neural network class."""

    def test_simple_nn_initialization(self):
        """Test SimpleNN initialization with default parameters."""
        model = SimpleNN()

        # Check that the model has the expected structure
        assert hasattr(model, "flatten")
        assert hasattr(model, "linear_relu_stack")

        # Check that flatten is a Flatten layer
        assert isinstance(model.flatten, torch.nn.Flatten)

        # Check that linear_relu_stack is a Sequential module
        assert isinstance(model.linear_relu_stack, torch.nn.Sequential)

        # Check the number of layers in the sequential module
        assert (
            len(model.linear_relu_stack) == 9
        )  # 3 Linear + 2 GELU + 2 BatchNorm1d + 2 Dropout

    def test_simple_nn_initialization_custom_params(self):
        """Test SimpleNN initialization with custom parameters."""
        input_dim = 5
        droprate = 0.2

        model = SimpleNN(input_dim=input_dim, droprate=droprate)

        # Check that the first linear layer has the correct input dimension
        first_linear = model.linear_relu_stack[0]
        assert isinstance(first_linear, torch.nn.Linear)
        assert first_linear.in_features == input_dim
        assert first_linear.out_features == 512

    def test_simple_nn_forward(self):
        """Test SimpleNN forward pass."""
        model = SimpleNN(input_dim=3)

        # Create test input
        batch_size = 10
        input_tensor = torch.randn(batch_size, 3)

        # Forward pass
        output = model(input_tensor)

        # Check output shape
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_simple_nn_different_input_sizes(self):
        """Test SimpleNN with different input sizes."""
        model = SimpleNN(input_dim=5)
        model.eval()  # Set to eval mode to avoid BatchNorm issues with batch size 1

        # Test with different batch sizes
        for batch_size in [1, 10, 100]:
            input_tensor = torch.randn(batch_size, 5)
            output = model(input_tensor)
            assert output.shape == (batch_size, 1)


class TestSubtractTop:
    """Test SubtractTop Lightning module."""

    def test_subtract_top_initialization(self):
        """Test SubtractTop initialization."""
        model = SubtractTop(input_dim=3)

        # Check that the model has the expected components
        assert hasattr(model, "model")
        assert hasattr(model, "loss_fn")

        # Check that model is a SimpleNN
        assert isinstance(model.model, SimpleNN)

        # Check that loss function is BCEWithLogitsLoss
        assert isinstance(model.loss_fn, torch.nn.BCEWithLogitsLoss)
        assert model.loss_fn.reduction == "none"

    def test_subtract_top_forward(self):
        """Test SubtractTop forward pass."""
        model = SubtractTop(input_dim=3)

        # Create test input
        batch_size = 10
        input_tensor = torch.randn(batch_size, 3)

        # Forward pass
        output = model(input_tensor)

        # Check output shape
        assert output.shape == (batch_size, 1)

    def test_subtract_top_training_step(self):
        """Test SubtractTop training step."""
        model = SubtractTop(input_dim=3)

        # Create test batch
        batch_size = 10
        x = torch.randn(batch_size, 3)
        y = torch.randint(0, 2, (batch_size, 1)).float()
        w = torch.ones(batch_size, 1)
        batch = (x, y, w)

        # Training step
        loss = model.training_step(batch, batch_idx=0)

        # Check that loss is a tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0  # Loss should be non-negative

    def test_subtract_top_validation_step(self):
        """Test SubtractTop validation step."""
        model = SubtractTop(input_dim=3)

        # Create test batch
        batch_size = 10
        x = torch.randn(batch_size, 3)
        y = torch.randint(0, 2, (batch_size, 1)).float()
        w = torch.ones(batch_size, 1)
        batch = (x, y, w)

        # Validation step
        loss = model.validation_step(batch, batch_idx=0)

        # Check that loss is a tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0  # Loss should be non-negative

    def test_subtract_top_predict_step(self):
        """Test SubtractTop predict step."""
        model = SubtractTop(input_dim=3)

        # Create test batch
        batch_size = 10
        x = torch.randn(batch_size, 3)
        y = torch.randint(0, 2, (batch_size, 1)).float()
        w = torch.ones(batch_size, 1)
        batch = (x, y, w)

        # Predict step
        output = model.predict_step(batch, batch_idx=0)

        # Check output shape
        assert output.shape == (batch_size, 1)

    def test_subtract_top_configure_optimizers(self):
        """Test SubtractTop optimizer configuration."""
        model = SubtractTop(input_dim=3)

        # Configure optimizers
        optimizer = model.configure_optimizers()

        # Check that optimizer is AdamW
        assert isinstance(optimizer, torch.optim.AdamW)

        # Check learning rate
        assert optimizer.param_groups[0]["lr"] == 1e-3

        # Check weight decay
        assert optimizer.param_groups[0]["weight_decay"] == 0.4


class TestPredictWeights:
    """Test predict_weights function."""

    def test_predict_weights(self, mock_model, mock_dataloader):
        """Test predict_weights function."""
        # Setup mock model
        mock_model.device = torch.device("cpu")
        mock_model.eval.return_value = None

        # Mock predictions
        mock_predictions = [torch.randn(50, 1), torch.randn(30, 1)]

        # Mock dataloader batches
        batches = []
        for i, pred in enumerate(mock_predictions):
            batch = (torch.randn(pred.shape[0], 10),)
            batches.append(batch)

        mock_dataloader.__iter__ = Mock(return_value=iter(batches))

        # Mock model forward pass - return predictions in order
        call_count = 0

        def mock_forward(x):
            nonlocal call_count
            result = mock_predictions[call_count]
            call_count += 1
            return result

        mock_model.side_effect = mock_forward

        # Test function
        weights = predict_weights(mock_model, mock_dataloader)

        # Verify results
        assert isinstance(weights, np.ndarray)
        assert len(weights) == sum(pred.numel() for pred in mock_predictions)
        assert np.all(weights > 0)  # Weights should be positive (exp of logits)

        # Verify model was set to eval mode
        mock_model.eval.assert_called_once()

    def test_predict_weights_empty_dataloader(self, mock_model):
        """Test predict_weights with empty dataloader."""
        mock_model.device = torch.device("cpu")
        mock_model.eval.return_value = None

        # Empty dataloader
        empty_dataloader = Mock()
        empty_dataloader.__iter__ = Mock(return_value=iter([]))

        # Test function
        weights = predict_weights(mock_model, empty_dataloader)

        # Should return empty array
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 0


class TestReplaceExits:
    """Test replace_exits function."""

    def test_replace_exits_no_exits(self):
        """Test replace_exits with no exit codes."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        original_data = data.copy()

        result = replace_exits(data)

        # Data should be unchanged
        np.testing.assert_array_equal(result, original_data)

    def test_replace_exits_with_exits(self):
        """Test replace_exits with exit codes."""
        data = np.array([[1.0, 2.0, 3.0], [-99.0, 5.0, 6.0], [7.0, -99.0, 9.0]])

        result = replace_exits(data)

        # Exit codes should be replaced with median
        assert not np.any(result == -99)

        # Check that medians are calculated correctly
        # Column 0: median of [1.0, 7.0] = 4.0
        # Column 1: median of [2.0, 5.0] = 3.5
        # Column 2: median of [3.0, 6.0, 9.0] = 6.0
        assert result[1, 0] == 4.0
        assert result[2, 1] == 3.5

    def test_replace_exits_all_exits(self):
        """Test replace_exits with all exit codes in a column."""
        data = np.array([[-99.0, 2.0], [-99.0, 5.0], [-99.0, 8.0]])

        result = replace_exits(data)

        # All exit codes should be replaced
        assert not np.any(result == -99)

        # Column 0 should have the same value for all entries
        # (0.0 when all are exit codes)
        assert result[0, 0] == result[1, 0] == result[2, 0] == 0.0


class TestParseArgs:
    """Test parse_args function."""

    def test_parse_args_required(self):
        """Test parsing required arguments."""
        test_args = [
            "subtract_top.py",
            "--data_path",
            "test_data.root",
            "--top_path",
            "test_top.root",
            "--output_file",
            "test_output.npz",
        ]

        with patch("sys.argv", test_args):
            args = parse_args()

            assert args.data_path == "test_data.root"
            assert args.top_path == "test_top.root"
            assert args.output_file == "test_output.npz"
            assert args.bootstrap is None
            assert args.split_seed == 42

    def test_parse_args_with_optional(self):
        """Test parsing with optional arguments."""
        test_args = [
            "subtract_top.py",
            "--data_path",
            "test_data.root",
            "--top_path",
            "test_top.root",
            "--output_file",
            "test_output.npz",
            "--bootstrap",
            "123",
            "--split_seed",
            "456",
        ]

        with patch("sys.argv", test_args):
            args = parse_args()

            assert args.data_path == "test_data.root"
            assert args.top_path == "test_top.root"
            assert args.output_file == "test_output.npz"
            assert args.bootstrap == 123
            assert args.split_seed == 456


class TestIntegration:
    """Integration tests for subtract_top module."""

    def test_model_training_workflow(self):
        """Test the complete model training workflow."""
        # Create model
        model = SubtractTop(input_dim=3)

        # Create sample data
        batch_size = 100
        x = torch.randn(batch_size, 3)
        y = torch.randint(0, 2, (batch_size, 1)).float()
        w = torch.ones(batch_size, 1)

        # Test training step
        loss = model.training_step((x, y, w), batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0

        # Test validation step
        val_loss = model.validation_step((x, y, w), batch_idx=0)
        assert isinstance(val_loss, torch.Tensor)
        assert val_loss >= 0

        # Test prediction step
        pred = model.predict_step((x, y, w), batch_idx=0)
        assert pred.shape == (batch_size, 1)

    def test_data_preprocessing_workflow(self):
        """Test data preprocessing functions together."""
        # Create test data with exit codes
        data = np.array([[1.0, 2.0, 3.0], [-99.0, 5.0, 6.0], [7.0, -99.0, 9.0]])

        # Test replace_exits
        cleaned_data = replace_exits(data)
        assert not np.any(cleaned_data == -99)

    def test_observables_consistency(self):
        """Test that observables are consistent with expected values."""
        # Test that the observables list matches the expected input dimension
        observables = [
            "isTop_logit",
            "HT_tracks",
            "Ntracks",
            "pT_trackj1",
            "m_trackj1",
        ]
        input_dim = len(observables)

        # Create model with this input dimension
        model = SubtractTop(input_dim=input_dim)

        # Test that model can handle this input size
        test_input = torch.randn(10, input_dim)
        output = model(test_input)
        assert output.shape == (10, 1)


class TestErrorHandling:
    """Test error handling in subtract_top module."""

    def test_invalid_input_dimensions(self):
        """Test behavior with invalid input dimensions."""
        # Test with negative input dimension
        with pytest.raises((ValueError, RuntimeError)):
            SimpleNN(input_dim=-1)

        # Test with zero input dimension
        with pytest.raises((ValueError, RuntimeError)):
            SimpleNN(input_dim=0)

    def test_invalid_batch_shapes(self):
        """Test behavior with invalid batch shapes."""
        model = SubtractTop(input_dim=3)

        # Test with wrong input shape
        x = torch.randn(10, 5)  # Wrong number of features
        y = torch.randint(0, 2, (10, 1)).float()
        w = torch.ones(10, 1)
        batch = (x, y, w)

        with pytest.raises((RuntimeError, ValueError)):
            model.training_step(batch, batch_idx=0)

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Test replace_exits with empty data
        empty_data = np.array([]).reshape(0, 3)
        result = replace_exits(empty_data)
        assert result.shape == empty_data.shape
