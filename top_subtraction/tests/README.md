# Top Subtraction Tests

This directory contains comprehensive unit tests for the top subtraction neural network scripts.

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_classify_top.py` - Tests for the top classifier training script
- `test_eval_classify_top.py` - Tests for the top classifier evaluation script
- `test_subtract_top.py` - Tests for the top subtraction script

## Running Tests

To run all tests:
```bash
cd top_subtraction
python -m pytest tests/ -v
```

To run specific test files:
```bash
python -m pytest tests/test_classify_top.py -v
python -m pytest tests/test_eval_classify_top.py -v
python -m pytest tests/test_subtract_top.py -v
```

To run tests with coverage:
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Unit Tests
- **Model Configuration**: Test dataclass configurations and parameter validation
- **Data Loading**: Test ROOT file loading and data preprocessing
- **Model Architecture**: Test neural network components and forward passes
- **Training Logic**: Test training steps, validation, and optimization
- **Utility Functions**: Test helper functions and data transformations

### Integration Tests
- **End-to-End Workflows**: Test complete training and evaluation pipelines
- **Data Consistency**: Test data flow between components
- **Error Handling**: Test graceful handling of edge cases and errors

### Mock Testing
- **External Dependencies**: Mock ROOT files, PyTorch Lightning, and other external libraries
- **File I/O**: Test file operations with temporary files
- **Network Operations**: Mock model loading and checkpointing

## Test Fixtures

The tests use several pytest fixtures defined in `conftest.py`:

- `sample_kinematics` - Mock kinematics data
- `sample_indices` - Mock object indices
- `sample_observables` - Mock observables data
- `sample_labels` - Mock labels
- `sample_weights` - Mock weights
- `mock_root_file` - Mock ROOT file and tree
- `temp_root_file` - Temporary ROOT file for testing
- `temp_npz_file` - Temporary NPZ file for testing
- `mock_model` - Mock neural network model
- `mock_trainer` - Mock PyTorch Lightning trainer
- `mock_dataloader` - Mock data loader

## Test Utilities

The test utilities are provided through pytest fixtures in `conftest.py`:

- **Data Generation**: Mock data generation for kinematics, indices, observables, labels, and weights
- **Mock Objects**: Functions to create mock models, trainers, and dataloaders
- **File I/O**: Temporary file creation and cleanup utilities

## Key Test Scenarios

### Classify Top Tests
- Configuration validation
- Data loading and filtering
- Dataset creation with balanced weights
- Model architecture and training setup
- Complete training workflow

### Eval Classify Top Tests
- Command line argument parsing
- Model loading and evaluation
- ROOT file copying with predictions
- Plotting functionality for pseudodata
- Error handling for missing files

### Subtract Top Tests
- Neural network architecture
- Training and validation steps
- Weight prediction and bootstrapping
- Data preprocessing (exit code replacement)
- Integration with PyTorch Lightning


## Dependencies

The tests require the following packages:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `numpy` - Numerical computations
- `torch` - PyTorch for neural networks
- `matplotlib` - Plotting (for plotting tests)
- `uproot` - ROOT file handling
- `awkward` - Array handling
- `unittest.mock` - Mocking framework

## Continuous Integration

These tests are designed to run in CI/CD pipelines and should:
- Complete in reasonable time (< 5 minutes)
- Not require external data files
- Use mocked external dependencies
- Provide clear error messages
- Have good coverage of critical functionality

## Adding New Tests

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Mock external dependencies
4. Test both success and failure cases
5. Include docstrings explaining test purpose
6. Use descriptive test names
7. Group related tests in classes

## Troubleshooting

Common issues and solutions:

- **Import Errors**: Ensure the parent directory is in the Python path
- **Mock Issues**: Check that mocks are properly configured for the test
- **File Permissions**: Ensure temporary files can be created and deleted
- **Memory Issues**: Use smaller test datasets for memory-constrained environments
- **Timing Issues**: Use fixed seeds for reproducible tests
