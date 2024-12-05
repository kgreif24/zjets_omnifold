"""
test_omnifold.py - Test suite for the Omnifolder class
"""

from cli.of_config import OfConfig
from omnifold import Omnifolder
import pytest

import numpy as np
import uproot
import awkward as ak


@pytest.mark.slow
def test_omnifold(tmp_path):

    # Create config object
    config = OfConfig(config_name="./assets/test_of.yml")

    # Overwrite the checkpoint dir with the tmp path
    config.mod_config("checkpoint_dir", tmp_path)

    # Write the config to a file
    config.create_template(template_path=f"{tmp_path}/test_of.yml")
    assert (tmp_path / "test_of.yml").exists()

    # Make omnifolder object
    of = Omnifolder(f"{tmp_path}/test_of.yml", use_slurm=False, index=1)
    of.run_of()

    # Check that we've gotten the correct file structure
    assert (tmp_path / "test-of" / "test-of-run_1").exists()
    assert (tmp_path / "test-of" / "test-of-run_1" / "weights").exists()
    assert (
        tmp_path / "test-of" / "test-of-run_1" / "weights" / "iteration_1_step_1.npz"
    ).exists()
    assert (
        tmp_path / "test-of" / "test-of-run_1" / "weights" / "iteration_1_step_2.npz"
    ).exists()
    assert (
        tmp_path / "test-of" / "test-of-run_1" / "weights" / "iteration_2_step_1.npz"
    ).exists()
    assert (
        tmp_path / "test-of" / "test-of-run_1" / "weights" / "iteration_2_step_2.npz"
    ).exists()
    assert (tmp_path / "test-of" / "test-of-run_1" / "pretrain_step_1").exists()
    assert (tmp_path / "test-of" / "test-of-run_1" / "pretrain_step_2").exists()
    assert (tmp_path / "test-of" / "test-of-run_1" / "iteration_1_step_1").exists()
    assert (
        tmp_path / "test-of" / "test-of-run_1" / "iteration_1_step_1" / "comp_plots"
    ).exists()
    assert (
        tmp_path / "test-of" / "test-of-run_1" / "iteration_1_step_1" / "test_plots"
    ).exists()
    assert (tmp_path / "test-of" / "test-of-run_1" / "iteration_1_step_2").exists()
    assert (tmp_path / "test-of" / "test-of-run_1" / "iteration_2_step_1").exists()
    assert (tmp_path / "test-of" / "test-of-run_1" / "iteration_2_step_2").exists()

    # Get root weights
    mc_test = uproot.open("./assets/evts_100_200.root")
    mc_test_tree = mc_test["OmniTree"]
    mc_test_p190 = ak.to_numpy(mc_test_tree["pass190"].array())
    mc_test_truth_p190 = ak.to_numpy(mc_test_tree["truth_pass190"].array())
    mc_test_weights = ak.to_numpy(mc_test_tree["weight"].array())

    # Check that the i1s1 weights are correct
    i1s1 = np.load(
        tmp_path / "test-of" / "test-of-run_1" / "weights" / "iteration_1_step_1.npz"
    )
    p190 = i1s1["source_pass190_test"]
    raw_test = i1s1["raw_test_output"]
    assert len(raw_test) == np.count_nonzero(p190)
    expected_net_weights = np.exp(raw_test)
    assert np.all(expected_net_weights == i1s1["network_test"])
    test_weights = i1s1["test"]
    assert np.all(
        test_weights[p190 == 1]
        == expected_net_weights * mc_test_weights[mc_test_p190 == 1]
    )
    assert np.all(test_weights[p190 == 0] == mc_test_weights[mc_test_p190 == 0])

    # Check that the i1s2 weights are correct
    i1s2 = np.load(
        tmp_path / "test-of" / "test-of-run_1" / "weights" / "iteration_1_step_2.npz"
    )
    truth_p190 = i1s2["source_truth_pass190_test"]
    raw_test = i1s2["raw_test_output"]
    assert len(raw_test) == np.count_nonzero(truth_p190)
    expected_net_weights = np.exp(raw_test)
    assert np.all(expected_net_weights == i1s2["network_test"])
    test_weights = i1s2["test"]
    assert np.all(
        test_weights[truth_p190 == 1]
        == expected_net_weights * mc_test_weights[mc_test_truth_p190 == 1]
    )
    assert np.all(
        test_weights[truth_p190 == 0] == mc_test_weights[mc_test_truth_p190 == 0]
    )

    # Check that the i2s1 weights are correct
    i2s1 = np.load(
        tmp_path / "test-of" / "test-of-run_1" / "weights" / "iteration_2_step_1.npz"
    )
    p190 = i2s1["source_pass190_test"]
    raw_test = i2s1["raw_test_output"]
    assert len(raw_test) == np.count_nonzero(p190)
    expected_net_weights = np.exp(raw_test)
    assert np.all(expected_net_weights == i2s1["network_test"])
    test_weights = i2s1["test"]
    assert np.all(
        test_weights[p190 == 1]
        == expected_net_weights * i1s2["test"][mc_test_p190 == 1]
    )
    assert np.all(test_weights[p190 == 0] == i1s2["test"][mc_test_p190 == 0])

    # Check that the i2s2 weights are correct
    i2s2 = np.load(
        tmp_path / "test-of" / "test-of-run_1" / "weights" / "iteration_2_step_2.npz"
    )
    truth_p190 = i2s2["source_truth_pass190_test"]
    raw_test = i2s2["raw_test_output"]
    assert len(raw_test) == np.count_nonzero(truth_p190)
    expected_net_weights = np.exp(raw_test)
    assert np.all(expected_net_weights == i2s2["network_test"])
    test_weights = i2s2["test"]
    assert np.all(
        test_weights[truth_p190 == 1]
        == expected_net_weights * i1s2["test"][mc_test_truth_p190 == 1]
    )
    assert np.all(
        test_weights[truth_p190 == 0] == i1s2["test"][mc_test_truth_p190 == 0]
    )
