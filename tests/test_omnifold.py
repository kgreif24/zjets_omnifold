"""
test_omnifold.py - Test suite for the Omnifolder class
"""

import os
import shutil
import glob
import pytest
import json
import numpy as np
import uproot
import awkward as ak

from cli.of_config import OfConfig
from omnifold import Omnifolder


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

    # Check for pretrain checkpoints
    step1_glob = glob.glob(f"{tmp_path}/test-of/test-of-run_1/pretrain_step_1/*.ckpt")
    assert len(step1_glob) == 2
    step2_glob = glob.glob(f"{tmp_path}/test-of/test-of-run_1/pretrain_step_2/*.ckpt")
    assert len(step2_glob) == 2

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


@pytest.mark.slow
def test_restart(tmp_path):

    # Create config object
    config = OfConfig(config_name="./assets/test_of.yml")

    # Overwrite the checkpoint dir with the tmp path, and make
    # it so we only run pre-training
    config.mod_config("checkpoint_dir", tmp_path)
    config.mod_config("num_iterations", 0)

    # Write the config to a file
    config.create_template(template_path=f"{tmp_path}/test_of.yml")
    assert (tmp_path / "test_of.yml").exists()

    # Copy status.json from assets to tmp_path
    shutil.copy("./assets/status.json", tmp_path)
    assert (tmp_path / "status.json").exists()

    # Make omnifolder object
    of = Omnifolder(f"{tmp_path}/test_of.yml", use_slurm=False, index=1)
    of.run_of()

    # Look for final pre-train checkpoint
    assert (tmp_path / "test-of" / "test-of-run_1" / "pretrain_step_2").exists()
    check_glob = glob.glob(f"{tmp_path}/test-of/test-of-run_1/pretrain_step_2/*.ckpt")
    assert len(check_glob) == 2


def test_checkpoint(tmp_path):

    # Create config object
    config = OfConfig(config_name="./assets/test_of.yml")

    # Overwrite the checkpoint dir with the tmp path, and make
    # it so we only run pre-training
    config.mod_config("checkpoint_dir", tmp_path)

    # Write the config to a file
    config.create_template(template_path=f"{tmp_path}/test_of.yml")
    assert (tmp_path / "test_of.yml").exists()

    # Create omnifolder object
    of = Omnifolder(f"{tmp_path}/test_of.yml", use_slurm=False, index=1)

    # Adjust the omnifolder state and save
    of.current_iteration = 3
    of.current_step = 2
    of.training = False
    of.run_id = 'test-run'
    of.seed = 1234
    of.save_status()

    print("The root directory is: ", of.root_dir)
    print("The contents of the root directory are: \n", os.listdir(of.root_dir))

    status_path = (tmp_path / "test-of" / "test-of-run_1" / "status.json")
    assert status_path.exists()
    with open(status_path, "r") as f:
        status = json.load(f)
    assert status["current_iteration"] == 3
    assert status["current_step"] == 2
    assert not status["training"]
    assert status["run_id"] == 'test-run'
    assert status["seed"] == 1234
