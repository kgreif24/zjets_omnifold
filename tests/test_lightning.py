"""
test_train_eval.py - Test suite for the OfTrain / OfEval classes
"""

import os
import pathlib
from lightning_train import OfTrain
from lightning_eval import OfEval
from cli.of_config import OfConfig


def test_pretrain(tmp_path):

    try:

        # Load OfConfig
        config = OfConfig(config_name="./assets/test_lightning.yml")

        # Overwrite the checkpoint_dir and write out a new file
        config.mod_config("checkpoint_dir", str(tmp_path))
        config_name = tmp_path / "config.yaml"
        config.create_template(template_path=config_name)

        # Initialize the class, being sure to configure it to run a fast development run
        # and to run pretraining
        trainer = OfTrain(config_name, 0, 1, unit_test=True)
        trainer.run()

    except Exception:

        assert False

    # Check that we made a best model checkpoint symlink
    assert (
        tmp_path / "test-of" / "test-of-run" / "pretrain_step_1" / "best_model.ckpt"
    ).exists()


def test_lightning_train(tmp_path):

    try:

        # Load OfConfig
        config = OfConfig(config_name="./assets/test_lightning.yml")

        # Overwrite the checkpoint_dir and write out a new file
        config.mod_config("checkpoint_dir", str(tmp_path))
        config_name = tmp_path / "config.yaml"
        config.create_template(template_path=config_name)

        # Initialize the class, being sure to configure it to run a fast development run
        trainer = OfTrain(config_name, 1, 1, unit_test=True)
        trainer.run()

    except Exception:

        assert False

    # Check that we made a best model checkpoint symlink
    assert (
        tmp_path / "test-of" / "test-of-run" / "iteration_1_step_1" / "best_model.ckpt"
    ).exists()


def test_checkpoint_behavior(tmp_path):

    # Load OfConfig
    config = OfConfig(config_name="./assets/test_lightning.yml")

    # Overwrite the checkpoint_dir and write out a new file
    config.mod_config("checkpoint_dir", str(tmp_path))
    config_name = tmp_path / "config.yaml"
    config.create_template(template_path=config_name)

    # Initialize the class
    trainer = OfTrain(config_name, 1, 1, unit_test=True)

    # Modify min steps
    trainer.min_steps = 2000

    # Add checkpoint names to the directory
    checkpoint_names = [
        "step=1_val_wasserstein=0.1234.ckpt",
        "step=2000_val_wasserstein=0.5678.ckpt",
        "step=3000_val_wasserstein=0.4.ckpt",
        "step=5000_val_wasserstein=2.ckpt",
        "val_wasserstein=3.testteststep=6000.ckpt",
    ]
    checkpoint_paths = [
        os.path.join(trainer.checkpoint_dir, name) for name in checkpoint_names
    ]
    for path in checkpoint_paths:
        pathlib.Path(path).touch()

    # Test extract info
    s1, w1 = trainer._extract_info_from_checkpoint(checkpoint_paths[0])
    s2, w2 = trainer._extract_info_from_checkpoint(checkpoint_paths[1])
    s5, w5 = trainer._extract_info_from_checkpoint(checkpoint_paths[4])
    assert s1 == 1
    assert w1 == 0.1234
    assert s2 == 2000
    assert w2 == 0.5678
    assert s5 == 6000
    assert w5 == 3.0

    # Test best checkpoint
    best_checkpoint = trainer._find_best_checkpoint()
    assert best_checkpoint == os.path.basename(checkpoint_paths[2])


def test_lightning_eval(tmp_path):

    try:

        # Load OfConfig
        config = OfConfig(config_name="./assets/test_lightning.yml")

        # Overwrite the checkpoint_dir and write out a new file
        config.mod_config("checkpoint_dir", str(tmp_path))
        config_name = tmp_path / "config.yaml"
        config.create_template(template_path=config_name)

        # Initialize the class, being sure to configure it to run a fast development run
        evaluator = OfEval(
            config_name,
            1,
            1,
            check_path="./assets/full_checkpoint/full_checkpoint.ckpt",
            store=tmp_path,
            unit_test=True,
        )
        evaluator.run_testing()

    except Exception:
        assert False
