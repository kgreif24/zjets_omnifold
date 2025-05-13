"""
test_train_eval.py - Test suite for the OfTrain / OfEval classes
"""

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
            check_path="./assets/full_checkpoint.ckpt",
            store=tmp_path,
            unit_test=True,
        )
        evaluator.run_testing()

    except Exception:
        assert False
