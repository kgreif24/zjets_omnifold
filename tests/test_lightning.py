"""
test_train_eval.py - Test suite for the OfTrain / OfEval classes
"""

from lightning_train import OfTrain
# from lightning_eval import OfEval
from cli.of_config import OfConfig


def test_lightning_train(tmp_path):

    try:

        # Load OfConfig
        config = OfConfig(config_name="./assets/test_lightning.yml")

        # Overwrite the checkpoint_dir and write out a new file
        config.mod_config("checkpoint_dir", str(tmp_path))
        config_name = tmp_path / "config.yaml"
        config.create_template(template_path=config_name)

        # Initialize the class, being sure to configure it to run a fast development run
        trainer = OfTrain(config_name, 0, 1, unit_test=True)
        rid, path = trainer.run()

    except Exception:

        assert False

    # Check that we made a best model checkpoint symlink
    assert (
        tmp_path / "test-of" / "test-of-run" / "pretrain_step_1" / "best_model.ckpt"
    ).exists()


"""
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
            "test",
            config_name,
            0,
            1,
            check_path="./assets/test_checkpoint.ckpt",
            store=tmp_path,
            unit_test=True,
        )
        evaluator.run()

    except Exception:
        assert False
"""
