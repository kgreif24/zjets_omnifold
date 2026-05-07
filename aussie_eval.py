"""aussie_eval.py - Evaluation / weight-writing for AUSSIE.

Runs prediction on the paired train and test MC files with a trained
LAussieUnfolder, and writes the per-event weights to a .npz file that
matches the OmniFold schema for drop-in compatibility with
ensemble_weights.py and the downstream plotting code.

Meant to be invoked programmatically by aussie_train.py after training
finishes, but also runnable as a standalone script.

Author: Kevin Greif
python3
"""

import os
import argparse

import numpy as np
import torch
import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment

from cli.of_config import OfConfig
from lightning_aussie_module import LAussieUnfolder
from lightning_aussie_data_module import LAussieData
from utils.weights_io import write_weights_npz


class AussieEval:
    """AussieEval - Predict log R_phi(z) on the paired MC train and test files
    and write the resulting weights to an .npz file.
    """

    def __init__(
        self,
        config_path,
        iteration=1,
        check_path=None,
        store=None,
        index=-1,
        unit_test=False,
    ):
        self.config = OfConfig(config_name=config_path)
        self.iteration = iteration
        self.unit_test = unit_test

        if index != -1:
            self.config.group_name = f"{self.config.group_name}_{index}"

        # Directory layout - AUSSIE outputs live in aussie/ subdir
        self.root_dir = (
            f"{self.config.checkpoint_dir}/"
            f"{self.config.project_name}/{self.config.group_name}"
        )
        self.aussie_root = f"{self.root_dir}/aussie"
        self.checkpoint_dir = (
            f"{self.aussie_root}/iteration_{self.iteration}_step_2"
        )
        self.weight_dir = f"{self.aussie_root}/weights"
        os.makedirs(self.weight_dir, exist_ok=True)

        if check_path is None:
            check_path = f"{self.checkpoint_dir}/best_model.ckpt"
        if not os.path.exists(check_path):
            raise FileNotFoundError(f"Could not find AUSSIE checkpoint at {check_path}")
        print(f"Evaluating AUSSIE with checkpoint: {check_path}")

        self.store = store

        # Load model
        self.model = LAussieUnfolder.load_from_checkpoint(
            check_path,
            debug=self.config.debug,
        )

        # Single-GPU trainer for prediction
        self.trainer = L.Trainer(
            accelerator="auto" if (self.config.debug or unit_test) else "gpu",
            num_nodes=1,
            devices=1,
            logger=False,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            fast_dev_run=unit_test,
            use_distributed_sampler=False,
            enable_progress_bar=True,
        )

    def _make_data_module(self, mc_path):
        return LAussieData(
            mc_file=mc_path,
            max_events=self.config.max_events_target,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.test_batch_size,
            split_seed=self.config.split_seed,
            dataloader_workers=10,
            inference_mode=True,
        )

    def _predict_all(self, d_module):
        """Predict over the full paired dataset (predict_dataloader yields
        events in a fixed sequential order so indices align with pass190).
        """
        preds = self.trainer.predict(self.model, d_module.predict_dataloader())
        preds = torch.cat([p.cpu().flatten() for p in preds])
        return preds.numpy()

    def run_prediction(self):
        """Run AUSSIE prediction on MC train and test files, compute the
        updated weights, and write the .npz file.
        """

        d_train = self._make_data_module(self.config.mc_train_path)
        d_test = self._make_data_module(self.config.mc_test_path)

        predictions_train = self._predict_all(d_train)
        predictions_test = self._predict_all(d_test)

        network_weights_train = np.exp(predictions_train)
        network_weights_test = np.exp(predictions_test)

        pass190_train = d_train.get_pass190()
        pass190_test = d_test.get_pass190()

        # AUSSIE iteration 1 has no prior weights -> start from 1s.
        start_weights_train = np.ones_like(pass190_train, dtype=np.float32)
        start_weights_test = np.ones_like(pass190_test, dtype=np.float32)

        updated_weights_train = start_weights_train.copy()
        updated_weights_train[pass190_train == 1] *= network_weights_train
        updated_weights_test = start_weights_test.copy()
        updated_weights_test[pass190_test == 1] *= network_weights_test

        if self.store is None:
            out_path = (
                f"{self.weight_dir}/iteration_{self.iteration}_step_2.npz"
            )
        else:
            out_path = self.store
        write_weights_npz(
            out_path,
            raw_train=predictions_train,
            raw_test=predictions_test,
            network_train=network_weights_train,
            network_test=network_weights_test,
            train=updated_weights_train,
            test=updated_weights_test,
        )
        print(f"Wrote AUSSIE weights to {out_path}")
        return out_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run AUSSIE evaluation")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the OfConfig YAML"
    )
    parser.add_argument(
        "--iteration", type=int, default=1, help="AUSSIE iteration (typically 1)"
    )
    parser.add_argument(
        "--check_path",
        type=str,
        default=None,
        help="Explicit checkpoint path (else use best_model.ckpt symlink)",
    )
    parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="Override the output directory for the weights npz",
    )
    parser.add_argument(
        "--index", type=int, default=-1, help="Ensemble index"
    )
    args, _ = parser.parse_known_args()

    evaluator = AussieEval(
        args.config_path,
        iteration=args.iteration,
        check_path=args.check_path,
        store=args.store,
        index=args.index,
    )
    evaluator.run_prediction()
