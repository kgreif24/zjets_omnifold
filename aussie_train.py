"""aussie_train.py - Training driver for the AUSSIE alternative to Omnifold
step 2. Structure mirrors lightning_train.py / OfTrain but:

  - Uses a pre-trained Omnifold iteration-1 step-1 classifier (frozen)
  - Trains a single unfolder network by the AUSSIE L1 gradient-norm loss
  - Writes outputs into an aussie/ subdirectory of the existing run dir
  - Runs on a single GPU (AUSSIE requires create_graph=True, which breaks
    cleanly with DDP)

After training finishes, the per-event weights are written via
aussie_eval.AussieEval.run_prediction so that downstream tooling sees a
drop-in .npz equivalent to iteration_1_step_2.npz from Omnifold.

Author: Kevin Greif
python3
"""

import re
import os
import time
import argparse
import glob
import signal

import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only

from cli.of_config import OfConfig
from lightning_aussie_module import LAussieUnfolder
from lightning_aussie_data_module import LAussieData
from aussie_eval import AussieEval


class AussieTrain:
    """AussieTrain - Set up and run a single AUSSIE training.

    The driver handles checkpoint discovery, directory layout, W&B logging,
    SLURM signal handling, and invocation of the eval step after training.
    """

    AUSSIE_ITERATION = 1
    AUSSIE_STEP = 2

    def __init__(
        self,
        config_path,
        iteration=1,
        seed=222,
        index=-1,
        unit_test=False,
        classifier_ckpt_override=None,
        run_name=None,
    ):
        """__init__

        Arguments:
            config_path (str) - Path to the OfConfig YAML
            iteration (int) - Should be 1 for AUSSIE (hardcoded but exposed).
                A warning is issued if anything else is passed.
            seed (int) - Split seed for the data module
            index (int) - Ensemble index (-1 means no ensembling)
            unit_test (bool) - If True, run a short fast_dev_run-like training
            classifier_ckpt_override (str) - Explicit path to the step-1
                classifier. Overrides config.aussie_classifier_override and
                the default derived path.
            run_name (str) - W&B run name. Defaults to
                "aussie_iter{iteration}" if not provided.
        """

        self.config = OfConfig(config_name=config_path)
        self.iteration = iteration
        if iteration != self.AUSSIE_ITERATION:
            rank_zero_info(
                f"Warning: AUSSIE is designed for iteration=1 but got "
                f"iteration={iteration}. Proceeding anyway."
            )
        self.split_seed = seed
        self.unit_test = unit_test

        if index != -1:
            self.config.group_name = f"{self.config.group_name}_{index}"

        # --- Directory layout ---
        root_dir = (
            f"{self.config.checkpoint_dir}/"
            f"{self.config.project_name}/{self.config.group_name}"
        )
        self.root_dir = root_dir
        self.aussie_root = f"{root_dir}/aussie"
        os.makedirs(self.aussie_root, exist_ok=True)

        run_name = run_name if run_name is not None else f"aussie_iter{self.iteration}"
        self.run_name = run_name
        self.checkpoint_dir = f"{self.aussie_root}/{run_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Resolve classifier checkpoint path ---
        classifier_ckpt = classifier_ckpt_override
        if classifier_ckpt is None:
            classifier_ckpt = self.config.aussie_classifier_override
        if classifier_ckpt is None:
            classifier_ckpt = (
                f"{root_dir}/iteration_{self.iteration}_step_1/best_model.ckpt"
            )
        if not os.path.exists(classifier_ckpt):
            raise FileNotFoundError(
                f"AUSSIE needs a step-1 classifier at {classifier_ckpt}. "
                "Run Omnifold (or at least its iteration-1 step-1 training) "
                "first, or pass --classifier_ckpt_override."
            )
        self.classifier_ckpt_path = classifier_ckpt

        # --- GPU count: force 1 (create_graph=True breaks under DDP) ---
        if self.config.num_gpus > 1 or self.config.num_nodes > 1:
            rank_zero_info(
                f"Warning: config requests "
                f"{self.config.num_gpus} GPUs x {self.config.num_nodes} nodes, "
                "but AUSSIE v1 is single-GPU only. Forcing devices=1."
            )

        # --- Checkpoint selection thresholds ---
        if not (self.config.debug or self.unit_test):
            self.min_steps = self.config.aussie_min_checkpoint_steps
            self.finish_steps = self.config.checkpoint_finish_steps
        else:
            self.min_steps = 0
            self.finish_steps = 99999999

        # --- Build the Lightning module ---
        block_params = {
            "dropout": self.config.block_dropout,
            "attn_dropout": self.config.block_attn_dropout,
            "activation_dropout": self.config.block_activation_dropout,
        }
        cls_block_params = {
            "dropout": self.config.cls_block_dropout,
            "attn_dropout": self.config.cls_block_attn_dropout,
            "activation_dropout": self.config.cls_block_activation_dropout,
        }

        if self.config.wandb:
            self.wandb_logger = WandbLogger(
                project=self.config.project_name,
                group=self.config.group_name,
                name=run_name,
                save_dir=self.checkpoint_dir,
            )
            run_id = self.wandb_logger.experiment.id
        else:
            self.wandb_logger = None
            run_id = "test_run"

        self.l_module = LAussieUnfolder(
            classifier_ckpt_path=self.classifier_ckpt_path,
            input_dim=self.config.input_dim,
            debug=self.config.debug,
            seed=seed,
            run_id=run_id,
            weight_decay=self.config.weight_decay,
            min_lr=self.config.aussie_min_lr,
            max_lr=self.config.aussie_max_lr,
            warmup_steps=self.config.aussie_warmup_steps,
            cos_steps=self.config.aussie_cos_steps,
            linear_steps=self.config.aussie_linear_steps,
            logit_clamp=self.config.aussie_logit_clamp,
            # OfTransformer kwargs
            num_classes=1,
            remove_self_pair=self.config.remove_self_pair,
            embed_dims=self.config.embed_dims,
            pair_input_dim=self.config.pair_input_dim,
            pair_embed_dims=self.config.pair_embed_dims,
            fc_nodes=self.config.fc_nodes,
            fc_dropout=self.config.fc_dropout,
            cls_block_params=cls_block_params,
            num_cls_layers=self.config.num_cls_layers,
            block_params=block_params,
            num_layers=self.config.num_layers,
        )

        # --- Callbacks & trainer ---
        self.lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
            logging_interval="step"
        )
        save_filename = "{epoch:02d}-{step:06d}-{val_wasserstein:.1f}"
        self.checkpoints = L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_wasserstein",
            filename=save_filename,
            save_top_k=self.config.top_k_checkpoints,
            mode="min",
            dirpath=self.checkpoint_dir,
        )
        self.early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor="val_wasserstein",
            patience=self.config.early_stopping_patience,
            mode="min",
        )

        self.trainer = L.Trainer(
            accelerator=(
                "auto" if (self.config.debug or self.unit_test) else "gpu"
            ),
            num_nodes=1,
            devices=1,
            logger=self.wandb_logger,
            callbacks=[self.lr_monitor, self.checkpoints, self.early_stopping],
            plugins=[SLURMEnvironment(auto_requeue=False)],
            default_root_dir=self.checkpoint_dir,
            max_steps=self.config.aussie_max_steps,
            val_check_interval=self.config.aussie_val_check_interval,
            gradient_clip_val=self.config.aussie_grad_clip,
            enable_progress_bar=self.config.interactive,
            use_distributed_sampler=False,
        )

        # Clear any leftover checkpoints (Lightning early-stopping + multiple
        # runs interact badly otherwise - mirror OfTrain behaviour).
        checkpoint_glob = glob.glob(f"{self.checkpoint_dir}/*.ckpt")
        if self.trainer.global_rank == 0 and len(checkpoint_glob) > 0:
            rank_zero_info(
                f"Found existing checkpoints in {self.checkpoint_dir}, removing them."
            )
            for checkpoint in checkpoint_glob:
                os.remove(checkpoint)

        # --- Data module ---
        self.d_module = LAussieData(
            mc_file=self.config.mc_train_path,
            max_events=self.config.max_events_target,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.aussie_batch_size,
            split_seed=seed,
            dataloader_workers=0,
        )

        self.config_path = config_path
        self.index = index

    def run(self):
        """Run AUSSIE training, then write the per-event weights."""

        self.trainer.fit(self.l_module, self.d_module)

        # Create the best_model.ckpt symlink
        self.cleanup_on_exit(natural_exit=True)

        if self.config.wandb:
            wandb.finish()

        # Finally, predict and write the .npz for downstream tools
        if self.trainer.global_rank == 0:
            rank_zero_info("Running AUSSIE prediction to write weights...")
            evaluator = AussieEval(
                self.config_path,
                iteration=self.iteration,
                index=self.index,
                unit_test=self.unit_test,
            )
            evaluator.run_prediction()

    @rank_zero_only
    def cleanup_on_exit(self, natural_exit=False):
        """Make the best_model.ckpt symlink (or skip if we haven't run
        long enough). Same semantics as OfTrain.cleanup_on_exit."""

        best_model_link = f"{self.checkpoint_dir}/best_model.ckpt"

        if natural_exit:
            best_checkpoint = self._find_best_checkpoint()
            if not best_checkpoint:
                raise RuntimeError(
                    "No checkpoints found on natural exit, cannot make symlink."
                )
            if os.path.lexists(best_model_link):
                os.remove(best_model_link)
            os.symlink(best_checkpoint, best_model_link)
        elif self.trainer.global_step >= self.finish_steps:
            best_checkpoint = self._find_best_checkpoint()
            if not best_checkpoint:
                rank_zero_info("No checkpoints found on timeout, not making symlink.")
                return
            if os.path.lexists(best_model_link):
                os.remove(best_model_link)
            os.symlink(best_checkpoint, best_model_link)
        else:
            rank_zero_info(
                f"Run did not reach finish steps ({self.finish_steps}), "
                "not making symlink to best model."
            )

    def _find_best_checkpoint(self):
        checkpoint_files = glob.glob(f"{self.checkpoint_dir}/*.ckpt")
        if not checkpoint_files:
            raise ValueError("No checkpoint files found in the directory.")
        checkpoint_files = [
            f
            for f in checkpoint_files
            if self._extract_info_from_checkpoint(f)[0] >= self.min_steps
        ]
        if not checkpoint_files:
            rank_zero_info(f"No checkpoint files found with steps >= {self.min_steps}.")
            return
        sorted_checkpoints = sorted(
            checkpoint_files,
            key=lambda x: self._extract_info_from_checkpoint(x)[1],
        )
        return os.path.basename(sorted_checkpoints[0])

    def _extract_info_from_checkpoint(self, checkpoint_path):
        step_match = re.search(r"step=(\d+)", os.path.basename(checkpoint_path))
        wasserstein_match = re.search(
            r"val_wasserstein=(\d+(?:\.\d+)?)", os.path.basename(checkpoint_path)
        )
        if step_match:
            step = int(step_match.group(1))
        else:
            raise ValueError(
                f"Could not extract step from checkpoint path: {checkpoint_path}"
            )
        if wasserstein_match:
            wasserstein = float(wasserstein_match.group(1))
        else:
            raise ValueError(
                f"Could not extract wasserstein from checkpoint path: {checkpoint_path}"
            )
        return step, wasserstein


# ------------------ MAIN FUNCTION ------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run AUSSIE training")
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Which Omnifold iteration's step-1 classifier to use (default 1)",
    )
    parser.add_argument(
        "--index", type=int, default=-1, help="Ensemble index (-1 means no ensembling)"
    )
    parser.add_argument(
        "--split_seed", type=int, default=222, help="Seed for the train/val split"
    )
    parser.add_argument(
        "--classifier_ckpt_override",
        type=str,
        default=None,
        help="Optional explicit path to the step-1 classifier checkpoint",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name (default: aussie_iter{iteration})",
    )
    args, unknown = parser.parse_known_args()

    trainer = AussieTrain(
        args.config_path,
        iteration=args.iteration,
        seed=args.split_seed,
        index=args.index,
        classifier_ckpt_override=args.classifier_ckpt_override,
        run_name=args.run_name,
    )

    def handle_signal(signum, frame):
        print(f"Received signal {signum}, running exit routine.")
        trainer.cleanup_on_exit(natural_exit=False)
        os.system("sync")
        time.sleep(20)
        if trainer.trainer.is_global_zero and signum == signal.SIGUSR1:
            print("Requeue job!")
            os.system("scontrol requeue $SLURM_JOB_ID")

    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    trainer.run()

    print("...")
    time.sleep(10)
