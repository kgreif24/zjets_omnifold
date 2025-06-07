"""lightning_train.py - This file defines the "OfTrain" class, which handles the setup
and training for an omnifolder classifier using pytorch lightning. It also defines the
"main" function, which will be called from the Omnifolder class as a subprocess for
easy parallelism.

Author: Kevin Greif
Last updated 05/29/2025
python3
"""

import re
import os
import time
import argparse
import atexit
import glob
import signal

import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only

from cli.of_config import OfConfig
from lightning_module import LOfTransformer
from lightning_data_module import LOfData
from utils.subprocess_utils import cleanup_resources


class OfTrain:
    """OfTrain - This class is meant to handle the setup and training of a single
    Omnifold classifier. The driver code for using this class is in the "main" function
    below.

    Warm starting is implemented in that any training with iteration >= 1 will search
    for a "best_model.ckpt" file in the checkpoint directory of the pretraining runs.
    If this file does not exist, will raise an exception.
    """

    def __init__(
        self,
        config_path,
        iteration,
        step,
        seed=222,
        index=-1,
        unit_test=False,
    ):
        """__init__ - The init function for this class. It takes the OfConfig object
        used for this run of Omnifold, plus the iteration and step of this training run.

        Arguments:
        config_path - The path of the of config file
        iteration - The iteration number for this training
        step - The step number for this training
        seed - The seed to use for the train / val split in this training
        index - The index of the ensemble to run. Add this number to the end of the
            group ID if it is not None
        unit_test - If true, trainer will just run a few steps of training and exit

        Returns:
        None
        """

        # ------------- Config and directory setup -------------

        # Store the configuration
        self.config = OfConfig(config_name=config_path)
        self.iteration = iteration
        self.step = step
        self.split_seed = seed
        self.unit_test = unit_test

        # Modify the group name if an index is provided
        if index != -1:
            self.config.group_name = f"{self.config.group_name}_{index}"

        # Make root directory for this run of Omnifold
        root_dir = (
            f"{self.config.checkpoint_dir}/"
            f"{self.config.project_name}/{self.config.group_name}"
        )
        os.makedirs(root_dir, exist_ok=True)

        # Set run name
        if self.iteration == 0:
            run_name = f"pretrain_step_{self.step}"
        else:
            run_name = f"iteration_{self.iteration}_step_{self.step}"

        # Set the checkpoint directory
        self.checkpoint_dir = f"{root_dir}/{run_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # If we are running itertaion >= 1, get warm start path
        if iteration >= 1:
            if self.config.pretrain_checkpoint is None:
                ws_path = f"{root_dir}/pretrain_step_1/best_model.ckpt"
            else:
                ws_path = self.config.pretrain_checkpoint
            if not os.path.exists(ws_path):
                raise FileNotFoundError(f"Could not find warm start path {ws_path}. ")
        else:
            ws_path = None

        # Set minimum and finish steps for checkpointing
        # Set the minimum steps
        if self.iteration > 0 and not (self.config.debug or self.unit_test):
            self.min_steps = self.config.min_checkpoint_steps
            self.finish_steps = self.config.checkpoint_finish_steps
        else:
            self.min_steps = 0
            self.finish_steps = 99999999

        # ---------------- Lightning setup ----------------

        # Get min/max learning rates and cycle rates depending on step
        if self.iteration == 0:
            min_lr = self.config.pt_min_lr
            max_lr = self.config.pt_max_lr
            max_steps = self.config.pt_max_steps
            warmup_steps = self.config.pt_warmup_steps
            cos_steps = self.config.pt_cos_steps
            linear_steps = self.config.pt_linear_steps
        elif self.step == 1:
            min_lr = self.config.s1_min_lr * (
                self.config.s1_lr_decay ** (self.iteration - 1)
            )
            max_lr = self.config.s1_max_lr * (
                self.config.s1_lr_decay ** (self.iteration - 1)
            )
            max_steps = self.config.s1_max_steps
            warmup_steps = self.config.s1_warmup_steps
            cos_steps = self.config.s1_cos_steps
            linear_steps = self.config.s1_linear_steps
        else:
            min_lr = self.config.s2_min_lr * (
                self.config.s2_lr_decay ** (self.iteration - 1)
            )
            max_lr = self.config.s2_max_lr * (
                self.config.s2_lr_decay ** (self.iteration - 1)
            )
            max_steps = self.config.s2_max_steps
            warmup_steps = self.config.s2_warmup_steps
            cos_steps = self.config.s2_cos_steps
            linear_steps = self.config.s2_linear_steps

        # Build lightning module from scratch if we are not given a warm start path
        if ws_path is None:

            # If using wandb, make a logger and get a new run ID
            if self.config.wandb:

                # Build the logger
                self.wandb_logger = WandbLogger(
                    project=self.config.project_name,
                    group=self.config.group_name,
                    name=run_name,
                    save_dir=self.checkpoint_dir,
                )

                # Get run ID
                run_id = self.wandb_logger.experiment.id

            # Else we use no logger and set a dummy run ID
            else:
                self.wandb_logger = None
                run_id = "test_run"

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

            self.l_module = LOfTransformer(
                input_dim=self.config.input_dim,
                debug=self.config.debug,
                # Include the seed and run ID just so they are logged to W&B
                seed=seed,
                run_id=run_id,
                # Include the OF step for plots
                step=self.step,
                weight_decay=self.config.weight_decay,
                min_lr=min_lr,
                max_lr=max_lr,
                warmup_steps=warmup_steps,
                cos_steps=cos_steps,
                linear_steps=linear_steps,
                # Everything below here are parameters for the network
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

        # Else load the model from the warm start path
        else:

            # Note we give preference to the restart path if it exists
            rank_zero_info(f"Loading model from path {ws_path}")
            self.l_module = LOfTransformer.load_from_checkpoint(
                ws_path,
                debug=self.config.debug,
                step=self.step,
                weight_decay=self.config.weight_decay,
                min_lr=min_lr,
                max_lr=max_lr,
                warmup_steps=warmup_steps,
                cos_steps=cos_steps,
                linear_steps=linear_steps,
            )

            # Reset seed to the one provided to the training job
            self.l_module.reset_seed(seed)

            # Build wandb logger
            # In case of warm start, need to reset the run ID in the module
            # to the ID assigned for this run by wandb
            if self.config.wandb:
                self.wandb_logger = WandbLogger(
                    project=self.config.project_name,
                    group=self.config.group_name,
                    name=run_name,
                    save_dir=self.checkpoint_dir,
                    resume="never",
                )
                run_id = self.wandb_logger.experiment.id
                self.l_module.reset_run_id(run_id)
            else:
                self.wandb_logger = None
                run_id = "test_run"

        # Initialise the callbacks
        self.lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
            logging_interval="step"
        )
        save_filename = "{epoch:02d}-{step:06d}-{val_wasserstein:.1f}"
        n_train_steps = 800 if (self.iteration == 0 and not self.unit_test) else None
        self.checkpoints = L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_wasserstein",
            filename=save_filename,
            save_top_k=self.config.top_k_checkpoints,
            mode="min",
            dirpath=self.checkpoint_dir,
            every_n_train_steps=n_train_steps,
        )
        self.early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor="val_wasserstein",
            patience=self.config.early_stopping_patience,
            mode="min",
        )

        # Build trainer
        val_check = 800 if (self.iteration == 0 and not self.unit_test) else None
        devices = (
            "auto" if (self.config.debug or self.unit_test) else self.config.num_gpus
        )
        self.trainer = L.Trainer(
            accelerator="auto" if (self.config.debug or self.unit_test) else "gpu",
            num_nodes=self.config.num_nodes,
            devices=devices,
            logger=self.wandb_logger,
            callbacks=[self.lr_monitor, self.checkpoints, self.early_stopping],
            plugins=[SLURMEnvironment(auto_requeue=False)],
            default_root_dir=self.checkpoint_dir,
            max_steps=max_steps,
            val_check_interval=val_check,
            enable_progress_bar=self.config.interactive,
            use_distributed_sampler=False,
        )

        # If the checkpoint directory contains existing checkpoints,
        # remove them and start training again from scratch
        # This is an unfortunate requirement given how lightning's early
        # stopping callback is implemented
        checkpoint_glob = glob.glob(f"{self.checkpoint_dir}/*.ckpt")
        if self.trainer.global_rank == 0 and len(checkpoint_glob) > 0:
            rank_zero_info(
                f"Found existing checkpoints in {self.checkpoint_dir}, removing them."
            )
            for checkpoint in checkpoint_glob:
                os.remove(checkpoint)

        # ---------------- Data setup ----------------

        # Get weights for use in training. Define (but do not make!) the weight
        # directory
        weight_dir = f"{root_dir}/weights"

        # Find the data and weight files to use for this iteration and step
        # For step one:
        if self.step == 1:
            use_syst_kw = self.config.syst_kw
            use_truth = False
            # If this is pre-training (iteration 0), use the MC train file
            # and Sherpa file
            if self.iteration == 0:
                source_file = self.config.pretrain_source_path
                target_file = self.config.pretrain_target_path
                source_weight_file = None
                target_weight_file = None
            # If this is the first iteration, use the weights from the ROOT file
            # for source and target
            elif self.iteration == 1:
                source_file = self.config.mc_train_path
                target_file = self.config.data_path
                source_weight_file = None
                target_weight_file = None
            # Otherwise use the weights from the previous step two for the source,
            # and weights from the ROOT file for target
            else:
                source_file = self.config.mc_train_path
                target_file = self.config.data_path
                source_weight_file = (
                    f"{weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                )
                target_weight_file = None
        # For step two:
        if self.step == 2:
            use_syst_kw = None
            use_truth = True
            # If this is pre-training (iteration 0), use the MC train file and
            # Sherpa file
            if self.iteration == 0:
                source_file = self.config.pretrain_source_path
                target_file = self.config.pretrain_target_path
                source_weight_file = None
                target_weight_file = None
            # If this is the first iteration, use the weights from step one for target,
            # and the weights from the root file as source.
            elif self.iteration == 1:
                source_file = self.config.mc_train_path
                target_file = self.config.mc_train_path
                source_weight_file = None
                target_weight_file = (
                    f"{weight_dir}/iteration_{self.iteration}_step_1.npz"
                )
            # Otherwise use weights from previous step 2 for source, and the weights
            # from the previous step one for target
            else:
                source_file = self.config.mc_train_path
                target_file = self.config.mc_train_path
                source_weight_file = (
                    f"{weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                )
                target_weight_file = (
                    f"{weight_dir}/iteration_{self.iteration}_step_1.npz"
                )

        # Build the data module
        self.d_module = LOfData(
            source_file=source_file,
            target_file=target_file,
            source_weight_path=source_weight_file,
            target_weight_path=target_weight_file,
            data_divisor=1,
            total_rank=int(self.config.num_nodes * self.config.num_gpus),
            rank=self.trainer.global_rank,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.batch_size,
            split_seed=seed,
            dataloader_workers=0,
            testing=False,
            use_truth=use_truth,
            max_events_target=self.config.max_events_target,
            syst_kw=use_syst_kw,
        )

    def run(self):
        """run - This function runs the training for the Omnifold classifier.
        No arguments or returns
        """

        # Run training, pickup from restart path if available
        # If we only have a warm start path, start from the beginning
        self.trainer.fit(
            self.l_module,
            self.d_module,
        )

        # On natural exit, make a symlink to the best model
        self.cleanup_on_exit(natural_exit=True)

        # Close W&B
        if self.config.wandb:
            wandb.finish()

    @rank_zero_only
    def cleanup_on_exit(self, natural_exit=False):
        """ cleanup_on_exit - This function is called in two cases:
            1. When training has finished, to make `best_model.ckpt` symlink
            2. When the training process is going to be timed out or preempted,
               to make a symlink if we are past the 'finish_steps' threshold.
               Else do nothing and let checkpoints be cleared by the restarted
               process.

        Arguments:
        natural_exit - If true, this is a normal exit after training has finished.
            If false, this is an exit due to timeout or preemption.

        No returns
        """

        # Name for the best model symlink
        best_model_link = f"{self.checkpoint_dir}/best_model.ckpt"

        # If this is a natural exit, just make a symlink to the best model
        if natural_exit:
            best_checkpoint = self._find_best_checkpoint()
            if not best_checkpoint:
                raise RuntimeError(
                    "No checkpoints found on natural exit, cannot make symlink."
                )
            if os.path.lexists(best_model_link):
                os.remove(best_model_link)
            os.symlink(best_checkpoint, best_model_link)

        # If this is a timeout or preemption, check if we are past the finish steps
        elif self.trainer.global_step >= self.finish_steps:
            # Find the best checkpoint
            best_checkpoint = self._find_best_checkpoint()
            if not best_checkpoint:
                raise RuntimeError(
                    "No checkpoints found on timeout, cannot make symlink."
                )
            if os.path.lexists(best_model_link):
                os.remove(best_model_link)
            os.symlink(best_checkpoint, best_model_link)

        # Else run timed out before reaching finish steps, do nothing
        else:
            rank_zero_info(
                f"Run did not reach finish steps ({self.finish_steps}), "
                "not making symlink to best model."
            )

    def _find_best_checkpoint(self):
        """_find_best_checkpoint - This function finds the best checkpoint
        in the checkpoint directory. It assumes that the checkpoints are named
        in the format 'epoch=XX-step=YYYY-val_wasserstein=ZZZ.ckpt'.

        Returns:
        best_checkpoint - The path to the best checkpoint file
        """

        # Get all checkpoint files in the directory
        checkpoint_files = glob.glob(f"{self.checkpoint_dir}/*.ckpt")
        if not checkpoint_files:
            raise ValueError("No checkpoint files found in the directory.")

        # Drop checkpoints that are not above the minimum steps
        checkpoint_files = [
            f
            for f in checkpoint_files
            if self._extract_info_from_checkpoint(f)[0] >= self.min_steps
        ]
        if not checkpoint_files:
            rank_zero_info(
                f"No checkpoint files found with steps >= {self.min_steps}."
            )
            return

        # Sort checkpoints above min_steps by their wasserstein distance
        sorted_checkpoints = sorted(
            checkpoint_files,
            key=lambda x: self._extract_info_from_checkpoint(x)[1],
        )

        # Return the best (lowest wasserstein) checkpoint
        return os.path.basename(sorted_checkpoints[0])

    def _extract_info_from_checkpoint(self, checkpoint_path):
        """_extract_info_from_checkpoint - This function extracts the step
        and validation wasserstein distance from the checkpoint path.
        It assume the checkpoint path is in the format hardcoded into
        the ModelCheckpoint callback above.

        Arguments:
        checkpoint_path - The path to the checkpoint file

        Returns
        step - The step number for this checkpoint
        wasserstein - The validation wasserstein distance for this checkpoint
        """

        # Assumes format like '...val_wasserstein=12.3...step=1234...' in the filename
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

# This function will be called as a subprocess from the Omnifolder class

if __name__ == "__main__":

    # Register GPU cleanup at exit
    atexit.register(cleanup_resources)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the omnifold algorithm")
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="The iteration number for this training run",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="The step number for this training run, either 1 or 2",
    )
    parser.add_argument(
        "--index", type=int, default=-1, help="The index of the ensemble to run"
    )
    parser.add_argument(
        "--split_seed", type=int, default=222, help="The seed to use for the data split"
    )
    args, unknown = parser.parse_known_args()

    # Build trainer
    trainer = OfTrain(
        args.config_path,
        args.iteration,
        args.step,
        seed=args.split_seed,
        index=args.index,
    )

    # Override SIGUSR1 and SIGTERM signals to requeue
    def handle_signal(signum, frame):
        """Signal handler for the program."""
        print(f"Received signal {signum}, running exit routine.")
        trainer.cleanup_on_exit(natural_exit=False)
        os.system("sync")
        time.sleep(20)
        # Requeue on timeout, not needed on preemption since it is done automatically
        if trainer.trainer.is_global_zero and signum == signal.SIGUSR1:
            print("Requeue job!")
            os.system("scontrol requeue $SLURM_JOB_ID")

    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run the training
    trainer.run()

    # Print something and sleep a bit to flush the output
    print("...")
    time.sleep(10)
