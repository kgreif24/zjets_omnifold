""" lightning_train.py - This file defines the "OfTrain" class, which handles the setup
and training for an omnifolder classifier using pytorch lightning. It also defines the
"main" function, which will be called from the Omnifolder class as a subprocess for
easy parallelism.

Author: Kevin Greif
Last updated 03/08/2024
python3
"""

import os
import time
import argparse
import atexit

import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from cli.of_config import OfConfig
from lightning_module import LOfTransformer
from lightning_data_module import LOfData
from utils.subprocess_utils import cleanup_resources


class OfTrain:
    """OfTrain - This class is meant to handle the setup and training of a single
    Omnifold classifier. The driver code for using this class is in the "main" function
    below.
    """

    def __init__(
        self,
        config_path,
        iteration,
        step,
        ws_path=None,
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
        ws_path - The path to a model checkpoint to warm start from, if left as None
            then model will be initialized from scratch
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

        # Modify the group name if an index is provided
        if index != -1:
            self.config.group_name = f"{self.config.group_name}_{index}"

        # Make root directory for this run of Omnifold
        root_dir = f"{self.config.checkpoint_dir}/\
                     {self.config.project_name}/{self.config.group_name}"
        os.makedirs(root_dir, exist_ok=True)

        # Set run name
        if self.iteration == 0:
            run_name = f"pretrain_step_{self.step}"
        else:
            run_name = f"iteration_{self.iteration}_step_{self.step}"

        # Set the checkpoint directory
        checkpoint_dir = f"{root_dir}/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # ---------------- Lightning setup ----------------

        # Initialise the wandb logger
        if self.config.wandb:

            # Build the logger
            self.wandb_logger = WandbLogger(
                project=self.config.project_name,
                group=self.config.group_name,
                name=run_name,
                save_dir=checkpoint_dir,
            )

            # Get run ID
            self.run_id = self.wandb_logger.experiment.id

        # Else we use no logger and set a dummy run ID
        else:
            self.wandb_logger = None
            self.run_id = "test_run"

        # Initialise the callbacks
        self.lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
            logging_interval="step"
        )
        self.checkpoints = L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=self.config.top_k_checkpoints,
            mode="min",
            dirpath=checkpoint_dir,
        )
        self.early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.config.early_stopping_patience, mode="min"
        )

        # Build trainer
        # Only need to reload dataloaders if we are pretraining and request in config
        reload_dataloaders = (
            True
            if (self.iteration == 0 and self.config.num_pretrain_pieces > 1)
            else False
        )
        self.trainer = L.Trainer(
            accelerator="auto" if (self.config.debug or unit_test) else "gpu",
            num_nodes=self.config.num_nodes,
            devices=(
                "auto" if (self.config.debug or unit_test) else self.config.num_gpus
            ),
            logger=self.wandb_logger,
            callbacks=[self.lr_monitor, self.checkpoints, self.early_stopping],
            max_epochs=self.config.max_epochs,
            enable_progress_bar=self.config.interactive,
            fast_dev_run=unit_test,
            reload_dataloaders_every_n_epochs=reload_dataloaders,
            use_distributed_sampler=False,
        )

        # Get min/max learning rates depending on step
        if self.iteration == 0:
            min_lr = self.config.pt_min_lr
            max_lr = self.config.pt_max_lr
        elif self.step == 1:
            min_lr = self.config.s1_min_lr
            max_lr = self.config.s1_max_lr * (self.config.s1_max_decay**self.iteration)
        else:
            min_lr = self.config.s2_min_lr
            max_lr = self.config.s2_max_lr * (self.config.s2_max_decay**self.iteration)

        # Build lightning module from scratch if we are not given a warm start parth
        if ws_path is None:

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
                log=self.config.wandb,
                debug=self.config.debug,
                # Include the seed just so it is logged to W&B
                seed=self.config.split_seed,
                # Include the OF step for plots
                step=self.step,
                min_lr=min_lr,
                max_lr=max_lr,
                cycle_steps=self.config.cycle_steps,
                warmup_steps=self.config.warmup_steps,
                gamma=self.config.gamma,
                # Everything below here are parameters for the network
                num_classes=1,
                trim=self.config.run_trimmer,
                remove_self_pair=self.config.remove_self_pair,
                embed_dims=self.config.embed_dims,
                pair_input_dim=self.config.pair_input_dim,
                pair_extra_dim=0,
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
            self.l_module = LOfTransformer.load_from_checkpoint(
                ws_path,
                log=self.config.wandb,
                debug=self.config.debug,
                step=self.step,
                min_lr=min_lr,
                max_lr=max_lr,
                cycle_steps=self.config.cycle_steps,
                warmup_steps=self.config.warmup_steps,
                gamma=self.config.gamma,
            )

        # ---------------- Data setup ----------------

        # Get weights for use in training. Define (but do not make!) the weight
        # directory
        weight_dir = f"{root_dir}/weights"

        # Find the data and weight files to use for this iteration and step
        # For step one:
        if self.step == 1:
            use_truth = False
            # If this is pre-training (iteration 0), use the MC train file
            # and Sherpa file
            if self.iteration == 0:
                source_file = self.config.mc_train_path
                target_file = self.config.pretrain_path
                source_weight_file = "root"
                target_weight_file = "root"
            # If this is the first iteration, use the weights from the root file
            # for source and no weights for the target
            elif self.iteration == 1:
                source_file = self.config.mc_train_path
                target_file = self.config.data_path
                source_weight_file = "root"
                target_weight_file = None
            # Otherwise use the weights from the previous step two for the source,
            # and no weights for the target
            else:
                source_file = self.config.mc_train_path
                target_file = self.config.data_path
                source_weight_file = (
                    f"{weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                )
                target_weight_file = None
        # For step two:
        if self.step == 2:
            use_truth = True
            # If this is pre-training (iteration 0), use the MC train file and
            # Sherpa file
            if self.iteration == 0:
                source_file = self.config.mc_train_path
                target_file = self.config.pretrain_path
                source_weight_file = "root"
                target_weight_file = "root"
            # If this is the first iteration, use the weights from step one for target,
            # and the weights from the root file as source.
            elif self.iteration == 1:
                source_file = self.config.mc_train_path
                target_file = self.config.mc_train_path
                source_weight_file = "root"
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
            data_divisor=(
                self.config.num_pretrain_pieces if self.iteration == 0 else 1
            ),  # Only need to use data divisor in pretraining
            total_rank=int(self.config.num_nodes * self.config.num_gpus),
            rank=self.trainer.global_rank,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.batch_size,
            split_seed=self.split_seed,
            dataloader_workers=10,
            testing=False,
            use_truth=use_truth,
        )

    def run(self):
        """run - This function runs the training for the Omnifold classifier.

        Arguments:
        None
        Returns:
        {string} - The run id for this training run
        """

        # Run training
        self.trainer.fit(self.l_module, self.d_module)

        # Close W&B
        if self.config.wandb:
            wandb.finish()

        # Return the run id and best checkpoint path
        return self.run_id, self.checkpoints.best_model_path


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
        "--ws_path",
        type=str,
        default=None,
        help="Path to a model checkpoint to warm start from",
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

    # Run the training
    trainer = OfTrain(
        args.config_path,
        args.iteration,
        args.step,
        seed=args.split_seed,
        index=args.index,
    )
    run_id, best_path = trainer.run()

    # Print the run id and best model path
    rank_zero_info(f"\n###RUN ID###\n{run_id}")
    rank_zero_info(f"\n###BEST MODEL PATH###\n{best_path}")

    # Print something and sleep a bit to flush the output
    print("...")
    time.sleep(10)
