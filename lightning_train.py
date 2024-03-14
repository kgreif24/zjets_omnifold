""" lightning_train.py - This file defines the "OfTrain" class, which handles the setup
and training for an omnifolder classifier using pytorch lightning. It also defines the
"main" function, which will be called from the Omnifolder class as a subprocess for 
easy parallelism.

Author: Kevin Greif
Last updated 03/08/2024
python3
"""

import os
import argparse

import numpy as np
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import *

from cli.of_config import OfConfig
from lightning_module import *
from plotting_utils import *


class OfTrain:
    """ OfTrain - This class is meant to handle the setup and training of a single
    Omnifold classifier. The driver code for using this class is in the "main" function
    below.
    """

    def __init__(self, config_path, iteration, step):
        """ __init__ - The init function for this class. It takes the OfConfig object
        used for this run of Omnifold, plus the iteration and step of this training run.

        Arguments:
        config_path - The path of the of config file
        iteration - The iteration number for this training
        step - The step number for this training

        Returns:
        None
        """

        # Store the configuration
        self.config = OfConfig(config_name=config_path)
        self.iteration = iteration
        self.step = step

        # Get weights for use in training. Define (but do not make!) the weight directory
        weight_dir = f"./{self.config.checkpoint_dir}/{self.config.project_name}/weights"

        # Find the weight file to use for this iteration and step
        # If this is iteration zero step one, use the weights from the root files
        if iteration == 0 and step == 1:
            weight_file = None
        # If this is step one but iteration > 0, use the weights from the previous step two
        elif iteration > 0 and step == 1:
            weight_file = f"{weight_dir}/iteration_{iteration-1}_step_2.npz"
        # If this is step two, use the weights from step one
        elif step == 2:
            weight_file = f"{weight_dir}/iteration_{iteration}_step_1.npz"
        else:
            raise ValueError("Invalid iteration and step combination")

        # Build data module, since eval is handled by a separate class only need one module
        self.d_module = LOfData(
            mc_file=self.config.mc_train_path,
            data_file=self.config.data_path,
            weight_path=weight_file,
            muon_only=self.config.debug,
            batch_size=self.config.batch_size,
            split_seed=self.config.split_seed,
            testing=False,
            max_tracks=self.config.max_tracks
        )

        # Initialise the wandb logger
        if self.config.wandb:

            run_name = f"iteration_{self.iteration}_step_{self.step}"
            self.wandb_logger = WandbLogger(
                project=self.config.project_name, 
                group=self.config.group_name,
                name=run_name, 
                save_dir=self.config.checkpoint_dir,
                resume=True
            )

            # Get run ID
            self.run_id = self.wandb_logger.experiment.id

        # Else we use no logger
        else:
            self.wandb_logger = None
            # Set a dummy run ID
            self.run_id = "test_run"

        # Make directories for saving validation plots
        val_dir = f'./{self.config.checkpoint_dir}/{self.config.project_name}/{self.run_id}/val_plots'
        os.makedirs(val_dir, exist_ok=True)

        # Initialise the callbacks
        self.lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        self.checkpoints = L.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch}-{val_loss:.4f}',
            save_top_k=self.config.top_k_checkpoints,
            mode='min'
        )
        self.early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            mode='min'
        )

        # Build trainer
        self.trainer = L.Trainer(
            accelerator='gpu',
            devices=self.config.num_gpus,
            logger=self.wandb_logger,
            callbacks=[self.lr_monitor, self.checkpoints, self.early_stopping],
            max_epochs=self.config.max_epochs,
            log_every_n_steps=50,
            enable_progress_bar=False
        )

        # Build lightning module
        block_params = {
            'dropout': self.config.block_dropout,
            'attn_dropout': self.config.block_attn_dropout,
            'activation_dropout': self.config.block_activation_dropout
        }
        cls_block_params = {
            'dropout': self.config.cls_block_dropout,
            'attn_dropout': self.config.cls_block_attn_dropout,
            'activation_dropout': self.config.cls_block_activation_dropout
        }

        self.l_module = LOfTransformer(
            input_dim=self.config.input_dim,
            val_plots=val_dir,
            debug=self.config.debug,
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
            # Include the seed just so it is logged to W&B
            seed=self.config.split_seed
        )


    def run(self):
        """ run - This function runs the training for the Omnifold classifier.

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



############## MAIN FUNCTION ##############
        
# This function will be called as a subprocess from the Omnifolder class

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the omnifold algorithm')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the configuration file')
    parser.add_argument('--iteration', type=int, default=None, help='The iteration number for this training run')
    parser.add_argument('--step', type=int, default=None, help='The step number for this training run')
    args, unknown = parser.parse_known_args()

    # Run the training
    trainer = OfTrain(args.config_path, args.iteration, args.step)
    run_id, best_path = trainer.run()

    # Print the run id and best model path
    rank_zero_info(f"\n###RUN ID###\n{run_id}")
    rank_zero_info(f"\n###BEST MODEL PATH###\n{best_path}")
