""" lightning_train.py - This file defines the "OfTrain" class, which handles the setup
and training for an omnifolder classifier using pytorch lightning. It also defines the
"main" function, which will be called from the Omnifolder class as a subprocess for 
easy parallelism.

Author: Kevin Greif
Last updated 03/08/2024
python3
"""

import sys, os
import argparse

import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import *

from cli.of_config import OfConfig
from lightning_module import *
from utils.plotting_utils import *


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
        weight_dir = f"./{self.config.checkpoint_dir}/{self.config.project_name}/{self.config.group_name}/weights"

        # Find the data and weight files to use for this iteration and step. Also set the maximum number of events
        # to use (two copies of MC used for step two currently does not fit in memory)
        # For step one:
        if self.step == 1:
            use_truth = False
            source_file = self.config.mc_train_path
            target_file = self.config.data_path
            # If this is the first iteration, use the weights from the root file for source 
            # and no weights for the target
            if self.iteration == 0:
                source_weight_file = 'root'
                target_weight_file = None
            # Otherwise use the weights from the previous step two for the source, and no
            # weights for the target
            else:
                source_weight_file = f"{weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                target_weight_file = None
            max_events_source = self.config.max_train_step_one
            max_events_target = self.config.max_train_step_one
        # For step two:
        if self.step == 2:
            use_truth = True
            source_file = self.config.mc_train_path
            target_file = self.config.mc_train_path
            # If this is the first iteration, use the weights from step one for target, and the
            # weights from the root file as source.
            if self.iteration == 0:
                source_weight_file = 'root'
                target_weight_file = f"{weight_dir}/iteration_{self.iteration}_step_1.npz"
            # Otherwise use the weights from the previous step two for source, and the weights
            # from the previous step one for target
            else:
                source_weight_file = f"{weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                target_weight_file = f"{weight_dir}/iteration_{self.iteration}_step_1.npz"
            max_events_source = self.config.max_train_step_two
            max_events_target = self.config.max_train_step_two

        # If split seed is -1, we generate a new seed for this run
        if self.config.split_seed == -1:
            split_seed = np.random.randint(0, 1000000)
        else:
            split_seed = self.config.split_seed

        # Build the data module
        self.d_module = LOfData(
            source_file=source_file,
            target_file=target_file,
            source_weight_path=source_weight_file,
            target_weight_path=target_weight_file,
            max_events_source=max_events_source,
            max_events_target=max_events_target,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.batch_size,
            split_seed=split_seed,
            dataloader_workers=10,
            load_all=False,
            testing=False,
            use_truth=use_truth
        )

        # Initialise the wandb logger
        if self.config.wandb:

            run_name = f"iteration_{self.iteration}_step_{self.step}"
            self.wandb_logger = WandbLogger(
                project=self.config.project_name, 
                group=self.config.group_name,
                name=run_name, 
                save_dir=self.config.checkpoint_dir
            )

            # Get run ID
            self.run_id = self.wandb_logger.experiment.id

        # Else we use no logger
        else:
            self.wandb_logger = None
            # Set a dummy run ID
            self.run_id = "test_run"

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
            enable_progress_bar=self.config.interactive
        )

        # Make directories for saving validation plots in the rank zero process
        val_dir = f'./{self.config.checkpoint_dir}/{self.config.project_name}/{self.run_id}/val_plots'
        if self.trainer.global_rank == 0 and self.config.plot_val:
            os.makedirs(val_dir, exist_ok=True)
        else:
            val_dir = None

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
            log=self.config.wandb,
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
            seed=split_seed,
            # Include the OF step for plots
            step=self.step
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
    parser.add_argument('--step', type=int, default=None, help='The step number for this training run, either 1 or 2')
    args, unknown = parser.parse_known_args()

    # Run the training
    trainer = OfTrain(args.config_path, args.iteration, args.step)
    run_id, best_path = trainer.run()

    # Print the run id and best model path
    rank_zero_info(f"\n###RUN ID###\n{run_id}")
    rank_zero_info(f"\n###BEST MODEL PATH###\n{best_path}")
