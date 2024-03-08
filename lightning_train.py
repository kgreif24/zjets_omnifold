""" lightning_train.py - This file defines the "OfTrain" class, which handles the setup
and training for an omnifolder classifier using pytorch lightning. It also defines the
"main" function, which will be called from the Omnifolder class as a subprocess for 
easy parallelism.

Author: Kevin Greif
Last updated 03/08/2024
python3
"""

import sys

import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger

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

        # Build data module, since eval is handled by a separate class only need one module
        self.d_module = LOfData(
            mc_file=self.config.mc_train_path,
            data_file=self.config.data_path,
            muon_only=self.config.debug,
            batch_size=self.config.batch_size,
            split_seed=self.config.split_seed,
            testing=False,
            max_tracks=self.config.max_tracks
        )

        # Initialise the wandb logger
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
            enable_progress_bar=self.config.debug
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
        wandb.finish()

        # Return the run id and best checkpoint path
        return self.run_id, self.checkpoints.best_model_path



############## MAIN FUNCTION ##############
        
# This function will be called as a subprocess from the Omnifolder class

def run_train(config_path, iteration, step, return_dict):
    """ run_train - This function is the main entry point for training an
    Omnifold classifier. It is meant to be run as a multiprocess from the Omnifolder
    class functions.

    Arguments:
    config - The path to the configuration file
    iteration - The iteration number for this training run
    step - The step number for this training run
    return_dict - A dictionary for returning information to the main process

    Returns:
    None
    """

    # Train the classifier
    trainer = OfTrain(config_path, iteration, step)
    run_id, best_path = trainer.run()

    # Update the return dictionary
    return_dict['run_id'] = run_id
    return_dict['best_model_path'] = best_path


