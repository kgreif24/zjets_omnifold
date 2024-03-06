""" omnifold.py - This file contains a class which implements the omnifold
algorithm. 

Author: Kevin Greif
Last updated 03/06/2024
python3
"""

import lightning as L
from lightning_module import *
from pytorch_lightning.loggers import WandbLogger
import wandb

import numpy as np
import os

import plotting_utils as pu


class Omnifolder():
    """ Omnifolder - At the moment I have no idea what I am doing...
    """

    def __init__(self, config):
        """ __init__ - This function initializes the omnifolder object.

        Arguments:
        config - An of_config object which contains all the hyperparameters for the omnifold algorithm.

        Returns:
        None
        """

        print("###################################################")
        print("############## Welcome to Omnifold!! ##############")
        print("###################################################")

        # Set config object as an instance variable
        self.cfg = config

        # Set some instance variables for tracking progress through the procedure
        self.current_interation = 0

        # Build lightning data modules. Only want one worker so long as all of the data fits in memory
        print("Build train / val data modules...")
        self.d_module_train = LOfData(
            mc_file=self.cfg.mc_train_path,
            data_file=self.cfg.data_path,
            muon_only=self.cfg.debug,
            batch_size=self.cfg.batch_size,
            dataloader_workers=1,
            split_seed=self.cfg.split_seed,
            testing=False,
            max_tracks=self.cfg.max_tracks
        )
        self.d_module_train.setup(stage='train')

        print("Build test data module...")
        self.d_module_test = LOfData(
            mc_file=self.cfg.mc_test_path,
            data_file=self.cfg.data_path,
            muon_only=self.cfg.debug,
            batch_size=self.cfg.batch_size,
            dataloader_workers=1,
            split_seed=self.cfg.split_seed,
            testing=True,
            max_tracks=self.cfg.max_tracks
        )

        # Lists for keeping track of the weights derived throughout
        self.push_weights_train_hist = []
        self.push_weights_val_hist = []
        self.push_weights_test_hist = []
        self.pull_weights_train_hist = []
        self.pull_weights_val_hist = []
        self.pull_weights_test_hist = []

        # Get starting weights from data modules
        start_weights_train = self.d_module_train.train_dataset[:][3].cpu().numpy().flatten()
        start_weights_val = self.d_module_train.val_dataset[:][3].cpu().numpy().flatten()
        start_weights_test = self.d_module_test.all_dataset[:][3].cpu().numpy().flatten()

        # Set current push weights to start weights for init
        self.push_weights_train = start_weights_train
        self.push_weights_val = start_weights_val
        self.push_weights_test = start_weights_test

        # Set current pull weights to one (will be updated)
        self.pull_weights_train = np.ones_like(start_weights_train)
        self.pull_weights_val = np.ones_like(start_weights_val)
        self.pull_weights_test = np.ones_like(start_weights_test)

        # Login to wandb
        wandb.login()


    def run_of(self):
        """ run_of - Run the whole Omnifold procedure from start to finish.
        Arguments: None
        Returns: None
        """

        print("############## Running Omnifold ##############")

        for i in range(self.cfg.num_iterations):
            self.current_interation = i
            print(f"\n\n ##### Running iteration {i+1} of {self.cfg.num_iterations} #####")
            self.step_one()
            break
            # self.step_two()


    def step_one(self):
        """ step_one - This function runs the first step of the omnifold algorithm.
        Arguments: None
        Returns: None
        """

        print("Running step one")

        # Setup amd run training
        name = f'iteration_{self.current_interation}_step1'
        lightning_module, trainer = self.setup_training(name)
        trainer.fit(lightning_module, self.d_module_train)

        # Recover best checkpoint
        lightning_module = LOfTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, debug=self.cfg.debug)

        # Run testing
        trainer.test(lightning_module, self.d_module_test)

        # Derive weights
        derived_weights_train, derived_weights_val, derived_weights_test = self.calc_all_weights(lightning_module, trainer)

        # Update pull weights
        self.pull_weights_train = self.push_weights_train * derived_weights_train
        self.pull_weights_val = self.push_weights_val * derived_weights_val
        self.pull_weights_test = self.push_weights_test * derived_weights_test
    
        # Append weights to histories
        self.pull_weights_train_hist.append(self.pull_weights_train)
        self.pull_weights_val_hist.append(self.pull_weights_val)
        self.pull_weights_test_hist.append(self.pull_weights_test)

        # Finish wandb
        wandb.finish()


    def setup_training(self, name):
        """ setup_training - This function sets up a single network training. It initializes
        a wandb logger, training callbacks, trainer object, and lightning module object.

        Arguments:
        name {string} - The name of this training run for logging to wandb
        Returns:
        l_module - A lightning module object
        trainer - A lightning trainer object
        """

        # Initialise the wandb logger and callbacks
        wandb_logger = WandbLogger(project=self.cfg.project_name, group=self.cfg.group_name, name=name, save_dir=self.cfg.checkpoint_dir)

        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        checkpoints = L.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch}-{val_loss:.4f}',
            save_top_k=self.cfg.top_k_checkpoints,
            mode='min'
        )
        early_stopping = L.pytorch.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.cfg.early_stopping_patience,
            mode='min'
        )

        # Setup directory structure for storing plots
        self.val_plot_store_location = f"{self.cfg.checkpoint_dir}/{self.cfg.project_name}/{wandb_logger.experiment._run_obj.run_id}/val_plots"
        self.test_plot_store_location = f"{self.cfg.checkpoint_dir}/{self.cfg.project_name}/{wandb_logger.experiment._run_obj.run_id}/test_plots"
        os.makedirs(self.val_plot_store_location)
        os.makedirs(self.test_plot_store_location)

        # Build trainer
        trainer = L.Trainer(
            accelerator='gpu',
            devices=self.cfg.num_gpus,
            logger=wandb_logger,
            callbacks=[lr_monitor, checkpoints, early_stopping],
            max_epochs=self.cfg.max_epochs,
            enable_progress_bar=self.cfg.debug
        )

        # Build lightning module
        block_params = {
            'dropout': self.cfg.block_dropout, 
            'attn_dropout': self.cfg.block_attn_dropout, 
            'activation_dropout': self.cfg.block_activation_dropout,
        }
        cls_block_params = {
            'dropout': self.cfg.cls_block_dropout, 
            'attn_dropout': self.cfg.cls_block_attn_dropout, 
            'activation_dropout': self.cfg.cls_block_activation_dropout,
        }

        l_module = LOfTransformer(
            input_dim=self.cfg.input_dim,
            val_plots=self.val_plot_store_location, 
            test_plots=self.test_plot_store_location,
            debug=self.cfg.debug,
            seed=self.cfg.split_seed,
            num_classes=1,
            trim=self.cfg.run_trimmer,
            remove_self_pair=self.cfg.remove_self_pair,
            pair_input_dim=self.cfg.pair_input_dim,
            pair_extra_dim=0,
            embed_dims=self.cfg.embed_dims,
            pair_embed_dims=self.cfg.pair_embed_dims,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_layers,
            num_cls_layers=self.cfg.num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            fc_nodes=self.cfg.fc_nodes,
            fc_dropout=self.cfg.fc_dropout
        )

        return l_module, trainer

    def calc_all_weights(self, module, trainer):
        """ calc_all_weights - This function runs prediction over the given lightning module for all of the 
        train / val / test sets. It returns the derived weights.

        Arguments:
        module - A lightning module object
        trainer - A lightning trainer object
        Returns:
        (array, array, array) - The derived weights for the train, val, and test sets, as numpy arrays
        """

        # Run predictions for the train, validation, and test sets
        predictions_train = trainer.predict(module, self.d_module_train.train_dataloader())
        predictions_val = trainer.predict(module, self.d_module_train.val_dataloader())
        predictions_test = trainer.predict(module, self.d_module_test)

        # Send predictions to CPU, convert to numpy, and concatenate
        predictions_train = np.concatenate([pred.cpu().numpy().flatten() for pred in predictions_train])
        predictions_val = np.concatenate([pred.cpu().numpy().flatten() for pred in predictions_val])
        predictions_test = np.concatenate([pred.cpu().numpy().flatten() for pred in predictions_test])

        # Calculate weights from predictions
        probs_train = 1 / (1 + np.exp(-predictions_train))
        derived_weights_train = probs_train / (1 - probs_train)
        probs_val = 1 / (1 + np.exp(-predictions_val))
        derived_weights_val = probs_val / (1 - probs_val)
        probs_test = 1 / (1 + np.exp(-predictions_test))
        derived_weights_test = probs_test / (1 - probs_test)

        # Return derived weights
        return derived_weights_train, derived_weights_val, derived_weights_test
