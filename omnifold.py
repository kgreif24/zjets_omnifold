""" omnifold.py - This file contains a class which implements the omnifold
algorithm. 

Author: Kevin Greif
Last updated 03/06/2024
python3
"""

import os, sys
import multiprocessing as mp
import numpy as np

import torch
import lightning as L
from lightning_module import *
from pytorch_lightning.loggers import WandbLogger
import wandb

from cli.of_config import OfConfig
import lightning_train as train
# import lightning_eval as eval
import plotting_utils as pu


class Omnifolder():
    """ Omnifolder - This class implements the omnifold algorithm.
    It is responsible for running the omnifold procedure, calculating weights,
    plotting results, etc.

    It is not responsible for running the training or inference of Omnifold
    classifiers. This is handled in processes spawned by this class.
    """

    def __init__(self, config_path):
        """ __init__ - This function initializes the omnifolder object.

        Arguments:
        config_path - Path to the config file for the omnifold algorithm

        Returns:
        None
        """

        # Print welcome message if rank zero
        print("\n\n###################################################")
        print("############## Welcome to Omnifold!! ##############")
        print("###################################################\n\n")

        # Set config path and config object as instance variables
        self.config_path = config_path
        self.cfg = OfConfig(config_name=config_path)

        # Set some instance variables for tracking progress through the procedure
        self.current_interation = 0

        # Lists for keeping track of the weights derived throughout
        self.push_weights_train_hist = []
        self.push_weights_val_hist = []
        self.push_weights_test_hist = []
        self.pull_weights_train_hist = []
        self.pull_weights_val_hist = []
        self.pull_weights_test_hist = []

        # Login to wandb
        wandb.login()

        # Set mp start method to spawn
        mp.set_start_method('spawn')


    def run_of(self):
        """ run_of - Run the whole Omnifold procedure from start to finish.
        Arguments: None
        Returns: None
        """

        print("\n############## Running Omnifold ##############\n")

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
        print("Running step one!")

        # Initialize manager for getting run_id from training process
        with mp.Manager() as manager:

            # Add dictionary to manager
            return_dict = manager.dict()

            # Run training as a subprocess
            p = mp.Process(target=train.run_train, args=(self.config_path, self.current_interation, 1, return_dict))
            p.start()
            p.join()

            # Get run_id from training process
            run_id = return_dict['run_id']

            # Get best model checkpoint path
            best_model_path = return_dict['best_model_path']

        print("Done with step one, run_id is:", run_id)
        print("Best model path is:", best_model_path)

        sys.exit()

        # Run evaluation as a subprocess, no need for any return information

        # Setup amd run training
        name = f'iteration_{self.current_interation}_step1'
        lightning_module, trainer = self.setup_training(name)
        trainer.fit(lightning_module, self.d_module_train)

        # At thisp point only want to use the rank zero process to run testing
        torch.distributed.destroy_process_group()
        print(f"I am a rank {trainer.local_rank} process")
        if trainer.is_global_zero:

            # Recover best checkpoint
            lightning_module = LOfTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, debug=self.cfg.debug)

            # Produce a new trainer for testing and predictions
            test_trainer = L.Trainer(
                accelerator='gpu',
                devices=1,
                logger=False,
                enable_progress_bar=self.cfg.debug
            )

            # Run testing
            test_trainer.test(lightning_module, self.d_module_test)

            # # Derive weights
            # derived_weights_train, derived_weights_val, derived_weights_test = self.calc_all_weights(lightning_module, test_trainer)

            # # Update pull weights
            # self.pull_weights_train = self.push_weights_train * derived_weights_train
            # self.pull_weights_val = self.push_weights_val * derived_weights_val
            # self.pull_weights_test = self.push_weights_test * derived_weights_test

            # # Append weights to histories
            # self.pull_weights_train_hist.append(self.pull_weights_train)
            # self.pull_weights_val_hist.append(self.pull_weights_val)
            # self.pull_weights_test_hist.append(self.pull_weights_test)

        # If these are not the rank zero process, kill them!
        else:
            sys.exit(0)

        # Finish wandb
        wandb.finish()


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

        # Garther predictions from all processes


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
