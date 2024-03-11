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
import lightning_eval as eval
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

        # Login to wandb
        if self.cfg.wandb:
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
            p = mp.Process(
                target=train.run_train, 
                args=(self.config_path, self.current_interation, 1, return_dict)
            )
            p.start()
            p.join()

            # Get run_id from training process
            run_id = return_dict['run_id']

            # Get best model checkpoint path
            best_model_path = return_dict['best_model_path']

        print("Done with step one, run_id was:", run_id)
        print("Best model path is:", best_model_path)

        # Run evaluation as a subprocess, no need for any return information as weight outputs
        # will be written to disk
        p = mp.Process(
            target=eval.run_eval, 
            args=(best_model_path, run_id, self.config_path, self.current_interation, 1)
        )
        p.start()
        p.join()

        print("Finished step one!!")
        sys.exit()
