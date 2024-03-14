""" omnifold.py - This file contains a class which implements the omnifold
algorithm. 

Author: Kevin Greif
Last updated 03/06/2024
python3
"""

import os, sys
import subprocess
import numpy as np

import torch
import lightning as L
from lightning_module import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import *
import wandb

from cli.of_config import OfConfig
import lightning_train as train
import lightning_eval as eval
import plotting_utils as pu
from subprocess_utils import capture_subprocess_output


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

        # Run training as a subprocess
        train_args = [
            "python", 
            "lightning_train.py", 
            "--config_path", 
            self.config_path, 
            "--iteration", 
            str(self.current_interation), 
            "--step", 
            "1"
        ]
        train_code, output = capture_subprocess_output(train_args)

        # Reverse search output for run_id and best model path
        lines = output.split("\n")
        found_id = False
        found_path = False
        for i in reversed(range(len(lines))):
            if "###RUN ID###" in lines[i] and i+1 < len(lines):
                run_id = lines[i + 1]
                found_id = True
            if "###BEST MODEL PATH###" in lines[i] and i+1 < len(lines):
                best_model_path = lines[i + 1]
                found_path = True
            if found_id and found_path:
                break

        # Run evaluation as a subprocess, no need to keep output
        eval_args = [
            "python", 
            "lightning_eval.py",
            "--check_path",
            best_model_path,
            "--run_id",
            run_id,
            "--config_path", 
            self.config_path, 
            "--iteration", 
            str(self.current_interation), 
            "--step", 
            "1"
        ]
        subprocess.run(eval_args, check=True)

        print("Finished step one!!")
        sys.exit()
