""" omnifold.py - This file contains a class which implements the omnifold
algorithm. 

Author: Kevin Greif
Last updated 03/06/2024
python3
"""

import sys

import torch
import lightning as L
from lightning_module import *
from pytorch_lightning.utilities.rank_zero import *
import wandb

from cli.of_config import OfConfig
from utils.subprocess_utils import capture_subprocess_output


class Omnifolder():
    """ Omnifolder - This class implements the omnifold algorithm.
    It is responsible for running the omnifold procedure, calculating weights,
    plotting results, etc.

    It is not responsible for running the training or inference of Omnifold
    classifiers. This is handled in processes spawned by this class.
    """

    def __init__(self, config_path, continue_iteration=0, continue_step_two=False):
        """ __init__ - This function initializes the omnifolder object.

        Arguments:
        config_path - Path to the config file for the omnifold algorithm

        Returns:
        None
        """

        # Print welcome message
        print("\n\n###################################################")
        print("############## Welcome to Omnifold!! ##############")
        print("###################################################\n\n")

        print("Is CUDA available: ", torch.cuda.is_available())
        print("CUDA device count: ", torch.cuda.device_count())
        print("CUDA device name: ", torch.cuda.get_device_name(0))

        if continue_iteration != 0:
            print("Continuing from iteration ", continue_iteration)
        if continue_step_two:
            print("Continuing from step two")

        # Set config path and config object as instance variables
        self.config_path = config_path
        self.cfg = OfConfig(config_name=config_path)

        # Set some instance variables for tracking progress through the procedure
        self.current_interation = continue_iteration
        self.continue_step_two = continue_step_two
        self.end_iteration = self.current_interation + self.cfg.num_iterations

        # Login to wandb
        if self.cfg.wandb:
            wandb.login()


    def run_of(self):
        """ run_of - Run the whole Omnifold procedure from start to finish.
        Arguments: None
        Returns: None
        """

        print("\n############## Running Omnifold ##############\n")

        first_iteration = True

        for i in range(self.current_iteration, self.end_iteration):
            self.current_interation = i
            print(f"\n\n ##### Running iteration {i+1} of {self.end_iteration} #####")
            if first_iteration and not self.continue_step_two:
                self.run_step(1)
                first_iteration = False
            self.run_step(2)

        print("\n############## Omnifold Finished!! ##############\n")

        # self.run_finish()


    def run_step(self, step):
        """ step_one - This function runs a step of the omnifold algorithm.
        Which step it runs is controlled by the step argument.

        Arguments:
            step - The step of the omnifold algorithm to run. 1 or 2.
        Returns: None
        """

        # Raise a value error if step is not 1 or 2
        if step not in [1, 2]:
            raise ValueError("Step must be 1 or 2!")

        print(f"\n########## Step {step} Training ##########\n")

        # Run training as a subprocess
        train_args = [
            "srun",
            "--ntasks-per-node", "4",
            "-c", "32",
            "--cpu_bind=cores",
            "-G", "4",
            "--gpu-bind=none",
            "python", 
            "lightning_train.py", 
            "--config_path", 
            self.config_path, 
            "--iteration", 
            str(self.current_interation), 
            "--step", 
            str(step)
        ]
        train_code, output = capture_subprocess_output(train_args)
        if train_code != 0:
            print("Error running training subprocess!")
            sys.exit(train_code)


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

        print(f"\n########## Step {step} Evaluating ##########\n")

        # Run evaluation as a subprocess, no need to keep output
        eval_args = [
            "srun",
            "--ntasks-per-node", "1",
            "-c", "32",
            "--cpu_bind=cores",
            "-G", "1",
            "--gpu-bind=none",
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
            str(step)
        ]
        test_code, _ = capture_subprocess_output(eval_args)
        if test_code != 0:
            print("Error running evaluation subprocess!")
            sys.exit(test_code)

        print(f"Finished step {step}!!")


    # def run_finish(self):
    #     """ run_finish - This function runs the plotting to evaluate the performance
    #     of this Omnifold run.
    #     """