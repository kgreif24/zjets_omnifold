""" omnifold.py - This file contains a class which implements the omnifold
algorithm. 

Author: Kevin Greif
Last updated 03/06/2024
python3
"""

import sys
import subprocess

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

    def __init__(self, config_path, continue_iteration=0, continue_step_two=False, index=None, use_slurm=True):
        """ __init__ - This function initializes the omnifolder object.

        Arguments:
        config_path - Path to the config file for the omnifold algorithm
        continue_iteration - The iteration to continue from
        continue_step_two - If true, continue from step two
        index - The index of the ensemble to run. Add this number to the end of the group ID
             if it is not None
        use_slurm - If true, preprend training and evaluation commands with slurm directives

        Returns:
        None
        """

        # Print welcome message
        print("\n\n###################################################")
        print("############## Welcome to Omnifold!! ##############")
        print("###################################################\n\n")

        cuda_available = torch.cuda.is_available()
        print("Is CUDA available: ", cuda_available)
        if cuda_available:
            print("CUDA device count: ", torch.cuda.device_count())
            print("CUDA device name: ", torch.cuda.get_device_name(0))

        # Set config path and config object as instance variables
        self.config_path = config_path
        print("Omnifolder class trying to load config from file", config_path)
        self.cfg = OfConfig(config_name=config_path)
        print("Checkpoint path: ", self.cfg.checkpoint_dir)

        # Handle paths for warm starting if we are not starting from scratch
        if continue_iteration != 0:
            print("Continuing from iteration ", continue_iteration)
            self.step_one_ws_path = self.cfg.pt_step_one_checkpoint
            self.step_two_ws_path = self.cfg.pt_step_two_checkpoint
        if continue_step_two:
            print("Continuing from step two")

        # Set some instance variables for tracking progress through the procedure
        self.current_iteration = continue_iteration
        self.continue_step_two = continue_step_two
        self.end_iteration = self.current_iteration + self.cfg.num_iterations
        self.index = index
        self.use_slurm = use_slurm

        # Login to wandb
        if self.cfg.wandb:
            wandb.login()


    def run_of(self):
        """ run_of - Run the whole Omnifold procedure from start to finish.
        Arguments: None
        Returns: None
        """

        print("\n############## Running Omnifold ##############\n")

        # Pre-train step if this is iteration 0
        if self.current_iteration == 0:
            self.pre_train()
            self.current_iteration = 1

        # Omnifold Loop
        first_iteration = True
        for i in range(self.current_iteration, self.end_iteration+1):  # 1-indexed
            self.current_iteration = i
            print(f"\n\n ##### Running iteration {i} of {self.end_iteration} #####")
            if first_iteration and self.continue_step_two:
                self.run_step(2)
                first_iteration = False
            else:
                self.run_step(1)
                self.run_step(2)

        print("\n############## Omnifold Finished!! ##############\n")


    def pre_train(self):
        """ pre_train - This function runs the pre-training step of the omnifold algorithm.
        It will train two networks, a step 1 and a step 2 network. These will then be
        used as the starting point for the trainings in the iterations.

        No arguments or returns
        """

        print("\n########## Pre-Training ##########\n")
        for step in [1, 2]:
            print("Running pre-training for step ", step)
            self.run_step(step, pt=True)


    def run_step(self, step, pt=False):
        """ step_one - This function runs a step of the omnifold algorithm.
        Which step it runs is controlled by the step argument.

        Arguments:
            step - The step of the omnifold algorithm to run. 1 or 2.
            pt - If true, indicates that this is a pre-training step and best model
                checkpoints will be saved to warm start all subsequent trainings
        Returns: None
        """

        # Raise a value error if step is not 1 or 2
        if step not in [1, 2]:
            raise ValueError("Step must be 1 or 2!")

        print(f"\n## Step {step} Training ##\n")

        # Determine seed for train / val split
        seed = self.cfg.split_seed
        if seed == -1:
            seed = np.random.randint(0, 10000)

        # Determine checkpoint path to warm start from, if any
        ws_path = None
        if not pt:
            if step == 1:
                ws_path = self.step_one_ws_path
            elif step == 2:
                ws_path = self.step_two_ws_path

        # Run training as a subprocess
        train_args = [
            "python", 
            "lightning_train.py", 
            "--config_path", 
            self.config_path, 
            "--iteration", 
            str(self.current_iteration), 
            "--step", 
            str(step),
            "--split_seed",
            str(seed),
            "--index",
            str(self.index)
        ]
        # Add warm start path if it exists
        if ws_path is not None:
            train_args += ["--ws_path", ws_path]
        # Add slurm args if requested
        if self.use_slurm:
            slurm_args = [
                "srun",
                "--ntasks-per-node", str(self.cfg.num_gpus),
                "-c", "32",
                "--cpu_bind=cores",
                "-G", str(self.cfg.num_gpus),
                "--gpu-bind=none",
                "--mem", "256G"
            ]
            train_args = slurm_args + train_args
        print(train_args)
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

        # Save best model path to warm start future trainings
        if pt:
            if step == 1:
                self.step_one_ws_path = best_model_path
            elif step == 2:
                self.step_two_ws_path = best_model_path

        print(f"\n## Step {step} Evaluating ##\n")

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
            str(self.current_iteration), 
            "--step", 
            str(step),
            "--index",
            str(self.index)
        ]
        if self.use_slurm:
            slurm_args = [
                "srun",
                "-n", "1",
                "-c", "64",
                "--cpu_bind=cores",
                "-G", "1",
                "--gpu-bind=none",
                "--mem", "256G"
            ]
            eval_args = slurm_args + eval_args
        print(eval_args)
        test_code = subprocess.run(eval_args)
        if test_code.returncode != 0:
            print("Error running evaluation subprocess!")
            sys.exit(test_code.returncode)

        print(f"Finished step {step}!!")
