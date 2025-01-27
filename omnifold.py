""" omnifold.py - This file contains a class which implements the omnifold
algorithm.

Author: Kevin Greif
Last updated 03/06/2024
python3
"""

import os
import sys
import time
import subprocess
import json

import numpy as np
import wandb

from cli.of_config import OfConfig
from utils.subprocess_utils import capture_subprocess_output


class Omnifolder:
    """Omnifolder - This class implements the omnifold algorithm.
    It is responsible for running the omnifold procedure, calculating weights,
    plotting results, etc.

    It is not responsible for running the training or inference of Omnifold
    classifiers. This is handled in processes spawned by this class.
    """

    def __init__(
        self,
        config_path,
        index=-1,
        use_slurm=True,
    ):
        """__init__ - This function initializes the omnifolder object.

        Arguments:
        config_path - Path to the config file for the omnifold algorithm
        checkpoint_file - Path to a json file which contains the status of a
            terminated run. If this is not None, the run will be resumed from
            the last saved status. If None, run Omnifold from scratch.
        index - The index of the ensemble to run. Add this number to the end of the
            group ID if it is not None

        Returns:
        None
        """

        # Print welcome message
        print("\n\n###################################################")
        print("############## Welcome to Omnifold!! ##############")
        print("###################################################\n\n")

        # Set config path and config object as instance variables
        self.config_path = config_path
        print("Loading config from path", config_path)
        self.cfg = OfConfig(config_name=config_path)

        # Modify the group name if an index is provided
        if index != -1:
            self.cfg.group_name = f"{self.cfg.group_name}_{index}"

        # Make root dir for this run of Omnifold
        self.root_dir = (
            f"{self.cfg.checkpoint_dir}/"
            f"{self.cfg.project_name}/{self.cfg.group_name}"
        )
        os.makedirs(self.root_dir, exist_ok=True)

        # Load status from checkpoint file if it exists in root dir
        status_file = os.path.join(self.root_dir, "status.json")
        if os.path.exists(status_file):
            print(f"Picking up progress from status file {status_file}")
            with open(status_file, "r") as f:
                status = json.load(f)
            self.current_iteration = status["current_iteration"]
            self.current_step = status["current_step"]
            self.training = status["training"]
            self.run_id = status["run_id"]
            self.seed = status["seed"]
            print(f"Removing status file {status_file}")
            os.remove(status_file)

        # Else configure to run from scratch
        else:
            self.current_iteration = 0
            self.current_step = 1
            self.training = True
            self.run_id = None
            self.seed = self.cfg.split_seed

        # Set some instance variables
        if self.current_iteration > self.cfg.num_iterations:
            raise ValueError(
                "Current iteration is greater than total number of iterations!"
            )
        self.end_iteration = self.cfg.num_iterations
        self.index = index
        self.use_slurm = use_slurm
        self.made_checkpoint = False
        self.head_node = os.getenv("SLURMD_NODENAME", None)
        print("Head node is ", self.head_node)

    def run_of(self):
        """run_of - Run the whole Omnifold procedure from start to finish.
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
        for i in range(self.current_iteration, self.end_iteration + 1):  # 1-indexed
            self.current_iteration = i
            print(f"\n\n ##### Running iteration {i} of {self.end_iteration} #####")
            if first_iteration and self.current_step == 2:
                self.run_step(2)
                first_iteration = False
            else:
                self.run_step(1)
                self.run_step(2)

        print("\n############## Omnifold Finished!! ##############\n")

    def pre_train(self):
        """pre_train - This function runs the pre-training step of the omnifold
        algorithm. It will train two networks, a step 1 and a step 2 network.
        These will then be used as the starting point for the trainings
        in the iterations.

        No arguments or returns
        """

        print("\n########## Pre-Training ##########\n")
        for step in [1, 2]:
            print("Running pre-training for step ", step)
            self.run_step(step)

    def run_step(self, step):
        """step_one - This function runs a step of the omnifold algorithm.
        Which step it runs is controlled by the step argument.

        Arguments:
            step - The step of the omnifold algorithm to run. 1 or 2.
        Returns: None
        """

        # Raise a value error if step is not 1 or 2
        if step not in [1, 2]:
            raise ValueError("Step must be 1 or 2!")

        # If we are done training, skip straight to evaluation
        if self.training:
            print(f"\n## Step {step} Training ##\n")

            # Determine seed for train / val split
            if self.seed == -1:
                self.seed = np.random.randint(0, 10000)

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
                str(self.seed),
                "--index",
                str(self.index),
            ]

            # Add slurm arguments if we are using
            if self.use_slurm:
                slurm_args = [
                    "srun",
                    "--nodes",
                    "1" if self.cfg.debug else str(self.cfg.num_nodes),
                    "--ntasks-per-node",
                    "1" if self.cfg.debug else str(self.cfg.num_gpus),
                    "--cpus-per-task",
                    "4",
                    "--cpu_bind=cores",
                    "--mem-per-cpu=10G",
                    "--gpus-per-task",
                    "0" if self.cfg.debug else "1",
                    "--gpu-bind=none",
                    "--overlap",
                ]
                train_args = slurm_args + train_args
            print(train_args)

            # Run training subprocess
            train_code, output = capture_subprocess_output(
                train_args,
            )

            # Exit on non-zero return code
            if train_code != 0:
                print(f"Error running training subprocess! Code {train_code}")
                sys.exit(train_code)

            # Sleep for a bit to ensure all resources are released
            print("Sleeping for 4 minutes")
            time.sleep(240)

            # Reverse search output for run_id
            lines = output.split("\n")
            for i in reversed(range(len(lines))):
                if "###RUN ID###" in lines[i] and i + 1 < len(lines):
                    self.run_id = lines[i + 1]
                    break

        # Only care about running evaluation if this is not a pre-training step
        if self.current_iteration > 0:
            print(f"\n## Step {step} Evaluating ##\n")

            # Run evaluation as a subprocess, no need to keep output
            eval_args = [
                "python",
                "lightning_eval.py",
                "--run_id",
                self.run_id,
                "--config_path",
                self.config_path,
                "--iteration",
                str(self.current_iteration),
                "--step",
                str(step),
                "--index",
                str(self.index),
            ]
            if self.use_slurm:
                slurm_args = [
                    "srun",
                    "--nodes",
                    "1",
                    "--nodelist",
                    str(self.head_node),
                    "--ntasks-per-node",
                    "1",
                    "--cpus-per-task",
                    "128",
                    "--cpu_bind=threads",
                    "--mem-per-cpu=1790M",
                    "--gpus-per-task",
                    "1",
                    "--gpu-bind=none",
                    "--overlap",
                ]
                eval_args = slurm_args + eval_args
            print(eval_args)
            process = subprocess.run(eval_args)
            if process.returncode != 0:
                print(f"Error running evaluation subprocess! Code {process.returncode}")
                sys.exit(process.returncode)

        # Set training flag to True and model paths / IDs to None
        self.training = True
        self.best_model_path = None
        self.run_id = None

        # Increment current step
        self.current_step = (self.current_step % 2) + 1

        print(f"Finished step {step}!!")

    def save_status(self):
        """ save_status - Saves the status of this Omnifold run to a
        json file. This file can then be used to resume the run at a
        later time.

        Arguments: None
        Returns: None
        """

        # Only make a check point once, either on SIGUSR1 or SIGTERM
        if not self.made_checkpoint:

            # Write json file with the config path, current iteration, and
            # any other relevant information
            status = {
                "current_iteration": self.current_iteration,
                "current_step": self.current_step,
                "training": self.training,
                "run_id": self.run_id,
                "seed": self.seed,
            }
            with open(f"{self.root_dir}/status.json", "w") as f:
                json.dump(status, f)

            print("Status saved to ", f"{self.root_dir}/status.json")
            self.made_checkpoint = True
