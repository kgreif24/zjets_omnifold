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
import numpy as np
import uproot
import awkward as ak

from cli.of_config import OfConfig


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

        # Make root dir and weight dir for this run of Omnifold
        self.root_dir = (
            f"{self.cfg.checkpoint_dir}/"
            f"{self.cfg.project_name}/{self.cfg.group_name}"
        )
        os.makedirs(self.root_dir, exist_ok=True)
        self.weight_dir = f"{self.root_dir}/weights"
        os.makedirs(self.weight_dir, exist_ok=True)

        # Infer run status from directory structure
        status = self._infer_next_step()
        self.current_iteration = status[0]
        self.current_step = status[1]
        self.training = status[2]

        # Set some instance variables
        if self.current_iteration > self.cfg.num_iterations:
            raise ValueError(
                "Current iteration is greater than total number of iterations!"
            )
        self.end_iteration = self.cfg.num_iterations
        self.index = index
        self.use_slurm = use_slurm
        self.made_checkpoint = False

        # Set seeds for running bootstraps if enabled
        self.data_bootstrap_path = None
        if self.cfg.bootstrap_data:
            self.data_bootstrap_path = self._generate_data_bootstrap()

    def run_of(self):
        """run_of - Run the whole Omnifold procedure from start to finish.
        Arguments: None
        Returns: None
        """

        print("\n############## Running Omnifold ##############\n")

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

            # Get seed
            seed = self.cfg.split_seed
            if seed == -1:
                seed = np.random.randint(10000)

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
                "--index",
                str(self.index),
                "--split_seed",
                str(seed),
            ]

            # Add bootstrap seeds if enabled
            if self.cfg.bootstrap_data and step == 1:
                train_args.append("--data_bootstrap_path")
                train_args.append(str(self.data_bootstrap_path))

            # Add slurm arguments if we are using
            if self.use_slurm:
                slurm_args = [
                    "srun",
                    "--nodes",
                    "1" if self.cfg.debug else str(self.cfg.num_nodes),
                    "--ntasks-per-node",
                    "1" if self.cfg.debug else str(self.cfg.num_gpus),
                    "--cpus-per-task",
                    "32",
                    "--cpu-bind=none",
                    "--mem-per-cpu=1790M",
                    "--gpus-per-task",
                    "0" if self.cfg.debug else "1",
                    "--gpu-bind=none",
                    "--overlap",
                ]
                train_args = slurm_args + train_args
            print(train_args)

            # Run training subprocess
            process = subprocess.run(train_args)
            if process.returncode != 0:
                print(f"Error running training subprocess! Code {process.returncode}")
                sys.exit(process.returncode)

            # Sleep for a bit to ensure all resources are released
            print("Sleeping for 10 seconds")
            time.sleep(10)

            # Set flag to mark training finished
            self.training = False

        # Run evaluation
        print(f"\n## Step {step} Evaluating ##\n")

        # Run testing and predictions as two separate subprocesses
        for i in range(2):
            eval_args = self._create_eval_args(is_test=i == 0)
            print(eval_args)
            process = subprocess.run(eval_args)
            if process.returncode != 0:
                print(f"Error running evaluation subprocess! Code {process.returncode}")
                sys.exit(process.returncode)

        # Set training flag to True
        self.training = True

        # Increment current step
        self.current_step = (self.current_step % 2) + 1

        print(f"Finished step {step}!!")

    def _create_eval_args(self, is_test):
        """_create_eval_args - This function creates the arguments for testing and
        prediction subprocesses. It is used to avoid code duplication.
        Arguments:
            is_test - True if we are running the test, False if we are running
                the prediction
        Returns:  
            {list} - List of arguments for the subprocess
        """

        # Create common arguments
        eval_args = [
            "python",
            "lightning_eval.py",
            "--config_path",
            self.config_path,
            "--iteration",
            str(self.current_iteration),
            "--step",
            str(self.current_step),
            "--index",
            str(self.index),
        ]
        slurm_args = [
            "srun",
            "--ntasks-per-node",
            "1",
            "--cpus-per-task",
            "128",
            "--cpu-bind=none",
            "--mem-per-cpu=1790M",
            "--gpus-per-task",
            "0" if self.cfg.debug else "1",
            "--gpu-bind=none",
            "--overlap",
        ]

        # Add test argument
        if is_test:
            eval_args.append("--run_test")
        if self.use_slurm:
            if not is_test:
                slurm_args += ["--nodes", str(self.cfg.num_nodes)]
            else:
                slurm_args += ["--nodes", "1"]
            eval_args = slurm_args + eval_args

        return eval_args

    def _infer_next_step(self):
        """_infer_next_step - This function will examine the directory structure for
        the run of Omnifold and determine what the next step in the procedure is.
        It will return the current iteration and step, as well as whether we should
        proceed with training or evaluation.

        No arguments
        Returns:
            {int} - Current iteration
            {int} - Current step
            {bool} - True if we are training, False if we are evaluating
        """

        # Loop through iterations
        for iteration in range(1, self.cfg.num_iterations + 1):
            # Loop through steps
            for step in range(1, 3):

                # Move on if we have weights for this iteration / step
                weight_path = f"{self.weight_dir}/iteration_{iteration}_step_{step}.npz"
                if os.path.exists(weight_path):
                    continue

                # If we don't have weights, we need to do work on this iteration / step
                # Check if we have a best model path, if so we are done training
                model_path = (
                    f"{self.root_dir}/iteration_{iteration}_step_{step}/best_model.ckpt"
                )
                if os.path.exists(model_path):
                    training = False
                else:
                    training = True

                # Return results
                return iteration, step, training

        # If loop concludes, we are done with the procedure
        return self.cfg.num_iterations, 2, False

    def _generate_data_bootstrap(self):
        """_generate_data_bootstrap - This function generates the data bootstrap to be
        used throuhgout the rest of the Omnifold procedure.

        Arguments: None
        Returns:
            {str} - The path to the bootstrap weights
        """

        # Load the data using uproot to understand the number of events
        f = uproot.open(self.cfg.data_path)
        tree = f["OmniTree"]
        p190 = ak.to_numpy(tree["pass190"].array())

        # Sample Poisson to get the bootstrap
        bs_seed = 100 + self.index
        rng = np.random.default_rng(bs_seed)
        bs_weights = rng.poisson(lam=1.0, size=len(p190))

        # Save the bootstrap weights to disk
        location = f"{self.root_dir}/bootstrap_s{bs_seed}.npy"
        np.save(location, bs_weights)

        # Return the location of the bootstrap weights
        return location


if __name__ == "__main__":

    # Create Omnifolder object and print state
    omnifolder = Omnifolder("./cli/base_ensemble.yml", index=6)
    print(omnifolder.current_iteration)
    print(omnifolder.current_step)
    print(omnifolder.training)
