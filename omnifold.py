""" omnifold.py - This file contains a class which implements the omnifold
algorithm. 

Author: Kevin Greif
Last updated 2/19/2024
python3
"""

from lightning_module import *
from cli.of_config import OfConfig


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

        # Set config object as an instance variable
        self.cfg = config

        # Set some instance variables for tracking progress through the procedure
        self.current_interation = 0

        # Build lightning data modules
        self.d_module_train = LOfData(
            mc_file=self.cfg.mc_train_path,
            data_file=self.cfg.data_path,
            muon_only=self.cfg.muon_only,
            batch_size=self.cfg.batch_size,
            dataloader_workers=1,
            split_seed=self.cfg_split_seed,
            testing=False,
            max_tracks=self.cfg.max_tracks
        )
        self.d_module_test = LOfData(
            mc_file=self.cfg.mc_test_path,
            data_file=self.cfg.data_path,
            muon_only=self.cfg.muon_only,
            batch_size=self.cfg.batch_size,
            dataloader_workers=1,
            split_seed=self.cfg_split_seed,
            testing=True,
            max_tracks=self.cfg.max_tracks
        )


    def run_of(self):
        """ run_of - Run the whole Omnifold procedure from start to finish.
        Arguments: None
        Returns: None
        """

        print("############## Running Omnifold ##############")

        for i in range(self.num_interations):
            self.current_interation = i
            print(f"Running iteration {i} of {self.num_interations}")
            self.step_one()
            self.step_two()


    def step_one(self):
        """ step_one - This function runs the first step of the omnifold algorithm.
        Arguments: None
        Returns: None
        """

        print("Running step one")

