""" run_omnifold.py - A script for exectuing the omnifold algorithm.
All parameters for the algorithm are controlled via yaml files and a command
line interface.

Author: Kevin Greif
Last updated 03/05/2024
python3
"""

import argparse
from cli.of_config import OfConfig
from omnifold import Omnifolder

if __name__ == '__main__':

    # Parse command line arguments, looking for passed in config file
    parser = argparse.ArgumentParser(description='Run the omnifold algorithm')
    parser.add_argument('--config', type=str, default=None, help='Path to the configuration file')

    # Make the config object
    config = OfConfig(existing_parser=parser)

    # Run Omnifold!
    of = Omnifolder(config)
    of.run_of()