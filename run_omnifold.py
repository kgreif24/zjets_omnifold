""" run_omnifold.py - A script for exectuing the omnifold algorithm.
All parameters for the algorithm are controlled via yaml files and a command
line interface.

Author: Kevin Greif
Last updated 03/05/2024
python3
"""

import sys
sys.path.append('of_transformer')
sys.path.append('utils')

import argparse
import numpy as np
from cli.of_config import OfConfig
from omnifold import Omnifolder

if __name__ == '__main__':

    # Disable numpy errors
    np.seterr(all='ignore')

    # Parse command line arguments, looking for passed in config file
    parser = argparse.ArgumentParser(description='Run the omnifold algorithm')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the configuration file')
    args = parser.parse_args()
    print(args.config_path)

    # Run Omnifold!
    of = Omnifolder(args.config_path)
    of.run_of()