""" run_omnifold.py - A script for exectuing the omnifold algorithm.
All parameters for the algorithm are controlled via yaml files and a command
line interface.

Author: Kevin Greif
Last updated 03/05/2024
python3
"""

import sys, os
# On perlmutter, need to set this environment variable to avoid a conflict
# between how numpy and torch handle multithreading. See this issue for more:
# https://github.com/pytorch/pytorch/issues/37377
os.environ['MKL_THREADING_LAYER'] = 'GNU'
sys.path.append('of_transformer')
sys.path.append('utils')

import argparse
from omnifold import Omnifolder

if __name__ == '__main__':

    # Parse command line arguments, looking for passed in config file
    parser = argparse.ArgumentParser(description='Run the omnifold algorithm')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the configuration file')
    parser.add_argument('--continue_iteration', type=int, default=0, help='The restart iteration number for this run')
    parser.add_argument('--continue_step_two', action='store_true', help='If true, will continue from step two and then proceed as usual')
    parser.add_argument('--ensemble_index', default=-1, type=int, help='The index of the ensemble to run')
    args = parser.parse_args()

    # Run Omnifold!
    of = Omnifolder(args.config_path, continue_iteration=args.continue_iteration, continue_step_two=args.continue_step_two, index=args.ensemble_index)
    of.run_of()