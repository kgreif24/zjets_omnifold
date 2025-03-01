""" run_omnifold.py - A script for exectuing the omnifold algorithm.
All parameters for the algorithm are controlled via yaml files and a command
line interface.

Author: Kevin Greif
Last updated 03/05/2024
python3
"""

import os
import argparse
import time
import signal
from omnifold import Omnifolder

# On perlmutter, need to set this environment variable to avoid a conflict
# between how numpy and torch handle multithreading. See this issue for more:
# https://github.com/pytorch/pytorch/issues/37377
os.environ["MKL_THREADING_LAYER"] = "GNU"

if __name__ == "__main__":

    # Parse command line arguments, looking for passed in config file
    parser = argparse.ArgumentParser(description="Run the omnifold algorithm")
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "--ensemble_index",
        default=-1,
        type=int,
        help="The index of the ensemble to run",
    )
    args = parser.parse_args()

    # Build Omnifolder class
    of = Omnifolder(
        args.config_path,
        index=args.ensemble_index,
    )

    # Signal handling function
    def handle_signal(signum, frame):
        if signum == signal.SIGTERM or signum == signal.SIGUSR1:
            print(f"Caught signal {signum}")
            print(f"At time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            print("Creating checkpoint...")
            of.save_status()

    # Register signal handlers
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run the omnifold algorithm
    of.run_of()
