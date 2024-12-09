""" run_omnifold.py - A script for exectuing the omnifold algorithm.
All parameters for the algorithm are controlled via yaml files and a command
line interface.

Author: Kevin Greif
Last updated 03/05/2024
python3
"""

import os
import argparse
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
        "--checkpoint_file",
        type=str,
        default=None,
        help="Path to the checkpoint json file which describes a terminated run",
    )
    parser.add_argument(
        "--ensemble_index",
        default=-1,
        type=int,
        help="The index of the ensemble to run",
    )
    args = parser.parse_args()

    # Global variable to track subprocesses
    subprocesses = []

    # Build Omnifolder class
    of = Omnifolder(
        args.config_path,
        checkpoint_file=args.checkpoint_file,
        index=args.ensemble_index,
        subprocesses=subprocesses,
    )

    # Signal handler function
    def handle_signal(signum, frame):
        print(f"Caught signal {signum}, making checkpoints and exiting")
        of.save_status()
        for process in subprocesses:
            if process.poll() is None:
                # os.killpg(os.getpgid(process.pid), signum)
                print(f"Have running process {process.pid}! Could propagate")

    # Register signal handler
    signal.signal(signal.SIGUSR1, handle_signal)

    # Run the omnifold algorithm
    of.run_of()
