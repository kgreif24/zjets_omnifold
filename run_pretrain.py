""" run_pretrain.py - This file uses the OfTrain class to run a pre-training, which
produces a model checkpoint that can be used as a starting point for all of the 
trainings within Omnifold.

The pretraining task is defined as discriminating MadGraph events from Sherpa events.

Author: Kevin Greif
Last updated 02/21/2025
python3
"""

import argparse
import atexit
import random
from lightning_train import OfTrain
from utils.subprocess_utils import cleanup_resources


# Register GPU cleanup at exit
atexit.register(cleanup_resources)

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Run a pre-training of the Omnifold algorithm."
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/pretrain.yaml",
    help="Path to the config file for the omnifold algorithm",
)
parser.add_argument(
    "--split_seed",
    type=int,
    default=333,
    help="Seed for the train / val split",
)
parser.add_argument(
    "--index",
    type=int,
    default=-1,
    help="Index of the run within an ensemble",
)
parser.add_argument(
    "--step",
    type=int,
    default=1,
    help="Step of the Omnifold algorithm",
)
args = parser.parse_args()

# Get random seed if requested
if args.split_seed == -1:
    args.split_seed = random.randint(0, 1000000)

# Build trainer
trainer = OfTrain(
    args.config,
    0,
    args.step,
    seed=args.split_seed,
    index=args.index,
)

# Run the pre-training
trainer.run()
