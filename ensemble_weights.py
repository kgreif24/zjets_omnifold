""" ensemble_weights.py - This is a script for calculating central value weights
over an ensemble of runs of Omnifold.

Author: Kevin Greif
Last updated 08/27/2024
python3
"""

import argparse
import glob
import numpy as np


# Parse arguments
parser = argparse.ArgumentParser(description='Calculate central value weights over an ensemble of Omnifold runs.')
parser.add_argument('--weight_card', type=str, help='Path to the weights, with ensemble taken over the wildcard *')
parser.add_argument('--reduction', type=str, help='Reduction method to use: mean or median', default='mean', choices=['mean', 'median'])
parser.add_argument('--output', type=str, help='Output file path')
args = parser.parse_args()

# Create glob of weight files
weight_files = sorted(glob.glob(args.weight_card))

# Get the vector names from the first file
vector_names = np.load(weight_files[0]).files

# Prepare results dictionary
results = {}

# Loop over vectors
for vector in vector_names:

    # Take mean or median
    if args.reduction == 'mean':
        results[vector] = np.mean([np.load(file)[vector] for file in weight_files], axis=0)
    elif args.reduction == 'median':
        results[vector] = np.median([np.load(file)[vector] for file in weight_files], axis=0)

# Save results
np.savez(args.output, **results)