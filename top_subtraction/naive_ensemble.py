""" naive_ensemble.py - This script will create a naive ensemble of weights for the
background subtraction.

It just clips the weights to be between 0 and 1, and then takes an average across
the ensemble

Author: Kevin Greif
Last updated September 13, 2025
"""

import argparse
import glob
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create a naive ensemble of weights")
    parser.add_argument(
        "--input_files", type=str, required=True, help="Path to the input files"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_files = glob.glob(args.input_files)
    print(f"Found {len(input_files)} input files")
    ensemble_weights = []
    for file in input_files:
        raw_weights = np.load(file)["pd_weights"]
        raw_weights = np.where(np.isinf(raw_weights), 1.0, raw_weights)
        raw_weights = np.clip(raw_weights, 0.0, 1.0)
        ensemble_weights.append(raw_weights)
    ensemble_weights = np.array(ensemble_weights)
    weights = np.mean(ensemble_weights, axis=0)
    np.savez(args.output_file, weight=weights)
