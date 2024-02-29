""" lightning_eval.py - This program will evaluate a model checkpoint
over a slice of the OF data that depends on the random seed passed to the lightning
data module. It will save a numpy array of the model outputs at a specified location.

Usage:
python lightning_eval.py --model ./path/to/checkpoint --save ./path/to/save --seed <seed>

Author: Kevin Greif
Last updated 02/02/2024
python3
"""

import os
import lightning as L
from lightning_module import *
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score

import plotting_utils as pu

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Evaluate a model checkpoint over a slice of the OF data')
predict_or_plot = parser.add_mutually_exclusive_group(required=True)
predict_or_plot.add_argument('--predictions', type=str, help='Path to the model predictions (if already made)')
predict_or_plot.add_argument('--model', type=str, help='Path to the model checkpoint')
parser.add_argument('--save', type=str, help='Path to save the model outputs')
parser.add_argument('--seed', type=int, default=420, help='Random seed for the data module, only needed if using validation set')
parser.add_argument('--validate', action='store_true', help='Use the validation set instead of the test set')
args = parser.parse_args()

# Make data module
d_module = LOfData(
    mc_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test.root',
    data_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Jan30_Combined_All.root',
    muon_only=False,
    batch_size=256,
    dataloader_workers=1,
    split_seed=args.seed,
    testing=not args.validate
)

# Load model checkpoint and run prediction
if args.model:
    model = LOfTransformer.load_from_checkpoint(args.model)

    # Make lightning trainer
    trainer = L.Trainer(accelerator='gpu', devices=1)

    # Run predictions
    predictions = trainer.predict(model, d_module)

    # Save predictions
    predictions = np.concatenate([pred.cpu().numpy().flatten() for pred in predictions])
    print("Stored {} predictioons".format(len(predictions)))
    np.save(args.save, predictions)

    # Reset save path to the directory so we save plots in the same place
    args.save = os.path.dirname(args.save)

# Else load predictions from file
else:
    predictions = np.load(args.predictions)

# Pull plotting info from data module
track_info = d_module.all_dataset[:][0][:,0:3,2:].cpu().numpy()
plotting = d_module.all_dataset[:][4].cpu().numpy()
labels = d_module.all_dataset[:][1].cpu().numpy().flatten()
start_weights = d_module.all_dataset[:][3].cpu().numpy().flatten()

# Run plotting
pu.make_logged_plots(
    plotting,
    labels,
    start_weights,
    predictions,
    save_location=args.save,
    display=False
)

# Run inclusive plots
pu.make_inclusive_track_plots(
    track_info,
    labels,
    start_weights,
    predictions,
    save_location=args.save,
    display=False
)

# Calculate probabilities
probabilities = 1 / (1 + np.exp(-predictions))

# Calculate AUC
auc = roc_auc_score(labels, probabilities)
print("AUC: {}".format(auc))