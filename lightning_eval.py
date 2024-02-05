""" lightning_eval.py - This program will evaluate a model checkpoint
over a slice of the OF data that depends on the random seed passed to the lightning
data module. It will save a numpy array of the model outputs at a specified location.

Usage:
python lightning_eval.py --model ./path/to/checkpoint --save ./path/to/save --seed <seed>

Author: Kevin Greif
Last updated 02/02/2024
python3
"""

import lightning as L
from lightning_module import *
import numpy as np
import argparse


# Parse the command line arguments
parser = argparse.ArgumentParser(description='Evaluate a model checkpoint over a slice of the OF data')
parser.add_argument('--model', type=str, help='Path to the model checkpoint')
parser.add_argument('--save', type=str, help='Path to save the model outputs')
parser.add_argument('--seed', type=int, help='Random seed for the data module')
parser.add_argument('--validate', action='store_true', help='Use the validation set instead of the test set')
args = parser.parse_args()

# Make data module
d_module = LOfData(
    mc_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test.root',
    data_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Dec15.root',
    muon_only=False,
    batch_size=256,
    dataloader_workers=1,
    seed=args.seed,
    testing=not args.validate
)

# Load model checkpoint
model = LOfTransformer.load_from_checkpoint(args.model)

# Make lightning trainer
trainer = L.Trainer(accelerator='gpu', devices=1)

# Run predictions
predictions = trainer.predict(model, d_module)

# Save predictions
predictions = np.concatenate([pred.cpu().numpy().flatten() for pred in predictions])
print("Stored {} predictioons".format(len(predictions)))
np.save(args.save, predictions)
