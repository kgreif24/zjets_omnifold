"""eval_classify_top.py - Evaluate a trained top classifier and possibly
append the top classifier outputs as an extra feature in the data
(or pseudodata) tree. This output can then be used as a feature in the actual
top subtraction.

Author: Kevin Greif
Last updated September 12, 2025
"""

import argparse
import sys

import awkward as ak
import lightning as L
import numpy as np
import torch
import uproot
import matplotlib.pyplot as plt

sys.path.append("..")
from lightning_module import LOfTransformer  # noqa: E402
from of_dataset import OfDataset  # noqa: E402
import utils.data_utils as du  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained top classifier")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint to evaluate",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file to evaluate"
    )
    parser.add_argument(
        "--is_pseudodata",
        action="store_true",
        help="If true, the data file is pseudodata and not data",
    )
    parser.add_argument(
        "--append_filename",
        type=str,
        default=None,
        help="Name of the copied data tree, containing the top classifier outputs",
    )
    args = parser.parse_args()
    return args


def copy_root_with_predictions(input_path, output_path, predictions):
    """
    Copy a ROOT file and add predictions vector as 'topClassLogit' branch.

    Args:
        input_path (str): Path to the input ROOT file
        output_path (str): Path for the output ROOT file
        predictions (np.ndarray): Predictions array to add as new branch
    """
    # Open the input file
    with uproot.open(input_path) as input_file:

        # Get the tree name (assuming 'OmniTree' based on the existing code)
        tree = input_file["OmniTree"]

        # Read all existing branches
        branches_data = {}
        for branch_name in tree.keys():
            branches_data[branch_name] = tree[branch_name].array()

        # Add the predictions as a new branch
        isTop_logit = np.mean(predictions) * np.ones((tree.num_entries))
        isTop_logit[branches_data["pass190"] == 1] = predictions
        branches_data["isTop_logit"] = isTop_logit

        # Create the output file with the new tree
        with uproot.recreate(output_path) as output_file:
            # Create the new tree with all branches including the new one
            output_file["OmniTree"] = branches_data


def main():
    args = parse_args()

    # Load the checkpoint
    model = LOfTransformer.load_from_checkpoint(args.checkpoint_path)

    # Load the data
    f = uproot.open(args.data_path)
    t = f["OmniTree"]
    kinematics, indices = du.get_kinematics(t, muon_only=False)
    if args.is_pseudodata:
        is_top = ak.to_numpy(t["isTop"].array())
        is_top = np.expand_dims(is_top, axis=1)
    else:
        is_top = np.ones((len(kinematics), 1), dtype=np.int32)

    # Make the dataset and dataloader
    dataset = OfDataset(
        kinematics=kinematics,
        labels=is_top,
        weights=np.ones((len(kinematics), 1)),
        object_indeces=indices,
        w1_obs=np.zeros((len(kinematics), 1)),  # Dummy W1 observable
        n_jets=5,
        max_tracks=264,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=False, collate_fn=du.null_collate
    )

    # Make trainer and run predictions
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=None,
    )
    predictions = trainer.predict(model, dataloader)
    predictions = np.concatenate([pred.cpu().flatten().numpy() for pred in predictions])

    # If this is psuedodata, make a plot of the predictions
    if args.is_pseudodata:

        # Also get predictions for the pure top events
        f_top = uproot.open(
            "/pscratch/sd/k/kgreif/data/"
            "ZjetOmnifold_14May2025_Background_Sherpa2212_AllTop_"
            "WithTracks_slim_Systematics.root"
        )
        t_top = f_top["OmniTree"]
        kinematics_top, indices_top = du.get_kinematics(t_top, muon_only=False)
        dataset_top = OfDataset(
            kinematics=kinematics_top,
            labels=np.ones((len(kinematics_top), 1)),
            weights=np.ones((len(kinematics_top), 1)),
            object_indeces=indices_top,
            w1_obs=np.zeros((len(kinematics_top), 1)),
        )
        dataloader_top = torch.utils.data.DataLoader(
            dataset_top, batch_size=1024, shuffle=False, collate_fn=du.null_collate
        )
        predictions_top = trainer.predict(model, dataloader_top)
        predictions_top = np.concatenate(
            [pred.cpu().flatten().numpy() for pred in predictions_top]
        )

        is_top = is_top.flatten()
        bins = np.linspace(-10, 10, 100)
        plt.hist(
            predictions[is_top == 0], bins=bins, alpha=0.5, label=" PD Z+jets"
        )
        plt.hist(
            predictions[is_top == 1],
            bins=bins,
            alpha=0.5,
            label="PD Top",
            color="red",
        )
        plt.hist(
            predictions_top,
            bins=bins,
            alpha=0.5,
            label="Sherpa Top",
            histtype="step",
            color="red",
        )
        plt.legend()
        plt.xlabel("Top Classifier Logit")
        plt.ylabel("Number of Events")
        plt.yscale("log")
        plt.savefig("./plot_storage/top_classifier_predictions.png", dpi=300)
        plt.close()

    # Copy the ROOT file with predictions as new branch
    if args.append_filename is not None:
        copy_root_with_predictions(args.data_path, args.append_filename, predictions)


if __name__ == "__main__":
    main()
