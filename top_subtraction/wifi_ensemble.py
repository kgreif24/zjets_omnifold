"""wifi_ensemble.py - This script will use the methods in
https://arxiv.org/abs/2506.00113 to produce a well motivated ensemble of weights for
the background subtraction.

We can compare the performance of this ensemble to the performance of the naive
ensemble that results from just taking the average over the produced weights.

Author: Kevin Greif
Last updated September 13, 2025
"""

import argparse
import glob
import uproot
import awkward as ak
import numpy as np
import torch
from torchmin import minimize


class WifiEnsemble:
    """WifiEnsemble - This class implements the wifi ensemble for a given set of
    reweightings that serve as the basis functions. In practice the basis functions
    must be pre-evaluated and their outputs stored in .npz files.
    """

    def __init__(self, basis_file_paths):
        """__init__ - This function initializes the WifiEnsemble object.

        Arguments:
            basis_file_paths (list): A list of paths to the basis functions, obtained
            for example by globbing over a wildcard pattern.
        """
        self.basis_file_paths = basis_file_paths
        num_pd, num_top = (
            np.load(basis_file_paths[0])["pd_weights"].shape[0],
            np.load(basis_file_paths[0])["top_weights"].shape[0],
        )
        # Remember the first element of each basis is a constant 1.0
        self.basis_pd = [np.ones(num_pd)]
        self.basis_top = [np.ones(num_top)]
        for basis_file_path in basis_file_paths:
            pd_weights = np.load(basis_file_path)["pd_weights"]
            # Replace any infinite values in pd weights with 1.0
            pd_weights = np.where(np.isinf(pd_weights), 1.0, pd_weights)
            self.basis_pd.append(pd_weights)
            top_weights = np.load(basis_file_path)["top_weights"]
            # Replace any infinite values in top weights with 1.0
            top_weights = np.where(np.isinf(top_weights), 1.0, top_weights)
            self.basis_top.append(top_weights)
        self.basis_pd = torch.tensor(np.array(self.basis_pd))
        self.basis_top = torch.tensor(np.array(self.basis_top))

    def forward_pd(self, ws):
        return ws @ self.basis_pd

    def forward_top(self, ws):
        return ws @ self.basis_top

    def forward(self, ws):
        """forward - This function returns the weighted output for the data
        (or pseudodata) and the top events.

        Arguments:
            ws (torch.Tensor): The weights for the ensemble.
            Length is n_basis_functions + 1

        Returns:
            tuple of torch.Tensor: The weighted output for the data (or pseudodata)
            and the top events.
        """
        pd = self.forward_pd(ws)
        top = self.forward_top(ws)
        num = torch.cat([pd, top], axis=0)
        return num, pd


def mlc_min(ws, ensemble, num_weights):
    """mlc_min - Loss function for minimizing the w's in the ensemble.
    Note we assume the denominator (data or pseudodata without top)
    always has weights of 1.0!!

    Implements eqn. 8 in the paper.

    Arguments:
        ws (torch.Tensor): The weights for the ensemble.
        ensemble (WifiEnsemble): The ensemble object.
        num_weights (torch.Tensor): The weights for the numerator of the MLC.

    Returns:
        torch.Tensor: The MLC loss
    """

    num, den = ensemble.forward(ws)
    mlc1 = (-num + (torch.exp(-num) - 1)) * num_weights
    mlc2 = den + (torch.exp(den) - 1)
    return torch.mean(mlc1) + torch.mean(mlc2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the wifi ensemble")
    parser.add_argument(
        "--top_path", type=str, required=True, help="Path to the top MC file"
    )
    parser.add_argument(
        "--basis_file_paths",
        type=str,
        required=True,
        help="Path to the basis file paths",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    basis_file_paths = glob.glob(args.basis_file_paths)
    print(f"Found {len(basis_file_paths)} basis file paths")

    # Need to get the weights for the numerator (PD - Top)
    num_pd = np.load(basis_file_paths[0])["pd_weights"].shape[0]
    f_top = uproot.open(args.top_path)
    t_top = f_top["OmniTree"]
    weights_top = ak.to_numpy(t_top["weight"].array())
    pass190_top = ak.to_numpy(t_top["pass190"].array())
    weights_top = weights_top[pass190_top == 1]
    num_weights = np.concatenate([np.ones(num_pd), -1.0 * weights_top])
    num_weights = torch.tensor(num_weights, dtype=torch.float64)

    # Optimize the wifi ensemble with the MLC loss
    print("Optimizing the wifi ensemble with the MLC loss")
    ensemble = WifiEnsemble(basis_file_paths)
    ws = torch.zeros(len(basis_file_paths) + 1, dtype=torch.float64)
    ws[0] = 1.0
    res_root = minimize(
        lambda w: mlc_min(w, ensemble, num_weights),
        # lambda w: mlc_min(torch.softmax(w, dim=0), ensemble, num_weights),
        x0=ws,
        method="newton-exact",
    )
    w0 = res_root.x
    # w0 = torch.softmax(res_root.x, dim=0)
    print(f"Optimized weights: {w0}")

    # Predict the weights for the data (or pseudodata) and write to output file
    pd_weights = ensemble.forward_pd(w0)
    np.savez(args.output_file, weights=pd_weights)
