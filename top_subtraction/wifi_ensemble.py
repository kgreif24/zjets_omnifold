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
            top_weights = np.load(basis_file_path)["top_weights"]

            # Clip extreme weights to prevent optimization instability
            # We can just clip the weights at 1.0 since we never want to add density
            pd_weights = np.where(np.isinf(pd_weights), 1.0, pd_weights)
            pd_weights = np.where(np.isnan(pd_weights), 1.0, pd_weights)
            pd_weights = np.clip(pd_weights, 1e-6, 1.0)

            top_weights = np.where(np.isinf(top_weights), 1.0, top_weights)
            top_weights = np.where(np.isnan(top_weights), 1.0, top_weights)
            top_weights = np.clip(top_weights, 1e-6, 1.0)

            try:
                assert bool(np.isinf(top_weights).any()) is False
                assert bool(np.isnan(top_weights).any()) is False
            except Exception as e:
                print(f"Basis file path: {basis_file_path}")
                print(f"Basis top: {top_weights[:10]}")
                print(f"Basis pd: {pd_weights[:10]}")
                raise Exception("Basis top or pd is inf or nan before log") from e

            self.basis_pd.append(pd_weights)
            self.basis_top.append(top_weights)

        self.basis_pd = torch.tensor(np.array(self.basis_pd), dtype=torch.float64)
        self.basis_top = torch.tensor(np.array(self.basis_top), dtype=torch.float64)

        # self.basis_top = torch.log(self.basis_top)
        # self.basis_pd = torch.log(self.basis_pd)

        try:
            assert bool(torch.isinf(self.basis_top).any()) is False
            assert bool(torch.isinf(self.basis_pd).any()) is False
            assert bool(torch.isnan(self.basis_top).any()) is False
            assert bool(torch.isnan(self.basis_pd).any()) is False
        except Exception as e:
            print(f"Basis top: {self.basis_top[:3,:10]}")
            print(f"Basis pd: {self.basis_pd[:3,:10]}")
            raise Exception("Basis top or pd is inf or nan after log") from e

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

    Implements eqn. 8 in the paper

    Arguments:
        ws (torch.Tensor): The weights for the ensemble.
        ensemble (WifiEnsemble): The ensemble object.
        num_weights (torch.Tensor): The weights for the numerator of the MLC.

    Returns:
        torch.Tensor: The MLC loss
    """

    num, den = ensemble.forward(ws)

    # MLC loss terms
    mlc1 = (-num + (torch.exp(-num) - 1)) * num_weights
    mlc2 = den + (torch.exp(den) - 1)

    return torch.mean(mlc1) + torch.mean(mlc2)


def bce_min(ws, ensemble, num_weights):
    """ bce_min - Loss function for minimizing the w's in the ensemble.
    Note we assume the denominator (data or pseudodata without top)
    always has weights of 1.0!!

    Implements the BCE loss.

    Arguments:
        ws (torch.Tensor): The weights for the ensemble.
        ensemble (WifiEnsemble): The ensemble object.
        num_weights (torch.Tensor): The weights for the numerator of the BCE.

    Returns:
        torch.Tensor: The BCE loss
    """

    num, den = ensemble.forward(ws)
    try:
        assert bool(torch.isinf(num).any()) is False
        assert bool(torch.isinf(den).any()) is False
        assert bool(torch.isnan(num).any()) is False
        assert bool(torch.isnan(den).any()) is False
    except Exception as e:
        print(f"Num: {num[:10]}")
        print(f"Den: {den[:10]}")
        raise Exception("Num or den is inf or nan") from e

    # BCE loss terms
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    bce1 = (bce(num, torch.ones_like(num)) * num_weights).mean()
    bce2 = bce(den, torch.zeros_like(den)).mean()

    return bce1 + bce2


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

    # Initialize ensemble
    ensemble = WifiEnsemble(basis_file_paths)

    # Initialize weights with small random values to break symmetry
    ws = torch.rand(len(basis_file_paths) + 1, dtype=torch.float64)

    # Minimize!
    res_root = minimize(
        lambda w: mlc_min(
            w, ensemble, num_weights
        ),
        # lambda w: bce_min(
        #     torch.softmax(w, dim=0), ensemble, num_weights
        # ),
        x0=ws,
        method="newton-exact",
    )

    # Apply softmax to get final normalized weights
    w0 = res_root.x
    print(f"Optimized weights: {w0}")
    print(f"Weight sum: {torch.sum(w0):.6f}")
    print(f"Max weight: {torch.max(w0):.6f}, Min weight: {torch.min(w0):.6f}")

    # Predict the weights for the data (or pseudodata) and write to output file
    pd_weights = ensemble.forward_pd(w0).clip(max=1.0)
    np.savez(args.output_file, weights=pd_weights)
