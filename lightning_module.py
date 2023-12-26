""" lightning_module.py - This file defines the LOfTransformer and LOfData
classes. These pytorch lightning modules that simplify the data processing
and training of Omnifold classifiers.

Author: Kevin Greif
1/05/2024
python3
"""

import torch
import lightning as L
import wandb

import numpy as np
import uproot
import awkward as ak

from of_transformer import OfTransformer
from data_utils import *


class LOfTransformer(L.LightningModule):
    """ LOfTransformer - This class is a wrapper for the Omnifold Transformer.
    It will initialize the model in the __init__ method. Any additional arguments
    that are passed to the __init__ method will be passed to the OfTransformer.

    For now, loss is hardcoded to the BCEWithLogitsLoss.
    Optimizer is hard coded to AdamW with a learning rate of 1e-3.

    """

    # Init function 
    def __init__(self, input_dim=3, param_dict={}, **kwargs):
        """ __init__ - This method initializes the LOfTransformer class.
        There are two required arguments. The first gives the input
        dimension for the model. The second is a dictionary of parameters
        to be passed as keyword arguments to the OfTransformer init function.

        Arguments:
            input_dim {int} -- The input dimension of the model.
            param_dict {dict} -- A dictionary of parameters to be passed to the
                OfTransformer init function.
        """

        # Initialize model and loss
        super().__init__(**kwargs)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.model = OfTransformer(input_dim, **param_dict)

        # Log hyperparameters
        self.save_hyperparameters(input_dim, param_dict)


    # Forward pass
    def forward(self, inputs, mask):
        return self.model(inputs, mask=mask)
    

    # Training step
    def training_step(self, batch, batch_idx):
        inputs, target, mask, weights = batch
        output = self(inputs, mask)
        loss = self.criterion(output, target) * weights
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True)
        return loss
    

    # Validation step
    def validation_step(self, batch, batch_idx):
        inputs, target, mask, weights = batch
        output = self(inputs, mask)
        loss = self.criterion(output, target) * weights
        loss = loss.mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    
    
    # Configure optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return optimizer

    

class LOfData(L.LightningDataModule):
    """ LOfData - This class handles all of the data processing for training
    and Omnifold classifier. It subclasses the LightningDataModule class.

    Init function takes the path to the data files as arguments, along with the
    max number of tracks we want to consider, and the batch size for the data loaders.
    It then goes through the process of loading data from disk using uproot, and applying
    the relevant preprocessing. The data is stored in the class as attributes, in the form 
    of pytorch datasets.

    "prepare_data" method is meant for downloading datasets. Since we already have
    the data in a shared filesystem this should not be necessary.

    "setup" method applies the train / test / val split to the data
    
    DataLoaders are produced in the relevant hook.
    """

    # Init function
    def __init__(self, mc_file=None, data_file=None, max_tracks=None, batch_size=256, **kwargs):
        """ __init__ - This method initializes the LOfData class. It takes
        the path to the Monte Carlo and data files as arguments.

        Arguments:
            mc_file {str} -- The path to the Monte Carlo file.
            data_file {str} -- The path to the data file.
            max_tracks {int} -- The maximum number of tracks to consider.
                Defaults to None, which means all tracks are considered.
            batch_size {int} -- The batch size for the data loaders. Defaults
                to 256.
        """

        # Set class attributes
        super().__init__(**kwargs)
        self.mc_file = mc_file
        self.data_file = data_file
        self.max_tracks = max_tracks
        self.batch_size = batch_size

        # Logging
        self.save_hyperparameters()

        # Load the data
        print("Loading data...")
        f_mc = uproot.open(self.mc_file)
        tree_mc = f_mc['OmniTree']
        f_pd = uproot.open(self.data_file)
        tree_pd = f_pd['OmniTree']

        # Pass 190 flags
        pass190_mc = ak.to_numpy(tree_mc['pass190'].array())
        print("We have a fracion {} of good events in mc".format(np.sum(pass190_mc) / len(pass190_mc)))
        pass190_pd = ak.to_numpy(tree_pd['pass190'].array())
        print("We have a fracion {} of good events in pseudodata".format(np.sum(pass190_pd) / len(pass190_pd)))

        # Load kinematics
        mc_kinematics, mc_mask = get_kinematics(tree_mc, filter=pass190_mc, max_tracks=self.max_tracks)
        # pseudo data kinematics will take the maximum # of tracks from MC
        pd_kinematics, pd_mask = get_kinematics(tree_pd, filter=pass190_pd, max_tracks=mc_kinematics.shape[2]-2)

        # MC weights
        mc_weights = ak.to_numpy(tree_mc['weight'].array())
        mc_weights /= np.mean(mc_weights)
        mc_weights = np.expand_dims(mc_weights[pass190_mc == 1], axis=1)

        # Pseudodata weights
        pd_weights = ak.to_numpy(tree_pd['weight'].array())
        pd_weights /= np.mean(pd_weights)
        pd_weights = np.expand_dims(pd_weights[pass190_pd == 1], axis=1)

        # Drop 11k weights at the end since we don't have tracks for those events
        pd_weights = pd_weights[:pd_kinematics.shape[0]]

        # Build labels
        mc_labels = np.zeros((mc_kinematics.shape[0], 1), dtype=np.float32)
        pd_labels = np.ones((pd_kinematics.shape[0], 1), dtype=np.float32)

        # Concatenate MC and pseudodata together, taking as many MC events as there are pseudodata events
        kinematics = np.concatenate([mc_kinematics[:pd_kinematics.shape[0],...], pd_kinematics], axis=0)
        mask = np.concatenate([mc_mask[:pd_mask.shape[0],...], pd_mask], axis=0)
        weights = np.concatenate([mc_weights[:pd_weights.shape[0],...], pd_weights], axis=0)
        labels = np.concatenate([mc_labels[:pd_labels.shape[0],...], pd_labels], axis=0)

        # Convert to torch tensors with float32 precision
        kinematics = torch.from_numpy(kinematics.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        weights = torch.from_numpy(weights.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.float32))

        # Build pytorch datasets
        self.all_dataset = torch.utils.data.TensorDataset(kinematics, labels, mask, weights)
        print("All done with data loading!")
        print("We have {} events in total".format(len(self.all_dataset)))


    # Setup function
    def setup(self, stage=None):
        """ setup - This method loads data from files using uproot, and 
        applies the relevant preprocessing. It then creates pytorch
        data sets. Data loaders are created in the relevant hook.

        No arguments or returns
        """

        # Make train, test, and validation datasets
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.all_dataset, [0.8, 0.1, 0.1])


    # Train dataloader
    def train_dataloader(self):
        """ train_dataloader - This method returns a pytorch dataloader
        for the training data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the training data.
        """
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    

    # Validation dataloader
    def val_dataloader(self):
        """ val_dataloader - This method returns a pytorch dataloader
        for the validation data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the validation data.
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    

    # Test dataloader
    def test_dataloader(self):
        """ test_dataloader - This method returns a pytorch dataloader
        for the test data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the test data.
        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
