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

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from of_transformer import OfTransformer
from simple_network import DumbNeuralNetwork
from data_utils import *
from plotting_utils import *


class LOfTransformer(L.LightningModule):
    """ LOfTransformer - This class is a wrapper for the Omnifold Transformer.
    It will initialize the model in the __init__ method. Any additional arguments
    that are passed to the __init__ method will be passed to the OfTransformer.

    For now, loss is hardcoded to the BCEWithLogitsLoss.
    Optimizer is hard coded to AdamW with a learning rate of 1e-3.

    """

    # Init function 
    def __init__(self, input_dim=3, plot_staging='./plot_storage', **kwargs):
        """ __init__ - This method initializes the LOfTransformer class.
        There is one required argument which gives the input dimension for the 
        transformer. This is the # of features per object (usually 3). 
        Any other keyword arguments are passed to the OfTransformer init function,
        and saved as hyperparameters of the module.

        Arguments:
            input_dim {int} -- The input dimension of the model.
            plot_staging {str} -- The path to the directory where plots will be stored for logging
            **kwargs {dict} -- A dictionary of keyword arguments to be passed
                to the OfTransformer init function.
        """

        # Initialize model and loss
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.model = OfTransformer(input_dim, **kwargs)

        # Save validation step data, labels, and outputs
        self.validation_step_data = []
        self.validation_step_labels = []
        self.validation_step_outputs = []

        # Log hyperparameters
        self.save_hyperparameters(ignore=['plot_staging'])

        # Plot staging
        self.plot_staging = plot_staging


    # Forward pass
    def forward(self, inputs, mask):
        return self.model(inputs, mask=mask)
    

    # Training step
    def training_step(self, batch, batch_idx):
        inputs, target, mask, weights = batch
        output = self(inputs, mask)
        loss = self.criterion(output, target) * weights
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    

    # Validation step
    def validation_step(self, batch, batch_idx):

        # Forward pass
        inputs, target, mask, weights = batch
        output = self(inputs, mask)

        # Calculate and log loss
        loss = self.criterion(output, target) * weights
        loss = loss.mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store data, labels, and outputs
        self.validation_step_data.append(inputs.detach().cpu())
        self.validation_step_labels.append(target.detach().cpu())
        self.validation_step_outputs.append(output.detach().cpu())


    # # Validation step end for logging reweighting plots to wandb
    # def on_validation_epoch_end(self):

    #     # Aggregate all inputs and predictions and send to numpy
    #     all_inputs = torch.cat(self.validation_step_data).numpy()
    #     all_labels = torch.cat(self.validation_step_labels).numpy().flatten()
    #     all_preds = torch.cat(self.validation_step_outputs).numpy().flatten()

    #     # Make plots
    #     # if not self.trainer.sanity_checking:
    #     plot_dict = make_logged_plots(all_inputs, all_labels, all_preds, save_location=self.plot_staging)

    #     # Log plots
    #     for key, histpath in plot_dict.items():
    #         wandb.log({key: wandb.Image(histpath)})

    #     # Clear memory
    #     self.validation_step_data.clear()
    #     self.validation_step_labels.clear()
    #     self.validation_step_outputs.clear()


    # Prediction step
    def predict_step(self, batch, batch_idx):
        inputs, target, mask, weights = batch
        return self(inputs, mask)


    # Configure optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=1e4,
            warmup_steps=1e3,
            max_lr=5e-3,
            min_lr=1e-4,
            gamma=0.5
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}
    

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
    def __init__(self, mc_file=None, data_file=None, max_tracks=None, batch_size=256, dataloader_workers=0, **kwargs):
        """ __init__ - This method initializes the LOfData class. It takes
        the path to the Monte Carlo and data files as arguments.

        Arguments:
            mc_file {str} -- The path to the Monte Carlo file.
            data_file {str} -- The path to the data file.
            max_tracks {int} -- The maximum number of tracks to consider.
                Defaults to None, which means all tracks are considered.
            batch_size {int} -- The batch size for the data loaders. Defaults
                to 256.
            dataloader_workers {int} -- The number of workers for the data loaders.
        """

        # Set class attributes
        super().__init__()
        self.mc_file = mc_file
        self.data_file = data_file
        self.max_tracks = max_tracks
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers

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
        mc_kinematics, mc_mask = get_kinematics(tree_mc, filter=pass190_mc, max_tracks=self.max_tracks, **kwargs)
        # pseudo data kinematics will take the maximum # of tracks from MC
        pd_kinematics, pd_mask = get_kinematics(tree_pd, filter=pass190_pd, max_tracks=mc_kinematics.shape[2]-2, **kwargs)

        # MC weights
        mc_weights = ak.to_numpy(tree_mc['weight'].array())
        mc_weights = np.expand_dims(mc_weights[pass190_mc == 1], axis=1)

        # Pseudodata weights
        pd_weights = ak.to_numpy(tree_pd['weight'].array())
        pd_weights = np.expand_dims(pd_weights[pass190_pd == 1], axis=1)

        # Drop 11k weights at the end since we don't have tracks for those events
        pd_weights = pd_weights[:pd_kinematics.shape[0]]

        # Build labels
        mc_labels = np.zeros((mc_kinematics.shape[0], 1), dtype=np.float32)
        pd_labels = np.ones((pd_kinematics.shape[0], 1), dtype=np.float32)

        # Concatenate MC and pseudodata together. Since class ratio with MC weights is about 1.0 just use all events
        kinematics = np.concatenate([mc_kinematics, pd_kinematics], axis=0)
        mask = np.concatenate([mc_mask, pd_mask], axis=0)
        weights = np.concatenate([mc_weights, pd_weights], axis=0)
        labels = np.concatenate([mc_labels, pd_labels], axis=0)

        # Make one-hot encoding dimensions identifying whether the object is a muon or track
        is_muon = np.concatenate([np.ones((kinematics.shape[0], 2)), np.zeros((kinematics.shape[0], kinematics.shape[2]-2))], axis=1)
        is_track = np.concatenate([np.zeros((kinematics.shape[0], 2)), np.ones((kinematics.shape[0], kinematics.shape[2]-2))], axis=1)
        one_hot = np.stack([is_muon, is_track], axis=1)

        # Concatenate one-hot encoding to kinematics
        kinematics = np.concatenate([kinematics, one_hot], axis=1)

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
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.dataloader_workers)
    

    # Validation dataloader
    def val_dataloader(self):
        """ val_dataloader - This method returns a pytorch dataloader
        for the validation data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the validation data.
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)
    

    # Test dataloader
    def test_dataloader(self):
        """ test_dataloader - This method returns a pytorch dataloader
        for the test data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the test data.
        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)

    # Predict dataloader
    def predict_dataloader(self):
        """ test_dataloader - This method returns a pytorch dataloader
        for the test data, but to run predictions. This is what we will usually be using.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the test data.
        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)
