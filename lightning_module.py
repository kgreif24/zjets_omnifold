""" lightning_module.py - This file defines the LOfTransformer and LOfData
classes. These pytorch lightning modules that simplify the data processing
and training of Omnifold classifiers.

Author: Kevin Greif
1/05/2024
python3
"""

import torch
import lightning as L
import torchmetrics
import wandb
from pytorch_lightning.utilities.rank_zero import *

import numpy as np
import uproot
import awkward as ak

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from of_transformer import OfTransformer
from simple_network import DumbNeuralNetwork
from wasserstein_metric import WassersteinOne
from data_utils import *
import plotting_utils as pu


class LOfTransformer(L.LightningModule):
    """ LOfTransformer - This class is a wrapper for the Omnifold Transformer.
    It will initialize the model in the __init__ method. Any additional arguments
    that are passed to the __init__ method will be passed to the OfTransformer.

    For now, loss is hardcoded to the BCEWithLogitsLoss.
    """

    # Init function 
    def __init__(self, 
                 input_dim=3, 
                 val_plots='./val_plot_storage', 
                 test_plots='./test_plot_storage', 
                 debug=False, 
                 seed=420, 
                 **kwargs):
        """ __init__ - This method initializes the LOfTransformer class.
        There is one required argument which gives the input dimension for the 
        transformer. This is the # of features per object (usually 3). 
        Any other keyword arguments are passed to the OfTransformer init function,
        and saved as hyperparameters of the module.

        Arguments:
            input_dim {int} -- The input dimension of the model.
            val_plots {str} -- The path to the directory where validation plots will be stored for logging
            test_plots {str} -- The path to the directory where testing plots will be stored for logging
            debug {bool} -- Set to true if we are running in debug mode, use simple network on muons only
            seed {int} -- The random seed to use for the train / val split. Only used for logging
            **kwargs {dict} -- A dictionary of keyword arguments to be passed
                to the OfTransformer init function.
        """

        # Debug flag and seed value
        self.debug = debug
        self.seed = seed

        # Initialize model and loss
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        if debug:
            self.model = DumbNeuralNetwork()
        else:
            self.model = OfTransformer(input_dim, **kwargs)

        # Performance metrics, note this also handles plotting and logging to wandb
        self.val_auc = torchmetrics.classification.AUROC(task='binary')
        self.test_auc = torchmetrics.classification.AUROC(task='binary')
        if not self.debug:
            self.wasserstein_train = WassersteinOne(draw_plots=not self.debug)
            self.wasserstein_val = WassersteinOne(draw_plots=not self.debug, save_location=val_plots)
            self.wasserstein_test = WassersteinOne(draw_plots=not self.debug, save_location=test_plots)

        # Log hyperparameters
        self.save_hyperparameters(ignore=['plot_staging', 'debug'])


    # Forward pass
    def forward(self, inputs, mask):
        tracks = inputs[:,:3,:]
        if self.debug:
            return self.model(tracks)
        else:
            return self.model(inputs, v=tracks, mask=mask)
    

    # Training step
    def training_step(self, batch, batch_idx):

        # Separate batch, make forward pass, calculate loss
        inputs, target, mask, weights, plotting = batch
        output = self(inputs, mask)
        loss = self.criterion(output, target) * weights
        loss = loss.mean()

        # Logging metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        if not self.debug:
            self.wasserstein_train.update(plotting, weights, output, target)

        return loss
    
    # Train epoch end for logging train metrics
    def on_train_epoch_end(self):
            
        # Log wasserstein metric
        if not self.debug:
            train_wass, _ = self.wasserstein_train.compute()
            self.log('train_wasserstein', train_wass, on_epoch=True, prog_bar=False, sync_dist=True)

            # Reset wasserstein metric
            self.wasserstein_train.reset()


    # Validation step
    def validation_step(self, batch, batch_idx):

        # Forward pass
        inputs, target, mask, weights, plotting = batch
        output = self(inputs, mask)

        # Calculate and log loss
        loss = self.criterion(output, target) * weights
        loss = loss.mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log AUC, note the AUROC class auto-applies sigmoid to logits
        self.val_auc(output, target)
        self.log('val_auc', self.val_auc, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        # Update wasserstein metric
        if not self.debug:
            self.wasserstein_val.update(plotting, weights, output, target)


    # Validation step end for logging reweighting plots to wandb
    def on_validation_epoch_end(self):

        # Just return if in debug mode
        if self.debug:
            return

        # Don't do anything but reset metric on validation sanity check
        if not self.trainer.sanity_checking:

            # Calculate wasserstein metric, and get dictionary of plots to log
            val_wass, plot_dict = self.wasserstein_val.compute()

            # Log wasserstein metric
            self.log('val_wasserstein', val_wass, on_epoch=True, prog_bar=False, sync_dist=True)

            # Log plots if this is the rank zero process only!
            if self.trainer.is_global_zero:
                for key, histpath in plot_dict.items():
                    log_name = 'val_' + key
                    self.logger.experiment.log({log_name: wandb.Image(histpath)}, step=self.trainer.global_step)
            
        # Reset metric
        self.wasserstein_val.reset()


    # Test step
    def test_step(self, batch, batch_idx):

        # Forward pass
        inputs, target, mask, weights, plotting = batch
        output = self(inputs, mask)

        # Calculate and log AUC
        self.test_auc(output, target)
        self.log('test_auc', self.test_auc, on_epoch=True, on_step=False, prog_bar=False, sync_dist=False)

        # Update wasserstein metric
        if not self.debug:
            self.wasserstein_test.update(plotting, weights, output, target)


    # Test epoch end for logging plots and metrics to wandb
    def on_test_epoch_end(self):

        # Just return if in debug mode
        if self.debug:
            return

        # Calculate wasserstein metric, and get dictionary of plots to log
        test_wass, plot_dict = self.wasserstein_test.compute()

        # Log wasserstein metric
        self.log('test_wasserstein', test_wass, on_epoch=True, prog_bar=False, sync_dist=False)

        # Log plots if this is the rank zero process only!
        if self.trainer.is_global_zero:
            for key, histpath in plot_dict.items():
                log_name = 'test_' + key
                self.logger.experiment.log({log_name: wandb.Image(histpath)})

        # Reset metric
        self.wasserstein_test.reset()
        

    # Prediction step
    def predict_step(self, batch, batch_idx):
        inputs, _, mask, _, _ = batch
        return self(inputs, mask)


    # Configure optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=1e4,
            warmup_steps=1e3,
            max_lr=1e-4,
            min_lr=1e-5,
            gamma=0.8
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
    def __init__(
        self,
        mc_file=None,
        data_file=None,
        max_tracks=None,
        weight_path=None,
        muon_only=False,
        batch_size=256,
        dataloader_workers=0,
        split_seed=420,
        testing=False,
        **kwargs
    ):
        """ __init__ - This method initializes the LOfData class. It takes
        the path to the Monte Carlo and data files as arguments.

        Arguments:
            mc_file {str} -- The path to the Monte Carlo file.
            data_file {str} -- The path to the data file.
            max_tracks {int} -- The maximum number of tracks to consider.
                Defaults to None, which means all tracks are considered.
            weight_path {string} -- Path to the weights .npy file to use. Defaults to None,
                in which case the weights in the root file are used.
            muon_only {bool} -- Set to true if we only want to consider muons.
            batch_size {int} -- The batch size for the data loaders. Defaults
                to 256.
            dataloader_workers {int} -- The number of workers for the data loaders.
            split_seed {int} - The random seed to use in making train / val split,
                ensure this is common between all processes
            testing {bool} - Set to true if this data module is for testing
        """

        # Set class attributes
        super().__init__()
        self.mc_file = mc_file
        self.data_file = data_file
        self.max_tracks = max_tracks
        self.weight_path = weight_path
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.split_seed = split_seed
        self.testing = testing

        # Load the data
        f_mc = uproot.open(self.mc_file)
        tree_mc = f_mc['OmniTree']
        f_pd = uproot.open(self.data_file)
        tree_pd = f_pd['OmniTree']

        # Pass 190 flags
        pass190_mc = ak.to_numpy(tree_mc['pass190'].array())
        pass190_pd = ak.to_numpy(tree_pd['pass190'].array())
        rank_zero_info("We have a fracion {} of good events in mc".format(np.sum(pass190_mc) / len(pass190_mc)))
        rank_zero_info("We have a fracion {} of good events in pseudodata".format(np.sum(pass190_pd) / len(pass190_pd)))

        # Load kinematics
        mc_kinematics, mc_mask = get_kinematics(tree_mc, filter=pass190_mc, max_tracks=self.max_tracks, muon_only=muon_only, one_hot=True, **kwargs)
        # pseudo data kinematics will take the maximum # of tracks from MC
        pd_kinematics, pd_mask = get_kinematics(tree_pd, filter=pass190_pd, max_tracks=mc_kinematics.shape[2]-2, muon_only=muon_only, one_hot=True, **kwargs)

        # Get MC weights
        if self.weight_path is not None:
            # Note this assumes we only store weights for events which pass reco level cuts
            mc_weight_file = np.load(self.weight_path)
            if self.testing:
                mc_weights = mc_weight_file['test']
            else:
                mc_weights = mc_weight_file['train']
        # Default to using weights from the root file
        else:
            mc_weights = ak.to_numpy(tree_mc['weight'].array())
            mc_weights = np.expand_dims(mc_weights[pass190_mc == 1], axis=1)
        self.mc_weights = mc_weights

        # Pseudo data weights are always 1, but load them from file for generality
        pd_weights = ak.to_numpy(tree_pd['weight'].array())
        pd_weights = np.expand_dims(pd_weights[pass190_pd == 1], axis=1)
        # Drop 11k weights at the end of pseudodata since we don't have tracks for those events
        pd_weights = pd_weights[:pd_kinematics.shape[0]]

        # Labels
        mc_labels = np.zeros((mc_kinematics.shape[0], 1), dtype=np.float32)
        pd_labels = np.ones((pd_kinematics.shape[0], 1), dtype=np.float32)

        # Load plotting data
        plotting_variables = [hist_dict['key'] for hist_dict in pu.default_settings]
        mc_plotting = get_plotting(tree_mc, vars=plotting_variables, filter=pass190_mc, muon_only=muon_only, **kwargs)
        pd_plotting = get_plotting(tree_pd, vars=plotting_variables, filter=pass190_pd, muon_only=muon_only, **kwargs)

        # Concatenate MC and pseudodata together
        kinematics = np.concatenate([mc_kinematics, pd_kinematics], axis=0)
        mask = np.concatenate([mc_mask, pd_mask], axis=0)
        weights = np.concatenate([mc_weights, pd_weights], axis=0)
        self.labels = np.concatenate([mc_labels, pd_labels], axis=0)  # Make instance variable for getting labels in eval script
        plotting = np.concatenate([mc_plotting, pd_plotting], axis=0)

        # Convert to torch tensors with float32 precision
        kinematics = torch.from_numpy(kinematics.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        weights = torch.from_numpy(weights.astype(np.float32))
        labels = torch.from_numpy(self.labels.astype(np.float32))
        plotting = torch.from_numpy(plotting.astype(np.float32))

        # Build pytorch datasets
        self.all_dataset = torch.utils.data.TensorDataset(kinematics, labels, mask, weights, plotting)
        rank_zero_info("We have {} MC events and {} pseudo data events".format(len(mc_kinematics), len(pd_kinematics)))
        rank_zero_info("We have {} events in total".format(len(self.all_dataset)))


    # Method for getting MC weights
    def get_mc_weights(self):
        return self.mc_weights.flatten()

    
    # Method for getting the labels
    def get_labels(self):
        return self.labels.flatten()


    # Setup function
    def setup(self, stage: str):
        """ setup - This method performs the train / validation split on the data
        loaded in the init function, unless we are using the data module for testing
        in which case no split is performed.

        No arguments or returns
        """

        # Make train and validation split if necessary
        if not stage == 'test':
            generator = torch.Generator().manual_seed(self.split_seed)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.all_dataset, 
                [0.8, 0.2],
                generator=generator
            )


    # Train dataloader
    def train_dataloader(self):
        """ train_dataloader - This method returns a pytorch dataloader
        for the training data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the training data.
        """
        assert not self.testing
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.dataloader_workers)
    

    # Validation dataloader
    def val_dataloader(self):
        """ val_dataloader - This method returns a pytorch dataloader
        for the validation data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the validation data.
        """
        assert not self.testing
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)


    # Test dataloader
    def test_dataloader(self):
        """ test_dataloader - This method returns a pytorch dataloader for running predictions. It always yeilds the full dataset.
        Be sure to only use this when testing is set to true.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader.
        """
        assert self.testing
        return torch.utils.data.DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)


    # Predict dataloader
    def predict_dataloader(self):
        """ predict_dataloader - This method returns a pytorch dataloader for running predictions. Yields either the validation or the 
        full data set depending on whether testing is set to true.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader.
        """
        if self.testing:
            return torch.utils.data.DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)
        else:
            return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)
