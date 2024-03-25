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
                 step=1,
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
            step {int} -- Whether this training is for OF step one or two, only effects plot labeling
            **kwargs {dict} -- A dictionary of keyword arguments to be passed
                to the OfTransformer init function.
        """

        # Debug flag and seed value
        self.debug = debug
        self.seed = seed

        # Set plotting names based on step argument
        if step == 1:
            self.names = ('RecoMC', 'RecoPD')
        elif step == 2:
            self.names = ('TruthMC', 'PulledWeightsMC')

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
            self.wasserstein_train = WassersteinOne(draw_plots=False)
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
        inputs, target, mask, start_weights, plotting = batch
        output = self(inputs, mask)
        loss = self.criterion(output, target) * start_weights
        loss = loss.mean()

        # Calculate new weights
        probs = 1 / (1 + torch.exp(-output))
        derived_weights = probs / (1 - probs)
        end_weights = derived_weights * start_weights

        # Logging metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        if not self.debug:
            self.wasserstein_train.update(plotting, start_weights, end_weights, target)

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
        inputs, target, mask, start_weights, plotting = batch
        output = self(inputs, mask)

        # Calculate new weights
        probs = 1 / (1 + torch.exp(-output))
        derived_weights = probs / (1 - probs)
        end_weights = derived_weights * start_weights

        # Calculate and log loss
        loss = self.criterion(output, target) * start_weights
        loss = loss.mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log AUC, note the AUROC class auto-applies sigmoid to logits
        self.val_auc(output, target)
        self.log('val_auc', self.val_auc, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        # Update wasserstein metric
        if not self.debug:
            self.wasserstein_val.update(plotting, start_weights, end_weights, target)


    # Validation step end for logging reweighting plots to wandb
    def on_validation_epoch_end(self):

        # Just return if in debug mode
        if self.debug:
            return

        # Don't do anything but reset metric on validation sanity check
        if not self.trainer.sanity_checking:

            # Calculate wasserstein metric, and get dictionary of plots to log
            val_wass, plot_dict = self.wasserstein_val.compute(names=self.names)

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
        inputs, target, mask, start_weights, plotting = batch
        output = self(inputs, mask)

        # Calculate new weights
        probs = 1 / (1 + torch.exp(-output))
        derived_weights = probs / (1 - probs)
        end_weights = derived_weights * start_weights

        # Calculate and log AUC
        self.test_auc(output, target)
        self.log('test_auc', self.test_auc, on_epoch=True, on_step=False, prog_bar=False, sync_dist=False)

        # Update wasserstein metric
        if not self.debug:
            self.wasserstein_test.update(plotting, start_weights, end_weights, target)


    # Test epoch end for logging plots and metrics to wandb
    def on_test_epoch_end(self):

        # Just return if in debug mode
        if self.debug:
            return

        # Calculate wasserstein metric, and get dictionary of plots to log
        test_wass, plot_dict = self.wasserstein_test.compute(names=self.names)

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

    Init function takes the path to the data files as arguments.
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
        source_file=None,
        target_file=None,
        max_tracks=None,
        source_weight_path=None,
        target_weight_path=None,
        muon_only=False,
        batch_size=256,
        dataloader_workers=0,
        split_seed=420,
        load_all=False,
        testing=False,
        use_truth=False,
        **kwargs
    ):
        """ __init__ - This method initializes the LOfData class. It takes
        the path to the Monte Carlo and data files as arguments.

        Arguments:
            source_file {str} -- The path to the file containing the source data.
            target_file {str} -- The path to the file containing the target data.
            max_tracks {int} -- The maximum number of tracks to consider.
                Defaults to None, which means all tracks are considered.
            source_weight_path {string} -- Path to a .npz file containing weights for the source data
            target_weight_path {string} -- Path to a .npz file containing weights for the target data
            muon_only {bool} -- Set to true if we only want to consider muons.
            batch_size {int} -- The batch size for the data loaders. Defaults
                to 256.
            dataloader_workers {int} -- The number of workers for the data loaders.
            split_seed {int} - The random seed to use in making train / val split,
                ensure this is common between all processes
            load_all {bool} - Set to true if data loader should load all data, if false it produces train / val split
            testing {bool} - Set to true for data module to load testing weights (not training)
            use_truth {bool} - Set to true if we want to use truth level information
                for the data module. Defaults to false.
        """

        # Set class attributes
        super().__init__()
        self.source_file = source_file
        self.target_file = target_file
        self.max_tracks = max_tracks
        self.source_weight_path = source_weight_path
        self.target_weight_path = target_weight_path
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.split_seed = split_seed
        self.load_all = load_all
        self.testing = testing
        self.use_truth = use_truth

        # Load the data
        f_source = uproot.open(self.source_file)
        tree_source = f_source['OmniTree']
        f_target = uproot.open(self.target_file)
        tree_target = f_target['OmniTree']

        # Set prefix for keys
        prekey = ""
        if self.use_truth:
            prekey = "truth_"

        # Pass 190 flags
        self.pass190_source = ak.to_numpy(tree_source[prekey+'pass190'].array())
        self.pass190_target = ak.to_numpy(tree_target[prekey+'pass190'].array())
        rank_zero_info("We have a fracion {} of good events in source".format(np.sum(self.pass190_source) / len(self.pass190_source)))
        rank_zero_info("We have a fracion {} of good events in target".format(np.sum(self.pass190_target) / len(self.pass190_target)))

        # Load kinematics
        source_kinematics, source_mask = get_kinematics(
            tree_source, 
            filter=self.pass190_source,
            max_tracks=self.max_tracks,
            muon_only=muon_only,
            one_hot=True,
            get_truth=self.use_truth,
            **kwargs
        )
        # Target kinematics will take the maximum # of tracks from source
        target_kinematics, target_mask = get_kinematics(
            tree_target,
            filter=self.pass190_target,
            max_tracks=source_kinematics.shape[2]-2,
            muon_only=muon_only,
            one_hot=True,
            get_truth=self.use_truth,
            **kwargs
        )

        ## TODO: Refactor this into a function!
        # Get source weights
        self.source_all_weights = self.load_weights(tree_source, path=self.source_weight_path, test=self.testing)
        self.source_weights = np.expand_dims(self.source_all_weights[self.pass190_source == 1], axis=1)

        # Get target weights
        self.target_all_weights = self.load_weights(tree_target, path=self.target_weight_path, test=self.testing)
        self.target_weights = np.expand_dims(self.target_all_weights[self.pass190_target == 1], axis=1)

        # Labels
        source_labels = np.zeros((source_kinematics.shape[0], 1), dtype=np.float32)
        target_labels = np.ones((target_kinematics.shape[0], 1), dtype=np.float32)

        # Load plotting data
        plotting_variables = [hist_dict['key'] for hist_dict in pu.default_settings.values()]
        source_plotting = get_plotting(tree_source, vars=plotting_variables, filter=self.pass190_source, muon_only=muon_only, get_truth=self.use_truth, **kwargs)
        target_plotting = get_plotting(tree_target, vars=plotting_variables, filter=self.pass190_target, muon_only=muon_only, get_truth=self.use_truth, **kwargs)

        # Concatenate source and pseudodata together
        self.kinematics = np.concatenate([source_kinematics, target_kinematics], axis=0)
        mask = np.concatenate([source_mask, target_mask], axis=0)
        weights = np.concatenate([self.source_weights, self.target_weights], axis=0)
        self.labels = np.concatenate([source_labels, target_labels], axis=0)  # Make instance variable for getting labels in eval script
        self.plotting = np.concatenate([source_plotting, target_plotting], axis=0)

        # Convert to torch tensors with float32 precision
        kinematics = torch.from_numpy(self.kinematics.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        weights = torch.from_numpy(weights.astype(np.float32))
        labels = torch.from_numpy(self.labels.astype(np.float32))
        plotting = torch.from_numpy(self.plotting.astype(np.float32))

        # Build pytorch datasets
        self.all_dataset = torch.utils.data.TensorDataset(kinematics, labels, mask, weights, plotting)
        rank_zero_info("We have {} source events and {} pseudo data events".format(len(source_kinematics), len(target_kinematics)))
        rank_zero_info("We have {} events in total".format(len(self.all_dataset)))


    def load_weights(self, tree, path=None, test=False):
        """ load_weights - This function implements the logic for loading weights to be used both in data loading, and
        in providing access to the weights for the purposes of calculating the next iteration of weights in the evaluation
        routine. The logic is as follows:
        
        1. If the path is 'root', then we load the weights from the root file
        2. If the path is not None, then we load the weights from the .npz file at the given path
        3. If the path is None, then we return a vector of ones
        
        Arguments:
            tree {uproot.tree.TTree} -- The uproot tree object
            path {str} -- The path to the weights file. If set to 'root', then we load the weights from the root file.
            test {bool} -- Set to true if we want to load the test weights. Defaults to false.
            
        Returns:
            np.ndarray -- A numpy array of weights
        """

        # Get weights from root tree
        root_weights = ak.to_numpy(tree['weight'].array())

        # Load weights directly from root file
        if path == 'root':
            all_weights = root_weights

        # Load weights from the path
        elif path is not None:
            weight_file = np.load(path)
            if test:
                all_weights = weight_file['test']
            else:
                all_weights = weight_file['train']

        # Otherwise create a vector of ones
        else:
            all_weights = np.ones_like(root_weights, dtype=np.float32)
        return all_weights


    # Method for getting source weights
    def get_source_weights(self):
        return self.source_weights.flatten()
    
    # Method for getting the labels
    def get_labels(self):
        return self.labels.flatten()

    # Method for getting track kinematics
    def get_track_kinematics(self):
        return self.kinematics[:,:3,2:]  # Gets pT, eta, phi, for all tracks (not muons)

    # Method for getting plotting data
    def get_plotting(self):
        return self.plotting

    # Method for getting source root weights
    def get_source_all_weights(self):
        return self.source_all_weights

    # Method for getting pass 190 flags
    def get_source_pass190(self):
        return self.pass190_source


    # Setup function
    def setup(self, stage: str):
        """ setup - This method performs the train / validation split on the data
        loaded in the init function, unless we are using the data module for testing
        in which case no split is performed.

        No arguments or returns
        """

        # Make train and validation split if necessary
        if not self.load_all:
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
        assert not self.load_all
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.dataloader_workers)
    

    # Validation dataloader
    def val_dataloader(self):
        """ val_dataloader - This method returns a pytorch dataloader
        for the validation data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the validation data.
        """
        assert not self.load_all
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)


    # Test dataloader
    def test_dataloader(self):
        """ test_dataloader - This method returns a pytorch dataloader for running predictions. It always yeilds the full dataset.
        Be sure to only use this when testing is set to true.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader.
        """
        assert self.load_all
        return torch.utils.data.DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)


    # Predict dataloader
    def predict_dataloader(self):
        """ predict_dataloader - This method returns a pytorch dataloader for running predictions. Yields either the validation or the 
        full data set depending on whether "load_all" is set to true.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader.
        """
        if self.load_all:
            return torch.utils.data.DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)
        else:
            return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers)
