""" lightning_module.py - This file defines the LOfTransformer and LOfData
classes. These pytorch lightning modules that simplify the data processing
and training of Omnifold classifiers.

Author: Kevin Greif
Last updated 5/26/2024
python3
"""

import sys
import numpy as np
import uproot
import awkward as ak

import torch
import lightning as L
import torchmetrics
import wandb
from pytorch_lightning.utilities.rank_zero import *

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from of_dataset import OfDataset
from of_transformer.of_transformer import OfTransformer
from of_transformer.simple_network import DumbNeuralNetwork
from wasserstein_metric import WassersteinOne
from utils.data_utils import *
import utils.plotting_utils as pu


class LOfTransformer(L.LightningModule):
    """ LOfTransformer - This class is a wrapper for the Omnifold Transformer.
    It will initialize the model in the __init__ method. Any additional arguments
    that are passed to the __init__ method will be passed to the OfTransformer.

    For now, loss is hardcoded to the BCEWithLogitsLoss.
    """

    # Init function 
    def __init__(self, 
                 input_dim=3,          
                 test_plots=None,
                 log=False,
                 debug=False,
                 no_w1=False,
                 seed=420,
                 step=1,
                 min_lr=1e-5,
                 max_lr=1e-4,
                 cycle_steps=30000,
                 warmup_steps=8000,
                 gamma=0.85,
                 **kwargs):
        """ __init__ - This method initializes the LOfTransformer class.
        There is one required argument which gives the input dimension for the 
        transformer. This is the # of features per object (usually 3). 
        Any other keyword arguments are passed to the OfTransformer init function,
        and saved as hyperparameters of the module.

        Arguments:
            input_dim {int} -- The input dimension of the model.
            test_plots {str} -- The path to the directory where testing plots will be stored for logging,
                None by default, in which case testing plots will not be drawn
            log {bool} -- Set to true if we want to log plots to wandb. False by default
            debug {bool} -- Set to true if we are running in debug mode, use simple network on muons only
            no_w1 {bool} -- Set to true if we want to disable the wasserstein metric
            seed {int} -- The random seed to use for the train / val split. Only used for logging
            step {int} -- Whether this training is for OF step one or two, only effects plot labeling
            min_lr {float} -- The minimum learning rate
            max_lr {float} -- The maximum learning rate
            cycle_steps {int} -- The number of steps in a cycle
            warmup_steps {int} -- The number of steps in the warmup
            gamma {float} -- The gamma parameter for the learning rate scheduler
            **kwargs {dict} -- A dictionary of keyword arguments to be passed
                to the OfTransformer init function.
        """

        # Set instance vars
        self.debug = debug
        self.no_w1 = no_w1
        self.seed = seed
        self.log_things = log
        self.step = step
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        # Set plotting names based on step argument
        if step == 1:
            self.names = ('RecoMC', 'RecoPD')
        elif step == 2:
            self.names = ('TruthMC', 'PulledWeightsMC')

        # Set 32 bit precision for all operations
        torch.set_float32_matmul_precision('medium')

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
        if not (self.debug or self.no_w1):
            self.wasserstein_val = WassersteinOne(pu.default_settings, draw_plots=False)
            self.draw_test = True if test_plots != None else False
            self.wasserstein_test = WassersteinOne(pu.default_settings, draw_plots=self.draw_test, save_location=test_plots)

        # Log hyperparameters
        self.save_hyperparameters(ignore=['test_plots', 'debug'])


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
        network_weights = torch.exp(output)
        end_weights = network_weights * start_weights

        # Log training loss
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        return loss


    # Validation step
    def validation_step(self, batch, batch_idx):

        # Forward pass
        inputs, target, mask, start_weights, plotting = batch
        output = self(inputs, mask)

        # Calculate new weights
        network_weights = torch.exp(output)
        end_weights = network_weights * start_weights

        # Calculate and log loss
        loss = self.criterion(output, target) * start_weights
        loss = loss.mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log AUC, note the AUROC class auto-applies sigmoid to logits
        self.val_auc(output, target)
        self.log('val_auc', self.val_auc, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        # Update wasserstein metric
        if not (self.debug or self.no_w1):
            self.wasserstein_val.update(plotting, start_weights, end_weights, target)


    # Validation step end for logging reweighting plots to wandb
    def on_validation_epoch_end(self):

        # Just return if in debug mode or not update wasserstein metrics
        if self.debug or self.no_w1:
            return

        # Don't do anything but reset metric on validation sanity check
        if not self.trainer.sanity_checking:

            # Calculate wasserstein metric, and get dictionary of plots to log
            val_wass, _ = self.wasserstein_val.compute(names=self.names)

            # Logging
            if self.log_things:
                self.log('val_wasserstein', val_wass, on_epoch=True, prog_bar=False, sync_dist=True)
            
        # Reset metric
        self.wasserstein_val.reset()


    # Test step
    def test_step(self, batch, batch_idx):

        # Forward pass
        inputs, target, mask, start_weights, plotting = batch
        output = self(inputs, mask)

        # Calculate new weights
        network_weights = torch.exp(output)
        end_weights = network_weights * start_weights

        # Calculate and log AUC
        self.test_auc(output, target)
        self.log('test_auc', self.test_auc, on_epoch=True, on_step=False, prog_bar=False, sync_dist=False)

        # Update wasserstein metric
        if not (self.debug or self.no_w1):
            self.wasserstein_test.update(plotting, start_weights, end_weights, target)


    # Test epoch end for logging plots and metrics to wandb
    def on_test_epoch_end(self):

        # Just return if in debug mode or not using wasserstein metrics
        if self.debug or self.no_w1:
            return

        # Calculate wasserstein metric, and get dictionary of plots to log
        test_wass, plot_dict = self.wasserstein_test.compute(names=self.names)

        # Logging
        if self.log_things:
            self.log('test_wasserstein', test_wass, on_epoch=True, prog_bar=False, sync_dist=False)
            if self.draw_test and self.trainer.is_global_zero:
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

        # Build and return optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.cycle_steps,
            warmup_steps=self.warmup_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            gamma=self.gamma
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}
    

class LOfData(L.LightningDataModule):
    """ LOfData - This class handles all of the data processing for training
    and Omnifold classifier. It subclasses the LightningDataModule class.

    Init function takes the path to the data files as arguments.
    It then goes through the process of loading data from disk using uproot, and applying
    the relevant preprocessing. The data is stored in the class as attributes, in the form 
    of pytorch datasets.

    The "data_divisor" argument is used to divide the data into pieces. This is useful for
    training on large datasets, where we may not want to load the entire dataset into memory.
    
    DataLoaders are produced in the relevant hook.
    """

    # Init function
    def __init__(
        self,
        source_file=None,
        target_file=None,
        source_weight_path=None,
        target_weight_path=None,
        data_divisor=1,
        n_jets=5,
        max_tracks=None,
        muon_only=False,
        batch_size=256,
        dataloader_workers=0,
        split_seed=2,
        testing=False,
        use_truth=False,
        **kwargs
    ):
        """ __init__ - This method initializes the LOfData class. It takes
        the path to the Monte Carlo and data files as arguments.

        Arguments:
            source_file {str} -- The path to the file containing the source data.
            target_file {str} -- The path to the file containing the target data.
            source_weight_path {string} -- Path to a .npz file containing weights for the source data
            target_weight_path {string} -- Path to a .npz file containing weights for the target data
            data_divisor {int} -- Divide the whole dataset into this many pieces. Default to 1,
                in which case the whole dataset is used. If >1, then the dataloaders will be configured
                to load only one piece of the data for each epoch.
            n_jets {int} -- The number of jets to consider in the data. Defaults to 5.
            max_tracks {int} -- The maximum number of tracks to consider in the data. Defaults to None,
                in which case all tracks are considered.
            muon_only {bool} -- Set to true if we only want to consider muons.
            batch_size {int} -- The batch size for the data loaders. Defaults
                to 256.
            dataloader_workers {int} -- The number of workers for the data loaders.
            split_seed {int} - The random seed to use in making train / val split,
                if set to -1 then a random integer is used.
            testing {bool} - Set to true for data module to load testing weights (not training)
            use_truth {bool} - Set to true if we want to use truth level information
                for the data module. Defaults to false.
            **kwargs - Passed to the OfDataset classes
        """

        # Set class attributes
        super().__init__()
        self.source_file = source_file
        self.target_file = target_file
        self.source_weight_path = source_weight_path
        self.target_weight_path = target_weight_path
        self.data_divisor = data_divisor
        self.max_tracks = max_tracks
        self.n_jets = n_jets
        self.muon_only = muon_only
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.split_seed = split_seed
        self.testing = testing
        self.use_truth = use_truth

        # Find total number of events in source and target
        self.num_source = uproot.open(self.source_file)['OmniTree'].num_entries
        self.num_target = uproot.open(self.target_file)['OmniTree'].num_entries

        # Determine start / stop indeces for each data piece
        source_start_indeces = np.arange(0, self.num_source, self.num_source // self.data_divisor)
        source_stop_indeces = np.roll(source_start_indeces, -1)
        source_stop_indeces[source_stop_indeces == 0] = self.num_source
        target_start_indeces = np.arange(0, self.num_target, self.num_target // self.data_divisor)
        target_stop_indeces = np.roll(target_start_indeces, -1)
        target_stop_indeces[target_stop_indeces == 0] = self.num_target

        self.start_indeces = list(zip(source_start_indeces, target_start_indeces))
        self.stop_indeces = list(zip(source_stop_indeces, target_stop_indeces))

        # By default, load the first piece. In the case where we are not using a data divisor,
        # this will be the one and only load operation
        self.current_piece = 0
        self.rebuild_datasets(piece=self.current_piece)


    def rebuild_datasets(self, piece=0):
        """ build_datasets - This function builds the pytorch datasets for the source and target data.
        It will load the kinematics, index, weight, label, and plotting info, and use them to build the "source dataset",
        which is used for prediction, and the "all_dataset", which is used for training / validation / testing.
        It will load a piece of the data based on the piece argument, and the data divisor set in the init function.

        Arguments:
            piece {int} -- The piece of the data to load. Defaults to 0, in which case the first piece is loaded.

        Returns:
            None
        """

        if piece >= self.data_divisor:
            raise ValueError("Piece number exceeds data divisor")

        # Get start and stops for source and target
        source_start, target_start = self.start_indeces[piece]
        source_stop, target_stop = self.stop_indeces[piece]

        # Get the data from files
        source_kinematics, source_indeces, source_weights, source_plotting, self.source_pass190, self.source_truth_pass190 = self.load_data_from_file(
            self.source_file, self.source_weight_path, start=source_start, stop=source_stop
        )
        target_kinematics, target_indeces, target_weights, target_plotting, self.target_pass190, self.target_truth_pass190 = self.load_data_from_file(
            self.target_file, self.target_weight_path, start=target_start, stop=target_stop
        )

        # Use the appropriate pass190 flags
        if self.use_truth:
            self.source_use190 = self.source_truth_pass190
            self.target_use190 = self.target_truth_pass190
        else:
            self.source_use190 = self.source_pass190
            self.target_use190 = self.target_pass190

        # Store all weights for use in prediction, then truncate and apply filter
        self.source_all_weights = source_weights
        source_weights = np.expand_dims(source_weights[self.source_use190 == 1], axis=1)

        self.target_all_weights = target_weights
        target_weights = np.expand_dims(target_weights[self.target_use190 == 1], axis=1)

        # Normalize weights so the class ratio is one but the sum of the weights is
        # the number of events in the whole dataset (so initial loss is log(2))
        source_divisor = 2 * np.sum(source_weights) / (len(source_weights) + len(target_weights))
        target_divisor = 2 * np.sum(target_weights) / (len(source_weights) + len(target_weights))
        source_weights /= source_divisor
        target_weights /= target_divisor

        # Labels
        source_labels = np.zeros((len(source_kinematics), 1), dtype=np.float32)
        target_labels = np.ones((len(target_kinematics), 1), dtype=np.float32)

        # Concatenate source and target data
        self.kinematics = ak.concatenate([source_kinematics, target_kinematics], axis=0)
        if not self.muon_only:  # Since we don't use one-hot encodings in debug mode
            self.indeces = ak.concatenate([source_indeces, target_indeces], axis=0)
        else:
            self.indeces = None
        weights = np.concatenate([source_weights, target_weights], axis=0)
        self.labels = np.concatenate([source_labels, target_labels], axis=0)
        self.plotting = np.concatenate([source_plotting, target_plotting], axis=0)

        # Build pytorch datasets
        self.source_dataset = OfDataset(
            source_kinematics,
            source_labels,
            source_weights,
            source_plotting,
            object_indeces=source_indeces,
            n_jets=self.n_jets,
            max_tracks=self.max_tracks
        )
        self.all_dataset = OfDataset(
            self.kinematics,
            self.labels,
            weights, 
            self.plotting,
            object_indeces=self.indeces,
            n_jets=self.n_jets,
            max_tracks=self.max_tracks
        )


    def load_data_from_file(self, path, weight_path, start=None, stop=None):
        """ load_data_from_file - This function loads data from a file using uproot, and applies the relevant preprocessing.
        It returns the kinematics, mask, plotting data, weights, and pass190 filter

        Arguments:
            path {str} -- The path to the file to load the data from.
            weight_path {str} -- The path to the weights file. Note this is for all events, without the pass190 filter
            start {int} -- The start index for the data. Defaults to None, in which case start from 0
            stop {int} -- The stop index for the data. Defaults to None, in which case stop at the end of the file.

        Returns:
            np.ndarray -- The kinematics data
            np.ndarray -- The mask data
            np.ndarray -- The weight data, for all events. Does not apply the pass190 filter!!
            np.ndarray -- The plotting data
            np.ndarray -- The pass190 filter
        """

        # Load the data
        f = uproot.open(path)
        tree = f['OmniTree']

        # Get pass 190 flags
        pass190 = ak.to_numpy(tree['pass190'].array(entry_start=start, entry_stop=stop))
        truth_pass190 = ak.to_numpy(tree['truth_pass190'].array(entry_start=start, entry_stop=stop))
        if self.use_truth:
            use190 = truth_pass190
        else:
            use190 = pass190
        rank_zero_info("We have a fraction {} of good events in this dataset".format(np.sum(use190) / len(use190)))

        # Get kinematics
        kinematics, indeces = get_kinematics(
            tree, 
            muon_only=self.muon_only,
            get_truth=self.use_truth,
            start=start,
            stop=stop
        )

        # Get weights, note this is for all events, without the pass190 filter
        weights = self.load_weights(tree, path=weight_path, test=self.testing, start=start, stop=stop)

        # Get plotting data
        plotting_variables = [hist_dict['key'] for hist_dict in pu.default_settings.values()]
        plotting = get_plotting(
            tree,
            vars=plotting_variables,
            muon_only=self.muon_only,
            get_truth=self.use_truth,
            start=start,
            stop=stop
        )

        return kinematics, indeces, weights, plotting, pass190, truth_pass190


    def load_weights(self, tree, path=None, test=False, start=None, stop=None):
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
            start {int} -- The start index for the weights. Defaults to None, in which case start from 0
            stop {int} -- The stop index for the weights. Defaults to None, in which case stop at the end of the file.
            
        Returns:
            np.ndarray -- A numpy array of weights
        """

        # Get weights from root tree
        root_weights = ak.to_numpy(tree['weight'].array(entry_start=start, entry_stop=stop))

        # Load weights directly from root file
        if path == 'root':
            all_weights = root_weights

        # Load weights from the path
        elif path is not None:
            weight_file = np.load(path)
            np_read_start = 0 if start is None else start
            if test:
                all_weights = weight_file['test']
                np_read_stop = len(all_weights) if stop is None else stop
                all_weights = all_weights[np_read_start:np_read_stop]
            else:
                all_weights = weight_file['train']
                np_read_stop = len(all_weights) if stop is None else stop
                all_weights = weight_file['train'][np_read_start:np_read_stop]

        # Otherwise create a vector of ones
        else:
            all_weights = np.ones_like(root_weights, dtype=np.float32)
            
        return all_weights
    
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

    # Methods for getting pass 190 flags
    def get_source_pass190(self):
        return self.source_use190
    def get_target_pass190(self):
        return self.target_use190
    def get_source_reco_pass190(self):
        return self.source_pass190
    def get_target_reco_pass190(self):
        return self.target_pass190
    def get_source_truth_pass190(self):
        return self.source_truth_pass190
    def get_target_truth_pass190(self):
        return self.target_truth_pass190


    # Train dataloader
    def train_dataloader(self):
        """ train_dataloader - This method returns a pytorch dataloader
        for the training data.

        Arguments:
            piece {int} -- The piece of the data to load.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the training data.
        """

        # Rebuild datasets if required by data divisor
        piece = (self.current_piece + 1) % self.data_divisor
        if piece != self.current_piece:
            self.rebuild_datasets(piece=piece)
            self.current_piece = piece

        # Make train / val split
        generator = torch.Generator().manual_seed(self.split_seed)
        train_dataset, _ = torch.utils.data.random_split(
            self.all_dataset, 
            [0.8, 0.2],
            generator=generator
        )

        # Return dataloader
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
            collate_fn=custom_collate
        )
    

    # Validation dataloader
    def val_dataloader(self):
        """ val_dataloader - This method returns a pytorch dataloader
        for the validation data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the validation data.
        """

        # Make train / val split
        generator = torch.Generator().manual_seed(self.split_seed)
        _, val_dataset = torch.utils.data.random_split(
            self.all_dataset, 
            [0.8, 0.2],
            generator=generator
        )

        # Return dataloader
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
            collate_fn=custom_collate
        )


    # Test dataloader
    def test_dataloader(self):
        """ test_dataloader - This method returns a pytorch dataloader for running predictions. It always yeilds
        the "all dataset". In the case that we are dividing the data into pieces, this will always just
        use the current piece since it doesn't matter which part of the data we use for testing.

        No arguments

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader.
        """

        return torch.utils.data.DataLoader(
            self.all_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
            collate_fn=custom_collate
        )


    # Predict dataloader
    def predict_dataloader(self):
        """ predict_dataloader - This method returns a pytorch dataloader for running predictions. 
        Only need to run predictions for the source data in general, so can just use the source dataset.

        Note the data modules used for prediction should never divide the data since
        we always want to predict for every event. Will include assertion that the data divisor is 1. 

        No arguments

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader.
        """

        assert self.data_divisor == 1
        return torch.utils.data.DataLoader(
            self.source_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
            collate_fn=custom_collate
        )
