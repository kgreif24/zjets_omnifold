""" lightning_data_module.py - This file defines the LOfData class.
It builds OfDatasets from the input ROOT files and provides dataloaders for training and testing.

Author: Kevin Greif
Last updated 11/15/2024
python3
"""

import torch
import lightning as L
from pytorch_lightning.utilities.rank_zero import *

import numpy as np
import uproot
import awkward as ak

from of_dataset import OfDataset
from utils.data_utils import *
import utils.plotting_utils as pu


class LOfData(L.LightningDataModule):
    """ LOfData - This class handles all of the data processing for training
    and Omnifold classifier. It subclasses the LightningDataModule class.

    Init function takes the path to the data files as arguments.
    It then goes through the process of loading data from disk using uproot, and applying
    the relevant preprocessing. The data is stored in the class as attributes, in the form 
    of pytorch datasets.

    The "data_divisor" argument is used to divide the data into pieces. One piece is trained on in a given
    epoch, and then the next piece will be loaded in the subsequent epoch. This is useful for
    training on large datasets, where the entire dataset may not fit in memory.

    The "total_rank" and "rank" arguments are used for distributed training. In this case all of the data
    in a given piece are divided by the number of GPUs being used in training. Each GPU will then load
    only the data it needs. This is much more efficient than loading the entire dataset on each GPU.
    
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
        total_rank=1,
        rank=0,
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
            total_rank {int} -- The total number of GPUs in use for distributed training. Defaults to 1.
            rank {int} -- The rank of the current GPU in use for distributed training. Defaults to 0.
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
        self.total_rank = total_rank
        self.rank = rank
        self.n_jets = n_jets
        self.max_tracks = max_tracks
        self.muon_only = muon_only
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.split_seed = split_seed
        self.testing = testing
        self.use_truth = use_truth

        # Find total number of events in source and target, and get the pass190 filters
        self.source_tree = uproot.open(self.source_file)['OmniTree']
        self.num_source = self.source_tree.num_entries
        self.source_pass190 = ak.to_numpy(self.source_tree['pass190'].array())
        self.source_truth_pass190 = ak.to_numpy(self.source_tree['truth_pass190'].array())

        self.target_tree = uproot.open(self.target_file)['OmniTree']
        self.num_target = self.target_tree.num_entries
        self.target_pass190 = ak.to_numpy(self.target_tree['pass190'].array())
        self.target_truth_pass190 = ak.to_numpy(self.target_tree['truth_pass190'].array())

        rank_zero_info(f"We have {self.num_source} source events and {self.num_target} target events")

        # Determine which filter to use
        if self.use_truth:
            self.source_use190 = self.source_truth_pass190
            self.target_use190 = self.target_truth_pass190
        else:
            self.source_use190 = self.source_pass190
            self.target_use190 = self.target_pass190

        # Calculate number of good events
        rank_zero_info(f"We have a fraction {np.sum(self.source_use190) / self.num_source} of good events in the source dataset")
        rank_zero_info(f"We have a fraction {np.sum(self.target_use190) / self.num_target} of good events in the target dataset")

        # Determine start / stop indeces for each data piece, note we don't trucate in the 
        # case of non-divisible data, since it is fine if epochs have slightly different lengths
        source_indeces = np.linspace(0, self.num_source, self.data_divisor + 1, dtype=int)
        source_start_indeces = source_indeces[:-1]
        source_stop_indeces = source_indeces[1:]
        target_indeces = np.linspace(0, self.num_target, self.data_divisor + 1, dtype=int)
        target_start_indeces = target_indeces[:-1]
        target_stop_indeces = target_indeces[1:]

        self.start_indeces = list(zip(source_start_indeces, target_start_indeces))
        self.stop_indeces = list(zip(source_stop_indeces, target_stop_indeces))

        # If we are sharding, make sure we didn't request a nonsensical rank
        if self.total_rank > 1:
            assert self.rank >= 0 and self.rank < self.total_rank

        # By default, load the first piece. In the case where we are not using a data divisor,
        # this will be the one and only load operation
        self.current_piece = 0
        self.rebuild_datasets(piece=self.current_piece)


    def rebuild_datasets(self, piece=0):
        """ build_datasets - This function builds the pytorch datasets for the source and target data.
        It will load the kinematics, index, weight, label, and plotting info, and use them to build the "source dataset",
        which is used for prediction, and the "all_dataset", which is used for training / validation / testing.
        It will load a piece of the data based on the piece argument, and the data divisor set in the init function.

        Note this function also "shards" the data depending on the number of GPUs in use.
        This is done by dividing the data piece into equal shards, and then only loading the chunk that corresponds
        to the rank of the current GPU.

        Arguments:
            piece {int} -- The piece of the data to load. Defaults to 0, in which case the first piece is loaded.

        Returns:
            None
        """

        ####################### Configure which data to read ########################

        if piece >= self.data_divisor:
            raise ValueError("Piece number exceeds data divisor")

        # Get start and stops for source and target
        source_start, target_start = self.start_indeces[piece]
        source_stop, target_stop = self.stop_indeces[piece]

        # If we are using more than one GPU, further shard the data depending on the rank
        # Calculate the start / stop indeces here
        if self.total_rank > 1:
            source_start, source_stop = self.calc_shard_indeces(source_start, source_stop, file='source')
            target_start, target_stop = self.calc_shard_indeces(target_start, target_stop, file='target')


        ####################### Load the data ########################

        # Get the data from files
        source_kinematics, source_indeces, source_weights, source_plotting = self.load_data_from_file(
            'source', self.source_weight_path, start=source_start, stop=source_stop
        )
        target_kinematics, target_indeces, target_weights, target_plotting = self.load_data_from_file(
            'target', self.target_weight_path, start=target_start, stop=target_stop
        )

        # Use the appropriate pass190 flags
        if self.use_truth:
            self.source_use190 = self.source_truth_pass190
            self.target_use190 = self.target_truth_pass190
        else:
            self.source_use190 = self.source_pass190
            self.target_use190 = self.target_pass190

        ####################### Process weights ##########################

        # Store all weights for use in prediction, then apply filter
        self.source_all_weights = source_weights.copy()
        self.target_all_weights = target_weights.copy()

        # Apply filter
        source_weights_filtered = source_weights[self.source_use190 == 1]
        target_weights_filtered = target_weights[self.target_use190 == 1]

        # Normalize weights so the class ratio is one but the sum of the weights is
        # the number of events in the whole dataset (so initial loss is log(2))
        source_divisor = 2 * np.sum(source_weights_filtered) / (len(source_weights_filtered) + len(target_weights_filtered))
        target_divisor = 2 * np.sum(target_weights_filtered) / (len(source_weights_filtered) + len(target_weights_filtered))
        source_weights_rescaled = source_weights_filtered / source_divisor
        target_weights_rescaled = target_weights_filtered / target_divisor

        # Push rescaling back to the full weights
        source_weights[self.source_use190 == 1] = source_weights_rescaled
        target_weights[self.target_use190 == 1] = target_weights_rescaled

        # Truncate both the weights and pass190 filters to this particular piece
        source_weights = source_weights[source_start:source_stop]
        target_weights = target_weights[target_start:target_stop]
        source_use190 = self.source_use190[source_start:source_stop]
        target_use190 = self.target_use190[target_start:target_stop]

        # Then finally filter out the weights within this piece, these we use
        source_weights = np.expand_dims(source_weights[source_use190 == 1], axis=1)
        target_weights = np.expand_dims(target_weights[target_use190 == 1], axis=1)

        ####################### Process labels ##########################

        source_labels = np.zeros((len(source_kinematics), 1), dtype=np.float32)
        target_labels = np.ones((len(target_kinematics), 1), dtype=np.float32)

        ####################### Concatentate data and build datasets ##########################

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


    def load_data_from_file(self, which_file='source', weight_path='root', start=None, stop=None):
        """ load_data_from_file - This function loads data from a file using uproot, and applies the relevant preprocessing.
        It returns the kinematics, mask, plotting data, weights, and pass190 filter

        Arguments:
            which_file {str} -- The file to load data from. Can be 'source' or 'target'
            weight_path {str} -- The path to the weights file. Note this is for all events, without the pass190 filter
                Defaults to 'root', in which case we load the weights from the ROOT file
            start {int} -- The start index for the data. Defaults to None, in which case start from 0
            stop {int} -- The stop index for the data. Defaults to None, in which case stop at the end of the file.

        Returns:
            np.ndarray -- The kinematics data
            np.ndarray -- The mask data
            np.ndarray -- The weight data, for all events, regardless of start/stop. Does not apply the pass190 filter!!
            np.ndarray -- The plotting data
        """

        # Get tree to use
        if which_file == 'source':
            tree = self.source_tree
        elif which_file == 'target':
            tree = self.target_tree
        else:
            raise ValueError("Invalid file argument")

        # Get kinematics
        kinematics, indeces = get_kinematics(
            tree, 
            muon_only=self.muon_only,
            get_truth=self.use_truth,
            start=start,
            stop=stop
        )

        # Get weights, note this is for all events, without the pass190 filter
        weights = self.load_weights(tree, path=weight_path, test=self.testing)

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

        return kinematics, indeces, weights, plotting


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


    def calc_shard_indeces(self, start, stop, file='source'):
        """ calc_shard_indeces - This function calculates the indeces of some shard within either
        the source or target root file, given the start and stop indeces of the piece, and a string
        argument specifying the whether to calculate for source or target.

        Function uses the pass190 filters and the rank as set in the init function.

        Arguments:
            start {int} -- The starting index for the piece
            stop {int} -- The stopping index for the piece
            file {str} -- The file to calculate indeces for. Can be 'source' or 'target'
        """

        if file == 'source':
            use190 = self.source_use190
        elif file == 'target':
            use190 = self.target_use190
        else:
            raise ValueError("Invalid file argument")

        # Get the pass190 filter for the piece
        use190_piece = use190[start:stop]
        print(use190)

        # Get total number of good events in the piece
        good_events = np.sum(use190_piece)
        print(f"Have {good_events} good events in this piece")

        # Truncate the good events to be divisible by the total rank
        # This is needed to make sure that each epoch is the same length across all GPUs
        good_events = (good_events // self.total_rank) * self.total_rank

        # Create shards out of the number of events that pass the filter
        shard_indeces = np.linspace(0, good_events, self.total_rank + 1, dtype=int)
        start_idx = shard_indeces[:-1]
        stop_idx = shard_indeces[1:]

        # Get start / stop index for this shard in space of events that pass the filter
        min_idx = start_idx[self.rank]
        max_idx = stop_idx[self.rank]
        print(f"Looking for good events {min_idx} to {max_idx}")

        # Conver to space of all events
        min_idx = self.pass_to_all(use190, start, min_idx)
        max_idx = self.pass_to_all(use190, start, max_idx)

        return min_idx, max_idx

    
    def pass_to_all(self, pass190, start, idx):
        """ pass_to_all - This function calculates an index within the space of all events
        based on a start index of a piece (in the space of all events) and an index within the space of 
        only good events in the piece.
        
        This is for use in calculating shard indeces! Will use recursion to do calculation efficiently.
        
        Arguments:
            pass190 {np.ndarray} -- The pass190 filter
            start {int} -- The start index within the space of all events
            idx {int} -- The index within the space of only good events in the piece

        Returns:
            int -- The index within the space of all events
        """

        acquired_good_evts = np.sum(pass190[start:start+idx])
        if acquired_good_evts < idx:
            start += idx
            idx -= acquired_good_evts
            return self.pass_to_all(pass190, start, idx)
        else:
            return start + idx


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
            sampler=torch.utils.data.RandomSampler(train_dataset, generator=generator),
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
            sampler=torch.utils.data.SequentialSampler(val_dataset),
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
            sampler=torch.utils.data.SequentialSampler(self.all_dataset),
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
            sampler=torch.utils.data.SequentialSampler(self.source_dataset),
            num_workers=self.dataloader_workers,
            collate_fn=custom_collate
        )