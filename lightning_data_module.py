""" lightning_data_module.py - This file defines the LOfData class.
It builds OfDatasets from the input ROOT files and provides dataloaders for training and testing.

Author: Kevin Greif
Last updated 11/15/2024
python3
"""

import torch
import lightning as L
from pytorch_lightning.utilities.rank_zero import rank_zero_info

import copy
import numpy as np
import uproot
import awkward as ak

from of_dataset import OfDataset
import utils.data_utils as du
import utils.plotting_utils as pu


class LOfData(L.LightningDataModule):
    """LOfData - This class handles all of the data processing for training
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
        source_file,
        target_file=None,
        source_weight_path=None,
        target_weight_path=None,
        max_events_target=99999999,
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
        **kwargs,
    ):
        """__init__ - This method initializes the LOfData class. It takes
        the path to the Monte Carlo and data files as arguments.

        Arguments:
            source_file {str} -- The path to the file containing the source data.
            target_file {str} -- The path to the file containing the target data. Optional, defaults to None.
            source_weight_path {string} -- Path to a .npz file containing weights for the source data
            target_weight_path {string} -- Path to a .npz file containing weights for the target data
            max_events_target {int} -- The maximum number of events to consider in the target data. Defaults to np.inf
                which means all events are considered.
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
        self.max_events_target = max_events_target
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
        # for the source dataset
        self.source_tree = uproot.open(self.source_file)["OmniTree"]
        self.num_source = self.source_tree.num_entries
        self.source_pass190 = ak.to_numpy(self.source_tree["pass190"].array())
        self.source_truth_pass190 = ak.to_numpy(
            self.source_tree["truth_pass190"].array()
        )
        if self.use_truth:
            self.source_use190 = self.source_truth_pass190
        else:
            self.source_use190 = self.source_pass190
        rank_zero_info(f"Loading source data from {self.source_file}")

        # If we have a target file, do the same for the target, else set to None
        if self.target_file is not None:
            self.target_tree = uproot.open(self.target_file)["OmniTree"]
            self.num_target = self.target_tree.num_entries
            if self.num_target > self.max_events_target:
                self.num_target = self.max_events_target
            self.target_pass190 = ak.to_numpy(
                self.target_tree["pass190"].array(entry_stop=self.num_target)
            )
            self.target_truth_pass190 = ak.to_numpy(
                self.target_tree["truth_pass190"].array(entry_stop=self.num_target)
            )
            if self.use_truth:
                self.target_use190 = self.target_truth_pass190
            else:
                self.target_use190 = self.target_pass190
            rank_zero_info(f"Loading target data from {self.target_file}")
        else:
            self.num_target = None
            self.target_pass190 = None
            self.target_truth_pass190 = None
            self.target_use190 = None

        rank_zero_info(f"We have {self.num_source} source events")
        if self.target_file is not None:
            rank_zero_info(f"We have {self.num_target} target events")

        # Calculate number of good events
        rank_zero_info(
            f"We have a fraction {np.sum(self.source_use190) / self.num_source} of good events in the source dataset"
        )
        if self.target_file is not None:
            rank_zero_info(
                f"We have a fraction {np.sum(self.target_use190) / self.num_target} of good events in the target dataset"
            )

        # Determine start / stop indeces for each data piece, note we don't trucate in the
        # case of non-divisible data, since it is fine if epochs have slightly different lengths
        self.source_indeces = self._setup_pieces(self.num_source)
        if self.target_file is not None:
            self.target_indeces = self._setup_pieces(self.num_target)

        # If we are sharding, make sure we didn't request a nonsensical rank
        if self.total_rank > 1:
            assert self.rank >= 0 and self.rank < self.total_rank

        # By default, load the first piece. In the case where we are not using a data divisor,
        # this will be the one and only load operation
        self.current_piece = 0
        self._rebuild_dataset("source", piece=self.current_piece)
        if self.target_file is not None:
            self._rebuild_dataset("target", piece=self.current_piece)
            self._concatenate_datasets(piece=self.current_piece)

    def _setup_pieces(self, num_events):
        """_setup_pieces - This function sets up the pieces for the data module. It is called
        in the __init__ function for the source data set by default, and optionally for the
        target data set if one is given to the module. It returns the start and stop indeces
        for the pieces of the data.

        Arguments:
            num_events {int} -- The total number of events in the dataset, this is either source
                or target.

        Returns:
            {list} - A list of tuples containing the start and stop indeces for each piece
        """

        indeces = np.linspace(0, num_events, self.data_divisor + 1, dtype=int)
        start_indeces = indeces[:-1]
        stop_indeces = indeces[1:]

        return list(zip(start_indeces, stop_indeces))

    def _rebuild_dataset(self, filename, piece=0):
        """rebuild_dataset - This function builds the pytorch datasets for the source or target data.
        It will load the kinematics, index, weight, label, and plotting info, and use them to build the "source dataset",
        which is used for prediction, and the "all_dataset", which is used for training / validation / testing.
        It will load a piece of the data based on the piece argument, and the data divisor set in the init function.

        Note this function also "shards" the data depending on the number of GPUs in use.
        This is done by dividing the data piece into equal shards, and then only loading the chunk that corresponds
        to the rank of the current GPU.

        Arguments:
            filename {str} -- The file to load data from. Can be 'source' or 'target'
            piece {int} -- The piece of the data to load. Defaults to 0, in which case the first piece is loaded.

        Returns:
            None
        """

        # Make sure we don't ask for target data if it doesn't exist
        if filename == "target" and self.target_file is None:
            raise ValueError("Target file not provided")

        ####################### Configure which data to read ########################

        if piece >= self.data_divisor:
            raise ValueError("Piece number exceeds data divisor")

        # Get start and stop indeces for the piece
        if filename == "source":
            start, stop = self.source_indeces[piece]
        elif filename == "target":
            start, stop = self.target_indeces[piece]

        # If we are using more than one GPU, further shard the data depending on the rank
        # Calculate the start / stop indeces here
        if self.total_rank > 1:
            start, stop = self._calc_shard_indeces(start, stop, file=filename)

        ####################### Load the data ########################

        # Get the data from file
        kinematics, indeces, weights, plotting = self._load_data_from_file(
            filename, self.source_weight_path, start=start, stop=stop
        )

        ####################### Process weights ##########################

        # Store all weights for use in prediction, then truncate and apply filter
        if filename == "source":
            self.source_all_weights = weights.copy()
            weights = weights[start:stop]
            piece190 = self.source_use190[start:stop]
            weights = weights[piece190 == 1]
        elif filename == "target":
            self.target_all_weights = weights.copy()
            weights = weights[start:stop]
            piece190 = self.target_use190[start:stop]
            weights = weights[piece190 == 1]

        ####################### Process labels ##########################

        if filename == "source":
            labels = np.zeros((len(kinematics), 1), dtype=np.float32)
        elif filename == "target":
            labels = np.ones((len(kinematics), 1), dtype=np.float32)

        ####################### Build dataset ##########################

        # Build pytorch datasets
        if filename == "source":
            self.source_kinematics = kinematics
            self.source_labels = labels
            self.source_plotting = plotting
            self.source_dataset = OfDataset(
                kinematics,
                labels,
                weights,
                plotting,
                object_indeces=indeces,
                n_jets=self.n_jets,
                max_tracks=self.max_tracks,
            )
        elif filename == "target":
            self.target_kinematics = kinematics
            self.target_labels = labels
            self.target_plotting = plotting
            self.target_dataset = OfDataset(
                kinematics,
                labels,
                weights,
                plotting,
                object_indeces=indeces,
                n_jets=self.n_jets,
                max_tracks=self.max_tracks,
            )

    def _concatenate_datasets(self, piece=0):
        """_concatenate_datasets - This function concatenates the source and target datasets
        into a single dataset. This is used for training and validation, where we want to use
        both source and target data.

        Arguments:
            piece {int} -- The piece of the data for which we need to concatenate the datasets.
                Defaults to 0, in which case the first piece is used.

        Returns:
            None
        """

        # Calculate normalized weights across the entire source + target dataset
        source_weights, target_weights = self._weight_norm()

        # Calculate start / stop indeces for this piece / shard
        source_start, source_stop = self.source_indeces[piece]
        target_start, target_stop = self.target_indeces[piece]
        if self.total_rank > 1:
            source_start, source_stop = self._calc_shard_indeces(
                source_start, source_stop, file="source"
            )
            target_start, target_stop = self._calc_shard_indeces(
                target_start, target_stop, file="target"
            )

        # Truncate the weights and pass190 filters to the piece / shard
        source_weights = source_weights[source_start:source_stop]
        target_weights = target_weights[target_start:target_stop]
        source_use190 = self.source_use190[source_start:source_stop]
        target_use190 = self.target_use190[target_start:target_stop]

        # Filter out the weights within this piece
        source_weights = np.expand_dims(source_weights[source_use190 == 1], axis=1)
        target_weights = np.expand_dims(target_weights[target_use190 == 1], axis=1)

        # Replace weights in data sets with the normalized ones
        self.source_dataset.set_weights(source_weights)
        self.target_dataset.set_weights(target_weights)

        # Concatenate the datasets
        self.all_dataset = copy.deepcopy(self.source_dataset)
        self.all_dataset.concatenate(self.target_dataset)

    def _load_data_from_file(
        self, which_file="source", weight_path="root", start=None, stop=None
    ):
        """load_data_from_file - This function loads data from a file using uproot, and applies the relevant preprocessing.
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
        if which_file == "source":
            tree = self.source_tree
        elif which_file == "target":
            tree = self.target_tree
        else:
            raise ValueError("Invalid file argument")

        # Get kinematics
        kinematics, indeces = du.get_kinematics(
            tree,
            muon_only=self.muon_only,
            get_truth=self.use_truth,
            start=start,
            stop=stop,
        )

        # Get weights, note this is for all events, without the pass190 filter
        weights = self._load_weights(
            tree, which_file=which_file, path=weight_path, test=self.testing
        )

        # Get plotting data
        plotting_variables = [
            hist_dict["key"] for hist_dict in pu.default_settings.values()
        ]
        plotting = du.get_plotting(
            tree,
            vars=plotting_variables,
            muon_only=self.muon_only,
            get_truth=self.use_truth,
            start=start,
            stop=stop,
        )

        return kinematics, indeces, weights, plotting

    def _load_weights(self, tree, which_file="source", path=None, test=False):
        """_load_weights - This function implements the logic for loading weights to be used both in data loading, and
        in providing access to the weights for the purposes of calculating the next iteration of weights in the evaluation
        routine. The logic is as follows:

        1. If the path is 'root', then we load the weights from the root file
        2. If the path is not None, then we load the weights from the .npz file at the given path
        3. If the path is None, then we return a vector of ones

        Note if we are loading weights from source, then we truncate the weights by max_events_target

        Arguments:
            tree {uproot.tree.TTree} -- The uproot tree object
            path {str} -- The path to the weights file. If set to 'root', then we load the weights from the root file.
            test {bool} -- Set to true if we want to load the test weights. Defaults to false.

        Returns:
            np.ndarray -- A numpy array of weights
        """

        # Get weights from root tree
        max_read = self.max_events_target if which_file == "target" else None
        root_weights = ak.to_numpy(tree["weight"].array(entry_stop=max_read))

        # Load weights directly from root file
        if path == "root":
            all_weights = root_weights

        # Load weights from the path
        elif path is not None:
            weight_file = np.load(path)
            if test:
                all_weights = weight_file["test"]
            else:
                all_weights = weight_file["train"]
                if which_file == "target":
                    all_weights = all_weights[: int(self.max_events_target)]

        # Otherwise create a vector of ones
        else:
            all_weights = np.ones_like(root_weights, dtype=np.float32)

        return all_weights

    def _weight_norm(self):
        """_weight_norm - This function normalizes the source
        and target weights so that the sum of the weights is equal to the number of
        events in the dataset. This is done so that the initial loss is log(2),
        and the class ratio is one.

        Note this function acts on ALL weights that pass the appropriate filter,
        not just the weights within a particular piece.

        Arguments:
            None

        Returns:
            {np.ndarray} - The normalized weights for the source dataset
            {np.ndarray} - The normalized weights for the target dataset
        """

        # Make sure we have target data, else this function doesn't make sense
        assert self.target_file is not None

        # Apply appropriate filter to the weights
        source_weights_filtered = self.source_all_weights[self.source_use190 == 1]
        target_weights_filtered = self.target_all_weights[self.target_use190 == 1]

        # Get divisors and normalize
        source_divisor = (
            2
            * np.sum(source_weights_filtered)
            / (len(source_weights_filtered) + len(target_weights_filtered))
        )
        target_divisor = (
            2
            * np.sum(target_weights_filtered)
            / (len(source_weights_filtered) + len(target_weights_filtered))
        )
        source_weights_rescaled = source_weights_filtered / source_divisor
        target_weights_rescaled = target_weights_filtered / target_divisor

        # Push rescaling back to the full weights
        source_weights = self.source_all_weights.copy()
        target_weights = self.target_all_weights.copy()
        source_weights[self.source_use190 == 1] = source_weights_rescaled
        target_weights[self.target_use190 == 1] = target_weights_rescaled

        # Return full weights
        return source_weights, target_weights

    def _calc_shard_indeces(self, start, stop, file="source"):
        """_calc_shard_indeces - This function calculates the indeces of some shard within either
        the source or target root file, given the start and stop indeces of the piece, and a string
        argument specifying the whether to calculate for source or target.

        Function uses the pass190 filters and the rank as set in the init function.

        Arguments:
            start {int} -- The starting index for the piece
            stop {int} -- The stopping index for the piece
            file {str} -- The file to calculate indeces for. Can be 'source' or 'target'
        """

        if file == "source":
            use190 = self.source_use190
        elif file == "target":
            use190 = self.target_use190
        else:
            raise ValueError("Invalid file argument")

        # Get the pass190 filter for the piece
        use190_piece = use190[start:stop]

        # Get total number of good events in the piece
        good_events = np.sum(use190_piece)

        # Truncate the good events to be divisible by the total rank
        good_events = (good_events // self.total_rank) * self.total_rank

        # Create shards out of the number of events that pass the filter
        shard_indeces = np.linspace(0, good_events, self.total_rank + 1, dtype=int)
        start_idx = shard_indeces[:-1]
        stop_idx = shard_indeces[1:]

        # Get start / stop index for this shard in space of events that pass the filter
        min_idx = start_idx[self.rank]
        max_idx = stop_idx[self.rank]

        # Conver to space of all events
        min_idx = self._pass_to_all(use190, start, min_idx)
        max_idx = self._pass_to_all(use190, start, max_idx)

        return min_idx, max_idx

    def _pass_to_all(self, pass190, start, idx):
        """_pass_to_all - This function calculates an index within the space of all events
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

        acquired_good_evts = np.sum(pass190[start : start + idx])
        if acquired_good_evts < idx:
            start += idx
            idx -= acquired_good_evts
            return self._pass_to_all(pass190, start, idx)
        else:
            return start + idx

    # Method for getting the labels
    def get_labels(self):
        return np.concatenate(
            [self.source_labels, self.target_labels], axis=0
        ).flatten()

    # Method for getting track kinematics
    def get_track_kinematics(self):
        all_kinematics = ak.concatenate(
            [self.source_kinematics, self.target_kinematics], axis=0
        )
        # Gets pT, eta, phi, for all tracks (not muons)
        return all_kinematics[:, :3, 2:]

    # Method for getting plotting data
    def get_plotting(self):
        return np.concatenate([self.source_plotting, self.target_plotting], axis=0)

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
        """train_dataloader - This method returns a pytorch dataloader
        for the training data.

        Arguments:
            piece {int} -- The piece of the data to load.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the training data.
        """

        # Rebuild datasets if required by data divisor
        piece = (self.current_piece + 1) % self.data_divisor
        if piece != self.current_piece:
            self._rebuild_dataset("source", piece=piece)
            self._rebuild_dataset("target", piece=piece)
            self._concatenate_datasets(piece=piece)
            self.current_piece = piece

        # Make train / val split
        generator = torch.Generator().manual_seed(self.split_seed)
        train_dataset, _ = torch.utils.data.random_split(
            self.all_dataset, [0.8, 0.2], generator=generator
        )

        # Return dataloader
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.RandomSampler(train_dataset, generator=generator),
            num_workers=self.dataloader_workers,
            collate_fn=du.null_collate,
        )

    # Validation dataloader
    def val_dataloader(self):
        """val_dataloader - This method returns a pytorch dataloader
        for the validation data.

        Returns:
            torch.utils.data.DataLoader -- A pytorch dataloader for the validation data.
        """

        # Make train / val split
        generator = torch.Generator().manual_seed(self.split_seed)
        _, val_dataset = torch.utils.data.random_split(
            self.all_dataset, [0.8, 0.2], generator=generator
        )

        # Return dataloader
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(val_dataset),
            num_workers=self.dataloader_workers,
            collate_fn=du.null_collate,
        )

    # Test dataloader
    def test_dataloader(self):
        """test_dataloader - This method returns a pytorch dataloader for running predictions. It always yeilds
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
            collate_fn=du.null_collate,
        )

    # Predict dataloader
    def predict_dataloader(self):
        """predict_dataloader - This method returns a pytorch dataloader for running predictions.
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
            collate_fn=du.null_collate,
        )
