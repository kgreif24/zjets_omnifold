""" of_dataset.py - This file contains the OfDataset class, which subclasses the pytorch
dataset class and implements a custom procedure for promoting data loaded as awkward
arrays to the format needed for training and evaluating models.

Authors: Kevin Greif
Last updated 11/20/2024
python3
"""

import torch
import numpy as np
import awkward as ak
import utils.data_utils as utils


class OfDataset(torch.utils.data.Dataset):
    """OfDataset - A custom subclass of the Pytorch dataset class for use in training
    Omnifold models. The class loads all data as awkward arrays in init, but
    only promotes the data to torch tensors (with zero padding and one-hot encodings)
    when data is requested. This saves memory and allows many more events to fit in memory
    at one time.

    Mask for zero-padded inputs will be generated on the fly when data is accessed.
    """

    def __init__(
        self,
        kinematics,
        labels,
        weights,
        plotting,
        object_indeces=None,
        n_jets=5,
        max_tracks=None,
    ):
        """__init__ - Custom init function for the class. The only important difference
        in usage from a standard Pytorch Tensor dataset is that "kinematics" should be
        an awkward array.

        Arguments:
        kinematics (ak.Array) - The kinematics of the muons / tracks in each event.
            Shape should be (n_events, n_features, VAR n_tracks)
        labels (np.ndarray) - The labels for the events
        weights (np.ndarray) - The weights for the events
        plotting (np.ndarray) - The plotting dimensions for the events
        object_indeces (np.ndarray) - The indeces of the objects to include in the dataset.
            If None, do not include one-hot encodings for the tracks in the data.
        n_jets (int) - The maximum number of jets to include in the one-hot encodings.
            If object_indeces is None, this is not used
        max_tracks (int) - The maximum number of tracks to include in the dataset. If
            None, all tracks are included. Note this is really the maximum number of
            tracks minus 2, since we count the muons as well.

        Returns:
        none
        """

        # Store kinematics and track indeces as awkward arrays
        self.kinematics = kinematics
        self.object_indeces = object_indeces

        # Make max tracks and n_jets a class variable
        self.max_tracks = max_tracks
        self.n_jets = n_jets

        # Send labels, weights, and plotting to torch tensors
        self.weights = torch.from_numpy(weights.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.plotting = torch.from_numpy(plotting.astype(np.float32))

        # Verify all of the datasets have the same shape in the 0th dimension
        try:
            assert (
                len(self.kinematics)
                == len(self.weights)
                == len(self.labels)
                == len(self.plotting)
            )
        except AssertionError:
            raise Exception(
                "Arguments passed to OfDataset class don't have the same number of events!"
            )

    def set_weights(self, weights):
        """set_weights - Set the weights for the dataset

        Arguments:
        weights (np.ndarray) - The weights for the events

        Returns:
        none
        """

        self.weights = torch.from_numpy(weights.astype(np.float32))

    def concatenate(self, dataset):
        """concatenate - Concatenate the data from another dataset to this dataset

        Arguments:
        dataset (OfDataset) - The dataset to concatenate to this dataset

        Returns:
        none
        """

        # Concatenate all of the data
        self.kinematics = ak.concatenate([self.kinematics, dataset.kinematics], axis=0)
        if self.object_indeces is not None:
            self.object_indeces = ak.concatenate(
                [self.object_indeces, dataset.object_indeces], axis=0
            )
        self.labels = torch.cat([self.labels, dataset.labels], dim=0)
        self.weights = torch.cat([self.weights, dataset.weights], dim=0)
        self.plotting = torch.cat([self.plotting, dataset.plotting], dim=0)

    def __len__(self):
        """__len__ - Return the length of the dataset

        No arguments

        Returns:
        (int) - The number of events in the dataset
        """
        return len(self.labels)

    def __getitem__(self, index):
        """__getitem__ - The get item function for the dataset. This is just a
        wrapper for the __getitems__ function that retrieves a single event at a time.

        Arguments:
        index (int) - The index of the event to retrieve

        Returns:
        (tuple) - A tuple containing the following information:
            kinematics - the kinematics of the muons and tracks in the event,
                concatenated with the relevant onehot encodings
            mask - a mask for the zero-padded inputs
            labels - the labels for the event
            weights - the weights for the event
            plotting - the plotting data for the event
        """

        indeces = [index]
        return self.__getitems__(indeces)

    def __getitems__(self, indeces):
        """__getitems__ - The get items function for the dataset. This function
        retrieves multiple events at once.

        Arguments:
        indeces (list) - The indeces of the events to retrieve

        Returns:
        (tuple) - A tuple containing the following information:
            kinematics - the kinematics of the muons and tracks in the event,
                concatenated with the relevant onehot encodings
            mask - a mask for the zero-padded inputs
            labels - the labels for the event
            weights - the weights for the event
            plotting - the plotting data for the event
        """

        # Flatten the indeces if necessary
        indeces = np.array(indeces).flatten()

        ################ Kinematics + One Hots ################

        # Slice kinematics
        kinematics = self.kinematics[indeces, ...]

        # Find max tracks for this batch
        batch_max_tracks = int(ak.max(ak.count(kinematics, axis=2)))
        if self.max_tracks is not None and batch_max_tracks > self.max_tracks:
            batch_max_tracks = self.max_tracks

        # Zero pad kinematicc
        # Result is a numpy array of shape (batch, n_features, batch_max_tracks)
        kinematics = utils.pad_kinematics(kinematics, max_tracks=batch_max_tracks)

        # Process one-hot encodings
        if self.object_indeces is not None:

            # Slice and zero pad the object indeces
            object_indeces = self.object_indeces[indeces, ...]
            object_indeces = utils.pad_kinematics(
                object_indeces, max_tracks=batch_max_tracks, fill=999
            )

            # Get one hot encodings
            one_hots = utils.get_one_hot(kinematics, object_indeces, n_jets=self.n_jets)

            # Concatenate kinematics with one hot encodings
            kinematics = np.concatenate([kinematics, one_hots], axis=1)

        # Convert kinematics to torch tensor
        kinematics = torch.from_numpy(kinematics.astype(np.float32))

        ################ Mask ################

        # Generate mask for zero-padded inputs
        # Assume pT is the 0th element along axis 1
        mask = torch.zeros_like(kinematics[:, 0, :], dtype=torch.bool)
        mask[kinematics[:, 0, :] != 0] = True
        mask = torch.unsqueeze(mask, 1)

        ################ Labels, Weights, Plotting ################
        labels = self.labels[indeces, ...]
        weights = self.weights[indeces, ...]
        plotting = self.plotting[indeces, ...]

        # Return the data as a tuple
        return kinematics, labels, mask, weights, plotting
