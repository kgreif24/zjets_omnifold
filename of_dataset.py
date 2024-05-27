""" of_dataset.py - This file contains the OfDataset class, which subclasses the pytorch
dataset class and implements a custom procedure for promoting data loaded as awkward
arrays to the format needed for training and evaluating models.

Authors: Kevin Greif
Last updated 05/24/2024
python3
"""

import torch
import numpy as np
import awkward as ak
import utils.data_utils as utils


class OfDataset(torch.utils.data.Dataset):
    """ OfDataset - A custom subclass of the Pytorch dataset class for use in training
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
            max_tracks=None
        ):
        """ __init__ - Custom init function for the class. The only important difference
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

        # Find the maximum number of tracks to use in padding
        if max_tracks is not None:
            self.max_tracks = max_tracks
        else:
            self.max_tracks = int(ak.max(ak.count(self.kinematics, axis=2)))
        self.n_jets = n_jets

        # Send labels, weights, and plotting to torch tensors
        self.weights = torch.from_numpy(weights.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.plotting = torch.from_numpy(plotting.astype(np.float32))

        # Verify all of the datasets have the same shape in the 0th dimension
        try:
            assert len(self.kinematics) == len(self.weights) == len(self.labels) == len(self.plotting)
        except:
            raise Exception("Arguments passed to OfDataset class don't have the same number of events!")


    def __len__(self):
        """ __len__ - Return the length of the dataset

        No arguments

        Returns:
        (int) - The number of events in the dataset
        """
        return len(self.labels)
    

    def __getitem__(self, index):
        """ __getitem__ - The get item function for the dataset. This is just a 
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
        """ __getitems__ - The get items function for the dataset. This function
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

        ################ Kinematics + One Hots ################

        # Slice and zero pad the kinematics (muon and track)
        # Result is a numpy array of shape (batch, n_features, max_tracks)
        kinematics = self.kinematics[indeces,...]
        kinematics = utils.pad_kinematics(kinematics, max_tracks=self.max_tracks)

        # Process one-hot encodings
        if self.object_indeces is not None:

            # Slice and zero pad the object indeces
            object_indeces = self.object_indeces[indeces,...]
            object_indeces = utils.pad_kinematics(object_indeces, max_tracks=self.max_tracks, fill=-1)

            # Get one hot encodings
            one_hots = utils.get_one_hot(kinematics, object_indeces, n_jets=self.n_jets)

            # Concatenate kinematics with one hot encodings
            kinematics = np.concatenate([kinematics, one_hots], axis=1)

        # Convert kinematics to torch tensor
        kinematics = torch.from_numpy(kinematics.astype(np.float32))

        ################ Mask ################

        # Generate mask for zero-padded inputs
        # Assume pT is the 0th element along axis 1
        mask = torch.zeros_like(kinematics[:,0,:], dtype=torch.bool)
        mask[kinematics[:,0,:] != 0] = True
        mask = torch.unsqueeze(mask, 1)

        ################ Labels, Weights, Plotting ################
        labels = self.labels[indeces,...]
        weights = self.weights[indeces,...]
        plotting = self.plotting[indeces,...]

        # Return the data as a tuple
        return kinematics, mask, labels, weights, plotting


class DummyDataset(torch.utils.data.Dataset):
    """ DummyDataset - A custom subclass of the Pytorch dataset class that generates
    random data for training and evaluating models.

    Arguments:
    size (int) - The number of samples in the dataset
    input_shape (tuple) - The shape of the input data
    output_shape (tuple) - The shape of the output data

    Returns:
    none
    """
    def __init__(self, size, input_shape, output_shape):
        self.size = size
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __len__(self):
        """ __len__ - Return the length of the dataset

        No arguments

        Returns:
        (int) - The number of samples in the dataset
        """
        return self.size

    def __getitem__(self, index):
        """ __getitem__ - The get item function for the dataset. This generates random
        input and output data for each sample.

        Arguments:
        index (int) - The index of the sample to retrieve

        Returns:
        (dict) - A dictionary containing the input and output data for the sample
        """
        input_data = torch.randn(self.input_shape)
        output_data = torch.randn(self.output_shape)
        return {
            "input": input_data,
            "output": output_data
        }