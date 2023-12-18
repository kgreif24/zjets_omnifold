""" data_utils.py - This file contains functions for preprocessing and handling 
data for the training of Omnifold discriminators.

Author: Kevin Greif
12/18/2023
python3
"""

import numpy as np
import awkward as ak

# Zero pad the track kinematic information such that arrays are square
def pad_kinematics(input_array, max_tracks=None) -> np.ndarray:
    """ pad_kinematics - This function will take an awkward array of track kinematic
    information, and returns a zero-padded array with the length of padding given by
    the max_tracks argument. Events with more tracks than max_tracks will be truncated.
    If max_track is None, the maximum number of tracks in the input array will be used.

    Arguments:
    input_array - awkward array of track kinematic information
    max_tracks - maximum number of tracks to pad to
    
    Returns:
    padded_array - zero-padded numpy array of track kinematic information
    """

    # If max_tracks is None, get the maximum number of tracks in the input array
    if max_tracks is None:
        max_tracks = int(np.max(ak.count(input_array, axis=1)))

    # Create zero-padded array
    input_array = ak.pad_none(input_array, max_tracks, axis=1, clip=True)
    input_array = ak.to_numpy(ak.fill_none(input_array, 0, axis=1))

    return input_array