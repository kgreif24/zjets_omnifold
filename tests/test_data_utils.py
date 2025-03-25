"""
test_data_utils.py - Unit test suite for the data loading utilities.
"""

import utils.data_utils as du

import uproot
import awkward as ak
import numpy as np


def test_pad_kinematics():

    input_array = ak.Array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2], [4, 5], [7, 8]], [[1], [4], [7]]]
    )
    max_tracks = 2
    fill = -1

    output_array = du.pad_kinematics(input_array, max_tracks=max_tracks, fill=fill)
    test_out = np.array(
        [
            [[1, 2], [4, 5], [7, 8]],
            [[1, 2], [4, 5], [7, 8]],
            [[1, -1], [4, -1], [7, -1]],
        ]
    )
    assert np.all(output_array == test_out)


def test_get_one_hot():

    # Test events with 2 muons and 4 tracks
    kinematics = np.array(
        [
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
        ]
    )
    indeces = np.array(
        [[[-1, -1, 0, 0, 1, 1]], [[-1, -1, 0, 1, 2, 2]], [[-1, -1, 0, 1, 1, 2]]]
    )

    one_hot = du.get_one_hot(kinematics, indeces, n_jets=2)
    test_out = np.array(
        [
            [
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
            ],
            [
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        ]
    )
    assert np.all(one_hot == test_out)


def test_get_kinematics():

    # Hardcode the location of the event sample
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    nt = ak.to_numpy(t["Ntracks"].array())
    p190 = ak.to_numpy(t["pass190"].array())

    # Filter the number of tracks
    nt = nt[p190 == 1]

    # Assert we have the expected number of events
    gk1, ind1 = du.get_kinematics(t)
    assert len(gk1) == np.sum(p190)
    assert len(ind1) == np.sum(p190)

    # Assert that we've taken the log of the pT (some negative values)
    assert np.any(gk1[:, 0, :] < 0)

    # Assert the number of tracks in each event is correct
    gk1_count = ak.to_numpy(ak.count(gk1[:, 0, :], axis=1))
    ind1_count = ak.to_numpy(ak.count(ind1[:, 0, :], axis=1))
    assert np.all(gk1_count == ind1_count)
    assert np.all(gk1_count == nt + 2)

    # Assert that we have two muons within each event
    assert np.all(np.count_nonzero(ind1[:, 0, :] == -1, axis=1) == 2)


def test_get_observables():

    # Hardcode the location of the event sample
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    p190 = ak.to_numpy(t["pass190"].array())

    # Assert we have 100 events and 2 vars
    plotting = du.get_observables(t, ["Ntracks", "pT_ll"])
    assert plotting.shape == (np.sum(p190), 2)


def test_stack():

    # Hardcode the location of the event sample
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    p190 = ak.to_numpy(t["pass190"].array())

    # Get kinematics
    gk1, ind1 = du.get_kinematics(t)

    # Zero pad kinematics
    gk1 = du.pad_kinematics(gk1, max_tracks=20, fill=0)

    # Zero pad indeces
    ind1 = du.pad_kinematics(ind1, max_tracks=20, fill=999)

    # Get one hots
    one_hot = du.get_one_hot(gk1, ind1, n_jets=5)

    # Verify each object has a single one-hot encoding
    one_hot_count = np.count_nonzero(one_hot, axis=1)
    assert np.all(one_hot_count == 1)

    # Concatenate one hot with kinematics
    gk1 = np.concatenate([gk1, one_hot], axis=1)
    assert gk1.shape == (np.sum(p190), 10, 20)
