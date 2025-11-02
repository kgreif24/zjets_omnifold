"""
test_of_dataset.py - Test suite for the OfDataset class.
"""

import copy
import of_dataset as ofd
import utils.data_utils as du

import uproot
import numpy as np
import awkward as ak


def test_of_dataset():

    # Hard code the event sample file
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    p190 = ak.to_numpy(t["pass190"].array())

    # Load the kinematics and indeces
    kinematics, ind, pdgids = du.get_kinematics(t, evt_filter=p190)

    # Create dummy weights
    weights = np.ones(len(kinematics))

    # Create dummy labels
    labels = np.random.randint(0, 2, len(kinematics))

    # Get observables for W1 calculations
    obs = du.get_observables(t, ["Ntracks", "pT_ll"], evt_filter=p190)

    # Make datasets
    d1 = ofd.OfDataset(
        kinematics,
        labels,
        weights,
        obs,
        pdgids,
        object_indeces=ind,
        max_tracks=20,
        n_jets=2,
    )
    d2 = ofd.OfDataset(
        kinematics,
        labels,
        weights,
        obs,
        pdgids,
        object_indeces=ind,
        max_tracks=None,
        n_jets=5,
    )

    # Get some items
    kin1, label1, mask1, weights1, obs1 = d1[0]
    kin2, label2, mask2, weights2, obs2 = d2.__getitems__([1, 2, 3, 4, 5])

    # Ensure we have the correct shapes, labels, mask
    assert kin1.shape == (1, 7, 20)
    assert kin2.shape[0] == 5
    assert kin2.shape[1] == 10
    assert weights1.numpy()[0] == weights[0]
    assert weights2.numpy()[0] == weights[1]
    assert label1.numpy()[0] == labels[0]
    assert label2.numpy()[0] == labels[1]
    assert (
        np.count_nonzero(mask1) == obs1[0, 0] + 2 if obs1[0, 0] < 20 else 20
    )
    assert np.all(
        np.count_nonzero(mask2[:, 0, :], axis=1) == obs2[:, 0].numpy() + 2
    )

    # Check the concatenate function
    c1 = copy.deepcopy(d1)
    c1.concatenate(d2)
    assert len(c1) == len(d1) + len(d2)
