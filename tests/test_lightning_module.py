"""
test_lightning_module.py - Test suite for the LOfTransformer and LOfData classes
"""

import sys
sys.path.append('.')
sys.path.append('../utils')
import lightning as L
import numpy as np
import uproot
import awkward as ak

from of_dataset import OfDataset
from lightning_module import *
import data_utils as du


def test_overfit(tmp_path):

    # Make and save some .npz files to use as weights
    w1 = np.ones(100, dtype=np.float32)
    w2 = 2 * np.ones(100, dtype=np.float32)
    np.savez(f"{tmp_path}/source.npz", train=w1)
    np.savez(f"{tmp_path}/target.npz", train=w2)

    # Get data class
    data_module = LOfData(
        source_file = './assets/small_evt_sample.root',
        target_file = './assets/small_evt_sample.root',
        source_weight_path = f'{tmp_path}/source.npz',
        target_weight_path = f'{tmp_path}/target.npz',
        batch_size=10,
        split_seed=-1,
        load_all=True,
        n_jets=4
    )

    # Get data loader
    dl = data_module.predict_dataloader()

    # Initialize model
    model = LOfTransformer(
        input_dim=9,
        min_lr=1e-3,
        max_lr=2e-3
    )

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=10,
        enable_progress_bar=True
    )

    # Overfit
    trainer.fit(model, dl)
    print("Done with quick train!")
    assert False


def test_lofdata(tmp_path):

    # Plant a random vector of weights in the tmp directory
    random_weights = np.random.rand(100)
    np.savez(f'{tmp_path}/weights.npz', train=random_weights)

    # Get standalone event sample
    sample = './assets/small_evt_sample.root'
    f = uproot.open(sample)
    t = f['OmniTree']
    p190 = ak.to_numpy(t['pass190'].array())
    tp190 = ak.to_numpy(t['truth_pass190'].array())

    # Get data class, both reco and truth, using root weights
    data_module = LOfData(
        source_file = './assets/small_evt_sample.root',
        target_file = './assets/small_evt_sample.root',
        source_weight_path = 'root',
        target_weight_path = 'root',
        batch_size=10,
        split_seed=1,
        load_all=False,
        max_tracks=20
    )
    data_module.setup('train')
    assert len(data_module.train_dataset) + len(data_module.val_dataset) == len(data_module.all_dataset)

    data_module_truth = LOfData(
        source_file = './assets/small_evt_sample.root',
        target_file = './assets/small_evt_sample.root',
        source_weight_path = 'root',
        target_weight_path = 'root',
        batch_size=10,
        split_seed=-1,
        load_all=True,
        use_truth=True,
        max_tracks=20
    )

    # Check attributes
    lbl = data_module.get_labels()
    assert np.mean(lbl) == 0.5
    kin = data_module.get_track_kinematics()
    kin_truth = data_module_truth.get_track_kinematics()
    assert len(kin) == 2 * np.sum(p190)
    assert len(kin_truth) == 2 * np.sum(tp190)
    assert ak.all(ak.ravel(ak.count(kin, axis=1)) == 3)
    plot = data_module.get_plotting()
    assert plot.shape == (2 * np.sum(p190), 25)
    src_p190 = data_module.get_source_pass190()
    assert np.all(src_p190 == p190)
    trg_p190 = data_module.get_target_pass190()
    assert np.all(trg_p190 == p190)
    src_weights = data_module.get_source_all_weights()
    assert len(src_weights) == 100

    # Check load weights
    root_wgts = data_module.load_weights(t, path='root')
    assert np.all(root_wgts == src_weights)
    save_wgts = data_module.load_weights(t, path=f'{tmp_path}/weights.npz')
    assert np.all(save_wgts == random_weights)
    one_wgts = data_module.load_weights(t, path=None)
    assert np.all(one_wgts == np.ones(100))


    
    