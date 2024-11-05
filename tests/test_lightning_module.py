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
import pytest

from lightning_module import *
import data_utils as du

@pytest.mark.slow
def test_overfit(tmp_path):

    # Get data class
    data_module = LOfData(
        source_file='./assets/evts_000_100.root',
        target_file='./assets/evts_200_300.root',
        source_weight_path=None,
        target_weight_path=None,
        batch_size=10,
        split_seed=-1,
        muon_only=True
    )

    # Get data loader
    dl_train = data_module.test_dataloader(shuffle=True)
    dl_pred = data_module.test_dataloader(shuffle=False)

    # Initialize model
    model = LOfTransformer(
        debug=True,
        input_dim=10,
        min_lr=1e-3,
        max_lr=2e-3,
        no_w1=True
    )

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=200,
        enable_progress_bar=False,
        default_root_dir=tmp_path
    )

    # Overfit
    trainer.fit(model, dl_train)

    # Check that we have a small loss
    final_loss = trainer.callback_metrics.get("train_loss")
    assert final_loss < 0.1

    # Run predict
    predictions = trainer.predict(model, dl_pred)
    predictions = np.concatenate([p.cpu().numpy().flatten() for p in predictions])

    # Pass predictions through sigmoid and check they are reasonable
    probs = 1 / (1 + np.exp(-predictions))
    right_answers_low = np.concatenate([np.zeros(82), np.ones(83)-0.5])
    right_answers_high = np.concatenate([np.zeros(82)+0.5, np.ones(83)])
    test = np.logical_and(probs >= right_answers_low, probs <= right_answers_high)
    assert np.all(test)


def test_lofdata(tmp_path):

    # Plant a random vector of weights in the tmp directory
    random_weights = np.random.rand(100)
    np.savez(f'{tmp_path}/weights.npz', train=random_weights)

    # Get standalone event sample
    sample = './assets/evts_000_100.root'
    f = uproot.open(sample)
    t = f['OmniTree']
    p190 = ak.to_numpy(t['pass190'].array())
    tp190 = ak.to_numpy(t['truth_pass190'].array())

    # Check data divisor, ensure we can use different chunks of the data
    data_module = LOfData(
        source_file = './assets/evts_000_100.root',
        target_file = './assets/evts_000_100.root',
        source_weight_path = 'root',
        target_weight_path = 'root',
        batch_size=1,
        split_seed=1,
        load_all=False,
        max_tracks=20,
        data_divisor=3,
    )
    tloader1 = data_module.train_dataloader()
    vloader1 = data_module.val_dataloader()
    assert len(tloader1) + len(vloader1) == len(data_module.all_dataset)
    tloader2 = data_module.train_dataloader()
    vloader2 = data_module.val_dataloader()
    assert len(tloader2) + len(vloader2) == len(data_module.all_dataset)
    b1 = next(iter(tloader1))
    b2 = next(iter(tloader2))
    assert b1[0][0,0,0] != b2[0][0,0,0]

    # Get data class, both reco and truth, using root weights
    data_module = LOfData(
        source_file = './assets/evts_000_100.root',
        target_file = './assets/evts_000_100.root',
        source_weight_path = 'root',
        target_weight_path = 'root',
        batch_size=1,
        split_seed=1,
        load_all=False,
        max_tracks=20,
        data_divisor=1,
    )

    data_module_truth = LOfData(
        source_file = './assets/evts_000_100.root',
        target_file = './assets/evts_000_100.root',
        source_weight_path = 'root',
        target_weight_path = 'root',
        batch_size=1,
        split_seed=-1,
        load_all=True,
        use_truth=True,
        max_tracks=20,
        data_divisor=1
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


    
    