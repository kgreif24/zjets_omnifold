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

    one_hot_reco = du.get_one_hot(kinematics, indeces, n_jets=2)
    test_out_reco = np.array(
        [
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
            ],
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        ]
    )
    assert np.all(one_hot_reco == test_out_reco)


def test_get_masses():

    # Test events with 2 muons and 4 tracks
    pdgids = np.array(
        [
            [[13, 211, -999, -999, -999]],
            [[13, 13, 211, -999, -999]],
        ]
    )
    masses = du.get_masses(pdgids)
    masses_test = ak.Array(
        [
            [
                [
                    0.105658,
                    0.13957,
                    0.0,
                    0.0,
                    0.0,
                ]
            ],
            [
                [
                    0.105658,
                    0.105658,
                    0.13957,
                    0.0,
                    0.0,
                ]
            ],
        ]
    )
    assert ak.all(ak.isclose(masses, masses_test))


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
    gk1, ind1, pdgids = du.get_kinematics(t)
    assert len(gk1) == np.sum(p190)
    assert len(ind1) == np.sum(p190)
    assert len(pdgids) == np.sum(p190)

    # Assert that we've taken the log of the pT (some negative values)
    assert np.any(gk1[:, 0, :] < 0)

    # Assert the number of tracks in each event is correct
    gk1_count = ak.to_numpy(ak.count(gk1[:, 0, :], axis=1))
    ind1_count = ak.to_numpy(ak.count(ind1[:, 0, :], axis=1))
    pdgids_count = ak.to_numpy(ak.count(pdgids[:, 0, :], axis=1))
    assert np.all(gk1_count == ind1_count)
    assert np.all(gk1_count == pdgids_count)
    assert np.all(gk1_count == nt + 2)

    # Assert that we have two muons within each event
    assert np.all(np.count_nonzero(ind1[:, 0, :] == -1, axis=1) == 2)
    assert np.all(np.count_nonzero(pdgids[:, 0, :] == 13, axis=1) == 2)

    # Assert that the pdgids contain only 13 and 211
    assert np.all(np.unique(ak.to_numpy(ak.flatten(pdgids, axis=None))) == [13, 211])


def test_get_kinematics_truth():

    # Hardcode the location of the event sample
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    nt = ak.to_numpy(t["truth_Ntracks"].array())
    p190 = ak.to_numpy(t["truth_pass190"].array())

    # Filter the number of tracks
    nt = nt[p190 == 1]

    # Assert we have the expected number of events
    gk1, ind1, pdgids = du.get_kinematics(t, get_truth=True)
    assert len(gk1) == np.sum(p190)
    assert len(ind1) == np.sum(p190)
    assert len(pdgids) == np.sum(p190)

    # Assert that we have two muons within each event
    assert np.all(np.count_nonzero(ind1[:, 0, :] == -1, axis=1) == 2)
    assert np.all(np.count_nonzero(pdgids[:, 0, :] == 13, axis=1) == 2)

    # Assert that the pdgids contain only 13 and 211
    assert np.all(np.unique(ak.to_numpy(ak.flatten(pdgids, axis=None))) == [13, 211])


def test_get_kinematics_syst():

    # Hardcode the location of the event sample
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    p190 = ak.to_numpy(t["pass190"].array())

    # Get nominal kinematics
    nominal_kinematics, nominal_indices, _ = du.get_kinematics(t)
    assert len(nominal_kinematics) == np.sum(p190)

    # Get systematic kinematics
    trackeff_kinematics, trackeff_indices, _ = du.get_kinematics(t, syst_kw="track_eff")
    assert len(trackeff_kinematics) == np.sum(p190)

    # Get track scale kinematics
    trackscale_kinematics, trackscale_indices, _ = du.get_kinematics(
        t, syst_kw="track_scale"
    )
    assert len(trackscale_kinematics) == np.sum(p190)

    # Count tracks and assert track efficiency varied data has fewer
    nominal_count = ak.to_numpy(ak.count(nominal_kinematics[:, 0, :], axis=1))
    trackeff_count = ak.to_numpy(ak.count(trackeff_kinematics[:, 0, :], axis=1))
    assert np.all(nominal_count >= trackeff_count)

    # Calculate HT and assert that nominal is greater than track efficiency
    nominal_ht = ak.to_numpy(np.sum(np.exp(nominal_kinematics[:, 0, :]), axis=1))
    trackeff_ht = ak.to_numpy(np.sum(np.exp(trackeff_kinematics[:, 0, :]), axis=1))
    assert np.all(nominal_ht >= trackeff_ht)

    # Ensure that the track scale pT is different from the nominal pT
    nominal_pt = nominal_kinematics[:, 0, :]
    trackscale_pt = trackscale_kinematics[:, 0, :]
    assert ak.any(nominal_pt != trackscale_pt)


def test_get_observables():

    # Hardcode the location of the event sample
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    p190 = ak.to_numpy(t["pass190"].array())

    # Assert we have 100 events and 2 vars
    plotting = du.get_observables(t, ["Ntracks", "pT_ll"])
    assert plotting.shape == (np.sum(p190), 2)


def test_get_observables_syst():

    # Hardcode the location of the event sample
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    p190 = ak.to_numpy(t["pass190"].array())
    mid_p190 = ak.to_numpy(t["pass190_syst_ID_Up"].array())

    # Get obserable keys
    nominal_keys = du.get_w1_obs()
    trackeff_keys = du.get_w1_obs(syst_kw="track_eff")
    trackscale_keys = du.get_w1_obs(syst_kw="track_scale")
    muonid_keys = du.get_w1_obs(syst_kw="muon_id")
    assert len(nominal_keys) == len(trackeff_keys)
    assert len(nominal_keys) == len(trackscale_keys)
    assert len(nominal_keys) == len(muonid_keys)

    # Filter out 3 keys
    nominal_3keys = ["Ntracks", "HT_tracks", "pT_ll"]
    trackeff_3keys = ["syst_TrackFilter_Ntracks", "syst_TrackFilter_HT_tracks", "pT_ll"]
    muonid_3keys = [
        "syst_pT_l1_ID_Up",
        "syst_pT_ll_ID_Up",
        "m_trackj1",
    ]
    assert all([key in nominal_keys for key in nominal_3keys])
    assert all([key in trackeff_keys for key in trackeff_3keys])
    assert all([key in muonid_keys for key in muonid_3keys])

    # Get nominal and syst varied observables
    nominal_obs = du.get_observables(t, nominal_3keys, syst_kw=None)
    trackeff_obs = du.get_observables(t, trackeff_3keys, syst_kw="track_eff")
    muonid_obs = du.get_observables(t, muonid_3keys, syst_kw="muon_id")
    assert nominal_obs.shape == trackeff_obs.shape == (np.sum(p190), 3)
    assert muonid_obs.shape == (np.sum(mid_p190), 3)

    # Verify some features of the track systematics
    assert np.all(nominal_obs[:, 0] >= trackeff_obs[:, 0])
    assert np.all(nominal_obs[:, 1] + 1e-4 >= trackeff_obs[:, 1])
    assert np.all(np.isclose(nominal_obs[:, 2], trackeff_obs[:, 2], rtol=1e-4))


def test_stack():

    # Hardcode the location of the event sample
    sample = "./assets/evts_000_100.root"
    f = uproot.open(sample)
    t = f["OmniTree"]
    p190 = ak.to_numpy(t["pass190"].array())

    # Get kinematics
    gk1, ind1, pdgids = du.get_kinematics(t)

    # Zero pad kinematics
    gk1 = du.pad_kinematics(gk1, max_tracks=10, fill=0)

    # Zero pad pdgids and get masses
    pdgids = du.pad_kinematics(pdgids, max_tracks=10, fill=-999)
    masses = du.get_masses(pdgids)

    # Zero pad indeces
    ind1 = du.pad_kinematics(ind1, max_tracks=10, fill=-999)

    # Get one hots
    one_hot = du.get_one_hot(gk1, ind1, n_jets=5)

    # Verify each object has a single one-hot encoding
    one_hot_count = np.count_nonzero(one_hot, axis=1)
    assert np.all(one_hot_count[:, 2:] == 1)

    # Concatenate one hot with kinematics
    gk1 = np.concatenate([gk1, masses, one_hot], axis=1)
    assert gk1.shape == (np.sum(p190), 10, 10)
