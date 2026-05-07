""" of_dataset.py - This file contains the OfDataset and OfPairedDataset classes.
OfDataset implements the standard per-event Omnifold training data format.
OfPairedDataset returns paired (reco, truth) kinematics for the same simulated
event, as required by the AUSSIE training procedure.

Authors: Kevin Greif
Last updated 8/2/2025
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
    when data is requested. This saves memory and allows many more events to fit in
    memory at one time.

    Mask for zero-padded inputs will be generated on the fly when data is accessed.
    """

    def __init__(
        self,
        kinematics,
        labels,
        weights,
        w1_obs,
        pdgids,
        object_indeces=None,
        n_jets=5,
        truth_level=False,
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
        w1_obs (np.ndarray) - The W1 observables for the events, used to calculate
            W1 metrics.
        pdgids (ak.Array) - The pdgids of the particles in the event.
            Shape should be (n_events, 1, VAR n_tracks)
        object_indeces (ak.Array) - The indeces of the objects to include in the
            dataset. Should be shape (n_events, 1, VAR n_tracks)
        n_jets (int) - The maximum number of jets to include in the one-hot encodings.
            If object_indeces is None, this is not used
        max_tracks (int) - The maximum number of tracks to include in the dataset. If
            None, all tracks are included. Note this is really the maximum number of
            tracks minus 2, since we count the muons as well.

        Returns:
        None
        """

        # Store kinematics, track indeces, and pdgids as awkward arrays
        self.kinematics = kinematics
        self.object_indeces = object_indeces
        self.pdgids = pdgids

        # Set class variables
        self.max_tracks = max_tracks
        self.n_jets = n_jets

        # Send labels, weights, and W1 observables to torch tensors
        self.weights = torch.from_numpy(weights.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.w1_obs = torch.from_numpy(w1_obs.astype(np.float32))

        # Verify all of the datasets have the same shape in the 0th dimension
        try:
            assert (
                len(self.kinematics)
                == len(self.weights)
                == len(self.labels)
                == len(self.w1_obs)
                == len(self.pdgids)
            )
        except AssertionError:
            raise Exception(
                "Arguments passed to OfDataset class don't have"
                "the same number of events!"
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
        if self.pdgids is not None:
            self.pdgids = ak.concatenate([self.pdgids, dataset.pdgids], axis=0)
        self.labels = torch.cat([self.labels, dataset.labels], dim=0)
        self.weights = torch.cat([self.weights, dataset.weights], dim=0)
        self.w1_obs = torch.cat([self.w1_obs, dataset.w1_obs], dim=0)

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
            w1_obs - the W1 observables for the event
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
            w1_obs - the W1 observables for the event
        """

        # Flatten the indeces if necessary
        indeces = np.array(indeces).flatten()

        # ------------- Kinematics + One Hots -------------

        # Slice kinematics and pdgids
        kinematics = self.kinematics[indeces, ...]

        # Find max tracks for this batch
        batch_max_tracks = int(ak.max(ak.count(kinematics, axis=2)))
        if self.max_tracks is not None and batch_max_tracks > self.max_tracks:
            batch_max_tracks = self.max_tracks

        # Zero pad kinematics
        # Result is a numpy array of shape (batch, n_features, batch_max_tracks)
        kinematics = utils.pad_kinematics(kinematics, max_tracks=batch_max_tracks)

        # Process pdgids, goal is to infer the object mass as
        # an extra kinematic feature
        pdgids = self.pdgids[indeces, ...]
        pdgids = utils.pad_kinematics(
            pdgids, max_tracks=batch_max_tracks, fill=-999
        )
        masses = utils.get_masses(pdgids)
        kinematics = np.concatenate([kinematics, masses], axis=1)

        # Process one-hot encodings for track indeces
        if self.object_indeces is not None:

            # Slice the indeces then run padding
            object_indeces = self.object_indeces[indeces, ...]
            object_indeces = utils.pad_kinematics(
                object_indeces, max_tracks=batch_max_tracks, fill=-999
            )

            # Get one hot encodings
            one_hots = utils.get_one_hot(
                kinematics,
                object_indeces,
                n_jets=self.n_jets,
            )

            # Concatenate kinematics with one hot encodings
            kinematics = np.concatenate([kinematics, one_hots], axis=1)

        # Convert kinematics to torch tensor
        kinematics = torch.from_numpy(kinematics.astype(np.float32))

        # ------------------ Mask ------------------

        # Generate mask for zero-padded inputs
        # Assume pT is the 0th element along axis 1
        mask = torch.zeros_like(kinematics[:, 0, :], dtype=torch.bool)
        mask[kinematics[:, 0, :] != 0] = True
        mask = torch.unsqueeze(mask, 1)

        # ------------- Labels, Weights, W1 Obs -------------
        labels = self.labels[indeces, ...]
        weights = self.weights[indeces, ...]
        w1_obs = self.w1_obs[indeces, ...]

        # Return the data as a tuple
        return kinematics, labels, mask, weights, w1_obs


def _build_kinematics_tensor(
    kinematics_ak,
    pdgids_ak,
    object_indeces_ak,
    max_tracks,
    n_jets,
):
    """_build_kinematics_tensor - Helper that performs the same
    padding / mass-appending / one-hot-encoding pipeline used in
    OfDataset.__getitems__. Returns a float32 torch tensor of shape
    (batch, n_features, batch_max_tracks) and a boolean mask tensor
    of shape (batch, 1, batch_max_tracks).
    """

    # Find max tracks for this batch
    batch_max_tracks = int(ak.max(ak.count(kinematics_ak, axis=2)))
    if max_tracks is not None and batch_max_tracks > max_tracks:
        batch_max_tracks = max_tracks

    # Zero pad kinematics
    kinematics = utils.pad_kinematics(kinematics_ak, max_tracks=batch_max_tracks)

    # Masses from pdgids
    pdgids = utils.pad_kinematics(
        pdgids_ak, max_tracks=batch_max_tracks, fill=-999
    )
    masses = utils.get_masses(pdgids)
    kinematics = np.concatenate([kinematics, masses], axis=1)

    # One-hot encodings for track jet indeces
    if object_indeces_ak is not None:
        object_indeces = utils.pad_kinematics(
            object_indeces_ak, max_tracks=batch_max_tracks, fill=-999
        )
        one_hots = utils.get_one_hot(
            kinematics, object_indeces, n_jets=n_jets,
        )
        kinematics = np.concatenate([kinematics, one_hots], axis=1)

    kinematics = torch.from_numpy(kinematics.astype(np.float32))
    mask = torch.zeros_like(kinematics[:, 0, :], dtype=torch.bool)
    mask[kinematics[:, 0, :] != 0] = True
    mask = torch.unsqueeze(mask, 1)

    return kinematics, mask


class OfPairedDataset(torch.utils.data.Dataset):
    """OfPairedDataset - A dataset that returns paired reco (x) and truth (z)
    kinematics for the same MC event, as needed by the AUSSIE training loop.

    The dataset is filtered at construction time so that only events which
    pass both reco pass190 and truth pass190 are kept. The reco and truth
    awkward arrays therefore have the same length and are index-aligned.

    Only nominal MC weights are supported in the v1 AUSSIE implementation
    (no syst_kw, no bootstraps).
    """

    def __init__(
        self,
        truth_kinematics,
        weights,
        w1_obs,
        truth_pdgids,
        reco_kinematics=None,
        reco_pdgids=None,
        reco_object_indeces=None,
        truth_object_indeces=None,
        n_jets=5,
        max_tracks=None,
    ):
        """__init__ - The arrays passed in here must already be filtered
        to the appropriate pass190 selection. The weights should be the
        nominal MC weights (weight_mc branch).

        Reco-level arrays (reco_kinematics, reco_pdgids, reco_object_indeces)
        are optional. When omitted (inference / truth-only mode) the dataset
        returns None for x_kin and x_mask; predict_step ignores those fields.

        Arguments:
            truth_kinematics (ak.Array) - Truth muons + tracks, shape
                (n_events, n_features, VAR n_tracks)
            weights (np.ndarray) - MC weights, one per paired event
            w1_obs (np.ndarray) - Reco-level observables for the Wasserstein metric
            truth_pdgids (ak.Array) - Truth pdgids, shape (n_events, 1, VAR n_tracks)
            reco_kinematics (ak.Array | None) - Reco muons + tracks; required for
                training and validation, may be None for inference.
            reco_pdgids (ak.Array | None) - Reco pdgids; required when
                reco_kinematics is provided.
            reco_object_indeces (ak.Array | None) - Reco track-jet indeces, optional
            truth_object_indeces (ak.Array | None) - Truth track-jet indeces, optional
            n_jets (int) - Max jets for one-hot encoding
            max_tracks (int) - Max number of tracks to pad to
        """

        self.reco_kinematics = reco_kinematics
        self.truth_kinematics = truth_kinematics
        self.reco_pdgids = reco_pdgids
        self.truth_pdgids = truth_pdgids
        self.reco_object_indeces = reco_object_indeces
        self.truth_object_indeces = truth_object_indeces

        self.max_tracks = max_tracks
        self.n_jets = n_jets

        self.weights = torch.from_numpy(weights.astype(np.float32))
        self.w1_obs = torch.from_numpy(w1_obs.astype(np.float32))

        lengths_to_check = [
            len(self.truth_kinematics),
            len(self.weights),
            len(self.w1_obs),
            len(self.truth_pdgids),
        ]
        if self.reco_kinematics is not None:
            lengths_to_check += [len(self.reco_kinematics), len(self.reco_pdgids)]
        if len(set(lengths_to_check)) != 1:
            raise Exception(
                "Arrays passed to OfPairedDataset don't have matching event counts!"
            )

    def set_weights(self, weights):
        self.weights = torch.from_numpy(weights.astype(np.float32))

    def precompute(self):
        """Materialize all events into fixed-size tensors upfront.

        After calling this, __getitems__ becomes pure tensor indexing.
        Eliminates per-batch awkward-array overhead and makes num_workers=0
        fast enough for inference (no worker fork / OOM risk).
        Requires max_tracks to be set so the output shape is deterministic.
        """
        if self.max_tracks is None:
            raise ValueError("max_tracks must be set to use precompute()")

        all_indices = np.arange(len(self.weights))

        if self.reco_kinematics is not None:
            reco_obj = (
                self.reco_object_indeces if self.reco_object_indeces is not None else None
            )
            self._pre_x_kin, self._pre_x_mask = _build_kinematics_tensor(
                self.reco_kinematics[all_indices, ...],
                self.reco_pdgids[all_indices, ...],
                reco_obj[all_indices, ...] if reco_obj is not None else None,
                self.max_tracks,
                self.n_jets,
            )
        else:
            self._pre_x_kin, self._pre_x_mask = None, None

        truth_obj = (
            self.truth_object_indeces if self.truth_object_indeces is not None else None
        )
        self._pre_z_kin, self._pre_z_mask = _build_kinematics_tensor(
            self.truth_kinematics[all_indices, ...],
            self.truth_pdgids[all_indices, ...],
            truth_obj[all_indices, ...] if truth_obj is not None else None,
            self.max_tracks,
            self.n_jets,
        )
        self._precomputed = True

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, index):
        return self.__getitems__([index])

    def __getitems__(self, indeces):
        """Return paired (x_kin, x_mask, z_kin, z_mask, weights, w1_obs)."""

        indeces = np.array(indeces).flatten()

        if getattr(self, "_precomputed", False):
            x_kin = self._pre_x_kin[indeces] if self._pre_x_kin is not None else None
            x_mask = self._pre_x_mask[indeces] if self._pre_x_mask is not None else None
            z_kin = self._pre_z_kin[indeces]
            z_mask = self._pre_z_mask[indeces]
            weights = self.weights[indeces, ...]
            w1_obs = self.w1_obs[indeces, ...]
            return x_kin, x_mask, z_kin, z_mask, weights, w1_obs

        # Reco (optional - None when running in inference / truth-only mode)
        if self.reco_kinematics is not None:
            reco_kin_slice = self.reco_kinematics[indeces, ...]
            reco_pdg_slice = self.reco_pdgids[indeces, ...]
            reco_obj_slice = (
                self.reco_object_indeces[indeces, ...]
                if self.reco_object_indeces is not None
                else None
            )
            x_kin, x_mask = _build_kinematics_tensor(
                reco_kin_slice,
                reco_pdg_slice,
                reco_obj_slice,
                self.max_tracks,
                self.n_jets,
            )
        else:
            x_kin, x_mask = None, None

        # Truth
        truth_kin_slice = self.truth_kinematics[indeces, ...]
        truth_pdg_slice = self.truth_pdgids[indeces, ...]
        truth_obj_slice = (
            self.truth_object_indeces[indeces, ...]
            if self.truth_object_indeces is not None
            else None
        )
        z_kin, z_mask = _build_kinematics_tensor(
            truth_kin_slice,
            truth_pdg_slice,
            truth_obj_slice,
            self.max_tracks,
            self.n_jets,
        )

        weights = self.weights[indeces, ...]
        w1_obs = self.w1_obs[indeces, ...]

        return x_kin, x_mask, z_kin, z_mask, weights, w1_obs
