"""lightning_aussie_data_module.py - LightningDataModule for the AUSSIE
training procedure. Provides paired (reco, truth) MC events through an
OfPairedDataset. The source file is the MC training file; there is no
target (the step-1 classifier plays that role implicitly).

Nominal MC only in v1 - no systematic variations or bootstraps.

Author: Kevin Greif
python3
"""

import torch
import lightning as L
from pytorch_lightning.utilities.rank_zero import rank_zero_info

import numpy as np
import uproot
import awkward as ak

from of_dataset import OfPairedDataset
import utils.data_utils as du
import utils.subprocess_utils as su


class LAussieData(L.LightningDataModule):
    """LAussieData - A Lightning data module that yields paired reco/truth
    events for AUSSIE training. Only the MC training file is used; the
    classifier handles the data distribution implicitly.

    Single GPU only in v1 - no sharding, no data_divisor.
    """

    def __init__(
        self,
        mc_file,
        max_events=99999999,
        n_jets=5,
        max_tracks=None,
        muon_only=False,
        batch_size=128,
        dataloader_workers=0,
        split_seed=2,
        inference_mode=False,
    ):
        """__init__

        Arguments:
            mc_file (str) - Path to the MC ROOT file (OmniTree inside)
            max_events (int) - Cap on the number of events to read
            n_jets (int) - Max jets for one-hot encoding
            max_tracks (int) - Max tracks for padding
            muon_only (bool) - Use only muons in the kinematics (debug mode)
            batch_size (int) - Batch size for all dataloaders
            dataloader_workers (int) - Number of dataloader workers
            split_seed (int) - Seed for the 80/20 train/val split
            inference_mode (bool) - When True, apply only the truth-level
                selection (truth_pass190 == 1) instead of requiring events
                to pass both reco and truth cuts. Use this for evaluation /
                weight-writing so that efficiency corrections are included.
        """

        super().__init__()
        self.mc_file = mc_file
        self.max_events = max_events
        self.n_jets = n_jets
        self.max_tracks = max_tracks
        self.muon_only = muon_only
        self.batch_size = batch_size
        self.dataloader_workers = dataloader_workers
        self.split_seed = split_seed
        self.inference_mode = inference_mode

        # Load the tree and determine how many events we'll consume
        self.tree = uproot.open(self.mc_file)["OmniTree"]
        self.num_events = self.tree.num_entries
        if self.num_events > self.max_events:
            self.num_events = self.max_events
        rank_zero_info(f"Loading paired MC data from {self.mc_file}")

        reco_pass190 = ak.to_numpy(
            self.tree["pass190"].array(entry_stop=self.num_events)
        )
        truth_pass190 = ak.to_numpy(
            self.tree["truth_pass190"].array(entry_stop=self.num_events)
        )

        if self.inference_mode:
            # Inference: only require truth-level selection so that
            # efficiency corrections (pass truth, fail reco) are captured.
            self.pass190 = truth_pass190.astype(np.int64)
            n_good = int(np.sum(self.pass190))
            rank_zero_info(
                f"We have {self.num_events} total MC events and "
                f"{n_good} pass truth pass190 [inference mode] "
                f"(fraction {n_good / max(1, self.num_events):.4f})"
            )
        else:
            # Training: event must pass both reco and truth cuts.
            self.pass190 = (reco_pass190 & truth_pass190).astype(np.int64)
            n_good = int(np.sum(self.pass190))
            rank_zero_info(
                f"We have {self.num_events} total MC events and "
                f"{n_good} pass both reco and truth pass190 "
                f"(fraction {n_good / max(1, self.num_events):.4f})"
            )

        # Build the dataset once in init
        self._build_dataset()

    def _build_dataset(self):
        """_build_dataset - Load reco and truth kinematics under the same
        event filter, load the MC weights, and build the paired dataset.

        In inference_mode, reco kinematics are skipped entirely (they would
        contain -99 sentinels for events that fail the reco selection).
        predict_step only uses truth-level inputs, so the reco side is
        left as None in the dataset.
        """

        # Truth kinematics/indeces/pdgids are always required
        truth_kin, truth_idx, truth_pdg = du.get_kinematics(
            self.tree,
            evt_filter=self.pass190,
            muon_only=self.muon_only,
            get_truth=True,
            stop=self.num_events,
        )

        # Reco kinematics are only loaded during training; in inference mode
        # events may fail reco selection (giving -99 sentinels) so we skip them.
        if not self.inference_mode:
            reco_kin, reco_idx, reco_pdg = du.get_kinematics(
                self.tree,
                evt_filter=self.pass190,
                muon_only=self.muon_only,
                get_truth=False,
                stop=self.num_events,
            )
        else:
            reco_kin, reco_idx, reco_pdg = None, None, None

        # MC weights (use truth level weight branch, weight_mc)
        all_weights = ak.to_numpy(
            self.tree["weight_mc"].array(entry_stop=self.num_events)
        ).astype(np.float32)
        self.all_root_weights = all_weights.copy()
        weights = all_weights[self.pass190 == 1]

        # Normalize weights so that Sum(w) = N for numerical stability of
        # the MLC loss gradient (see AUSSIE paper eq. 12, 20).
        n_paired = len(weights)
        total = np.sum(weights)
        if total > 0:
            weights = weights * (n_paired / total)
        weights = np.expand_dims(weights, axis=1)

        # Reco-level observables for the Wasserstein validation metric.
        # Not used during inference, but we still load them (scalar branches,
        # no -99 sentinels) so the dataset shape stays consistent.
        w1_keys = du.get_w1_obs(get_truth=False)
        w1_observables = du.get_observables(
            self.tree,
            w1_keys,
            evt_filter=self.pass190,
            stop=self.num_events,
        )

        self.paired_dataset = OfPairedDataset(
            truth_kinematics=truth_kin,
            weights=weights,
            w1_obs=w1_observables,
            truth_pdgids=truth_pdg,
            reco_kinematics=reco_kin,
            reco_pdgids=reco_pdg,
            reco_object_indeces=reco_idx,
            truth_object_indeces=truth_idx,
            n_jets=self.n_jets,
            max_tracks=self.max_tracks,
        )

        # Pre-compute padded tensors for inference so that predict_dataloader
        # can run with num_workers=0 (pure tensor indexing) without being
        # dataloader-bound, and without the per-worker OOM risk from forking
        # the full awkward-array dataset.
        if self.inference_mode and self.max_tracks is not None:
            rank_zero_info("Pre-computing padded tensors for inference...")
            self.paired_dataset.precompute()
            rank_zero_info("Pre-computation done.")

    # Simple getters that callers (aussie_eval.py) may need
    def get_pass190(self):
        return self.pass190

    def get_root_weights(self):
        return self.all_root_weights

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.split_seed)
        train_dataset, _ = torch.utils.data.random_split(
            self.paired_dataset, [0.8, 0.2], generator=generator
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.RandomSampler(train_dataset, generator=generator),
            num_workers=self.dataloader_workers,
            collate_fn=du.null_collate,
            worker_init_fn=su.worker_init_fn,
        )

    def val_dataloader(self):
        generator = torch.Generator().manual_seed(self.split_seed)
        _, val_dataset = torch.utils.data.random_split(
            self.paired_dataset, [0.8, 0.2], generator=generator
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(val_dataset),
            num_workers=self.dataloader_workers,
            collate_fn=du.null_collate,
            worker_init_fn=su.worker_init_fn,
        )

    def predict_dataloader(self):
        """Full paired dataset (no split) in a fixed order for writing weights."""
        # When tensors are pre-computed, __getitems__ is pure tensor indexing
        # so num_workers=0 is fast and avoids worker-fork OOM.
        workers = (
            0
            if getattr(self.paired_dataset, "_precomputed", False)
            else self.dataloader_workers
        )
        return torch.utils.data.DataLoader(
            self.paired_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(self.paired_dataset),
            num_workers=workers,
            collate_fn=du.null_collate,
            worker_init_fn=su.worker_init_fn,
        )
