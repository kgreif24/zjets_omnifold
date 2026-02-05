"""classify_top.py - Classify top events from pseudodata.

The classifier will use the full-phase space (tracks and muons) as opposed
to only using a few summary features as is done in the actual subtraction.

The classifier output will be used as an input to the subtraction classifiers.

Author: Kevin Greif
Last updated September 12, 2025
"""

import sys
import argparse
import copy
from dataclasses import dataclass
from typing import Tuple

import awkward as ak
import lightning as L
import numpy as np
import torch
import uproot
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, random_split

sys.path.append("..")
from lightning_module import LOfTransformer  # noqa: E402
from of_dataset import OfDataset  # noqa: E402
import utils.data_utils as du  # noqa: E402


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""

    input_dim: int = 10
    seed: int = 43
    pair_input_dim: int = 4
    embed_dims: list = None
    pair_embed_dims: list = None
    fc_nodes: list = None
    num_classes: int = 1
    num_heads: int = 8
    num_layers: int = 6
    num_cls_layers: int = 2
    fc_dropout: float = 0.0
    activation: str = "gelu"
    remove_self_pair: bool = False
    max_lr: float = 1e-5
    min_lr: float = 1e-6

    def __post_init__(self):
        if self.embed_dims is None:
            self.embed_dims = [128, 512, 128]
        if self.pair_embed_dims is None:
            self.pair_embed_dims = [64, 64, 64]
        if self.fc_nodes is None:
            self.fc_nodes = [256, 256]


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    batch_size: int = 256
    max_epochs: int = 70
    num_nodes: int = 1
    devices: int = 4
    test_split: float = 0.2
    patience: int = 20
    n_jets: int = 5
    max_tracks: int = 264


def load_and_filter_data(
    file_path: str, is_pseudodata: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and filter data from ROOT file.

    Args:
        file_path: Path to the ROOT file
        is_pseudodata: Whether this is pseudodata (affects filtering logic)

    Returns:
        Tuple of (kinematics, indices, observables)
    """
    f = uproot.open(file_path)
    t = f["OmniTree"]

    # Load pass190 and isTop flags
    pass190 = ak.to_numpy(t["pass190"].array())
    if is_pseudodata:
        is_top = ak.to_numpy(t["isTop"].array())
        pass190 = np.logical_and(pass190, ~is_top)

    rank_zero_info(f"{'PD' if is_pseudodata else 'Top'} events: {np.sum(pass190)}")
    rank_zero_info(f"Fraction of good events: {np.sum(pass190) / len(pass190):.3f}")

    # Load kinematics and observables, filtering on pass190 and isTop if pseudodata
    kinematics, indices, pdgids = du.get_kinematics(t, pass190, muon_only=False)
    observables = du.get_observables(t, ["Ntracks", "HT_tracks", "pT_ll"], pass190)

    return kinematics, indices, observables, pdgids


def create_datasets(
    pd_data: Tuple, top_data: Tuple, config: TrainingConfig
) -> Tuple[OfDataset, OfDataset]:
    """Create datasets for pseudodata and top events.

    Args:
        pd_data: Tuple of (kinematics, indices, observables) for pseudodata
        top_data: Tuple of (kinematics, indices, observables) for top events
        config: Training configuration

    Returns:
        Tuple of (pseudodata dataset, top dataset)
    """
    pd_kinematics, pd_indices, pd_obs, pd_pdgids = pd_data
    top_kinematics, top_indices, top_obs, top_pdgids = top_data

    # Create labels: top is 1, pseudodata is 0
    top_labels = np.ones((len(top_kinematics), 1), dtype=np.float32)
    pd_labels = np.zeros((len(pd_kinematics), 1), dtype=np.float32)

    # Calculate balanced weights
    total_events = len(top_kinematics) + len(pd_kinematics)
    nf_top = 2 * len(top_kinematics) / total_events
    nf_pd = 2 * len(pd_kinematics) / total_events

    weight_top = np.ones((len(top_kinematics), 1)) / nf_top
    weight_pd = np.ones((len(pd_kinematics), 1)) / nf_pd

    # Create datasets
    top_dataset = OfDataset(
        kinematics=top_kinematics,
        labels=top_labels,
        weights=weight_top,
        object_indeces=top_indices,
        w1_obs=top_obs,
        pdgids=top_pdgids,
        n_jets=config.n_jets,
        max_tracks=config.max_tracks,
    )

    pd_dataset = OfDataset(
        kinematics=pd_kinematics,
        labels=pd_labels,
        weights=weight_pd,
        object_indeces=pd_indices,
        w1_obs=pd_obs,
        pdgids=pd_pdgids,
        n_jets=config.n_jets,
        max_tracks=config.max_tracks,
    )

    return pd_dataset, top_dataset


def create_data_loaders(
    combined_dataset: OfDataset, config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders.

    Args:
        combined_dataset: Combined dataset to split
        config: Training configuration

    Returns:
        Tuple of (train_loader, test_loader)
    """
    total_size = len(combined_dataset)
    test_size = int(config.test_split * total_size)
    train_size = total_size - test_size

    rank_zero_info(f"Total dataset size: {total_size}")
    rank_zero_info(f"Training size: {train_size}")
    rank_zero_info(f"Testing size: {test_size}")

    # Split dataset
    train_dataset, test_dataset = random_split(
        combined_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    rank_zero_info(f"Train dataset length: {len(train_dataset)}")
    rank_zero_info(f"Test dataset length: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=du.null_collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=du.null_collate,
    )

    return train_loader, test_loader


def create_model(config: ModelConfig) -> LOfTransformer:
    """Create and return the transformer model.

    Args:
        config: Model configuration

    Returns:
        Configured transformer model
    """
    block_params = {
        "dropout": 0.02,
        "attn_dropout": 0.02,
        "activation_dropout": 0.0,
    }

    return LOfTransformer(
        input_dim=config.input_dim,
        seed=config.seed,
        no_w1=True,  # No Wasserstein One metric given we are not reweighting
        pair_input_dim=config.pair_input_dim,
        embed_dims=config.embed_dims,
        pair_embed_dims=config.pair_embed_dims,
        fc_nodes=config.fc_nodes,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_cls_layers=config.num_cls_layers,
        block_params=block_params,
        cls_block_params=block_params,
        fc_dropout=config.fc_dropout,
        activation=config.activation,
        remove_self_pair=config.remove_self_pair,
        max_lr=config.max_lr,
        min_lr=config.min_lr,
    )


def setup_trainer(run_name: str, config: TrainingConfig) -> L.Trainer:
    """Set up the PyTorch Lightning trainer.

    Args:
        run_name: Name for this training run
        config: Training configuration

    Returns:
        Configured trainer
    """
    store_location = f"../checkpoints/top-subtraction/classifier/{run_name}"

    # Set up callbacks
    checkpoints = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch:02d}-{step:06d}-{val_loss:.1f}",
        save_top_k=1,
        mode="min",
        dirpath=store_location,
    )

    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        mode="min",
    )

    # Set up logger
    logger = WandbLogger(
        project="top-subtraction",
        group="classifier",
        name=run_name,
        save_dir=store_location,
    )

    # Create trainer
    trainer = L.Trainer(
        accelerator="auto",
        num_nodes=config.num_nodes,
        devices=config.devices,
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[checkpoints, early_stopping],
        default_root_dir=store_location,
    )

    return trainer


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a neural network classifier for top events"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to pseudodata file"
    )
    parser.add_argument(
        "--top_path", type=str, required=True, help="Path to top MC file"
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Name for this training run"
    )
    args = parser.parse_args()

    # Load and filter data
    rank_zero_info("Loading pseudodata...")
    pd_data = load_and_filter_data(args.data_path, is_pseudodata=True)

    rank_zero_info("Loading top MC data...")
    top_data = load_and_filter_data(args.top_path, is_pseudodata=False)

    # Create datasets
    rank_zero_info("Creating datasets...")
    training_config = TrainingConfig()
    pd_dataset, top_dataset = create_datasets(pd_data, top_data, training_config)

    # Combine datasets
    combined_dataset = copy.deepcopy(top_dataset)
    combined_dataset.concatenate(pd_dataset)

    # Create data loaders
    rank_zero_info("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(combined_dataset, training_config)

    # Create model
    rank_zero_info("Creating model...")
    model_config = ModelConfig()
    model = create_model(model_config)

    # Set up trainer
    rank_zero_info("Setting up trainer...")
    trainer = setup_trainer(args.run_name, training_config)

    # Train model
    rank_zero_info("Starting training...")
    trainer.fit(model, train_loader, test_loader)

    rank_zero_info("Training completed!")


if __name__ == "__main__":
    main()
