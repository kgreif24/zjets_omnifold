#!/usr/bin/env python

import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np
import argparse

from lightning_module import *
from plotting_utils import *


# Parse command line args
parser = argparse.ArgumentParser(description='Train a transformer model on Z+jets data')
parser.add_argument('--seed', type=int, default=420, help='Seed for the train / validation split')
parser.add_argument('--debug', action='store_true', help='Run in debug mode (single device, muons only)')
args = parser.parse_args()

# Define a single flag for entering debug mode (simple network, muons only)
if args.debug:
    n_devices = 4
else:
    n_devices = 4

# Build data module
d_module = LOfData(
    mc_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train.root',
    data_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Dec15.root',
    muon_only=args.debug,
    batch_size=256,
    dataloader_workers=1,
    split_seed=args.seed,
    testing=False,
    max_tracks=150
)

# Login to wandb
wandb.login()

# Initialise the wandb logger and callbacks
wandb_logger = WandbLogger(project='test-of-project', name='track_wass_1', save_dir='./checkpoints')


lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
checkpoints = L.pytorch.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filename='{epoch}-{val_loss:.4f}',
    save_top_k=-1,
    mode='min'
)
early_stopping = L.pytorch.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    mode='min'
)

# Build trainer
trainer = L.Trainer(
    accelerator='gpu',
    devices=n_devices,
    logger=wandb_logger,
    callbacks=[lr_monitor, checkpoints, early_stopping],
    max_epochs=70,
    log_every_n_steps=50,
    enable_progress_bar=False
)

# Build lightning module
l_module = LOfTransformer(
    10,
    debug=args.debug,
    num_classes=1,
    trim=False,
    remove_self_pair=True,
    embed_dims=[256, 256, 128],
    pair_input_dim=4,
    pair_extra_dim=0,
    pair_embed_dims=[32, 64, 128],
    fc_params=[(256, 0.05), (256, 0.05)],
    cls_block_params={'dropout': 0.05, 'attn_dropout': 0.05, 'activation_dropout': 0.05, 'num_heads': 8},
    num_cls_layers=3,
    block_params={'dropout': 0.1, 'attn_dropout': 0.1, 'activation_dropout': 0.1, 'num_heads': 8},
    num_layers=6,
    # Include the seed just so it is logged to W&B
    seed=args.seed
)

# Run training
trainer.fit(l_module, d_module)

# Close W&B
wandb.finish()


