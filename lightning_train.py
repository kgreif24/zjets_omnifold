#!/usr/bin/env python

import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np

from lightning_module import *
from plotting_utils import *


# Define a single flag for entering debug mode (simple network, muons only)
debug = False

# Make a random integer between 0 and 1000 for the seed
rng = np.random.default_rng()
seed = int(rng.integers(0, 1000))

# Build data module
d_module = LOfData(
    mc_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train.root',
    data_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Dec15.root',
    muon_only=debug,
    batch_size=256,
    dataloader_workers=1,
    seed=seed,
    testing=False
)

# Login to wandb
wandb.login()

# Initialise the wandb logger and callbacks
if not debug:
    wandb_logger = WandbLogger(project='test-of-project', name='all_tracks_8', save_dir='./checkpoints')
else:
    wandb_logger = None

lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
checkpoints = L.pytorch.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filename='{epoch}-{val_loss:.4f}',
    save_top_k=3,
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
    devices=4,
    logger=wandb_logger,
    reload_dataloaders_every_n_epochs=1,
    callbacks=[lr_monitor, checkpoints],
    max_epochs=70,
    log_every_n_steps=50,
    enable_progress_bar=False
)

# Build lightning module
l_module = LOfTransformer(
    5,
    debug=debug,
    num_classes=1,
    trim=False,
    embed_dims=[128, 256, 128],
    fc_params=[(256, 0.0)],
    pair_embed_dims=None,
    cls_block_params={'dropout': 0.0, 'attn_dropout': 0.0, 'activation_dropout': 0.0, 'num_heads': 8},
    num_cls_layers=2,
    block_params={'dropout': 0.05, 'attn_dropout': 0.05, 'activation_dropout': 0.05, 'num_heads': 8},
    num_layers=5
)

# Run training
trainer.fit(l_module, d_module)

# Close W&B
wandb.finish()


