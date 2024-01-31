#!/usr/bin/env python

import lightning as L
from lightning_module import *
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from plotting_utils import *



d_module = LOfData(
    mc_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train.root',
    data_file='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Dec15.root',
    muon_only=False,
    batch_size=256,
    dataloader_workers=31
)


# Login to wandb
wandb.login()

# Initialise the wandb logger and callbacks for logging
wandb_logger = WandbLogger(project='test-of-project', name='all_tracks_2', save_dir='./checkpoints')
lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
checkpoints = L.pytorch.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filename='{epoch}-{val_loss:.4f}',
    save_top_k=3,
    mode='min'
)

trainer = L.Trainer(
    accelerator='gpu',
    devices=4,
    logger=wandb_logger,
    callbacks=[lr_monitor, checkpoints],
    max_epochs=50,
    log_every_n_steps=50
)

l_module = LOfTransformer(
    5,
    num_classes=1,
    trim=False,
    embed_dims=[128, 128, 128],
    fc_params=[(256, 0.0)],
    pair_embed_dims=None,
    cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0, 'num_heads': 32},
    num_cls_layers=1,
    block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0, 'num_heads': 32},
    num_layers=3
)

trainer.fit(l_module, d_module)

wandb.finish()



# # Load best checkpoint and run evaluation over test set
# model = LOfTransformer.load_from_checkpoint(
#     checkpoints.best_model_path
#     # './checkpoints/test-of-project/4kh4hur8/checkpoints/epoch=1-val_loss=0.6975.ckpt'
# )
# evaluator = L.Trainer(
#     accelerator='gpu',
#     devices=1
# )
# results = evaluator.predict(model, datamodule=d_module, return_predictions=True)


