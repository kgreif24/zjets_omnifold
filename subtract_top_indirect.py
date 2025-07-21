"""subtract_top.py - Subtract the top contribution to the fiducial
volume in 2 steps:

1. Train classifier to subtract 10x the top contribution
2. Train classifier to add back 9x the top contribution

Both of these classifiers will use a single floating point number as input.
This is the isTop logit from the original top classifier.
"""

import argparse
import uproot
import awkward as ak
import numpy as np
import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from of_transformer.simple_network import DumbNeuralNetwork
from utils.data_utils import get_observables


class SubtractTop(L.LightningModule):
    def __init__(self, input_dim=1):
        super().__init__()
        self.model = DumbNeuralNetwork(input_dim)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) * w
        loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) * w
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.2)


@rank_zero_only
def predict_and_save(model, dataloader, output_file, labels):
    preds = []
    for batch in dataloader:
        x, y, w = batch
        y_hat = model(x.to(model.device))
        preds.append(y_hat)
    preds = np.concatenate([p.detach().cpu().numpy().flatten() for p in preds])
    source_preds = np.exp(preds[labels.flatten() == 0])
    print(f"Mean of weights in {output_file}: {np.mean(source_preds)}")
    np.savez(output_file, reweighting=source_preds)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--add9", action="store_true", help="Use 9x addition instead of 10x"
)
args = parser.parse_args()


# Set the observables to use
observables = ["isTop_logit", "HT_tracks", "Ntracks"]
input_dim = len(observables)

# Load pseudodata file
f_pd = uproot.open(
    "/pscratch/sd/k/kgreif/data/"
    "Pseudodata_WithSherapaNoReweighting_12May2025_topLogit.root"
)
t_pd = f_pd["OmniTree"]

# Load pass 190 and isTop logit
pd_data = get_observables(t_pd, observables)

# Load top MC file
f_top = uproot.open(
    "/pscratch/sd/k/kgreif/data/ZjetOmnifold_14May2025_Background"
    "_Sherpa2212_AllTop_WithTracks_slim_Systematics_topLogit.root"
)
t_top = f_top["OmniTree"]

# Load top MC data
pass190_top = ak.to_numpy(t_top["pass190"].array())
print(f"top events: {np.sum(pass190_top)}")
print(
    "We have a fracion {} of good events in top".format(
        np.sum(pass190_top) / len(pass190_top)
    )
)
top_data = get_observables(t_top, ["isTop_logit", "HT_tracks", "Ntracks"])
top_weights = ak.to_numpy(t_top["weight"].array())
top_weights = top_weights[pass190_top == 1]
if args.add9:
    target_data = top_data
    target_weights = 9.0 * top_weights
else:
    target_data = np.concatenate([pd_data, top_data], axis=0)
    target_weights = np.concatenate([np.ones(len(pd_data)), -10.0 * top_weights])

# Combine the data
data = np.concatenate([pd_data, target_data], axis=0)
labels = np.expand_dims(
    np.concatenate([np.zeros(len(pd_data)), np.ones(len(target_data))]), axis=1
)
weights = np.expand_dims(
    np.concatenate([np.ones(len(pd_data)), target_weights]), axis=1
)
print("data shapes:")
print(f"data: {data.shape}")
print(f"labels: {labels.shape}")
print(f"weights: {weights.shape}")

# Normalize the data
data = (data - data.mean(axis=0)) / data.std(axis=0)

# Make pytorch dataset and dataloader
dataset = torch.utils.data.TensorDataset(
    torch.tensor(data, dtype=torch.float32),
    torch.tensor(labels, dtype=torch.float32),
    torch.tensor(weights, dtype=torch.float32),
)
all_dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=16,
)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

# Load the model
top_model = SubtractTop(input_dim)

# Train the model
group = "add-9x" if args.add9 else "sub-10x"
trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    max_epochs=15,
    logger=WandbLogger(project="top-subtraction", group=group),
    callbacks=[
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, save_last=True
        )
    ],
)
trainer.fit(top_model, train_dataloader, val_dataloader)

# Re-load the best models
print(f"Loading best model from {trainer.checkpoint_callback.best_model_path}")
top_model_10sub = SubtractTop.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path, input_dim=input_dim
)
top_model_9add = SubtractTop.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path, input_dim=input_dim
)

# Predict and save the reweighting
out_name = "reweighting_add9.npz" if args.add9 else "reweighting_sub10.npz"
predict_and_save(top_model, all_dataloader, out_name, labels)

print("All done!")
