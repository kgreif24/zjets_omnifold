"""subtract_top.py - Subtract the top contribution to the fiducial
volume by training a classifier to distinguish between the top subtracted
pseudodata and the raw pseudodata.

Because the top contribution is very subtle, subtraction will only be differential
in 3 observables: Ntracks, HT_tracks, and the isTop logit.

Author: Kevin Greif
Last updated September 12, 2025
"""

import sys
import argparse
import uproot
import awkward as ak
import numpy as np
import torch
import lightning as L
from sklearn.model_selection import train_test_split

# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_info

sys.path.append("..")
from utils.data_utils import get_observables  # noqa: E402


class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim=3, droprate=0.0):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(droprate),
            torch.nn.Linear(512, 512),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(droprate),
            torch.nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class SubtractTop(L.LightningModule):
    def __init__(self, input_dim=1, droprate=0.0):
        super().__init__()
        self.model = SimpleNN(input_dim, droprate=droprate)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) * w
        loss = loss.mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) * w
        loss = loss.mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, w = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.4)


@rank_zero_only
def predict_weights(model, dataloader):
    with torch.no_grad():
        model.eval()
        preds = []
        for batch in dataloader:
            x = batch[0]
            y_hat = model(x.to(model.device))
            preds.append(y_hat)
        if not preds:
            # Handle empty dataloader case
            return np.array([])
        preds = np.concatenate([p.detach().cpu().numpy().flatten() for p in preds])
        weights = np.exp(preds)
    print(f"Mean of weights: {np.mean(weights)}")
    return weights


def bootstrap_sample(data, weights, bootstrap):
    raise NotImplementedError("Bootstrap sampling is not implemented yet")


def replace_exits(data):
    """replace_exits - Some of the observables used for training the subtraction
    networks contain exit codes (-99). This function replaces them with the median
    of the observable."""

    for i in range(data.shape[1]):
        non_exit_mask = data[:, i] != -99
        if np.any(non_exit_mask):
            # Replace exit codes with median of non-exit values
            median_val = np.median(data[non_exit_mask, i])
            data[data[:, i] == -99, i] = median_val
        else:
            # If all values are exit codes, replace with 0
            data[data[:, i] == -99, i] = 0.0
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subtract the top contribution to the fiducial volume"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file"
    )
    parser.add_argument(
        "--top_path", type=str, required=True, help="Path to the top MC file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        help="Enable bootstrapping (sample with replacement)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Seed for the train / test split",
    )
    return parser.parse_args()


# Main function
if __name__ == "__main__":

    args = parse_args()

    # Set the observables to use
    observables = [
        "isTop_logit",
        "HT_tracks",
        "Ntracks",
        "pT_trackj1",
        "m_trackj1",
    ]
    input_dim = len(observables)

    # Load pseudodata file and data
    f_pd = uproot.open(args.data_path)
    t_pd = f_pd["OmniTree"]
    pd_data = get_observables(t_pd, observables)
    pd_data = replace_exits(pd_data)
    pd_data_full = pd_data.copy()
    pd_weights = np.ones(len(pd_data))

    # Perform train / test split on pseudodata
    pd_data_train, pd_data_test, pd_weights_train, pd_weights_test = train_test_split(
        pd_data, pd_weights, test_size=0.2, random_state=args.split_seed
    )

    # Load top MC file and data
    f_top = uproot.open(args.top_path)
    t_top = f_top["OmniTree"]
    pass190_top = ak.to_numpy(t_top["pass190"].array())
    rank_zero_info(f"top events: {np.sum(pass190_top)}")
    rank_zero_info(
        "We have a fracion {} of good events in top".format(
            np.sum(pass190_top) / len(pass190_top)
        )
    )
    top_data = get_observables(t_top, observables)
    top_data = replace_exits(top_data)
    top_data_full = top_data.copy()
    top_weights = ak.to_numpy(t_top["weight"].array())
    top_weights = top_weights[pass190_top == 1]

    # Perform train / test split on top data
    top_splits = train_test_split(
        top_data, top_weights, test_size=0.2, random_state=args.split_seed
    )
    top_data_train, top_data_test, top_weights_train, top_weights_test = top_splits

    # Concatenate top data onto the pseudodata
    pd_minus_top_data_train = np.concatenate([pd_data_train, top_data_train], axis=0)
    pd_minus_top_weights_train = np.concatenate(
        [pd_weights_train, -1.0 * top_weights_train]
    )
    pd_minus_top_data_test = np.concatenate([pd_data_test, top_data_test], axis=0)
    pd_minus_top_weights_test = np.concatenate(
        [pd_weights_test, -1.0 * top_weights_test]
    )

    # Bootstrap sampling of training data if requested
    if args.bootstrap is not None:
        pd_data_train, pd_weights_train = bootstrap_sample(
            pd_data_train, pd_weights_train, args.bootstrap
        )
        top_data_train, top_weights_train = bootstrap_sample(
            top_data_train, top_weights_train, args.bootstrap
        )

    # Combine the data
    data_train = np.concatenate([pd_data_train, pd_minus_top_data_train], axis=0)
    data_test = np.concatenate([pd_data_test, pd_minus_top_data_test], axis=0)
    weights_train = np.expand_dims(
        np.concatenate([np.ones(len(pd_data_train)), pd_minus_top_weights_train]),
        axis=1,
    )
    weights_test = np.expand_dims(
        np.concatenate([np.ones(len(pd_data_test)), pd_minus_top_weights_test]), axis=1
    )
    labels_train = np.concatenate(
        [np.zeros((len(pd_data_train), 1)), np.ones((len(pd_minus_top_data_train), 1))],
        axis=0,
    )
    labels_test = np.concatenate(
        [np.zeros((len(pd_data_test), 1)), np.ones((len(pd_minus_top_data_test), 1))],
        axis=0,
    )
    rank_zero_info("data shapes:")
    rank_zero_info(f"data_train: {data_train.shape}")
    rank_zero_info(f"data_test: {data_test.shape}")
    rank_zero_info(f"labels_train: {labels_train.shape}")
    rank_zero_info(f"labels_test: {labels_test.shape}")
    rank_zero_info(f"weights_train: {weights_train.shape}")
    rank_zero_info(f"weights_test: {weights_test.shape}")

    # Normalize the data
    # Remember to apply the same normalization to the stand-alone pseudodata
    means = data_train.mean(axis=0)
    stds = data_train.std(axis=0)
    data_train = (data_train - means) / stds
    data_test = (data_test - means) / stds
    pd_data_full = (pd_data_full - means) / stds

    # Make pytorch datasets
    dataset_train = torch.utils.data.TensorDataset(
        torch.tensor(data_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.float32),
        torch.tensor(weights_train, dtype=torch.float32),
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.tensor(data_test, dtype=torch.float32),
        torch.tensor(labels_test, dtype=torch.float32),
        torch.tensor(weights_test, dtype=torch.float32),
    )

    # Make dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=256,
        shuffle=True,
        num_workers=16,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=256, shuffle=False
    )

    # Load the model
    top_model = SubtractTop(input_dim, droprate=0.0)

    # Train the model
    trainer = L.Trainer(
        accelerator="gpu",
        devices=4,
        max_epochs=30,
        logger=None,
        enable_progress_bar=False,
        # logger=WandbLogger(project="top-subtraction", group="sub-1x"),
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1, save_last=True
            )
        ],
    )
    trainer.fit(top_model, train_dataloader, val_dataloader)

    # Re-load the best model
    rank_zero_info(
        f"Loading best model from {trainer.checkpoint_callback.best_model_path}"
    )
    top_model = SubtractTop.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, input_dim=input_dim
    )

    # Predict and save the reweighting
    pd_dataset = torch.utils.data.TensorDataset(
        torch.tensor(pd_data_full, dtype=torch.float32),
    )
    pd_dataloader = torch.utils.data.DataLoader(
        pd_dataset, batch_size=256, shuffle=False
    )
    top_dataset = torch.utils.data.TensorDataset(
        torch.tensor(top_data_full, dtype=torch.float32),
    )
    top_dataloader = torch.utils.data.DataLoader(
        top_dataset, batch_size=256, shuffle=False
    )
    pd_weights = predict_weights(top_model, pd_dataloader)
    top_weights = predict_weights(top_model, top_dataloader)
    np.savez(args.output_file, pd_weights=pd_weights, top_weights=top_weights)

    rank_zero_info("All done!")
