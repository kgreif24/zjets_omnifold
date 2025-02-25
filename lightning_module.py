""" lightning_module.py - This file defines the LOfTransformer class.
This is a pytorch lightning module for training networks in Omnifold.

Author: Kevin Greif
Last updated 11/15/2024
python3
"""

import torch
import lightning as L
import torchmetrics
import wandb

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from pytorch_optimizer import Lion

from of_transformer.of_transformer import OfTransformer
from of_transformer.simple_network import DumbNeuralNetwork
from wasserstein_metric import WassersteinOne
import utils.plotting_utils as pu


class LOfTransformer(L.LightningModule):
    """LOfTransformer - This class is a wrapper for the Omnifold Transformer.
    It will initialize the model in the __init__ method. Any additional arguments
    that are passed to the __init__ method will be passed to the OfTransformer.

    For now, loss is hardcoded to the BCEWithLogitsLoss.
    """

    # Init function
    def __init__(
        self,
        input_dim=3,
        test_plots=None,
        debug=False,
        no_w1=False,
        seed=420,
        step=1,
        min_lr=1e-5,
        max_lr=1e-4,
        optimizer=None,
        weight_decay=0.01,
        cycle_steps=30000,
        warmup_steps=8000,
        gamma=0.85,
        **kwargs
    ):
        """__init__ - This method initializes the LOfTransformer class.
        There is one required argument which gives the input dimension for the
        transformer. This is the # of features per object (usually 3).
        Any other keyword arguments are passed to the OfTransformer init function,
        and saved as hyperparameters of the module.

        Arguments:
            input_dim {int} -- The input dimension of the model.
            test_plots {str} -- The path to the directory where testing plots will be
                stored for logging. None by default, in which case testing plots will
                not be drawn
            debug {bool} -- Set to true if we are running in debug mode, use simple
                network on muons only
            no_w1 {bool} -- Set to true if we want to disable the wasserstein metric
            seed {int} -- The random seed to use for the train / val split. Only used
                for logging
            step {int} -- Whether this training is for OF step one or two, only effects
                plot labeling
            min_lr {float} -- The minimum learning rate
            max_lr {float} -- The maximum learning rate
            weight_decay {float} -- The weight decay for the optimizer
            cycle_steps {int} -- The number of steps in a cycle
            warmup_steps {int} -- The number of steps in the warmup
            gamma {float} -- The gamma parameter for the learning rate scheduler
            **kwargs {dict} -- A dictionary of keyword arguments to be passed
                to the OfTransformer init function.
        """

        # Set instance vars
        self.debug = debug
        self.no_w1 = no_w1
        self.seed = seed
        self.step = step
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.optimizer = optimizer  # Just here for backwards comptability
        self.weight_decay = weight_decay
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        # Set plotting names based on step argument
        if step == 1:
            self.names = ("RecoMC", "RecoPD")
        elif step == 2:
            self.names = ("TruthMC", "PulledWeightsMC")

        # Set 32 bit precision for all operations
        torch.set_float32_matmul_precision("medium")

        # Initialize model and loss
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        if debug:
            self.model = DumbNeuralNetwork()
        else:
            self.model = OfTransformer(input_dim, **kwargs)

        # Performance metrics, note this also handles plotting and logging to wandb
        self.val_auc = torchmetrics.classification.AUROC(task="binary")
        self.test_auc = torchmetrics.classification.AUROC(task="binary")
        if not (self.debug or self.no_w1):
            self.wasserstein_val = WassersteinOne(pu.default_settings, draw_plots=False)
            self.draw_test = True if test_plots is not None else False
            self.wasserstein_test = WassersteinOne(
                pu.default_settings, draw_plots=self.draw_test, save_location=test_plots
            )

        # Log hyperparameters
        self.save_hyperparameters(ignore=["test_plots", "debug"])

    # Forward pass
    def forward(self, inputs, mask):
        tracks = inputs[:, :3, :]
        if self.debug:
            return self.model(tracks)
        else:
            return self.model(inputs, v=tracks, mask=mask)

    # Training step
    def training_step(self, batch, batch_idx):

        # Separate batch, make forward pass, calculate loss
        inputs, target, mask, start_weights, _ = batch
        output = self(inputs, mask)
        loss = self.criterion(output, target) * start_weights
        loss = loss.mean()

        # Log training loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):

        # Forward pass
        inputs, target, mask, start_weights, plotting = batch
        output = self(inputs, mask)

        # Calculate new weights
        network_weights = torch.exp(output)
        end_weights = network_weights * start_weights

        # Calculate and log loss
        loss = self.criterion(output, target) * start_weights
        loss = loss.mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log AUC, note the AUROC class auto-applies sigmoid to logits
        self.val_auc.update(output, target)

        # Update wasserstein metric
        if not (self.debug or self.no_w1):
            self.wasserstein_val.update(plotting, start_weights, end_weights, target)

    # Validation step end for logging reweighting plots to wandb
    def on_validation_epoch_end(self):

        # Just return if in debug mode or not update wasserstein metrics
        if self.debug or self.no_w1:
            return

        # Don't do anything but reset metric on validation sanity check
        if not self.trainer.sanity_checking:

            # Calculate AUROC and log
            val_auc = self.val_auc.compute()
            self.log(
                "val_auc",
                val_auc,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.val_auc.reset()

            # Calculate wasserstein metric and log
            if not (self.debug or self.no_w1):
                val_wass, _ = self.wasserstein_val.compute(names=self.names)
                self.log(
                    "val_wasserstein",
                    val_wass,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.wasserstein_val.reset()

    # Test step
    def test_step(self, batch, batch_idx):

        # Forward pass
        inputs, target, mask, start_weights, plotting = batch
        output = self(inputs, mask)

        # Calculate new weights
        network_weights = torch.exp(output)
        end_weights = network_weights * start_weights

        # Update AUC
        self.test_auc.update(output, target)

        # Update wasserstein metric
        if not (self.debug or self.no_w1):
            self.wasserstein_test.update(plotting, start_weights, end_weights, target)

    # Test epoch end for logging plots and metrics to wandb
    def on_test_epoch_end(self):

        # Just return if in debug mode or not using wasserstein metrics
        if self.debug or self.no_w1:
            return

        # Calculate and log AUC
        test_auc = self.test_auc.compute()
        self.log(
            "test_auc",
            test_auc,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            sync_dist=False,
        )
        self.test_auc.reset()

        # Calculate and log wasserstein metric
        if not (self.debug or self.no_w1):
            test_wass, plot_dict = self.wasserstein_test.compute(names=self.names)
            self.log(
                "test_wasserstein",
                test_wass,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )
            if self.draw_test and self.trainer.is_global_zero:
                for key, histpath in plot_dict.items():
                    log_name = "test_" + key
                    self.logger.experiment.log({log_name: wandb.Image(histpath)})
            self.wasserstein_test.reset()

    # Prediction step
    def predict_step(self, batch, batch_idx):
        inputs, _, mask, _, _ = batch
        return self(inputs, mask)

    # Configure optimizer
    def configure_optimizers(self):

        # Build and return optimizer and scheduler
        optimizer = Lion(
            self.model.parameters(), lr=self.max_lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.cycle_steps,
            warmup_steps=self.warmup_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            gamma=self.gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
