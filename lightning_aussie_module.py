"""lightning_aussie_module.py - Lightning module for the AUSSIE second step.

AUSSIE replaces the iterative Omnifold procedure with a single step that
minimises the L1 norm of the gradient of an MLC-like loss with respect to
the frozen step-1 classifier parameters (arXiv:2602.24282, eq. 20):

    L_MLC = E_{(x, z) ~ p_sim} [ R_theta(x) - R_phi(z) * log R_theta(x) ]
    L_AutoDiff[R_phi] = sum_i | d L_MLC / d theta_i |

The classifier R_theta is loaded from a completed Omnifold step-1 checkpoint
and held frozen. Only R_phi's parameters are optimised.

Author: Kevin Greif
python3
"""

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import lightning as L
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_info

from lightning_module import LOfTransformer
from of_transformer.of_transformer import OfTransformer
from of_transformer.simple_network import DumbNeuralNetwork
from wasserstein_metric import WassersteinOne


class LAussieUnfolder(L.LightningModule):
    """LAussieUnfolder - A Lightning module that trains an unfolder network
    R_phi(z) against a frozen classifier R_theta(x) via the AUSSIE L1
    gradient-norm objective.
    """

    def __init__(
        self,
        classifier_ckpt_path,
        input_dim=3,
        debug=False,
        no_w1=False,
        seed=420,
        run_id=None,
        weight_decay=0.01,
        min_lr=1e-7,
        max_lr=1e-5,
        warmup_steps=500,
        cos_steps=3000,
        linear_steps=15000,
        logit_clamp=10.0,
        **kwargs,
    ):
        """__init__ - Build the unfolder and load the frozen classifier.

        Arguments:
            classifier_ckpt_path (str) - Path to a completed Omnifold step-1
                LOfTransformer checkpoint. Its underlying model will be loaded
                and used as the frozen classifier R_theta.
            input_dim (int) - Input dimension passed to the unfolder network
            debug (bool) - Use DumbNeuralNetwork in place of OfTransformer
            no_w1 (bool) - Skip the Wasserstein validation metric
            seed (int) - Train/val split seed, stored for logging
            run_id (str) - W&B run id, stored for logging
            weight_decay (float) - Weight decay for the Lion optimiser
            min_lr (float) - Cosine floor learning rate
            max_lr (float) - Cosine peak learning rate
            warmup_steps (int) - Linear warmup steps before cosine
            cos_steps (int) - Steps of cosine annealing
            linear_steps (int) - Steps of linear cooldown after cosine
            logit_clamp (float) - Clamp on log R_theta(x) for numerical
                stability. Classifier tails can be extreme on unseen events.
            **kwargs - Passed to the underlying OfTransformer
        """

        self.classifier_ckpt_path = classifier_ckpt_path
        self.debug = debug
        self.no_w1 = no_w1
        self.seed = seed
        self.run_id = run_id
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.cos_steps = cos_steps
        self.linear_steps = linear_steps
        self.logit_clamp = logit_clamp

        torch.set_float32_matmul_precision("medium")

        for unused in ("optimizer", "cycle_steps", "gamma", "trim", "test_plots"):
            if unused in kwargs:
                kwargs.pop(unused)

        super().__init__()

        # Plot-label naming for the Wasserstein metric
        self.names = ("TruthMC_unfoldedReco", "TruthMC_step1Reco")

        # --- Build the unfolder (trainable) ---
        if debug:
            self.unfolder = DumbNeuralNetwork(input_dim=8)
        else:
            self.unfolder = OfTransformer(input_dim, **kwargs)

        # --- Build the classifier (frozen) ---
        # We load the full LOfTransformer to guarantee the saved hyperparameters
        # rebuild the exact architecture, then extract the underlying network.
        rank_zero_info(
            f"Loading frozen AUSSIE classifier from {classifier_ckpt_path}"
        )
        l_classifier = LOfTransformer.load_from_checkpoint(
            classifier_ckpt_path,
            debug=debug,
            step=1,
            map_location="cpu",
        )
        self.classifier = l_classifier.model
        self.classifier.eval()
        # Keep requires_grad=True so torch.autograd.grad can traverse the
        # classifier, but we will exclude these params from the optimiser.

        if not self.no_w1:
            self.wasserstein_val = WassersteinOne()
            self.wasserstein_test = WassersteinOne()

        # Save hyperparameters (classifier_ckpt_path included so checkpoint
        # round-trips correctly; run_id/debug excluded per LOfTransformer
        # convention).
        self.save_hyperparameters(ignore=["run_id", "debug"])

    # ---- Checkpoint hooks ----
    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        checkpoint["seed"] = self.seed
        checkpoint["run_id"] = self.run_id
        checkpoint["classifier_ckpt_path"] = self.classifier_ckpt_path

    @rank_zero_only
    def on_load_checkpoint(self, checkpoint):
        self.seed = checkpoint.get("seed", 222)
        self.run_id = checkpoint.get("run_id", "no_run_id")
        saved_cls = checkpoint.get("classifier_ckpt_path", None)
        if saved_cls is not None and saved_cls != self.classifier_ckpt_path:
            raise ValueError(
                "Checkpoint was created with classifier "
                f"{saved_cls} but LAussieUnfolder was instantiated with "
                f"{self.classifier_ckpt_path}. Refusing to silently mix "
                "classifier states."
            )
        # Keep the classifier in eval mode after loading
        self.classifier.eval()

    @rank_zero_only
    def reset_seed(self, seed):
        self.seed = seed
        self.save_hyperparameters({"seed": seed})

    @rank_zero_only
    def reset_run_id(self, run_id):
        self.run_id = run_id
        self.save_hyperparameters({"run_id": run_id})

    # Keep the classifier in eval mode even after Lightning flips the module
    # into train() at the start of each epoch.
    def train(self, mode=True):
        super().train(mode)
        self.classifier.eval()
        return self

    # ---- Forward helpers ----
    def _forward_net(self, net, inputs, mask):
        """Mirror LOfTransformer.forward - the underlying OfTransformer
        expects (x, v=tracks, mask) whereas DumbNeuralNetwork takes only
        the sliced tracks tensor.
        """
        tracks = inputs[:, :4, :]
        if self.debug:
            return net(tracks)
        return net(inputs, v=tracks, mask=mask)

    def forward(self, z_kin, z_mask):
        """Forward pass used by predict_step - returns log R_phi(z)."""
        return self._forward_net(self.unfolder, z_kin, z_mask)

    # ---- Training step: the AUSSIE L1 gradient-norm loss ----
    def training_step(self, batch, batch_idx):
        x_kin, x_mask, z_kin, z_mask, w, _ = batch

        # Force the math SDPA backend for the classifier's forward. The
        # efficient/flash backends have no double-backward kernel, so
        # autograd.grad(..., create_graph=True) below would fail with
        # "derivative for aten::_scaled_dot_product_..._backward is not
        # implemented" once .backward() is called on the outer loss.
        # Only the classifier forward needs this; the unfolder uses
        # whatever backend PyTorch picks by default.
        with sdpa_kernel([SDPBackend.MATH]):
            log_R_theta = self._forward_net(self.classifier, x_kin, x_mask)
        log_R_theta = torch.clamp(log_R_theta, -self.logit_clamp, self.logit_clamp)
        log_R_phi = self._forward_net(self.unfolder, z_kin, z_mask)

        R_theta = torch.exp(log_R_theta)
        R_phi = torch.exp(log_R_phi)

        # ell_i = R_theta(x_i) - R_phi(z_i) * log R_theta(x_i)
        ell = R_theta - R_phi * log_R_theta

        # Weighted mean over the batch
        wsum = w.sum().clamp_min(1e-12)
        L_MLC = (w * ell).sum() / wsum

        # Gradient of L_MLC w.r.t. the frozen classifier parameters.
        # create_graph=True so the outer .backward() can differentiate
        # through these gradients to update the unfolder.
        classifier_params = list(self.classifier.parameters())
        grads = torch.autograd.grad(
            L_MLC,
            classifier_params,
            create_graph=True,
            retain_graph=True,
        )

        # L1 norm over all classifier-gradient components
        loss = sum(g.abs().sum() for g in grads)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_L_MLC", L_MLC.detach(), prog_bar=False, sync_dist=True)
        return loss

    # ---- Validation: reco-level Wasserstein between AUSSIE-folded weights
    # and the step-1 pseudodata target. ----
    def validation_step(self, batch, batch_idx):
        x_kin, x_mask, z_kin, z_mask, w, plotting = batch

        # Both nets evaluated without grads - we only need the scalar weights
        with torch.no_grad():
            log_R_theta = self._forward_net(self.classifier, x_kin, x_mask)
            log_R_theta = torch.clamp(
                log_R_theta, -self.logit_clamp, self.logit_clamp
            )
            log_R_phi = self._forward_net(self.unfolder, z_kin, z_mask)

            R_theta = torch.exp(log_R_theta)
            R_phi = torch.exp(log_R_phi)

            # Per-event reco-level weights for the two distributions we
            # want to compare:
            #   (i)  MC reco weighted by R_phi(z)*w - AUSSIE forward folded
            #   (ii) MC reco weighted by R_theta(x)*w - step-1 target
            unfolded_weights = (R_phi * w).flatten()
            step1_weights = (R_theta * w).flatten()

            ell = R_theta - R_phi * log_R_theta
            wsum = w.sum().clamp_min(1e-12)
            L_MLC = (w * ell).sum() / wsum

        self.log("val_L_MLC", L_MLC, on_epoch=True, prog_bar=False, sync_dist=True)

        if not self.no_w1:
            # Build a single labelled set for WassersteinOne: label 0 = AUSSIE,
            # label 1 = step-1 target. Observables come from the reco-level
            # w1_obs in the batch.
            B = plotting.shape[0]
            device = plotting.device
            zeros = torch.zeros(B, 1, device=device)
            ones = torch.ones(B, 1, device=device)
            obs_concat = torch.cat([plotting, plotting], dim=0)
            wts_concat = torch.cat(
                [unfolded_weights.unsqueeze(1), step1_weights.unsqueeze(1)], dim=0
            )
            lbl_concat = torch.cat([zeros, ones], dim=0)
            self.wasserstein_val.update(obs_concat, wts_concat, lbl_concat)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        if not self.no_w1:
            val_wass = self.wasserstein_val.compute()
            self.log(
                "val_wasserstein",
                val_wass,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.wasserstein_val.reset()

    # ---- Predict step: return log R_phi(z) on every paired event ----
    def predict_step(self, batch, batch_idx):
        _, _, z_kin, z_mask, _, _ = batch
        return self._forward_net(self.unfolder, z_kin, z_mask)

    # ---- Optimiser & scheduler ----
    def configure_optimizers(self):
        # Only the unfolder's parameters are updated
        optimizer = torch.optim.AdamW(
            self.unfolder.parameters(),
            lr=self.max_lr,
            weight_decay=self.weight_decay,
        )
        lin_start_lr_factor = self.min_lr / self.max_lr
        lin_end_lr_factor = lin_start_lr_factor * 0.8
        if self.warmup_steps > 0:
            warmup_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=lin_start_lr_factor,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            wu_milestone = self.warmup_steps
        else:
            warmup_schedule = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=1
            )
            wu_milestone = 1
        cos_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cos_steps,
            eta_min=self.min_lr,
        )
        linear_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=lin_start_lr_factor,
            end_factor=lin_end_lr_factor,
            total_iters=self.linear_steps,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_schedule, cos_schedule, linear_schedule],
            milestones=[wu_milestone, self.cos_steps + wu_milestone],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
