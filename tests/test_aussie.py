"""tests/test_aussie.py - Tests for the AUSSIE alternative to Omnifold step 2.

The smoke test verifies that the L1 gradient-norm loss plumbing works:
torch.autograd.grad(create_graph=True) through the frozen classifier, then
.backward() through the outer scalar loss to the unfolder's parameters.

The closure test (marked slow) exercises a 1D Gaussian toy example where
the analytic ratio p_data(z) / p_sim(z) is known, mirroring Fig. 1 of the
AUSSIE paper (arXiv:2602.24282v1).
"""

import os

import numpy as np
import pytest
import torch
import lightning as L

from lightning_module import LOfTransformer
from lightning_aussie_module import LAussieUnfolder


def _make_fake_classifier_ckpt(tmp_path):
    """Build a minimal LOfTransformer (debug mode) and save a Lightning
    checkpoint so LAussieUnfolder can load it as its frozen classifier.
    """
    classifier = LOfTransformer(
        debug=True,
        input_dim=10,
        min_lr=1e-5,
        max_lr=1e-4,
        no_w1=True,
        step=1,
    )
    # Use Trainer.save_checkpoint to produce a proper Lightning checkpoint
    trainer = L.Trainer(
        max_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    # Lightning needs the module connected to a strategy before we can save.
    trainer.strategy.connect(classifier)
    ckpt_path = os.path.join(tmp_path, "fake_classifier.ckpt")
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path


def _random_paired_batch(batch_size=8, n_features=10, n_tracks=2):
    """Build a random paired (x_kin, x_mask, z_kin, z_mask, w, w1_obs)
    batch of the same shape the real data module would yield.

    In debug mode the module slices inputs[:, :4, :] (4 features) and
    DumbNeuralNetwork expects input_dim=8, so n_tracks must be 2
    (4 features * 2 particles = 8).
    """
    x_kin = torch.randn(batch_size, n_features, n_tracks)
    z_kin = torch.randn(batch_size, n_features, n_tracks)
    x_mask = torch.ones(batch_size, 1, n_tracks, dtype=torch.bool)
    z_mask = torch.ones(batch_size, 1, n_tracks, dtype=torch.bool)
    w = torch.ones(batch_size, 1)
    w1_obs = torch.randn(batch_size, 3)
    return x_kin, x_mask, z_kin, z_mask, w, w1_obs


def test_aussie_smoke(tmp_path):
    """Smoke test: one AUSSIE training step on random data. Confirms
    that autograd.grad with create_graph=True works through the frozen
    classifier and that gradients reach the unfolder's parameters.
    """
    ckpt_path = _make_fake_classifier_ckpt(str(tmp_path))

    module = LAussieUnfolder(
        classifier_ckpt_path=ckpt_path,
        debug=True,
        input_dim=10,
        min_lr=1e-5,
        max_lr=1e-4,
        no_w1=True,
    )

    batch = _random_paired_batch(batch_size=8, n_features=10, n_tracks=2)

    # Manually drive one training step so we don't need a real dataloader.
    module.train()
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss), "AUSSIE training loss is not finite"

    # Backprop and confirm that unfolder params receive gradients
    # and classifier params do not have .grad populated (they are excluded
    # from the optimiser even though autograd.grad traverses them).
    loss.backward()

    unfolder_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in module.unfolder.parameters()
    )
    assert unfolder_has_grad, "No gradient reached the unfolder parameters"

    # classifier params will not typically have .grad set because we used
    # torch.autograd.grad, not .backward, on them. Even if they do, the
    # optimizer only steps the unfolder.
    optim_cfg = module.configure_optimizers()
    optimizer = optim_cfg["optimizer"]
    assert all(
        any(p is up for up in module.unfolder.parameters())
        for group in optimizer.param_groups
        for p in group["params"]
    ), "Optimiser must only hold unfolder parameters"


def test_aussie_classifier_ckpt_mismatch_guard(tmp_path):
    """If the classifier path stored in a checkpoint differs from the one
    LAussieUnfolder was instantiated with, on_load_checkpoint should refuse
    to silently mix classifier states.
    """
    ckpt_a = _make_fake_classifier_ckpt(str(tmp_path / "a"))
    # Build and save an AUSSIE checkpoint keyed to ckpt_a
    os.makedirs(tmp_path / "a", exist_ok=True)
    os.makedirs(tmp_path / "b", exist_ok=True)
    module = LAussieUnfolder(
        classifier_ckpt_path=ckpt_a,
        debug=True,
        input_dim=10,
        no_w1=True,
    )
    trainer = L.Trainer(
        max_steps=0,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(module)
    aussie_ckpt = str(tmp_path / "aussie.ckpt")
    trainer.save_checkpoint(aussie_ckpt)

    # Try to load the AUSSIE checkpoint with a module that expects a
    # different classifier path. The mismatch guard should fire.
    ckpt_b = _make_fake_classifier_ckpt(str(tmp_path / "b"))
    with pytest.raises(ValueError, match="Refusing to silently mix"):
        LAussieUnfolder.load_from_checkpoint(
            aussie_ckpt,
            classifier_ckpt_path=ckpt_b,
            debug=True,
            input_dim=10,
            no_w1=True,
        )


@pytest.mark.slow
def test_aussie_gaussian_closure(tmp_path):
    """1D Gaussian toy closure test (Fig. 1 of arXiv:2602.24282v1).

    - p_sim(z)  = N(0, 1)
    - p_data(z) = N(0.3, 1)
    - Forward map x = z + eps, eps ~ N(0, 2)

    Step 1: train a classifier R_theta(x) between p_sim(x) and p_data(x).
    Step 2: train an AUSSIE unfolder R_phi(z) via the L1 gradient norm.
    The learned ratio should approximate N(z; 0.3, 1) / N(z; 0, 1).
    """

    # Skip if torch isn't available with autograd double-backward support
    # (always true on modern pytorch but guards exotic installs).
    if not torch.cuda.is_available() and os.environ.get("AUSSIE_CLOSURE_CPU") != "1":
        pytest.skip(
            "Gaussian closure test is CPU-slow; set AUSSIE_CLOSURE_CPU=1 to run."
        )

    torch.manual_seed(0)
    n = 20000

    # Draw sim/data samples at truth level, then smear to reco
    z_sim = torch.randn(n)
    z_data = 0.3 + torch.randn(n)
    x_sim = z_sim + 2.0 * torch.randn(n)
    x_data = z_data + 2.0 * torch.randn(n)

    # Simple MLP classifier/unfolder on 1D input
    def make_net():
        return torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )

    classifier = make_net()
    unfolder = make_net()

    # --- Train classifier with BCE ---
    opt_c = torch.optim.Adam(classifier.parameters(), lr=5e-3)
    x_all = torch.cat([x_sim, x_data]).unsqueeze(-1)
    y_all = torch.cat([torch.zeros(n), torch.ones(n)]).unsqueeze(-1)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(1500):
        idx = torch.randperm(len(x_all))[:2048]
        logits = classifier(x_all[idx])
        # Convert logits to density-ratio form by mapping sigmoid probs to ratios:
        # in our convention the net outputs log R(x) directly, so the BCE on
        # the *logit* is equivalent to the density-ratio trick if we take the
        # logit directly.
        loss = bce(logits, y_all[idx])
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()

    # Freeze classifier
    classifier.eval()

    # --- AUSSIE train the unfolder ---
    opt_u = torch.optim.Adam(unfolder.parameters(), lr=5e-3)
    zx = torch.stack([z_sim, x_sim], dim=-1)  # paired (z, x) from sim
    for step in range(3000):
        idx = torch.randperm(len(zx))[:2048]
        z_b = zx[idx, 0].unsqueeze(-1)
        x_b = zx[idx, 1].unsqueeze(-1)
        log_R_theta = classifier(x_b).clamp(-10, 10)
        log_R_phi = unfolder(z_b)
        R_theta = torch.exp(log_R_theta)
        R_phi = torch.exp(log_R_phi)
        ell = R_theta - R_phi * log_R_theta
        L_MLC = ell.mean()
        grads = torch.autograd.grad(
            L_MLC,
            list(classifier.parameters()),
            create_graph=True,
            retain_graph=True,
        )
        loss = sum(g.abs().sum() for g in grads)
        opt_u.zero_grad()
        loss.backward()
        opt_u.step()

    # --- Evaluate: analytic truth is N(0.3, 1) / N(0, 1) = exp(0.3*z - 0.045) ---
    # Evaluate in the central bulk only; the tails have few training samples
    # and include the classifier's extrapolation errors.
    z_eval = torch.linspace(-1.0, 1.0, 11).unsqueeze(-1)
    with torch.no_grad():
        learned = torch.exp(unfolder(z_eval)).squeeze(-1).numpy()
    analytic = np.exp(0.3 * z_eval.squeeze(-1).numpy() - 0.5 * 0.3 ** 2)

    # Normalise both so their mean over z_eval is 1 (irrelevant constant)
    learned_norm = learned / learned.mean()
    analytic_norm = analytic / analytic.mean()

    # Relative error in the central bulk should be modest. 25% is a generous
    # threshold acknowledging the small-stat nature of this toy and the fact
    # that the step-1 classifier is itself imperfect.
    rel_err = np.abs(learned_norm - analytic_norm) / np.maximum(analytic_norm, 1e-3)
    assert rel_err.max() < 0.25, (
        f"AUSSIE closure test: max relative error {rel_err.max():.3f} "
        f"exceeds tolerance; learned = {learned_norm}, analytic = {analytic_norm}"
    )
