import random

import lpips
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from framework.D_Net import D_Net
from framework.hrnet import HRNet


def tensor_normalization(data: torch.Tensor) -> torch.Tensor:
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / (_range + 1e-6)


class LinearDecayPositionEmbedding(nn.Module):
    """Sinusoidal position embedding with linearly decaying frequencies.

    Maps a scalar RGT volume ``(B, 1, T, H, W)`` to a multi-channel
    embedding ``(B, C, T, H, W)`` where each channel uses one frequency
    from ``frequencies``. Even-indexed channels use ``sin``, odd-indexed
    channels use ``cos``.
    """

    def __init__(self, frequencies=(2.0, 1.0, 0.5), discret: int = 256):
        super().__init__()
        self.frequencies = list(frequencies)
        self.discret = discret
        self.num_channels = len(self.frequencies)

    def forward(self, rgt_pos: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = rgt_pos.shape
        device = rgt_pos.device

        normalized_pos = (rgt_pos + 1) / 2
        scaled_pos = normalized_pos * (self.discret - 1) + 1
        scaled_pos = scaled_pos.squeeze(1)

        pos_embedding = torch.zeros(b, self.num_channels, t, h, w, device=device)
        for c in range(self.num_channels):
            angles = scaled_pos * self.frequencies[c]
            if c % 2 == 0:
                pos_embedding[:, c] = torch.sin(angles)
            else:
                pos_embedding[:, c] = torch.cos(angles)
        return pos_embedding


class Framework(pl.LightningModule):
    """Adversarial training framework for RGT estimation.

    Generator (``G_model``) is an HRNet that takes ``[seismic, horizon_RGT,
    mask]`` and predicts a dense RGT volume. Discriminator (``D_model``)
    is a 3D PatchGAN that operates on the position-embedded RGT
    conditioned on the seismic.

    Loss terms:
        - regression loss on the RGT (L1 + 0.5 * MSE), plus L1 on the
          position-embedded RGT;
        - relativistic hinge adversarial loss;
        - 3D LPIPS perceptual loss on slices sampled along each axis.
    """

    def __init__(
        self,
        restore_path,
        adv_train_start_step: int = 0,
        save_every_steps: int = 200,
        gan_factor: float = 0.01,
        reg_factor: float = 5.0,
        lpips_factor: float = 5.0,
        tv_factor: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.adv_train_start_step = adv_train_start_step
        self.save_every_steps = save_every_steps

        self.G_model = HRNet(c=48)
        self.G_model.train()

        self.D_model = D_Net(in_channels=4, base=100)

        self.adv_criterion = AdversarialLoss("rahinge")
        self.lpips_criterion = LPIPS3D(net="alex", k=32)
        self.lpips_criterion_1c = LPIPS3D_1c(net="alex", k=32)
        self.tv_criterion = TVLoss3D if tv_factor != 0 else None
        self.pos_emb = LinearDecayPositionEmbedding(discret=256)

        self.gan_factor = gan_factor
        self.reg_factor = reg_factor
        self.tv_factor = tv_factor
        self.lpips_factor = lpips_factor

        self.optim_g_state = None
        self.optim_d_state = None
        self.start_global_step = 0

        if restore_path:
            self._restore_from_checkpoint(restore_path)

    def _restore_from_checkpoint(self, restore_path: str) -> None:
        try:
            restore = torch.load(restore_path, map_location="cpu")

            pretrained_weights = restore["G_model"]
            model_dict = self.G_model.state_dict()
            mismatched_layers = []
            filtered_weights = {}
            for k, v in pretrained_weights.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_weights[k] = v
                else:
                    mismatched_layers.append(k)
            print("Skipped layers due to shape mismatch:", mismatched_layers)
            model_dict.update(filtered_weights)
            self.G_model.load_state_dict(model_dict)

            self.D_model.load_state_dict(restore["D_model"], strict=True)

            if "optimG" in restore:
                self.optim_g_state = restore["optimG"]
                print("Loaded G optimizer state.")
            if "optimD" in restore:
                self.optim_d_state = restore["optimD"]
                print("Loaded D optimizer state.")
            if "global_step" in restore:
                self.start_global_step = restore["global_step"]
                print(f"Resumed from global step: {self.start_global_step}")

            print(f"Successfully loaded weights from: {restore_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    def training_step(self, batch, batch_idx):
        seis, target, horiz = batch

        if random.randint(0, 1):
            seis = seis.permute(0, 1, 2, 4, 3)
            target = target.permute(0, 1, 2, 4, 3)
            horiz = horiz.permute(0, 1, 2, 4, 3)

        mask = (horiz != 0).to(horiz.dtype)
        mask_target = horiz

        self.last_batch = {
            "seis": seis,
            "target": target,
            "mask": mask,
            "mask_target": mask_target,
        }

        optimizer_g, optimizer_d = self.optimizers()

        # Generator step.
        self.toggle_optimizer(optimizer_g)
        for param in self.D_model.parameters():
            param.requires_grad = False

        pred_target = self.G_model(torch.cat([seis, mask_target, mask], dim=1))
        pred_target = torch.tanh(pred_target)

        pred_pos = self.pos_emb(pred_target)
        target_pos = self.pos_emb(target)

        reg_loss = logReg()(pred_target, target) * 2 + F.l1_loss(pred_pos, target_pos) * 0.2

        G_logits = self.D_model(torch.cat([pred_pos, seis], dim=1))
        G_logits_real = self.D_model(torch.cat([target_pos, seis], dim=1)).detach()
        G_loss = self.adv_criterion(d_real=G_logits_real, d_fake=G_logits, is_disc=False)

        if self.lpips_criterion is not None:
            lpips_loss = (
                self.lpips_criterion(pred_pos, target_pos) * 0.5
                + self.lpips_criterion(pred_pos[:, [2, 0, 1]], target_pos[:, [2, 0, 1]]) * 0.5
                + self.lpips_criterion_1c(pred_target, target)
            )
        else:
            lpips_loss = torch.tensor(0.0, device=self.device)

        self.log("reg_loss", reg_loss, prog_bar=True)
        self.log("G_loss", G_loss, prog_bar=True)
        self.log("tv_loss", torch.tensor(0.0, device=self.device), prog_bar=True)
        self.log("lpips_loss", lpips_loss, prog_bar=True)

        optimizer_g.zero_grad()
        self.manual_backward(
            reg_loss * self.reg_factor
            + G_loss * self.gan_factor
            + lpips_loss * self.lpips_factor
        )
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        # Discriminator step.
        self.toggle_optimizer(optimizer_d)
        for param in self.D_model.parameters():
            param.requires_grad = True

        D_logits_fake = self.D_model(torch.cat([pred_pos.detach(), seis], dim=1))
        D_logits_real = self.D_model(torch.cat([target_pos, seis], dim=1))
        D_loss = self.adv_criterion(d_real=D_logits_real, d_fake=D_logits_fake, is_disc=True)

        self.log("d_loss", D_loss, prog_bar=True)
        optimizer_d.zero_grad()
        self.manual_backward(D_loss * self.gan_factor)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.last_pred_target = pred_target

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G_model.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.D_model.parameters(), lr=1e-4, betas=(0.0, 0.9))

        if self.optim_g_state:
            try:
                opt_g.load_state_dict(self.optim_g_state)
                print("Loaded G optimizer state into optimizer.")
            except Exception as e:
                print(f"Failed to load G optimizer state: {e}")

        if self.optim_d_state:
            try:
                opt_d.load_state_dict(self.optim_d_state)
                print("Loaded D optimizer state into optimizer.")
            except Exception as e:
                print(f"Failed to load D optimizer state: {e}")

        return [opt_g, opt_d], []


class AdversarialLoss(nn.Module):
    """Relativistic adversarial losses: ``rasgan``, ``ralsgan``, ``rahinge``."""

    def __init__(self, type: str = "ralsgan"):
        super().__init__()
        self.type = type.lower()

    def __call__(self, d_real, d_fake, is_disc: bool = True):
        if self.type == "rasgan":
            if is_disc:
                return (
                    F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), torch.ones_like(d_real))
                    + F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), torch.zeros_like(d_fake))
                ) / 2
            return (
                F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), torch.zeros_like(d_real))
                + F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), torch.ones_like(d_fake))
            ) / 2

        if self.type == "ralsgan":
            if is_disc:
                return (
                    ((d_real - d_fake.mean() - 1) ** 2).mean()
                    + ((d_fake - d_real.mean() + 1) ** 2).mean()
                ) / 2
            return (
                ((d_real - d_fake.mean() + 1) ** 2).mean()
                + ((d_fake - d_real.mean() - 1) ** 2).mean()
            ) / 2

        if self.type == "rahinge":
            if is_disc:
                return (
                    F.relu(1.0 - (d_real - d_fake.mean())).mean()
                    + F.relu(1.0 + (d_fake - d_real.mean())).mean()
                ) / 2
            return (
                F.relu(1.0 + (d_real - d_fake.mean())).mean()
                + F.relu(1.0 - (d_fake - d_real.mean())).mean()
            ) / 2

        raise NotImplementedError(f"Unsupported loss type: {self.type}")


class TVLoss3D(nn.Module):
    """3D total variation loss for tensors shaped ``(N, C, D, H, W)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.diff(dim=2).pow(2).mean()
            + x.diff(dim=3).pow(2).mean()
            + x.diff(dim=4).pow(2).mean()
        )


class LPIPS3D_1c(nn.Module):
    """Single-channel 3D LPIPS.

    Randomly samples ``k`` slices along each of the D / H / W axes,
    expands each slice to 3 channels by repetition, and computes 2D LPIPS.
    The final loss is the mean over the three axes.
    """

    def __init__(self, net: str = "alex", k: int = 8):
        super().__init__()
        self.lpips_fun = lpips.LPIPS(net=net)
        self.k = k

    def forward(self, cube1: torch.Tensor, cube2: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = cube1.shape
        device = cube1.device

        idx_d = torch.randperm(D, device=device)[: self.k]
        s1_d = rearrange(cube1[:, :, idx_d, :, :], "b c k h w -> (b k) c h w").repeat(1, 3, 1, 1)
        s2_d = rearrange(cube2[:, :, idx_d, :, :], "b c k h w -> (b k) c h w").repeat(1, 3, 1, 1)
        loss_d = self.lpips_fun(s1_d, s2_d).mean()

        idx_h = torch.randperm(H, device=device)[: self.k]
        s1_h = rearrange(cube1[:, :, :, idx_h, :], "b c d k w -> (b k) c d w").repeat(1, 3, 1, 1)
        s2_h = rearrange(cube2[:, :, :, idx_h, :], "b c d k w -> (b k) c d w").repeat(1, 3, 1, 1)
        loss_h = self.lpips_fun(s1_h, s2_h).mean()

        idx_w = torch.randperm(W, device=device)[: self.k]
        s1_w = rearrange(cube1[:, :, :, :, idx_w], "b c d h k -> (b k) c d h").repeat(1, 3, 1, 1)
        s2_w = rearrange(cube2[:, :, :, :, idx_w], "b c d h k -> (b k) c d h").repeat(1, 3, 1, 1)
        loss_w = self.lpips_fun(s1_w, s2_w).mean()

        return (loss_d + loss_h + loss_w) / 3.0


class LPIPS3D(nn.Module):
    """Multi-channel 3D LPIPS.

    Same axis-sampling scheme as ``LPIPS3D_1c``, but assumes the input
    already has the channel dimension expected by 2D LPIPS (no channel
    expansion is performed).
    """

    def __init__(self, net: str = "alex", k: int = 8):
        super().__init__()
        self.lpips_fun = lpips.LPIPS(net=net)
        self.k = k

    def forward(self, cube1: torch.Tensor, cube2: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = cube1.shape
        device = cube1.device

        idx_d = torch.randperm(D, device=device)[: self.k]
        s1_d = rearrange(cube1[:, :, idx_d, :, :], "b c k h w -> (b k) c h w")
        s2_d = rearrange(cube2[:, :, idx_d, :, :], "b c k h w -> (b k) c h w")
        loss_d = self.lpips_fun(s1_d, s2_d).mean()

        idx_h = torch.randperm(H, device=device)[: self.k]
        s1_h = rearrange(cube1[:, :, :, idx_h, :], "b c d k w -> (b k) c d w")
        s2_h = rearrange(cube2[:, :, :, idx_h, :], "b c d k w -> (b k) c d w")
        loss_h = self.lpips_fun(s1_h, s2_h).mean()

        idx_w = torch.randperm(W, device=device)[: self.k]
        s1_w = rearrange(cube1[:, :, :, :, idx_w], "b c d h k -> (b k) c d h")
        s2_w = rearrange(cube2[:, :, :, :, idx_w], "b c d h k -> (b k) c d h")
        loss_w = self.lpips_fun(s1_w, s2_w).mean()

        return (loss_d + loss_h + loss_w) / 3.0


class logReg(nn.Module):
    """Regression loss combining L1 and (half-weighted) MSE."""

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(img1, img2) + F.mse_loss(img1, img2) * 0.5
