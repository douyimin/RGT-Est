import os
import time

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch


class EMACallback(pl.Callback):
    """Exponential moving average of model parameters.

    Tracks a shadow copy of every trainable parameter (and optionally the
    running statistics of BN layers) under either the whole LightningModule
    or a single submodule named by ``ema_scope``.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        ema_device=None,
        ema_scope=None,
        update_bn: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.ema_device = ema_device
        self.ema_scope = ema_scope
        self.update_bn = update_bn

        self.shadow = {}
        self.backup = {}
        self.is_ema_active = False

    def on_train_start(self, trainer, pl_module):
        for name, param in self._get_model_parameters(pl_module):
            self.shadow[name] = param.data.clone()
            if self.ema_device:
                self.shadow[name] = self.shadow[name].to(self.ema_device)

    def _get_model_parameters(self, pl_module):
        """Yield ``(name, tensor)`` pairs that should track an EMA."""
        if self.ema_scope is None:
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    yield name, param
            if self.update_bn:
                for name, buf in pl_module.named_buffers():
                    if "running_mean" in name or "running_var" in name:
                        yield name, buf
        else:
            prefix = self.ema_scope + "."
            model = getattr(pl_module, self.ema_scope)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    yield prefix + name, param
            if self.update_bn:
                for name, buf in model.named_buffers():
                    if "running_mean" in name or "running_var" in name:
                        yield prefix + name, buf

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in self._get_model_parameters(pl_module):
            self.shadow[name] = self.shadow[name] * self.decay + param.data * (1 - self.decay)

    def apply_ema(self, pl_module):
        """Replace live weights with EMA weights (saving the originals)."""
        if self.is_ema_active:
            return
        self.is_ema_active = True
        for name, param in self._get_model_parameters(pl_module):
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    def restore(self, pl_module):
        """Restore the original (non-EMA) weights."""
        if not self.is_ema_active:
            return
        self.is_ema_active = False
        for name, param in self._get_model_parameters(pl_module):
            param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "backup": self.backup,
            "is_ema_active": self.is_ema_active,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
        self.backup = state_dict["backup"]
        self.is_ema_active = state_dict["is_ema_active"]


class EMASwitchContext:
    """Context manager that temporarily swaps in EMA weights and ``eval`` mode."""

    def __init__(self, callback: EMACallback, pl_module):
        self.callback = callback
        self.pl_module = pl_module
        self.training = {}

    def __enter__(self):
        self.callback.apply_ema(self.pl_module)
        if self.callback.ema_scope is None:
            self.training["model"] = self.pl_module.training
            self.pl_module.eval()
        else:
            model = getattr(self.pl_module, self.callback.ema_scope)
            self.training[self.callback.ema_scope] = model.training
            model.eval()
        return self.pl_module

    def __exit__(self, *args):
        self.callback.restore(self.pl_module)
        if self.callback.ema_scope is None:
            if self.training.get("model", True):
                self.pl_module.train()
        else:
            model = getattr(self.pl_module, self.callback.ema_scope)
            if self.training.get(self.callback.ema_scope, True):
                model.train()


class CustomCallback(pl.Callback):
    """Logging, checkpointing, and visualization callback.

    On the rank-0 process this callback:
        - tracks a moving-average of the per-step losses and prints
          summaries every ``print_every_n_steps`` steps;
        - saves three orthogonal slices of the inputs/target/prediction
          every ``viz_every_n_steps`` steps;
        - saves both the standard and the EMA model weights (if an
          ``EMACallback`` is registered) every ``save_weights_every_n_steps``
          steps.
    """

    def __init__(
        self,
        print_every_n_steps: int = 10,
        save_weights_every_n_steps: int = 1000,
        save_dir: str = "model_weights",
        viz_every_n_steps: int = 500,
        results_dir: str = "results",
        viz_samples: int = 1,
        avg_window: int = 10,
    ):
        super().__init__()
        self.print_every_n_steps = print_every_n_steps
        self.save_weights_every_n_steps = save_weights_every_n_steps
        self.viz_every_n_steps = viz_every_n_steps
        self.viz_samples = viz_samples
        self.save_dir = save_dir
        self.results_dir = results_dir
        self.avg_window = avg_window

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.last_print_time = time.time()

        self.reg_loss_history = []
        self.g_loss_history = []
        self.d_loss_history = []
        self.tv_loss_history = []
        self.lpips_loss_history = []

    def on_fit_start(self, trainer, pl_module):
        trainer.enable_progress_bar = False

    def show_results(self, tensor_dict: dict, save_path: str, sample_idx: int = 0):
        """Plot three orthogonal mid-slices for each tensor and save to disk."""
        dict_len = len(tensor_dict)
        plt.figure(figsize=(30, 30))

        for idx, key in enumerate(tensor_dict):
            data = tensor_dict[key][sample_idx, 0]
            t, h, w = data.shape
            cmap = "gray" if key == "seis" else "jet"

            plt.subplot(3, dict_len, idx + 1)
            plt.title(key)
            plt.imshow(data[:, int(h / 2)], cmap=cmap)

            plt.subplot(3, dict_len, idx + dict_len + 1)
            plt.title(key)
            plt.imshow(data[:, :, int(w / 2)], cmap=cmap)

            plt.subplot(3, dict_len, idx + dict_len * 2 + 1)
            plt.title(key)
            plt.imshow(data[int(t / 2), :, :], cmap=cmap)

        plt.savefig(save_path)
        plt.close()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        # Collect losses and convert to floats.
        reg_loss = trainer.callback_metrics.get("reg_loss", torch.tensor(0.0))
        g_loss = trainer.callback_metrics.get("G_loss", torch.tensor(0.0))
        d_loss = trainer.callback_metrics.get("d_loss", torch.tensor(0.0))
        tv_loss = trainer.callback_metrics.get("tv_loss", torch.tensor(0.0))
        lpips_loss = trainer.callback_metrics.get("lpips_loss", torch.tensor(0.0))

        if isinstance(reg_loss, torch.Tensor):
            reg_loss = reg_loss.item()
        if isinstance(g_loss, torch.Tensor):
            g_loss = g_loss.item()
        if isinstance(d_loss, torch.Tensor):
            d_loss = d_loss.item()
        if isinstance(tv_loss, torch.Tensor):
            tv_loss = tv_loss.item()
        if isinstance(lpips_loss, torch.Tensor):
            lpips_loss = lpips_loss.item()

        self.reg_loss_history.append(reg_loss)
        self.g_loss_history.append(g_loss)
        self.d_loss_history.append(d_loss)
        self.tv_loss_history.append(tv_loss)
        self.lpips_loss_history.append(lpips_loss)

        self.reg_loss_history = self.reg_loss_history[-self.avg_window:]
        self.g_loss_history = self.g_loss_history[-self.avg_window:]
        self.d_loss_history = self.d_loss_history[-self.avg_window:]
        self.tv_loss_history = self.tv_loss_history[-self.avg_window:]
        self.lpips_loss_history = self.lpips_loss_history[-self.avg_window:]

        # Periodic loss printing.
        if pl_module.global_step % self.print_every_n_steps == 0 and self.reg_loss_history:
            current_time = time.time()
            elapsed_time = current_time - self.last_print_time
            self.last_print_time = current_time

            avg_reg_loss = sum(self.reg_loss_history) / len(self.reg_loss_history)
            avg_g_loss = sum(self.g_loss_history) / len(self.g_loss_history)
            avg_d_loss = sum(self.d_loss_history) / len(self.d_loss_history)
            avg_tv_loss = sum(self.tv_loss_history) / len(self.tv_loss_history)
            avg_lpips_loss = sum(self.lpips_loss_history) / len(self.lpips_loss_history)

            print(
                f"Step {pl_module.global_step}: time taken: {elapsed_time:.2f}, "
                f"reg_loss={avg_reg_loss:.4f}, "
                f"g_loss={avg_g_loss:.4f}, "
                f"d_loss={avg_d_loss:.4f}, "
                f"lpips_loss={avg_lpips_loss:.4f}, "
                f"tv_loss={avg_tv_loss:.4f}"
            )

        # Periodic visualization.
        if (
            pl_module.global_step % self.viz_every_n_steps == 0
            and pl_module.global_step > 0
            and hasattr(pl_module, "last_batch")
            and hasattr(pl_module, "last_pred_target")
        ):
            self._save_visualizations(trainer, pl_module)

        # Periodic checkpointing.
        if (
            pl_module.global_step % self.save_weights_every_n_steps == 0
            and pl_module.global_step > 0
        ):
            self._save_checkpoints(trainer, pl_module)

    def _save_visualizations(self, trainer, pl_module):
        seis = pl_module.last_batch["seis"]
        target = pl_module.last_batch["target"]
        mask = pl_module.last_batch["mask"]
        mask_target = pl_module.last_batch["mask_target"]
        pred_target = pl_module.last_pred_target

        batch_size = seis.size(0)
        actual_samples = min(self.viz_samples, batch_size)
        print(f"Saving {actual_samples} visualization sample(s).")

        ema_callback = None
        for callback in trainer.callbacks:
            if hasattr(callback, "apply_ema") and hasattr(callback, "restore"):
                ema_callback = callback
                break

        for sample_idx in range(actual_samples):
            step_str = str(pl_module.global_step).zfill(8)
            if actual_samples > 1:
                save_path = os.path.join(
                    self.results_dir,
                    f"result_step_{step_str}_sample_{sample_idx + 1}.png",
                )
            else:
                save_path = os.path.join(self.results_dir, f"result_step_{step_str}.png")

            normal_pred = pred_target.detach().cpu().float().numpy()

            tensors = {
                "seis": seis.detach().cpu().float().numpy(),
                "mask": mask.detach().cpu().float().numpy(),
                "target": target.detach().cpu().float().numpy(),
                "pred_target": normal_pred,
            }

            if ema_callback:
                with EMASwitchContext(ema_callback, pl_module):
                    with torch.no_grad():
                        ema_pred = torch.tanh(
                            pl_module.G_model(torch.cat([seis, mask_target, mask], dim=1))
                        ).mean(dim=1, keepdim=True)
                    tensors["ema_pred"] = ema_pred.detach().cpu().float().numpy()
                    self.show_results(tensors, save_path, sample_idx)
            else:
                self.show_results(tensors, save_path, sample_idx)

        print(f"Saved visualizations to: {self.results_dir}")

    def _save_checkpoints(self, trainer, pl_module):
        opt_g, opt_d = pl_module.optimizers()

        # Standard checkpoint.
        param = {
            "G_model": pl_module.G_model.state_dict(),
            "D_model": pl_module.D_model.state_dict(),
            "optimG": opt_g.state_dict(),
            "optimD": opt_d.state_dict(),
            "global_step": pl_module.global_step,
        }
        weights_path = os.path.join(self.save_dir, f"model_step_{pl_module.global_step}.pth")
        torch.save(param, weights_path)
        print(f"Saved standard checkpoint to: {weights_path}")

        # EMA checkpoint if an EMA callback is present.
        ema_callback = None
        for callback in trainer.callbacks:
            if hasattr(callback, "apply_ema") and hasattr(callback, "restore"):
                ema_callback = callback
                break

        if ema_callback:
            with EMASwitchContext(ema_callback, pl_module):
                ema_param = {
                    "G_model": pl_module.G_model.state_dict(),
                    "D_model": pl_module.D_model.state_dict(),
                    "optimG": opt_g.state_dict(),
                    "optimD": opt_d.state_dict(),
                    "global_step": pl_module.global_step,
                    "is_ema": True,
                }
                ema_weights_path = os.path.join(
                    self.save_dir, f"model_ema_step_{pl_module.global_step}.pth"
                )
                torch.save(ema_param, ema_weights_path)
                print(f"Saved EMA checkpoint to: {ema_weights_path}")
