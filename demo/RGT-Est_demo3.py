"""
RGT-Est demo 3 — runs the pretrained RGT-Est model on a 3D seismic volume
without any horizon constraints (the two auxiliary input channels are filled
with zeros).

Workflow:
    1. Load and normalize the seismic volume.
    2. Resample to the inference grid (400, 512, 512) and run the model.
    3. Resample the predicted RGT back to the original grid.
    4. Trace horizons as iso-surfaces of the RGT volume.
    5. Show three linked 3D views side by side (cigvis).

NOTE: the inference grid MUST be (400, 512, 512). Other sizes will degrade the
prediction or cause inference to fail. For arbitrary-size support please
contact xinmwu@ustc.edu.cn or douyimin@ustc.edu.cn.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cigvis
from cigvis import colormap

from rgt_helper import horizons_from_rgt, horizon_image


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def normalization(data: np.ndarray) -> np.ndarray:
    """Linearly rescale ``data`` into ``[0, 1]``."""
    rng = np.max(data) - np.min(data)
    return (data - np.min(data)) / (rng + 1e-6)


def z_score_clip(data: np.ndarray, clp_s: float = 3.0) -> np.ndarray:
    """Z-score normalize, clip at ``±clp_s`` sigma, then rescale to ``[0, 1]``."""
    z = (data - np.mean(data)) / (np.std(data) + 1e-8)
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))


# ---------------------------------------------------------------------------
# 1. Load seismic
# ---------------------------------------------------------------------------
seis = np.load(r"../RealData/Netherlands3.npy").astype(np.float32)
seis = z_score_clip(seis, clp_s=2).transpose(1, 2, 0) * 2 - 1   # → (H, W, T) in [-1, 1]
h, w, t = seis.shape
print("seis shape (H, W, T):", seis.shape)


# ---------------------------------------------------------------------------
# 2. RGT inference
# ---------------------------------------------------------------------------
model = torch.jit.load("../pretrianedModel/RGT-Est_CIG-Benchmark.pt").to(device)
model.eval()

infer_shape = (400, 512, 512)

seis_tensor = torch.from_numpy(seis.transpose(2, 0, 1))[None, None]      # (1, 1, T, H, W)
seis_tensor = F.interpolate(seis_tensor, infer_shape, mode="trilinear")

# Stack with two zero channels (horizon RGT values + labelled-voxel mask).
input_tensor = torch.cat(
    [seis_tensor, torch.zeros_like(seis_tensor), torch.zeros_like(seis_tensor)],
    dim=1,
).to(device)

with torch.no_grad():
    with torch.autocast(device_type=device):
        padded = nn.ReplicationPad3d(8)(input_tensor)
        rgt = model(padded)[:, :, 8:-8, 8:-8, 8:-8]

print("RGT raw output shape:", tuple(rgt.shape))


# ---------------------------------------------------------------------------
# 3. Resample RGT back to the original grid and normalize
# ---------------------------------------------------------------------------
rgt_volume = F.interpolate(rgt.float(), (t, h, w), mode="trilinear").cpu().numpy()[0, 0]
rgt_volume = z_score_clip(rgt_volume)
print("rgt_volume shape (T, H, W):", rgt_volume.shape)


# ---------------------------------------------------------------------------
# 4. Trace horizons from the RGT volume
# ---------------------------------------------------------------------------
ux = rgt_volume.transpose(1, 2, 0)                # (H, W, T)
n3, n2, n1 = ux.shape

hu = np.linspace(0.0, 1.0, 100)
hzs = horizons_from_rgt(sig=1.0, hu=hu, ux=ux)

rgt_mask = horizon_image(n1=n1, n2=n2, n3=n3, d=3, sig=0.0, x1=hzs)
rgt_mask = (rgt_mask > 0).astype(np.float32) * ux   # keep RGT value on each horizon
print("rgt_mask shape (H, W, T):", rgt_mask.shape)


# ---------------------------------------------------------------------------
# 5. Three linked 3D views: seismic / RGT / seismic + horizons
# ---------------------------------------------------------------------------
fg_cmap = colormap.set_alpha_except_min("AI", alpha=1)

nodes_seis    = cigvis.create_slices(seis, cmap="gray")
nodes_rgt     = cigvis.create_slices(normalization(rgt_volume.transpose(1, 2, 0)), cmap="AI")
nodes_overlay = cigvis.create_slices(seis, cmap="gray")
cigvis.add_mask(nodes_overlay, rgt_mask, cmaps=fg_cmap, interpolation="nearest")

cigvis.plot3D([nodes_seis, nodes_rgt, nodes_overlay], grid=(1, 3), share=True)
