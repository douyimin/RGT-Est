# -*- coding: utf-8 -*-
"""
RGT-Est: Learning Stratigraphically Consistent Relative Geologic Time
from 3D Seismic Data via Sinusoidal Mapping.

Authors:  Yimin Dou, Xinming Wu, Hui Gao, Zhengfa Bi
Contact:  Xinming Wu <xinmwu@ustc.edu.cn>

Copyright (c) 2026 Yimin Dou, Xinming Wu, Hui Gao, Zhengfa Bi.

License
-------
Source code in this file is released under the MIT License.
Pretrained model weights and datasets associated with this project
(distributed via Zenodo and Baidu Netdisk) are released separately
under the Creative Commons Attribution 4.0 International License
(CC BY 4.0). See the LICENSE and LICENSE-DATA files in the repository
root for the full terms.

If you use this software, please cite:
    Dou, Y., Wu, X., Gao, H., & Bi, Z. (2026).
    Learning Stratigraphically Consistent Relative Geologic Time
    from 3D Seismic Data via Sinusoidal Mapping.
"""



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
