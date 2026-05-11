import torch
import torch.nn as nn


class D_Net(nn.Module):
    """3D PatchGAN discriminator."""

    def __init__(
        self,
        in_channels: int = 4,
        base: int = 80,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = [
            nn.Conv3d(in_channels, base, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base
        for _ in range(1, num_layers):
            out_channels = min(channels * 2, 512)
            layers.extend([
                nn.Conv3d(channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            channels = out_channels

        layers.append(nn.Conv3d(channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
