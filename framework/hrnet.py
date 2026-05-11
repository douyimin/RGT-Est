from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


GroupNorm = partial(nn.GroupNorm, 16)


class PixelShuffle3d(nn.Module):
    """Channel-to-space upsampling for 3D feature maps.

    Equivalent to ``nn.PixelShuffle`` extended to volumetric tensors:
    rearranges ``(N, C * s^3, D, H, W)`` into ``(N, C, D*s, H*s, W*s)``.
    """

    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.size()
        s = self.scale
        n_out = c // (s ** 3)
        x = x.contiguous().view(b, n_out, s, s, s, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(b, n_out, d * s, h * s, w * s)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = GroupNorm(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = GroupNorm(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = GroupNorm(planes * self.expansion)
        self.act_fun = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.act_fun(self.norm1(self.conv1(x)))
        out = self.act_fun(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.act_fun(out + residual)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = GroupNorm(planes)
        self.act_fun = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = GroupNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.act_fun(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.act_fun(out + residual)


class StageModule(nn.Module):
    """One multi-resolution stage of HRNet.

    Each of ``stage`` parallel branches runs four ``BasicBlock`` layers
    at its own resolution; outputs are then fused across the first
    ``output_branches`` branches via 1x1 conv + upsample (for higher-res
    targets) or strided conv chains (for lower-res targets).
    """

    def __init__(self, stage: int, output_branches: int, c: int):
        super().__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            self.branches.append(nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
            ))

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                        GroupNorm(c * (2 ** i)),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode="nearest"),
                    ))
                else:  # i > j: progressive downsampling chain.
                    ops = []
                    for _ in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                            GroupNorm(c * (2 ** j)),
                            nn.SiLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                        GroupNorm(c * (2 ** i)),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.act_fun = nn.SiLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)
        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        return [self.act_fun(f) for f in x_fused]


class HRNet(nn.Module):
    """3D HRNet generator used as ``G_model`` in the RGT-Est framework."""

    def __init__(self, c: int = 48):
        super().__init__()

        # Stem.
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = GroupNorm(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = GroupNorm(64)
        self.act_fun = nn.SiLU(inplace=True)

        # Stage 1: four bottleneck blocks at a single resolution.
        downsample = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False),
            GroupNorm(256),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Transition 1: split into two parallel branches.
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, c, kernel_size=3, stride=1, padding=1, bias=False),
                GroupNorm(c),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(
                nn.Conv3d(256, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
                GroupNorm(c * 2),
                nn.SiLU(inplace=True),
            )),
        ])

        # Stage 2: two two-branch modules.
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c),
            StageModule(stage=2, output_branches=2, c=c),
        )

        # Transition 2: derive a new (lower-res) branch from the last branch only.
        self.transition2 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
                GroupNorm(c * 4),
                nn.SiLU(inplace=True),
            )),
        ])

        # Stage 3: five three-branch modules + one collapse-to-two-branches module.
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=2, c=c),
        )

        # Decoder: PixelShuffle3d twice to restore the original resolution.
        self.reg_decoder = nn.Sequential(
            nn.Conv3d(c + c * 2, c * 8, kernel_size=3, padding=1),
            GroupNorm(c * 8),
            nn.SiLU(inplace=True),
            PixelShuffle3d(2),
            nn.Conv3d(c, 16 * 8, kernel_size=3, padding=1),
            GroupNorm(16 * 8),
            nn.SiLU(inplace=True),
            PixelShuffle3d(2),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.act_fun(self.norm1(self.conv1(x)))
        x = self.act_fun(self.norm2(self.conv2(x)))

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1]),
        ]

        x = self.stage3(x)
        x = torch.cat([x[0], F.interpolate(x[1], scale_factor=2)], dim=1)
        return self.reg_decoder(x)


if __name__ == "__main__":
    model = HRNet(32).cuda()
    y = model(torch.ones(1, 3, 384, 288, 288).cuda())
    print(y.shape)
    print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())
