# RGT-Est

**Learning Stratigraphically Consistent Relative Geologic Time from 3D Seismic Data via Sinusoidal Mapping**

Yimin Dou, Xinming Wu*, Hui Gao, Zhengfa Bi

Released as the RGT-estimation entry of **CIGbench**.

---

## Abstract

Relative Geologic Time (RGT) estimation from seismic data underpins subsurface structural modeling, depositional analysis, and reservoir characterization. Accurate RGT estimation remains challenging because RGT is a topologically constrained continuous field — local errors readily propagate globally through topological coupling and distort the overall result. Conventional methods rely heavily on prior information and manual interaction, while existing deep-learning approaches predominantly use MSE/MAE regression, which struggles to recover thin horizons and to capture the stratigraphic semantics of the RGT field.

We propose **RGT-Est**, a deep-learning framework that transfers the optimization target from the topologically constrained continuous field into a **differentiable sinusoidal space**. This representation explicitly encodes the periodic stratigraphic semantics of RGT and alleviates the over-smoothing of fine horizons inherent in direct regression. Pointwise, perceptual, and adversarial losses are jointly imposed in this space to enforce local fidelity, inter-layer consistency, and global structural plausibility. An optional horizon-guidance module accepts sparse 2D or 3D horizons as priors.

Trained on synthetic data and evaluated on field surveys with dense faulting, large unconformities, steeply dipping strata, folded deformations, and clinoforms, RGT-Est achieves state-of-the-art performance among AI-based methods, and attains substantially higher horizon-correlation accuracy and topological consistency when sparse priors are incorporated.

![RGT-Est framework](figures/01.jpg)

---

## Idea

Pixel-wise MSE/MAE losses treat every voxel as an independent number, so the loss is blind to *where* the error sits — an error at a thin horizon is penalized the same as an error in a homogeneous interior. The result is over-smoothed thin layers and unstable stratigraphic ordering.

**RGT-Est lifts the prediction into a sinusoidal phase space before measuring the loss.** Three sinusoidal channels with linearly decreasing frequencies `(2.0, 1.0, 0.5)` encode the predicted RGT at three stratigraphic scales — the high-frequency channel resolves thin layers, the low-frequency channel anchors the large-scale framework, and together they give every RGT value a unique phase fingerprint. The same prediction error then produces a **~62× larger gradient for L1 and ~16× larger for LPIPS**, concentrated at layer boundaries rather than the depth bulk.

![Gradient analysis](figures/02.jpg)

---

## Contributions

1. **A sinusoidal-space modeling paradigm for RGT estimation.** We reformulate RGT estimation from continuous scalar-field regression into a multi-scale phase optimization problem. Three sinusoidal channels with linearly decreasing frequencies explicitly encode the periodic stratigraphic semantics of RGT and yield a unique representation of any RGT value, fundamentally alleviating the over-smoothing of thin layers caused by MSE/MAE losses.

2. **A multi-loss collaborative mechanism for global topological constraints.** We jointly impose adversarial, perceptual, and MAE losses in the sinusoidal space, constraining the network from three complementary perspectives — distributional consistency, structural fidelity, and pointwise accuracy — and equipping it with both fine-horizon discrimination and robust global stratigraphic awareness.

3. **Optional sparse horizon guidance.** An optional Horizon Guidance module accepts sparse 2D or 3D horizons as priors. RGT-Est operates fully automatically without any prior; once horizons are provided, it delivers substantially higher precision and naturally preserves lateral consistency in slice-by-slice 3D prediction.

4. **Systematic multi-scenario generalization evaluation.** We evaluate RGT-Est on multiple structurally complex field seismic datasets covering unconformities, densely faulted systems, steeply dipping structures, and strong structural superposition, substantially outperforming publicly available AI-based RGT estimation methods.

![Comparison with voxel-space regression](figures/03.jpg)
![Horizon-guided RGT](figures/04.jpg)
![Challenging field surveys](figures/05.jpg)

---

## Repository

```
RGT_Est/
├── train.py          # Training entry point (PyTorch Lightning + DDP).
├── framework/        # Generator, discriminator, losses, callbacks.
├── seisDataset/      # Placeholder for the training dataset module.
└── demo/             # Inference notebooks + RGT → horizon utilities.
```

Quick inference:

```python
import torch, torch.nn as nn, torch.nn.functional as F

model = torch.jit.load("RGT-Est_CIG-Benchmark.pt").to(device).eval()

# 3-channel input [seismic, horizon, mask]; zero channels 1, 2 for automatic mode.
x = F.interpolate(torch.cat([seis, horiz, mask], dim=1), (400, 512, 512), mode="nearest")
with torch.no_grad(), torch.autocast(device_type=device):
    rgt = model(nn.ReflectionPad3d(8)(x))[:, :, 8:-8, 8:-8, 8:-8]
```

See `demo/RGT-Est_demo.ipynb` and `demo/RGT-Est_horizConstra_demo.ipynb` for end-to-end examples. Pretrained weights ship separately with the CIGbench release.

---

## Citation

```bibtex
@article{dou2026learning,
  title={Learning Stratigraphically Consistent Relative Geologic Time from 3D Seismic Data via Sinusoidal Mapping},
  author={Dou, Yimin and Wu, Xinming and Gao, Hui and Bi, Zhengfa},
  journal={arXiv preprint arXiv:2605.01273},
  year={2026}
}
```

Correspondence: Xinming Wu — `xinmwu@ustc.edu.cn`.


