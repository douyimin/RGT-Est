# seisDataset

This directory is intentionally a placeholder for the training-time
dataset module. The training script `train.py` imports

```python
from seisDataset.RGT_dataset import RGTDataset
```

`RGTDataset` is expected to be a `torch.utils.data.Dataset` that yields
`(seis, target, horiz)` tuples, where each tensor is shaped
`(C, T, H, W)`:

- `seis`   — input seismic volume (1 channel).
- `target` — ground-truth RGT volume in `[-1, 1]` (1 channel).
- `horiz`  — sparse horizon RGT values, with `0` marking unlabelled
  voxels (1 channel). At training time, `mask = (horiz != 0)` is built
  on the fly and fed to the generator together with `seis` and `horiz`.

The dataset implementation is omitted from this release because the
training corpora used in the paper are not redistributable. Drop a
compatible `RGT_dataset.py` here to enable training.
