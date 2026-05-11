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



import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from framework.framework import Framework
from framework.custom_callback import CustomCallback, EMACallback
from seisDataset.RGT_dataset import RGTDataset


torch.backends.cudnn.benchmark = True


def main():
    train_dataset = RGTDataset(
        rgt_dir=r"",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    restore_path = ""
    model = Framework(restore_path=restore_path)

    ddp_strategy = DDPStrategy(find_unused_parameters=True)

    os.makedirs("results", exist_ok=True)
    os.makedirs("model_weights", exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16",
        logger=TensorBoardLogger("logs", name="RGT_Est"),
        callbacks=[
            CustomCallback(
                print_every_n_steps=10,
                save_weights_every_n_steps=2500,
                viz_every_n_steps=500,
                viz_samples=4,
                save_dir="model_weights",
                results_dir="results",
            ),
            EMACallback(
                decay=0.9996,
                ema_scope="G_model",
                update_bn=True,
            ),
        ],
        log_every_n_steps=10,
        enable_progress_bar=False,
        strategy=ddp_strategy,
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
