from pathlib import Path

import hydra
import pyrootutils
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning_vocoders.models.hifigan.lightning_module import HiFiGANLightningModule
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_name=None, config_path="../configs")
def main(cfg: DictConfig):
    seed_everything(172957)
    lightning_module = hydra.utils.instantiate(cfg.model.lightning_module, cfg)
    if cfg.compile:
        lightning_module = torch.compile(lightning_module, dynamic=True)
    callbacks = [LearningRateMonitor(logging_interval="step")]
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, cfg)
    loggers = [hydra.utils.instantiate(logger) for logger in cfg.train.loggers]
    trainer = hydra.utils.instantiate(cfg.train.trainer, logger=loggers, callbacks=callbacks)
    trainer.fit(lightning_module, datamodule, ckpt_path=cfg.train.ckpt_path)


if __name__ == "__main__":
    main()
