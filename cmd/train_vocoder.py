import pathlib

import hydra
from omegaconf import DictConfig

from miipher_2.train import hifigan as _core


@hydra.main(version_base=None, config_path="../configs", config_name="hifigan_finetune")
def main(cfg: DictConfig) -> None:
    args = _core.parse()
    args.stage = "finetune"
    args.wav_list = cfg.dataset.list
    args.adapter_ckpt = pathlib.Path(cfg.adapter_ckpt)
    args.resume = pathlib.Path(cfg.pretrained)  # <- 事前学習済 G
    args.steps = cfg.steps
    args.lr = cfg.lr
    args.save = pathlib.Path(cfg.save_dir)
    _core.main(args)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
