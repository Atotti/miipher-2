import hydra
from omegaconf import DictConfig

from miipher_2.train.hifigan import train_hifigan


@hydra.main(version_base=None, config_path="../configs", config_name="hifigan_finetune")
def main(cfg: DictConfig) -> None:  # pylint:disable=no-value-for-parameter
    train_hifigan(cfg)


if __name__ == "__main__":
    main()
