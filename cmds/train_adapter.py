import hydra
from omegaconf import DictConfig

from miipher_2.train.adapter import train_adapter


@hydra.main(version_base=None, config_path="../configs", config_name="adapter")
def main(cfg: DictConfig) -> None:  # pylint:disable=no-value-for-parameter
    train_adapter(cfg)


if __name__ == "__main__":
    main()
