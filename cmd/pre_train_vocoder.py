import hydra
from omegaconf import DictConfig

from miipher_2.train.pre_train_vocoder import pre_train_vocoder


@hydra.main(version_base=None, config_path="../configs", config_name=None)
def main(cfg: DictConfig) -> None:
    pre_train_vocoder(cfg)


if __name__ == "__main__":
    main()
