import hydra
from omegaconf import DictConfig

from miipher_2.train.adapter import train_adapter


@hydra.main(version_base=None, config_path="../configs", config_name="adapter")  # pyproject の layout に応じて調整
def main(cfg: DictConfig) -> None:
    train_adapter(cfg)


if __name__ == "__main__":
    main()
