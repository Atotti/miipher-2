import hydra
from lightning.pytorch import seed_everything
from omegaconf import DictConfig

from miipher_2.preprocess import Preprocessor


@hydra.main(version_base=None, config_path="../configs/", config_name=None)  # type: ignore
def main(cfg: DictConfig) -> None:
    seed_everything(172957)
    preprocessor = Preprocessor(cfg=cfg)
    preprocessor.build_from_path()


if __name__ == "__main__":
    main()
