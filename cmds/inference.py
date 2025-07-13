import hydra
from omegaconf import DictConfig

# `main`関数を`run_inference`にリネームして、より責務を明確にします
from miipher_2.utils.infer import run_inference


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    """
    Hydra経由で設定を読み込み、推論処理を実行するエントリーポイント
    """
    run_inference(cfg)


if __name__ == "__main__":
    main()
