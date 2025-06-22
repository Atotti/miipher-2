import hydra
from omegaconf import DictConfig

# 新しく作成する一括処理用の関数をインポート
from miipher_2.utils.infer import run_inference_dir


@hydra.main(version_base=None, config_path="../configs", config_name="infer_dir")
def main(cfg: DictConfig) -> None:
    """
    Hydra経由で設定を読み込み、ディレクトリ単位の音声修復を実行する
    """
    run_inference_dir(cfg)


if __name__ == "__main__":
    main()
