import hydra
from omegaconf import DictConfig

from miipher_2.train import adapter as _core


@hydra.main(version_base=None, config_path="../configs", config_name="adapter")
def main(cfg: DictConfig) -> None:
    # Hydra → argparse 変換して既存モジュール main() を呼び出す
    args = _core.parse()
    args.wav_list = cfg.dataset.list
    args.epochs = cfg.epochs
    args.lr = cfg.optim.lr
    args.save = cfg.save_dir
    _core.main(args)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter

# cmd/train_hubert.py と cmd/train_vocoder.py は、cmd/preprocess.py と同様に、hydra を使って設定ファイルを読み込むように修正してください。
