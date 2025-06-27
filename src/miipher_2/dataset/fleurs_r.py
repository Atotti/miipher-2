import re
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class FleursRCorpus(Dataset):
    """
    fleurs-r コーパス用のPyTorch Datasetクラス。
    複数の言語やサブセット(train, dev, test)を横断してデータをロードできます。
    """

    def __init__(self, root: str, subset: str | list[str] = "all", language: str | list[str] = "all") -> None:
        """
        Args:
            root (str): fleurs-r データセットのルートディレクトリ。
            subset (str | List[str]): 使用するサブセット。"train", "dev", "test"のいずれか、
                                       そのリスト、または"all"を指定。デフォルトは"all"。
            language (str | List[str]): 使用する言語 (例: "ja_jp")。
                                        そのリスト、または"all"を指定。デフォルトは"all"。
        """
        super().__init__()
        self.root = Path(root).resolve()
        self.samples: list[dict[str, str]] = []
        data_path = self.root / "data"

        # 1. 処理対象の言語を決定
        if language == "all":
            # xx_xx または xxx_xxxx 形式のディレクトリ名を持つ言語ディレクトリをすべて取得
            lang_dirs = [
                p for p in data_path.iterdir() if p.is_dir() and re.match(r"^[a-z]{2,3}(_[a-z]{2,4}){1,2}$", p.name)
            ]
            target_languages = sorted([p.name for p in lang_dirs])
        elif isinstance(language, list):
            target_languages = language
        else:
            target_languages = [language]

        # 2. 処理対象のサブセットを決定
        if subset == "all":
            target_subsets = ["train", "dev", "test"]
        elif isinstance(subset, list):
            target_subsets = subset
        else:
            target_subsets = [subset]

        print(f"Target languages: {len(target_languages)}")
        print(f"Target subsets: {target_subsets}")

        # 3. 全ての対象データをループで読み込み
        for lang in target_languages:
            for sub in target_subsets:
                lang_dir = data_path / lang
                tsv_path = lang_dir / f"{sub}.tsv"
                audio_dir = lang_dir / "audio" / sub / sub

                if not (tsv_path.exists() and audio_dir.exists()):
                    continue

                print(f"Processing: {lang}/{sub}")
                try:
                    metadata = pd.read_csv(
                        tsv_path,
                        sep="\t",
                        header=None,
                        usecols=[0, 1, 2],
                        names=["speaker_id", "wav_name", "raw_text"],
                        on_bad_lines="warn",
                    )
                except Exception as e:
                    print(f"Warning: Could not read TSV {tsv_path}: {e}")
                    continue

                lang_code_out = "jpn" if lang == "ja_jp" else lang.split("_")[0]

                for _, row in metadata.iterrows():
                    speaker = str(row["speaker_id"])
                    wav_name = row["wav_name"]
                    wav_path = audio_dir / wav_name

                    if wav_path.is_file():
                        self.samples.append(
                            {
                                "wav_path": str(wav_path),
                                "speaker": speaker,
                                "clean_text": row["raw_text"],
                                "basename": f"{lang}_{sub}_{speaker}_{wav_path.stem}",
                                "lang_code": lang_code_out,
                            }
                        )

        print(f"Total samples found: {len(self.samples)}")

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def speaker_dict(self) -> dict[str, int]:
        """一意の話者IDと整数のインデックスをマッピングした辞書を返します。"""
        if not hasattr(self, "_speaker_dict"):
            speakers = sorted({str(sample["speaker"]) for sample in self.samples})
            self._speaker_dict = {speaker: idx for idx, speaker in enumerate(speakers)}
        return self._speaker_dict
