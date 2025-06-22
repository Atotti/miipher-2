from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class FleursRCorpus(Dataset):
    """
    fleurs-r コーパス用のPyTorch Datasetクラス。

    {root}/data/ja_jp/audio/train/train/xxxx.wav
    """

    def __init__(self, root: str, subset: str = "train", language: str = "ja_jp") -> None:
        """
        Args:
            root (str): fleurs-r データセットのルートディレクトリ。
            subset (str): 使用するデータセットのサブセット ("train", "dev", "test" のいずれか)。
            language (str): 使用する言語 (例: "ja_jp")。
        """
        super().__init__()
        self.root = Path(root).resolve()
        self.subset = subset
        self.language = language

        self.data_dir = self.root / "data" / self.language
        tsv_path = self.data_dir / f"{self.subset}.tsv"

        self.audio_dir = self.data_dir / "audio" / self.subset / self.subset

        if not self.audio_dir.exists():
            msg = (
                f"音声ディレクトリが見つかりません: {self.audio_dir}。 "
                f"音声アーカイブ ({subset}.tar.gz など) が展開されていることを確認してください。"
            )
            raise FileNotFoundError(msg)

        # tsvファイルからメタデータを読み込む
        # カラム: id, path, raw_text, ... (ヘッダーなし)
        try:
            # 最初の3カラム (speaker_id, wav_filename, raw_text)のみが必要
            self.metadata = pd.read_csv(
                tsv_path,
                sep="\t",
                header=None,
                usecols=[0, 1, 2],
                names=["speaker_id", "wav_name", "raw_text"],
                on_bad_lines="warn",
            )
        except Exception as e:  # noqa: BLE001
            msg = f"TSVファイルの読み込みエラー: {tsv_path}: {e}"
            raise OSError(msg)  # noqa: B904

        self.lang_code_out = self.language.split("_")[0]

        # JVSCorpus の出力形式に合わせてサンプル情報を事前に生成
        self.samples = []
        for _, row in self.metadata.iterrows():
            speaker = str(row["speaker_id"])
            wav_name = row["wav_name"]
            wav_path = self.audio_dir / wav_name

            # 互換性のため、ファイルの存在を確認してから追加する
            if wav_path.is_file():
                self.samples.append(
                    {
                        "wav_path": str(wav_path),
                        "speaker": speaker,
                        "clean_text": row["raw_text"],
                        "basename": f"{self.subset}_{speaker}_{wav_path.stem}",
                        "lang_code": self.lang_code_out,
                    }
                )

    def __getitem__(self, index: int) -> dict[str, str]:
        """
        単一のサンプルデータを辞書形式で返します。
        この形式は JVSCorpus の __getitem__ の出力と互換性があります。
        注意: JVSCorpus の実装と同様に、ここでは音声はロードせず、パスのみを提供します。
        """
        return self.samples[index]

    def __len__(self) -> int:
        """データセットの総サンプル数を返します。"""
        return len(self.samples)

    @property
    def speaker_dict(self) -> dict[str, int]:
        """一意の話者IDと整数のインデックスをマッピングした辞書を返します。"""
        speakers = sorted(self.metadata["speaker_id"].astype(str).unique())
        return {speaker: idx for idx, speaker in enumerate(speakers)}

    @property
    def lang_code(self) -> str:
        """データセットの言語コードを返します。"""
        return self.lang_code_out
