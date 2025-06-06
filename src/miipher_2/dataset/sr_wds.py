import io

import torch
import torchaudio
import webdataset as wds
from torch.utils.data import IterableDataset


class SpeechRestoreWDS(IterableDataset):
    """
    yield = {
        "input_values":  Tensor [T]   (degraded waveform, float32, 16 kHz),
        "labels":        Tensor [T]   (clean waveform, same length)
    }
    """

    def __init__(self, tar_pattern: str, sr: int = 16_000, shuffle: bool = True) -> None:
        def decode_wav(bytes_):
            return torchaudio.load(io.BytesIO(bytes_))[0].squeeze(0)
        self.pipeline = (
            wds.WebDataset(tar_pattern, resampled=False).shuffle(10_000) if shuffle else wds.WebDataset(tar_pattern)
        )
        self.pipeline = (
            self.pipeline.decode(  # 拡張子ごとにデコーダを割り当て
                {
                    "wav": decode_wav,
                    "pth": torch.load,  # 使わないなら省略可
                }
            )
            .to_tuple("degraded_speech.wav", "speech.wav")  # 返す順序を固定
            .map_tuple(lambda x: x.float(), lambda y: y.float())  # dtype 統一
        )
        self.sr = sr

    def __iter__(self):
        for degraded, clean in self.pipeline:
            # → 必要ならここでリサンプリングや整形
            yield {
                "input_values": degraded,
                "labels": clean,
            }
