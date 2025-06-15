import torchaudio
import webdataset as wds
from torch.utils.data import IterableDataset


# Adapter学習用: 全て16kHzに変換する
class AdapterDataset(IterableDataset):
    def __init__(self, pattern: str, shuffle: int = 1000) -> None:
        self.dataset = wds.WebDataset(pattern, resampled=True).shuffle(shuffle).decode(wds.torch_audio)
        self.target_sr = 16000

    def __iter__(self):
        for sample in self.dataset:
            # webdatasetからロードされる音声は22.05kHz
            clean_22k, sr = sample["speech.wav"]
            noisy_22k, _ = sample["degraded_speech.wav"]

            # 両方とも16kHzにリサンプリング
            clean_16k = torchaudio.functional.resample(clean_22k, orig_freq=sr, new_freq=self.target_sr)
            noisy_16k = torchaudio.functional.resample(noisy_22k, orig_freq=sr, new_freq=self.target_sr)

            yield noisy_16k.mean(0, keepdim=True), clean_16k.mean(0, keepdim=True)


# Vocoder学習用: 劣化音声は16kHz、クリーン音声は22.05kHzで出力
class VocoderDataset(IterableDataset):
    def __init__(self, pattern: str, shuffle: int = 1000) -> None:
        self.dataset = wds.WebDataset(pattern, resampled=True).shuffle(shuffle).decode(wds.torch_audio)
        self.input_sr = 16000
        self.target_sr = 22050

    def __iter__(self):
        for sample in self.dataset:
            # webdatasetからロードされる音声は22.05kHz
            clean_22k, sr = sample["speech.wav"]
            noisy_22k, _ = sample["degraded_speech.wav"]

            # 劣化音声はmHuBERTに入力するため16kHzにリサンプリング
            noisy_16k = torchaudio.functional.resample(noisy_22k, orig_freq=sr, new_freq=self.input_sr)

            # クリーン音声は教師信号なので22.05kHzのまま
            # サンプリングレートが完全一致しない場合があるので、念のためリサンプリング
            if sr != self.target_sr:
                clean_22k = torchaudio.functional.resample(clean_22k, orig_freq=sr, new_freq=self.target_sr)

            yield noisy_16k.mean(0, keepdim=True), clean_22k.mean(0, keepdim=True)
