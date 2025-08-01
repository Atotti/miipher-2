from collections.abc import Iterator

import torch
import torchaudio
import webdataset as wds
from braceexpand import braceexpand
from torch.utils.data import IterableDataset


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """音声テンソルが必ず [channels, length] の2次元になるように保証する"""
    if tensor.dim() == 1:
        # テンソルが1次元の場合、チャンネル次元を追加する
        return tensor.unsqueeze(0)
    return tensor


class AdapterDataset(IterableDataset):
    """Adapter学習用: 全て16kHzに変換する"""

    def __init__(self, pattern: str | list[str], shuffle: int = 1000) -> None:
        # 複数のパターンに対応
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern
        
        # ブレース展開を適用
        expanded_patterns = []
        for p in patterns:
            expanded_patterns.extend(list(braceexpand(p)))

        self.dataset = (
            wds.WebDataset(
                expanded_patterns,
                resampled=True,
                shardshuffle=True,
            )
            .shuffle(shuffle)
            .decode(wds.torch_audio)
        )
        self.target_sr = 16000

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for sample in self.dataset:
            clean_wav, clean_sr = sample["speech.wav"]
            noisy_wav, noisy_sr = sample["degraded_speech.wav"]

            # ロードした直後に次元数を2Dに統一する
            clean_wav = _ensure_2d(clean_wav)
            noisy_wav = _ensure_2d(noisy_wav)

            # それぞれの正しいsrを使って16kHzにリサンプリング
            clean_16k = torchaudio.functional.resample(clean_wav, orig_freq=clean_sr, new_freq=self.target_sr)
            noisy_16k = torchaudio.functional.resample(noisy_wav, orig_freq=noisy_sr, new_freq=self.target_sr)

            # .mean(0, keepdim=True)はステレオ音声をモノラルに変換する安全策として残しておく
            yield noisy_16k.mean(0, keepdim=True), clean_16k.mean(0, keepdim=True)


class VocoderDataset(IterableDataset):
    """Vocoder学習用: 劣化音声は16kHz、クリーン音声は22.05kHzで出力

    Args:
        IterableDataset (_type_): _description_
    """

    def __init__(self, pattern: str | list[str], shuffle: int = 1000) -> None:
        # 複数のパターンに対応
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern
        
        # ブレース展開を適用
        expanded_patterns = []
        for p in patterns:
            expanded_patterns.extend(list(braceexpand(p)))

        self.dataset = wds.WebDataset(expanded_patterns, resampled=True, shardshuffle=True).shuffle(shuffle).decode(wds.torch_audio)
        self.input_sr = 16000
        self.target_sr = 22050

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for sample in self.dataset:
            clean_wav, clean_sr = sample["speech.wav"]
            noisy_wav, noisy_sr = sample["degraded_speech.wav"]

            # ロードした直後に次元数を2Dに統一する
            clean_wav = _ensure_2d(clean_wav)
            noisy_wav = _ensure_2d(noisy_wav)

            # 劣化音声はHuBERTに入力するため16kHzにリサンプリング
            noisy_16k = torchaudio.functional.resample(noisy_wav, orig_freq=noisy_sr, new_freq=self.input_sr)

            # クリーン音声は教師信号なので22.05kHzのまま
            if clean_sr != self.target_sr:
                clean_22k = torchaudio.functional.resample(clean_wav, orig_freq=clean_sr, new_freq=self.target_sr)
            else:
                clean_22k = clean_wav

            # .mean(0, keepdim=True)はステレオ音声をモノラルに変換する安全策として残しておく
            yield noisy_16k.mean(0, keepdim=True), clean_22k.mean(0, keepdim=True)


class CleanVocoderDataset(IterableDataset):
    """Vocoder事前学習用: クリーン音声を16kHzと22.05kHzの両方で出力"""

    def __init__(self, pattern: str | list[str], shuffle: int = 1000) -> None:
        # 複数のパターンに対応
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern
        
        # ブレース展開を適用
        expanded_patterns = []
        for p in patterns:
            expanded_patterns.extend(list(braceexpand(p)))

        self.dataset = wds.WebDataset(expanded_patterns, resampled=True, shardshuffle=True).shuffle(shuffle).decode(wds.torch_audio)
        self.input_sr = 16000
        self.target_sr = 22050

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for sample in self.dataset:
            # noisy_speech.wav の代わりに speech.wav を使う
            clean_wav, clean_sr = sample["speech.wav"]

            # ロードした直後に次元数を2Dに統一する
            clean_wav = _ensure_2d(clean_wav)

            # HuBERTに入力するため16kHzにリサンプリング
            clean_16k = torchaudio.functional.resample(clean_wav, orig_freq=clean_sr, new_freq=self.input_sr)

            # 教師信号なので22.05kHzのまま
            if clean_sr != self.target_sr:
                clean_22k = torchaudio.functional.resample(clean_wav, orig_freq=clean_sr, new_freq=self.target_sr)
            else:
                clean_22k = clean_wav

            # .mean(0, keepdim=True)はステレオ音声をモノラルに変換する安全策
            yield clean_16k.mean(0, keepdim=True), clean_22k.mean(0, keepdim=True)
