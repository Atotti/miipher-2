# ------------------------------------------------------------
#  miipher_2/data/dataloader.py
# ------------------------------------------------------------
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from miipher_2.utils.audio import SR, add_noise, add_reverb, codec


class CleanNoisyDataset(Dataset):
    """
    戻り値: noisy (B,1,T), clean (B,1,T)
    """

    def __init__(self, wav_files: list[str | Path]) -> None:
        self.wav_files = [Path(f) for f in wav_files]

    def __len__(self) -> int:
        return len(self.wav_files)

    def _degrade(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            wav = add_reverb(wav, rt60=random.uniform(0.2, 0.5))
        wav = add_noise(wav, snr_db=random.uniform(5, 30))
        if random.random() < 0.8:
            wav = codec(wav, random.choice(["mp3", "opus", "vorbis", "alaw", "amr"]))
        return wav

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.wav_files[idx])
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(0, keepdim=True)  # mono
        noisy = self._degrade(wav.clone())
        return noisy, wav  # (1,T), (1,T)
