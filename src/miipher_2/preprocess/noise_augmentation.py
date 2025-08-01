import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
from omegaconf import DictConfig
from tqdm import tqdm


def align_waveform(wav1: torch.Tensor, wav2: torch.Tensor) -> tuple[int, torch.Tensor]:
    assert wav2.size(1) >= wav1.size(1)
    diff = wav2.size(1) - wav1.size(1)
    min_mse = float("inf")
    best_i = -1

    for i in range(diff):
        segment = wav2[:, i : i + wav1.size(1)]
        mse: float = torch.mean((wav1 - segment).pow(2)).item()
        if mse < min_mse:
            min_mse = mse
            best_i = i

    return best_i, wav2[:, best_i : best_i + wav1.size(1)]


class DegradationApplier:
    def __init__(self, cfg: DictConfig) -> None:
        self.format_encoding_pairs = cfg.format_encoding_pairs
        self.reverb_conditions = cfg.reverb_conditions
        self.background_noise = cfg.background_noise
        self.cfg = cfg
        self.rirs: list[torch.Tensor] = []
        self.prepare_rir(cfg.n_rirs)
        self.noise_audio_paths = []
        for root, pattern in self.cfg.background_noise.patterns:
            self.noise_audio_paths.extend(list(Path(root).glob(pattern)))

    def applyCodec(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if len(self.format_encoding_pairs) == 0:
            return waveform
        param: dict = random.choice(self.format_encoding_pairs)
        audio_format: str = param["format"]
        compression: int | None = param.get("compression")
        codec_config = torchaudio.io.CodecConfig(compression_level=compression) if compression else None
        eff = torchaudio.io.AudioEffector(format=audio_format, codec_config=codec_config)
        wav_tc = waveform.transpose(0, 1)
        aug_tc = eff.apply(wav_tc, sample_rate)
        augmented = aug_tc.transpose(0, 1).contiguous()
        # mp3 encoding may increase the length of the waveform by zero-padding
        if waveform.size(1) != augmented.size(1):
            best_idx, augmented = align_waveform(waveform, augmented)
        return augmented.float()

    def applyReverb(self, waveform: torch.Tensor) -> torch.Tensor:
        if len(self.rirs) == 0:
            raise RuntimeError
        rir = random.choice(self.rirs)
        augmented = torchaudio.functional.fftconvolve(waveform, rir)
        # rir convolution may increase the length of the waveform
        if waveform.size(1) != augmented.size(1):
            augmented = augmented[:, : waveform.size(1)]
        return augmented.float()

    def prepare_rir(self, n_rirs: int) -> None:
        for _ in tqdm(range(n_rirs)):
            xy_min_max = self.reverb_conditions.room_xy
            z_min_max = self.reverb_conditions.room_z
            x = random.uniform(xy_min_max.min, xy_min_max.max)
            y = random.uniform(xy_min_max.min, xy_min_max.max)
            z = random.uniform(z_min_max.min, z_min_max.max)
            corners = np.array([[0, 0], [0, y], [x, y], [x, 0]]).T
            room = pra.Room.from_corners(corners, **self.reverb_conditions.room_params)
            room.extrude(z)
            room.add_source(self.cfg.reverb_conditions.source_pos)
            room.add_microphone(self.cfg.reverb_conditions.mic_pos)

            room.compute_rir()
            rir = torch.tensor(np.array(room.rir[0]))
            rir = rir / rir.norm(p=2)
            self.rirs.append(rir)

    def applyBackgroundNoise(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        snr_max, snr_min = self.background_noise.snr.max, self.background_noise.snr.min
        snr = random.uniform(snr_min, snr_max)

        noise_path = random.choice(self.noise_audio_paths)
        noise, noise_sr = torchaudio.load(noise_path)
        noise /= noise.norm(p=2)
        if noise.size(0) > 1:
            noise = noise[0].unsqueeze(0)
        noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)
        if not noise.size(1) < waveform.size(1):
            start_idx = random.randint(0, noise.size(1) - waveform.size(1))
            end_idx = start_idx + waveform.size(1)
            noise = noise[:, start_idx:end_idx]
        else:
            noise = noise.repeat(1, waveform.size(1) // noise.size(1) + 1)[:, : waveform.size(1)]
        if noise.abs().max() > 0:
            augmented = torchaudio.functional.add_noise(waveform=waveform, noise=noise, snr=torch.tensor([snr]))
        else:
            augmented = waveform
        return augmented

    def process(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        org_len = waveform.size(1)
        waveform = self.applyBackgroundNoise(waveform, sample_rate)
        if random.random() > self.cfg.reverb_conditions.p:
            waveform = self.applyReverb(waveform)
        waveform = self.applyCodec(waveform, sample_rate)
        assert org_len == waveform.size(1), f"{org_len}, {waveform.size(1)}"
        return waveform.squeeze()

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.process(waveform, sample_rate)
