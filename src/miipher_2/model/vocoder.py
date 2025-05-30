import torch
from speechbrain.inference.vocoders import HIFIGAN
from torch import nn


class SpeechBrainHiFiGAN(nn.Module):
    """Thin wrapper around SpeechBrain's pretrained HiFi-GAN."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        super().__init__()
        self.vocoder: HIFIGAN = HIFIGAN.from_hparams(source=model_id, run_opts={"device": str(device)})

    def forward(self, mel_or_latent: torch.Tensor) -> torch.Tensor:  # (B, T, n_mel)
        return self.vocoder.decode_batch(mel_or_latent).squeeze(1)  # (B, T)
