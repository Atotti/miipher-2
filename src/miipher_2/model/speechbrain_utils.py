"""
Utilities for SpeechBrain HiFi-GAN integration with Miipher-2
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn


class SpeechBrainHiFiGAN(nn.Module):
    """
    Wrapper for SpeechBrain HiFi-GAN vocoder
    """

    def __init__(self, model_id: str = "speechbrain/hifigan-hubert-k1000-LibriTTS", device: str = "cpu"):
        super().__init__()
        self.model_id = model_id
        self.device = device
        # Initialize attributes
        self.hifigan: Any | None = None
        self.generator: nn.Module | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load SpeechBrain HiFi-GAN model"""
        try:
            # Try importing speechbrain
            from speechbrain.pretrained import HIFIGAN

            # Load the pretrained HiFi-GAN model
            self.hifigan = HIFIGAN.from_hparams(source=self.model_id, run_opts={"device": self.device})

            # Extract the generator for direct use
            self.generator = self.hifigan.hifi_gan

        except ImportError:
            warnings.warn(
                "SpeechBrain is not installed. Using fallback generator. Install with: pip install speechbrain"
            )
            self._create_fallback_generator()
        except Exception as e:
            warnings.warn(f"Failed to load {self.model_id}: {e}")
            # Fallback to basic generator
            self._create_fallback_generator()

    def _create_fallback_generator(self) -> None:
        """Create a fallback generator if SpeechBrain model fails to load"""
        # Simple CNN-based generator as fallback
        self.generator = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(512, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from features

        Args:
            features: Input features (B, T, D) or (B, D, T)

        Returns:
            waveform: Generated waveform (B, T_wav)
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized")

        # Ensure features are in the right format (B, D, T)
        if features.dim() == 3 and features.size(-1) != features.size(1):
            features = features.transpose(1, 2)

        # Generate waveform
        if self.hifigan is not None and hasattr(self.hifigan, "decode_batch"):
            # Use SpeechBrain's decode method
            waveform = self.hifigan.decode_batch(features)
            if isinstance(waveform, list):
                waveform = torch.stack(waveform)
        else:
            # Use generator directly
            waveform = self.generator(features)
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)  # Remove channel dimension

        return waveform


def load_speechbrain_hifigan(
    model_id: str = "speechbrain/hifigan-hubert-k1000-LibriTTS", device: str = "cpu"
) -> SpeechBrainHiFiGAN:
    """
    Load SpeechBrain HiFi-GAN model

    Args:
        model_id: SpeechBrain model identifier
        device: Device to load the model on

    Returns:
        SpeechBrainHiFiGAN instance
    """
    return SpeechBrainHiFiGAN(model_id=model_id, device=device)


def create_hifigan_loss() -> dict[str, nn.Module]:
    """
    Create loss functions for HiFi-GAN training

    Returns:
        Dictionary containing loss functions
    """

    class MelSpectrogramLoss(nn.Module):
        """Mel-spectrogram reconstruction loss"""

        def __init__(self, sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 80):
            super().__init__()
            self.sample_rate = sample_rate
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.n_mels = n_mels

        def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
            # Compute mel spectrograms
            pred_mel = self._audio_to_mel(pred_audio)
            target_mel = self._audio_to_mel(target_audio)

            # L1 loss on mel spectrograms
            return nn.functional.l1_loss(pred_mel, target_mel)

        def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
            # Convert audio to mel spectrogram
            import torchaudio.transforms as T

            mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
            ).to(audio.device)

            return mel_transform(audio)

    class FeatureMatchingLoss(nn.Module):
        """Feature matching loss"""

        def forward(self, pred_features: list[torch.Tensor], target_features: list[torch.Tensor]) -> torch.Tensor:
            if not pred_features:
                return torch.tensor(0.0)

            loss = torch.tensor(0.0, device=pred_features[0].device)
            for pred_feat, target_feat in zip(pred_features, target_features, strict=False):
                loss += nn.functional.l1_loss(pred_feat, target_feat)
            return loss / len(pred_features)

    return {
        "mel_loss": MelSpectrogramLoss(),
        "feature_matching_loss": FeatureMatchingLoss(),
        "l1_loss": nn.L1Loss(),
        "mse_loss": nn.MSELoss(),
    }
