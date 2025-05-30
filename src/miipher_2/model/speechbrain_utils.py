"""
Utilities for integrating SpeechBrain models, particularly the HiFiGAN vocoder.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn


class SpeechBrainHiFiGAN(nn.Module):
    """
    Wrapper for SpeechBrain's HiFiGAN model for vocoding.
    """

    def __init__(self, model_id: str = "speechbrain/hifigan-hubert-k1000-LibriTTS", device: str = "cpu") -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.generator = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the SpeechBrain HiFiGAN model."""
        try:
            # Try to import speechbrain
            try:
                # pylint: disable=import-outside-toplevel
                from speechbrain.inference.vocoders import HiFiGAN

                self.generator = HiFiGAN.from_hparams(source=self.model_id)
                self.generator.to(self.device)
                self.generator.eval()

            except ImportError:
                warnings.warn(
                    "SpeechBrain is not installed. Using fallback generator. Install with: pip install speechbrain",
                    stacklevel=2,
                )
                self._create_fallback_generator()
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to load {self.model_id}: {e}"
            warnings.warn(msg, stacklevel=2)
            # Fallback to basic generator
            self._create_fallback_generator()

    def _create_fallback_generator(self) -> None:
        """Create a basic fallback vocoder if SpeechBrain is not available."""
        # This is a simple fallback - in practice you might want something more sophisticated

        class BasicVocoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Simple conv layers for basic vocoding
                self.conv_layers = nn.Sequential(
                    nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(128, 1, kernel_size=4, stride=2, padding=1),
                    nn.Tanh(),
                )

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                # Simple upsampling
                if features.dim() == 3:  # noqa: PLR2004
                    features = features.transpose(1, 2)  # (B, T, D) -> (B, D, T)
                return self.conv_layers(features)

        self.generator = BasicVocoder()
        self.generator.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from features.

        Args:
            features: Input features (B, T, D) or (B, D, T)

        Returns:
            Generated waveform (B, T)
        """
        if self.generator is None:
            msg = "Generator not initialized"
            raise RuntimeError(msg)

        # Ensure features are in the right format (B, D, T)
        if features.dim() == 3 and features.size(-1) != features.size(1):  # noqa: PLR2004
            features = features.transpose(1, 2)

        # Generate waveform
        with torch.no_grad():
            if hasattr(self.generator, "decode_batch"):
                # SpeechBrain interface
                waveform = self.generator.decode_batch(features)
                if isinstance(waveform, list):
                    waveform = waveform[0]  # Get first batch item if list
            else:
                # Use generator directly
                waveform = self.generator(features)
                if waveform.dim() == 3:  # noqa: PLR2004
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
        """
        Mel-spectrogram reconstruction loss for training vocoders.
        """

        class MelTransform(nn.Module):
            """Mel-spectrogram reconstruction loss"""

            def __init__(
                self, sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 80
            ) -> None:
                super().__init__()
                self.sample_rate = sample_rate
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.n_mels = n_mels

            def forward(self, pred_waveform: torch.Tensor, target_waveform: torch.Tensor) -> torch.Tensor:
                """
                Calculate mel-spectrogram loss between predicted and target waveforms.

                Args:
                    pred_waveform: Predicted waveform (B, T)
                    target_waveform: Target waveform (B, T)

                Returns:
                    Mel-spectrogram L1 loss
                """
                pred_mel = self._audio_to_mel(pred_waveform)
                target_mel = self._audio_to_mel(target_waveform)

                return torch.nn.functional.l1_loss(pred_mel, target_mel)

            def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
                # Convert audio to mel spectrogram
                from torchaudio import transforms

                mel_transform = transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
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
