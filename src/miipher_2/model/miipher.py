from collections.abc import Iterator
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .modules import ParallelAdapter
from .speechbrain_utils import SpeechBrainHiFiGAN


class Miipher2(nn.Module):
    """
    Miipher-2: A Universal Speech Restoration Model

    Key improvements over Miipher:
    1. Uses frozen Universal Speech Model (USM) as feature extractor (rinna/hubert-large)
    2. Uses Parallel Adapters instead of Conformer-based feature cleaner
    3. Uses SpeechBrain HiFi-GAN vocoder for high-quality audio synthesis
    4. No conditioning on text or speaker ID required
    """

    def __init__(
        self,
        usm_model: nn.Module,
        usm_layer_idx: int = 13,
        pa_hidden_dim: int = 1024,
        pa_input_output_dim: int = 1024,  # Updated for HuBERT hidden size
        freeze_usm: bool = True,
        hifigan_model_id: str = "speechbrain/hifigan-hubert-k1000-LibriTTS",
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # Frozen USM feature extractor (HuBERT)
        self.usm_model = usm_model
        self.usm_layer_idx = usm_layer_idx

        if freeze_usm:
            for param in self.usm_model.parameters():
                param.requires_grad = False

        # Get number of layers from HuBERT encoder
        try:
            # For HuBERT model structure
            if hasattr(self.usm_model, "encoder") and hasattr(self.usm_model.encoder, "layers"):
                num_layers = len(self.usm_model.encoder.layers)
            else:
                # Fallback to reasonable default
                num_layers = 24  # Default for hubert-large
        except Exception:
            num_layers = 24

        # Parallel adapters for feature cleaning
        self.parallel_adapters = nn.ModuleList()

        # Add parallel adapters to each layer
        for _layer_idx in range(num_layers):
            adapter = ParallelAdapter(
                input_dim=pa_input_output_dim, hidden_dim=pa_hidden_dim, output_dim=pa_input_output_dim
            )
            self.parallel_adapters.append(adapter)

        # SpeechBrain HiFi-GAN vocoder
        self.hifigan = SpeechBrainHiFiGAN(model_id=hifigan_model_id, device=device)

    def get_pa_parameters(self) -> Iterator[nn.Parameter]:
        """Get Parallel Adapter parameters for stage-specific training."""
        for adapter in self.parallel_adapters:
            yield from adapter.parameters()

    def forward(
        self, noisy_waveform: torch.Tensor, attention_mask: torch.Tensor | None = None, use_vocoder: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of Miipher-2

        Args:
            noisy_waveform: Input noisy waveform (B, T)
            attention_mask: Optional attention mask for variable length sequences
            use_vocoder: Whether to use vocoder for final synthesis

        Returns:
            clean_waveform: Restored clean waveform (B, T) if use_vocoder=True
            clean_features: Predicted clean USM features (B, T', D) if use_vocoder=False
        """
        # Extract USM features with parallel adapters
        clean_features = self.extract_clean_features(noisy_waveform, attention_mask)

        if not use_vocoder:
            return clean_features

        # Synthesize clean waveform using SpeechBrain HiFi-GAN
        return self.hifigan(clean_features)

    def extract_clean_features(
        self, noisy_waveform: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Extract clean USM features using parallel adapters

        Args:
            noisy_waveform: Input noisy waveform (B, T)
            attention_mask: Optional attention mask

        Returns:
            clean_features: Predicted clean USM features (B, T', D)
        """
        # Extract features from HuBERT with parallel adapters
        clean_features = torch.empty(1)  # Initialize for return

        context_manager = torch.no_grad() if not self.training else torch.enable_grad()
        with context_manager:
            # Get HuBERT hidden states at all layers
            outputs = self.usm_model(noisy_waveform, output_hidden_states=True)
            hidden_states_list = outputs.hidden_states

            # Apply parallel adapters to each layer
            adapted_states = []
            for layer_idx, (hidden_state, adapter) in enumerate(
                zip(hidden_states_list, self.parallel_adapters, strict=False)
            ):
                # Apply parallel adapter and add residual connection
                adapter_output = adapter(hidden_state)
                adapted_state = hidden_state + adapter_output
                adapted_states.append(adapted_state)

                # Use output from specified layer for vocoder
                if layer_idx == self.usm_layer_idx - 1:  # 0-indexed
                    clean_features = adapted_state

        return clean_features

    def inference(self, noisy_waveform: torch.Tensor, chunk_length: int | None = None) -> torch.Tensor:
        """
        Efficient inference for long sequences with optional chunking

        Args:
            noisy_waveform: Input noisy waveform (B, T)
            chunk_length: Optional chunk length for processing long sequences

        Returns:
            clean_waveform: Restored clean waveform (B, T)
        """
        self.eval()
        with torch.no_grad():
            if chunk_length is None:
                return self.forward(noisy_waveform, use_vocoder=True)
            # Process in chunks for memory efficiency
            return self._chunked_inference(noisy_waveform, chunk_length)

    def _chunked_inference(self, noisy_waveform: torch.Tensor, chunk_length: int) -> torch.Tensor:
        """
        Process long sequences in chunks

        Args:
            noisy_waveform: Input noisy waveform (B, T)
            chunk_length: Length of each chunk

        Returns:
            clean_waveform: Restored clean waveform (B, T)
        """
        B, T = noisy_waveform.shape
        clean_chunks = []

        for start_idx in range(0, T, chunk_length):
            end_idx = min(start_idx + chunk_length, T)
            chunk = noisy_waveform[:, start_idx:end_idx]

            clean_chunk = self.forward(chunk, use_vocoder=True)
            clean_chunks.append(clean_chunk)

        return torch.cat(clean_chunks, dim=1)

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device


class Miipher2Loss(nn.Module):
    """
    Loss function for Miipher-2 training

    Combines L1, L2, and spectral convergence losses for feature prediction
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 1.0, sc_weight: float = 1.0) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sc_weight = sc_weight

    def forward(
        self,
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        feature_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute combined loss for feature prediction

        Args:
            predicted_features: Predicted clean features (B, T, D)
            target_features: Target clean features (B, T, D)
            feature_mask: Optional mask for variable length sequences

        Returns:
            total_loss: Combined loss value
        """
        if feature_mask is not None:
            # Apply mask to ignore padded regions
            predicted_features = predicted_features * feature_mask.unsqueeze(-1)
            target_features = target_features * feature_mask.unsqueeze(-1)

        # L1 loss
        l1_loss = F.l1_loss(predicted_features, target_features)

        # L2 loss
        l2_loss = F.mse_loss(predicted_features, target_features)

        # Spectral convergence loss
        sc_loss = self._spectral_convergence_loss(predicted_features, target_features)

        return self.l1_weight * l1_loss + self.l2_weight * l2_loss + self.sc_weight * sc_loss

    def _spectral_convergence_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral convergence loss

        Args:
            predicted: Predicted features (B, T, D)
            target: Target features (B, T, D)

        Returns:
            sc_loss: Spectral convergence loss
        """
        # Compute FFT
        pred_fft = torch.fft.fft(predicted, dim=1)
        target_fft = torch.fft.fft(target, dim=1)

        # Compute spectral convergence
        numerator = torch.norm(pred_fft - target_fft, p="fro")
        denominator = torch.norm(target_fft, p="fro")

        return numerator / (denominator + 1e-8)
