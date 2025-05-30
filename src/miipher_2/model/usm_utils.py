"""
Utilities for Universal Speech Model (USM) integration with Miipher-2.
"""

from typing import Any

import torch
from torch import nn

from .miipher import Miipher2


def load_usm_model(
    model_name: str = "google/usm-base",
    device: str = "cpu",
    *,
    trust_remote_code: bool = True,
) -> torch.nn.Module:
    """
    Load Universal Speech Model (USM) from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        trust_remote_code: Whether to trust remote code

    Returns:
        Loaded USM model

    Raises:
        ImportError: If transformers library is not installed
    """
    try:
        # pylint: disable=import-outside-toplevel
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        model.to(device)
        model.eval()
        return model

    except ImportError as e:
        msg = "transformers library required for USM model loading. Install with: pip install transformers"
        raise ImportError(msg) from e


def create_miipher2_with_usm(
    usm_model_name: str = "google/usm-base",
    pa_hidden_dim: int = 1024,
    pa_input_output_dim: int = 1532,
    *,
    freeze_usm: bool = True,
    device: str = "cpu",
) -> Miipher2:
    """
    Create Miipher2 model with pre-loaded USM encoder.

    Args:
        usm_model_name: HuggingFace USM model name
        pa_hidden_dim: Hidden dimension for Parallel Adapter
        pa_input_output_dim: Input/output dimension for Parallel Adapter
        freeze_usm: Whether to freeze USM parameters
        device: Device to create model on

    Returns:
        Miipher2 model with USM encoder
    """
    # Load USM model
    usm_model = load_usm_model(usm_model_name, device)

    # Create Miipher2 model
    config = {
        "usm_model": usm_model,
        "pa_hidden_dim": pa_hidden_dim,
        "pa_input_output_dim": pa_input_output_dim,
        "freeze_usm": freeze_usm,
    }

    model = Miipher2(**config)
    model.to(device)

    return model


def extract_usm_features(
    model: torch.nn.Module,
    audio: torch.Tensor,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Extract features from audio using USM model.

    Args:
        model: USM model
        audio: Audio tensor (B, T) or (T,)
        sample_rate: Sample rate of audio

    Returns:
        Extracted features (B, T, D)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add batch dimension

    # Ensure audio is in the right format for USM
    with torch.no_grad():
        features = model(audio)

    # Handle different output formats
    if hasattr(features, "last_hidden_state"):
        features = features.last_hidden_state
    elif isinstance(features, tuple):
        features = features[0]

    return features


def save_miipher2_checkpoint(
    model: Miipher2,
    checkpoint_path: str,
    *,
    include_usm: bool = False,
    epoch: int | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    """
    Save Miipher2 model checkpoint.

    Args:
        model: Miipher2 model to save
        checkpoint_path: Path to save checkpoint
        include_usm: Whether to include USM weights in checkpoint
        epoch: Training epoch number
        optimizer_state: Optimizer state dict
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config() if hasattr(model, "get_config") else {},
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state

    # Optionally exclude USM weights to reduce checkpoint size
    if not include_usm:
        model_state = checkpoint["model_state_dict"]
        filtered_state = {k: v for k, v in model_state.items() if not k.startswith("usm_model.")}
        checkpoint["model_state_dict"] = filtered_state

    torch.save(checkpoint, checkpoint_path)


def load_miipher2_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    *,
    load_usm: bool = True,
) -> tuple[Miipher2, dict[str, Any]]:
    """
    Load Miipher2 model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        load_usm: Whether to load USM weights

    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model from config
    model_config = checkpoint.get("model_config", {})
    model = Miipher2(**model_config)

    # Load state dict
    if load_usm:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Load only non-USM weights
        state_dict = checkpoint["model_state_dict"]
        filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("usm_model.")}
        model.load_state_dict(filtered_state, strict=False)

    model.to(device)

    # Return model and checkpoint info
    checkpoint_info = {
        "epoch": checkpoint.get("epoch", 0),
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
    }

    return model, checkpoint_info
