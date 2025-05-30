"""
Utilities for loading USM model and building Miipher-2
"""

import warnings
from typing import Any, Dict, Optional

import torch
from torch import nn

from .miipher import Miipher2


def load_usm_model(model_path: str | None = None, device: str = "cpu") -> nn.Module:
    """
    Load Universal Speech Model (USM) - using rinna/hubert-large

    Args:
        model_path: Path to USM model checkpoint. If None, will load rinna/hubert-large from HuggingFace
        device: Device to load the model on

    Returns:
        USM model instance
    """
    try:
        # Try to load from HuggingFace transformers
        from transformers import HubertModel

        if model_path is None:
            # Use rinna/hubert-large as the USM model
            model = HubertModel.from_pretrained("rinna/hubert-large")
        else:
            model = torch.load(model_path, map_location=device)

        model.to(device)
        model.eval()
        return model

    except ImportError:
        raise ImportError("transformers library required for USM model loading. Install with: pip install transformers")


def create_miipher2_model(
    usm_model: nn.Module,
    usm_layer_idx: int = 13,
    pa_hidden_dim: int = 1024,
    pa_input_output_dim: int = 1532,
    freeze_usm: bool = True,
    device: str = "cpu",
) -> Miipher2:
    """
    Create Miipher-2 model with given USM model

    Args:
        usm_model: Pre-trained USM model
        usm_layer_idx: Layer index to extract features from (1-indexed)
        pa_hidden_dim: Hidden dimension for parallel adapters
        pa_input_output_dim: Input/output dimension for parallel adapters
        freeze_usm: Whether to freeze USM parameters
        device: Device to create the model on

    Returns:
        Miipher2 model instance
    """
    model = Miipher2(
        usm_model=usm_model,
        usm_layer_idx=usm_layer_idx,
        pa_hidden_dim=pa_hidden_dim,
        pa_input_output_dim=pa_input_output_dim,
        freeze_usm=freeze_usm,
    )

    model.to(device)
    return model


def load_miipher2_checkpoint(checkpoint_path: str, usm_model: nn.Module, device: str = "cpu") -> Miipher2:
    """
    Load Miipher-2 model from checkpoint

    Args:
        checkpoint_path: Path to Miipher-2 checkpoint
        usm_model: Pre-trained USM model
        device: Device to load the model on

    Returns:
        Loaded Miipher2 model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    model_config = checkpoint.get("model_config", {})

    # Create model
    model = create_miipher2_model(
        usm_model=usm_model,
        usm_layer_idx=model_config.get("usm_layer_idx", 13),
        pa_hidden_dim=model_config.get("pa_hidden_dim", 1024),
        pa_input_output_dim=model_config.get("pa_input_output_dim", 1532),
        freeze_usm=model_config.get("freeze_usm", True),
        device=device,
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def save_miipher2_checkpoint(
    model: Miipher2,
    checkpoint_path: str,
    epoch: int = 0,
    optimizer_state: dict[str, Any] | None = None,
    additional_info: dict[str, Any] | None = None,
):
    """
    Save Miipher-2 model checkpoint

    Args:
        model: Miipher2 model to save
        checkpoint_path: Path to save checkpoint
        epoch: Training epoch
        optimizer_state: Optimizer state dict
        additional_info: Additional information to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "usm_layer_idx": model.usm_layer_idx,
            "pa_hidden_dim": model.parallel_adapters[0].ffn[0].in_features,
            "pa_input_output_dim": model.parallel_adapters[0].ffn[-1].out_features,
            "freeze_usm": not any(p.requires_grad for p in model.usm_model.parameters()),
        },
    }

    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state

    if additional_info is not None:
        checkpoint.update(additional_info)

    torch.save(checkpoint, checkpoint_path)


def get_model_info(model: Miipher2) -> dict[str, Any]:
    """
    Get information about Miipher-2 model

    Args:
        model: Miipher2 model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    usm_params = sum(p.numel() for p in model.usm_model.parameters())
    adapter_params = sum(p.numel() for p in model.parallel_adapters.parameters())
    wavefit_params = sum(p.numel() for p in model.wavefit.parameters())

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "usm_parameters": usm_params,
        "adapter_parameters": adapter_params,
        "wavefit_parameters": wavefit_params,
        "usm_frozen": not any(p.requires_grad for p in model.usm_model.parameters()),
        "usm_layer_index": model.usm_layer_idx,
        "num_adapters": len(model.parallel_adapters),
    }
