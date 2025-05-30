"""
Training module for Miipher-2 model.
This module provides utilities for training both PA and vocoder components.
"""

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .miipher import Miipher2


class Miipher2Trainer:
    """
    Trainer for Miipher-2 model with support for different training stages.

    Supports:
    - PA (Parallel Adapter) training
    - Vocoder fine-tuning
    """

    def __init__(
        self,
        model: Miipher2,
        vocoder: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        training_stage: str = "PA",
    ) -> None:
        self.model = model
        self.vocoder = vocoder
        self.device = device
        self.training_stage = training_stage
        self.optimizer: torch.optim.Optimizer

        # Move models to device
        self.model.to(device)
        self.vocoder.to(device)

        # Setup optimizers based on training stage
        self._setup_optimizers(learning_rate)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def _setup_optimizers(self, learning_rate: float) -> None:
        """Setup optimizers based on training stage."""
        if self.training_stage == "PA":
            # Only optimize PA parameters
            pa_params = []
            for name, param in self.model.named_parameters():
                if "parallel_adapter" in name:
                    pa_params.append(param)
                else:
                    param.requires_grad = False

            self.optimizer = torch.optim.AdamW(pa_params, lr=learning_rate)

        elif self.training_stage == "vocoder_finetune":
            # Optimize both PA and vocoder
            all_params = list(self.model.parameters()) + list(self.vocoder.parameters())
            self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate)

        else:
            # Default: optimize all model parameters
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor, Any]) -> dict[str, float]:
        """
        Single training step.

        Args:
            batch: Tuple of (noisy_features, clean_features, metadata)

        Returns:
            Dictionary of loss values
        """
        if self.training_stage == "PA":
            return self._train_pa_step(batch)
        if self.training_stage == "vocoder_finetune":
            return self._train_vocoder_finetune_step(batch)

        msg = f"Unknown training stage: {self.training_stage}"
        raise ValueError(msg)

    def _train_pa_step(self, batch: tuple[torch.Tensor, torch.Tensor, Any]) -> dict[str, float]:
        """Training step for PA (Parallel Adapter) stage."""
        noisy_features, clean_features, _ = batch

        # Move to device
        noisy_features = noisy_features.to(self.device)
        clean_features = clean_features.to(self.device)

        # Forward pass through model
        self.optimizer.zero_grad()

        # Get USM features and PA predictions
        usm_features = self.model.encode_features(noisy_features)
        pa_output = self.model.parallel_adapter(usm_features)

        # Loss: PA output should predict clean features
        pa_loss = self.mse_loss(pa_output, clean_features)

        # Backward pass
        pa_loss.backward()
        self.optimizer.step()

        return {"pa_loss": pa_loss.item(), "total_loss": pa_loss.item()}

    def _train_vocoder_finetune_step(self, batch: tuple[torch.Tensor, torch.Tensor, Any]) -> dict[str, float]:
        """Training step for vocoder fine-tuning stage."""
        noisy_features, clean_features, metadata = batch

        # Move to device
        noisy_features = noisy_features.to(self.device)
        clean_features = clean_features.to(self.device)

        self.optimizer.zero_grad()

        # Get target waveform from metadata (if available)
        target_waveform = None
        if hasattr(metadata, "waveform") and metadata.waveform is not None:
            target_waveform = metadata.waveform.to(self.device)

        # Forward pass: noisy -> clean features -> waveform
        usm_features = self.model.encode_features(noisy_features)
        pa_output = self.model.parallel_adapter(usm_features)

        # Generate waveform from clean features
        pred_waveform = self.vocoder(pa_output)

        # Losses
        pa_loss = self.mse_loss(pa_output, clean_features)

        total_loss = pa_loss

        # Add waveform loss if target is available
        if target_waveform is not None:
            # Ensure same length
            min_len = min(pred_waveform.size(-1), target_waveform.size(-1))
            pred_waveform_aligned = pred_waveform[..., :min_len]
            target_waveform_aligned = target_waveform[..., :min_len]

            waveform_loss = self.l1_loss(pred_waveform_aligned, target_waveform_aligned)
            total_loss = pa_loss + 0.1 * waveform_loss

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            return {
                "pa_loss": pa_loss.item(),
                "waveform_loss": waveform_loss.item(),
                "total_loss": total_loss.item(),
            }

        # Backward pass (PA loss only)
        total_loss.backward()
        self.optimizer.step()

        return {"pa_loss": pa_loss.item(), "total_loss": total_loss.item()}

    def validate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        """Validation loop."""
        self.model.eval()
        self.vocoder.eval()

        total_val_loss = 0.0
        total_pa_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    metrics = self.train_step(batch)
                    total_val_loss += metrics.get("total_loss", 0.0)
                    total_pa_loss += metrics.get("pa_loss", 0.0)
                    num_batches += 1

                except Exception as e:  # noqa: BLE001
                    print(f"Validation batch failed: {e}")
                    continue

        # Set back to training mode
        self.model.train()
        self.vocoder.train()

        if num_batches == 0:
            return {"val_loss": 0.0, "val_pa_loss": 0.0}

        return {
            "val_loss": total_val_loss / num_batches,
            "val_pa_loss": total_pa_loss / num_batches,
        }

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "vocoder_state_dict": self.vocoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stage": self.training_stage,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.vocoder.load_state_dict(checkpoint["vocoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    def train(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None = None,
        num_epochs: int = 100,
        save_dir: str = "checkpoints",
        save_every: int = 10,
        validate_every: int = 5,
    ) -> dict[str, list[float]]:
        """
        Train the model.

        Returns:
            Training history dictionary
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            self.vocoder.train()

            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                try:
                    # Training step
                    metrics = self.train_step(batch)

                    batch_loss = metrics["pa_loss"] if self.training_stage == "PA" else metrics["total_loss"]

                    epoch_loss += batch_loss
                    num_batches += 1
                    global_step += 1

                    # Log progress
                    if global_step % 100 == 0:
                        print(f"Epoch {epoch}, Step {global_step}, Loss: {batch_loss:.6f}")

                except Exception as e:  # noqa: BLE001
                    print(f"Training batch failed: {e}")
                    continue

                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_path = Path(save_dir) / f"checkpoint_step_{global_step}.pt"
                    self.save_checkpoint(str(checkpoint_path), epoch)

                # Check if training is complete
                if (self.training_stage == "PA" and global_step >= 10000) or (  # PA training: 10k steps
                    self.training_stage == "vocoder_finetune" and global_step >= 5000
                ):  # Vocoder: 5k steps
                    print("All training stages completed!")
                    final_path = Path(save_dir) / "final_model.pt"
                    self.save_checkpoint(str(final_path), epoch)
                    return history

            # Record epoch loss
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            history["train_loss"].append(avg_epoch_loss)

            # Validation
            if val_loader is not None and epoch % validate_every == 0:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                print(f"Epoch {epoch}: Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_metrics['val_loss']:.6f}")

        return history


def create_trainer(
    model: Miipher2, vocoder: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu", **kwargs: Any
) -> Miipher2Trainer:
    """
    Create a trainer instance with the given model and vocoder.

    Args:
        model: Miipher2 model instance
        vocoder: Vocoder model instance
        device: Device to run training on
        **kwargs: Additional arguments for trainer

    Returns:
        Configured trainer instance
    """
    return Miipher2Trainer(model=model, vocoder=vocoder, device=device, **kwargs)
