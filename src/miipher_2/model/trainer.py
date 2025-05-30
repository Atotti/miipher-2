"""
Miipher2 Training and Fine-tuning implementation
Following the paper's methodology for speech restoration with sequential training
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn, optim
from torch.utils.data import DataLoader

from .miipher import Miipher2
from .speechbrain_utils import create_hifigan_loss


class Miipher2Trainer:
    """
    Trainer for Miipher2 model with sequential training stages

    Stage 1: PA training (800k steps)
    Stage 2: Vocoder fine-tuning (675k steps)
    """

    def __init__(
        self,
        model: Miipher2,
        vocoder: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.vocoder = vocoder
        self.device = device
        self.grad_clip_norm = grad_clip_norm

        # Move models to device
        self.model.to(device)
        self.vocoder.to(device)

        # Stage tracking (2-stage training with pre-trained vocoder)
        self.training_stage = "PA"  # "PA", "vocoder_finetune"
        self.stage_steps = {"PA": 800000, "vocoder_finetune": 675000}
        self.current_stage_step = 0

        # Get PA parameters as list
        pa_params = list(self.model.get_pa_parameters())

        # Separate optimizers for different stages
        self.pa_optimizer = torch.optim.AdamW(pa_params, lr=learning_rate, weight_decay=weight_decay)
        self.vocoder_optimizer = torch.optim.AdamW(
            self.vocoder.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Current optimizer
        self.optimizer = self.pa_optimizer

        # Stage-specific schedulers
        self.pa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.pa_optimizer, T_max=self.stage_steps["PA"])
        self.vocoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.vocoder_optimizer, T_max=self.stage_steps["vocoder_finetune"]
        )
        self.scheduler = self.pa_scheduler

        # Initialize loss functions
        self.loss_functions = create_hifigan_loss()
        for loss_fn in self.loss_functions.values():
            loss_fn.to(device)

    def switch_training_stage(self) -> None:
        """Switch to the next training stage."""
        if self.training_stage == "PA":
            print(f"Switching from PA training to vocoder fine-tuning at step {self.current_stage_step}")
            self.training_stage = "vocoder_finetune"
            self.current_stage_step = 0

            # Update optimizer to include both PA and vocoder parameters
            pa_params = list(self.model.get_pa_parameters())
            all_params = pa_params + list(self.vocoder.parameters())

            self.optimizer = torch.optim.AdamW(
                all_params, lr=self.pa_optimizer.param_groups[0]["lr"], weight_decay=1e-5
            )
            self.scheduler = self.vocoder_scheduler

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor, Any]) -> dict[str, float]:
        """Execute one training step based on current stage."""
        if self.training_stage == "PA":
            return self._train_pa_step(batch)
        if self.training_stage == "vocoder_finetune":
            return self._train_vocoder_finetune_step(batch)
        raise ValueError(f"Unknown training stage: {self.training_stage}")

    def _train_pa_step(self, batch: tuple[torch.Tensor, torch.Tensor, Any]) -> dict[str, float]:
        """Stage 1: Train Parallel Adapters only."""
        self.optimizer.zero_grad()

        noisy_audio, clean_audio, text = batch
        noisy_audio = noisy_audio.to(self.device)
        clean_audio = clean_audio.to(self.device)

        # Extract clean USM features (target)
        with torch.no_grad():
            clean_features = self.model.extract_clean_features(clean_audio)

        # Predict clean features from noisy audio using PA
        predicted_features = self.model.extract_clean_features(noisy_audio)

        # PA loss: L1 + L2 + spectral convergence
        loss = self._compute_pa_loss(predicted_features, clean_features)

        loss.backward()
        pa_params = list(self.model.get_pa_parameters())
        torch.nn.utils.clip_grad_norm_(pa_params, self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        self.current_stage_step += 1
        if self.current_stage_step >= self.stage_steps["PA"]:
            self.switch_training_stage()

        return {"stage": "PA", "pa_loss": loss.item()}

    def _train_vocoder_finetune_step(self, batch: tuple[torch.Tensor, torch.Tensor, Any]) -> dict[str, float]:
        """Stage 2: Fine-tune vocoder with PA-predicted features."""
        self.optimizer.zero_grad()

        noisy_audio, clean_audio, text = batch
        noisy_audio = noisy_audio.to(self.device)
        clean_audio = clean_audio.to(self.device)

        # Get PA-predicted features
        predicted_features = self.model.extract_clean_features(noisy_audio)

        # Generate audio from predicted features
        generated_audio = self.vocoder(predicted_features)

        # Combined losses
        mel_loss = self._compute_mel_loss(generated_audio, clean_audio)
        feature_loss = self._compute_feature_matching_loss(generated_audio, clean_audio)
        l1_loss = F.l1_loss(generated_audio, clean_audio)

        total_loss = mel_loss + 0.5 * feature_loss + 0.1 * l1_loss

        total_loss.backward()
        pa_params = list(self.model.get_pa_parameters())
        all_params = pa_params + list(self.vocoder.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        self.current_stage_step += 1

        return {
            "stage": "vocoder_finetune",
            "mel_loss": mel_loss.item(),
            "feature_loss": feature_loss.item(),
            "l1_loss": l1_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _compute_pa_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute PA loss: L1 + L2 + spectral convergence."""
        l1_loss = F.l1_loss(predicted, target)
        l2_loss = F.mse_loss(predicted, target)

        # Spectral convergence loss
        pred_norm = torch.norm(predicted, dim=-1, keepdim=True)
        target_norm = torch.norm(target, dim=-1, keepdim=True)
        spectral_loss = F.mse_loss(pred_norm, target_norm) / torch.mean(target_norm**2)

        return l1_loss + l2_loss + spectral_loss

    def _compute_mel_loss(self, generated_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """Compute mel-spectrogram loss."""
        # Use HiFi-GAN mel loss if available
        if "mel_loss" in self.loss_functions:
            return self.loss_functions["mel_loss"](generated_audio, target_audio)
        # Fallback to L1 loss
        return F.l1_loss(generated_audio, target_audio)

    def _compute_feature_matching_loss(self, generated_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """Compute feature matching loss."""
        # Use HiFi-GAN feature matching loss if available
        if "feature_loss" in self.loss_functions:
            return self.loss_functions["feature_loss"](generated_audio, target_audio)
        # Fallback to L1 loss
        return F.l1_loss(generated_audio, target_audio)

    def _compute_vocoder_loss(
        self, generated_audio: torch.Tensor, target_audio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute combined vocoder loss."""
        mel_loss = self._compute_mel_loss(generated_audio, target_audio)
        feature_loss = self._compute_feature_matching_loss(generated_audio, target_audio)
        l1_loss = F.l1_loss(generated_audio, target_audio)

        return {"mel_loss": mel_loss, "feature_loss": feature_loss, "l1_loss": l1_loss}

    def validate(self, val_dataloader: DataLoader) -> float:
        """Validation loop."""
        self.model.eval()
        self.vocoder.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    if self.training_stage == "PA":
                        # PA validation
                        noisy_audio, clean_audio, _ = batch
                        noisy_audio = noisy_audio.to(self.device)
                        clean_audio = clean_audio.to(self.device)

                        clean_features = self.model.extract_clean_features(clean_audio)
                        predicted_features = self.model.extract_clean_features(noisy_audio)
                        loss = self._compute_pa_loss(predicted_features, clean_features)
                        total_loss += loss.item()

                    else:
                        # Vocoder validation
                        noisy_audio, clean_audio, _ = batch
                        noisy_audio = noisy_audio.to(self.device)
                        clean_audio = clean_audio.to(self.device)

                        generated_audio = self.model(noisy_audio, use_vocoder=True)
                        loss_components = self._compute_vocoder_loss(generated_audio, clean_audio)
                        total_loss += sum(loss.item() for loss in loss_components.values())

                    num_batches += 1

                except Exception as e:
                    print(f"Validation batch failed: {e}")
                    continue

        self.model.train()
        self.vocoder.train()
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: str, epoch: int | None = None) -> None:
        """Save training checkpoint with stage information."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "vocoder_state_dict": self.vocoder.state_dict(),
            "pa_optimizer_state_dict": self.pa_optimizer.state_dict(),
            "vocoder_optimizer_state_dict": self.vocoder_optimizer.state_dict(),
            "pa_scheduler_state_dict": self.pa_scheduler.state_dict(),
            "vocoder_scheduler_state_dict": self.vocoder_scheduler.state_dict(),
            "training_stage": self.training_stage,
            "current_stage_step": self.current_stage_step,
            "epoch": epoch,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path} (Stage: {self.training_stage}, Step: {self.current_stage_step})")

    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.vocoder.load_state_dict(checkpoint["vocoder_state_dict"])
        self.pa_optimizer.load_state_dict(checkpoint["pa_optimizer_state_dict"])
        self.vocoder_optimizer.load_state_dict(checkpoint["vocoder_optimizer_state_dict"])
        self.pa_scheduler.load_state_dict(checkpoint["pa_scheduler_state_dict"])
        self.vocoder_scheduler.load_state_dict(checkpoint["vocoder_scheduler_state_dict"])

        self.training_stage = checkpoint.get("training_stage", "PA")
        self.current_stage_step = checkpoint.get("current_stage_step", 0)

        # Set current optimizer and scheduler based on stage
        if self.training_stage == "PA":
            self.optimizer = self.pa_optimizer
            self.scheduler = self.pa_scheduler
        elif self.training_stage == "vocoder_finetune":
            pa_params = list(self.model.get_pa_parameters())
            all_params = pa_params + list(self.vocoder.parameters())
            self.optimizer = torch.optim.AdamW(
                all_params, lr=self.vocoder_optimizer.param_groups[0]["lr"], weight_decay=1e-5
            )
            self.scheduler = self.vocoder_scheduler

        print(f"Checkpoint loaded: {path} (Stage: {self.training_stage}, Step: {self.current_stage_step})")
        return checkpoint.get("epoch", 0)

    def fine_tune(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 10,
        save_dir: str = "./checkpoints",
        validate_every: int = 1000,
        save_every: int = 5000,
    ) -> dict[str, list[float]]:
        """
        Fine-tune the model following the sequential training strategy

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            validate_every: Validation frequency (steps)
            save_every: Checkpoint saving frequency (steps)

        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        global_step = 0

        print(f"Starting sequential training with stage: {self.training_stage}")
        print(f"Target steps - PA: {self.stage_steps['PA']}, Vocoder: {self.stage_steps['vocoder_finetune']}")

        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_dataloader:
                # Execute training step based on current stage
                metrics = self.train_step(batch)

                if self.training_stage == "PA":
                    batch_loss = metrics["pa_loss"]
                else:
                    batch_loss = metrics["total_loss"]

                epoch_loss += batch_loss
                num_batches += 1
                global_step += 1

                # Logging
                if global_step % 100 == 0:
                    print(f"Step {global_step}, Stage: {self.training_stage}, Loss: {batch_loss:.6f}")

                # Validation
                if val_dataloader and global_step % validate_every == 0:
                    val_loss = self.validate(val_dataloader)
                    history["val_loss"].append(val_loss)
                    print(f"Validation Loss: {val_loss:.6f}")

                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pt")
                    self.save_checkpoint(checkpoint_path, epoch)

                # Check if all stages completed
                if (
                    self.training_stage == "vocoder_finetune"
                    and self.current_stage_step >= self.stage_steps["vocoder_finetune"]
                ):
                    print("All training stages completed!")
                    final_path = os.path.join(save_dir, "final_model.pt")
                    self.save_checkpoint(final_path, epoch)
                    return history

            avg_epoch_loss = epoch_loss / num_batches
            history["train_loss"].append(avg_epoch_loss)
            print(f"Epoch {epoch + 1} completed, Avg Loss: {avg_epoch_loss:.6f}")

        print("Training completed!")
        return history


def create_trainer(
    model: Miipher2, vocoder: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu", **kwargs: Any
) -> Miipher2Trainer:
    """
    Create a Miipher2 trainer instance

    Args:
        model: Miipher2 model
        vocoder: Vocoder model
        device: Device to use for training
        **kwargs: Additional trainer arguments

    Returns:
        Miipher2Trainer instance
    """
    return Miipher2Trainer(model, vocoder, device, **kwargs)
