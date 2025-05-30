#!/usr/bin/env python3
"""
Miipher-2 Vocoder Fine-tuning Command

Usage:
    python -m cmd.train_vocoder
    python -m cmd.train_vocoder hydra.run.dir=outputs/vocoder_training
"""

import sys
from pathlib import Path

import hydra
import torch
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from example_finetune import AudioDataset
from miipher import Miipher2
from speechbrain_utils import load_speechbrain_vocoder
from trainer import Miipher2Trainer


@hydra.main(version_base=None, config_path="../configs/", config_name="train_vocoder")
def main(cfg: DictConfig) -> None:
    """Main function for vocoder fine-tuning."""

    # Set random seed
    seed_everything(cfg.get("seed", 172957))

    print("=== Miipher-2 Vocoder Fine-tuning ===")
    print(f"Config: {cfg}")

    # Setup device
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and vocoder
    print("Loading model and vocoder...")
    model = Miipher2(device=device)
    vocoder = load_speechbrain_vocoder(device=device)

    # Initialize trainer
    trainer = Miipher2Trainer(model=model, vocoder=vocoder, device=device, learning_rate=cfg.training.learning_rate)

    # Force vocoder fine-tuning stage
    trainer.training_stage = "vocoder_finetune"
    trainer.current_stage_step = 0
    print(f"Training stage: {trainer.training_stage}")

    # Load PA checkpoint if specified
    if cfg.get("pa_checkpoint"):
        print(f"Loading PA weights from: {cfg.pa_checkpoint}")
        trainer.load_checkpoint(cfg.pa_checkpoint)
        trainer.training_stage = "vocoder_finetune"  # Reset to vocoder stage
        trainer.current_stage_step = 0
    else:
        print("Warning: No PA checkpoint specified. Training from scratch.")

    # Setup datasets
    print("Setting up datasets...")
    train_dataset = setup_dataset(cfg.data, train=True)
    val_dataset = setup_dataset(cfg.data, train=False)

    # Debug mode
    if cfg.get("debug", False):
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(20, len(val_dataset))))
        print("Debug mode: Using reduced dataset")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    # Resume from vocoder checkpoint if specified
    if cfg.get("resume_checkpoint"):
        print(f"Resuming from checkpoint: {cfg.resume_checkpoint}")
        trainer.load_checkpoint(cfg.resume_checkpoint)

    # Training loop
    print("Starting vocoder fine-tuning...")
    run_vocoder_training(trainer, train_loader, val_loader, cfg)

    print("Vocoder fine-tuning completed!")


def setup_dataset(data_cfg: DictConfig, train: bool = True):
    """Setup dataset based on configuration."""

    if train:
        return AudioDataset(
            clean_dir=data_cfg.clean_dir,
            noisy_dir=data_cfg.noisy_dir,
            sample_rate=data_cfg.sample_rate,
            segment_length=data_cfg.get("segment_length", 4.0),
            snr_range=data_cfg.get("snr_range", [5, 30]),
            noise_types=data_cfg.get("noise_types", ["white", "pink"]),
            augment=True,
        )
    return AudioDataset(
        clean_dir=data_cfg.val_clean_dir,
        noisy_dir=data_cfg.val_noisy_dir,
        sample_rate=data_cfg.sample_rate,
        segment_length=data_cfg.get("segment_length", 4.0),
        augment=False,
    )


def run_vocoder_training(trainer, train_loader, val_loader, cfg: DictConfig):
    """Run vocoder fine-tuning loop."""

    # Training parameters
    max_steps = trainer.stage_steps["vocoder_finetune"]  # 675k steps
    log_interval = cfg.training.get("log_interval", 100)
    save_interval = cfg.training.get("save_interval", 5000)
    val_interval = cfg.training.get("val_interval", 1000)

    print(f"Target vocoder steps: {max_steps}")
    print(f"Log interval: {log_interval}")
    print(f"Save interval: {save_interval}")
    print(f"Validation interval: {val_interval}")

    total_steps = trainer.current_stage_step
    best_val_loss = float("inf")

    # Training loop
    try:
        epoch = 0
        while total_steps < max_steps:
            epoch += 1
            print(f"\n=== Epoch {epoch} ===")

            for batch_idx, batch in enumerate(train_loader):
                if total_steps >= max_steps:
                    break

                # Training step (joint PA + vocoder)
                metrics = trainer._train_vocoder_step(batch)
                total_steps += 1

                # Logging
                if total_steps % log_interval == 0:
                    print(f"Step {total_steps}/{max_steps}")
                    for key, value in metrics.items():
                        print(f"  {key}: {value:.6f}")

                # Validation
                if total_steps % val_interval == 0:
                    print("Running validation...")
                    val_loss = validate_vocoder(trainer, val_loader)
                    print(f"Validation Loss: {val_loss:.6f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        trainer.save_checkpoint("best_vocoder_model.pt", epoch)
                        print("New best vocoder model saved!")

                # Regular checkpoint saving
                if total_steps % save_interval == 0:
                    trainer.save_checkpoint(f"vocoder_checkpoint_step_{total_steps}.pt", epoch)
                    print(f"Vocoder checkpoint saved at step {total_steps}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint("vocoder_interrupted.pt", epoch)
        print("Saved interrupted vocoder checkpoint")

    # Final checkpoint
    trainer.save_checkpoint("vocoder_final.pt", epoch)
    print("Final vocoder checkpoint saved")


def validate_vocoder(trainer, val_loader):
    """Validate vocoder performance."""
    trainer.model.eval()
    trainer.vocoder.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            try:
                # Full pipeline validation
                noisy_audio, clean_audio, _ = batch
                noisy_audio = noisy_audio.to(trainer.device)
                clean_audio = clean_audio.to(trainer.device)

                # Generate clean audio through full pipeline
                generated_audio = trainer.model(noisy_audio, use_vocoder=True)

                # Compute full loss (mel + feature + L1)
                loss_components = trainer._compute_vocoder_loss(generated_audio, clean_audio)
                total_loss += sum(loss_components.values())
                num_batches += 1

            except Exception as e:
                print(f"Validation batch failed: {e}")
                continue

    trainer.model.train()
    trainer.vocoder.train()
    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    main()
