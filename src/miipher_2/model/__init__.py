"""
Miipher-2 Model Package

Implementation of Miipher-2: A Universal Speech Restoration Model

Components:
- Miipher2: Main model class with USM + Parallel Adapters + SpeechBrain HiFi-GAN
- Miipher2Loss: Loss function for training
- Miipher2Trainer: Sequential training implementation (PA → Vocoder fine-tuning)
- USM utilities for HuBERT model loading
- SpeechBrain utilities for HiFi-GAN vocoder
- Modules for Parallel Adapters
"""

from .miipher import Miipher2, Miipher2Loss
from .modules import ParallelAdapter
from .speechbrain_utils import SpeechBrainHiFiGAN, create_hifigan_loss
from .trainer import Miipher2Trainer, create_trainer
from .usm_utils import load_usm_model

__all__ = [
    "Miipher2",
    "Miipher2Loss",
    "Miipher2Trainer",
    "ParallelAdapter",
    "SpeechBrainHiFiGAN",
    "create_hifigan_loss",
    "create_trainer",
    "load_usm_model",
]
