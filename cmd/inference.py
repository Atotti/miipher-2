#!/usr/bin/env python3
"""
Miipher-2 Inference Command

Usage:
    python -m cmd.inference
    python -m cmd.inference input_audio=/path/to/noisy.wav output_audio=/path/to/clean.wav
"""

import hydra
from omegaconf import DictConfig
import torch
import torchaudio
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from miipher import Miipher2
from speechbrain_utils import load_speechbrain_vocoder


@hydra.main(version_base=None, config_path="../configs/", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Main function for inference."""

    print("=== Miipher-2 Inference ===")
    print(f"Config: {cfg}")

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input audio
    input_path = Path(cfg.input_audio)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    print(f"Loading input audio: {input_path}")
    noisy_audio, sample_rate = torchaudio.load(input_path)

    # Resample if necessary
    if sample_rate != cfg.sample_rate:
        print(f"Resampling from {sample_rate}Hz to {cfg.sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, cfg.sample_rate)
        noisy_audio = resampler(noisy_audio)
        sample_rate = cfg.sample_rate

    # Convert to mono if necessary
    if noisy_audio.shape[0] > 1:
        print("Converting to mono")
        noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)

    # Initialize model and vocoder
    print("Loading model and vocoder...")
    model = Miipher2(device=device)
    vocoder = load_speechbrain_vocoder(device=device)

    # Load checkpoint
    if cfg.checkpoint:
        print(f"Loading checkpoint: {cfg.checkpoint}")
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'vocoder_state_dict' in checkpoint:
            vocoder.load_state_dict(checkpoint['vocoder_state_dict'])
    else:
        print("Warning: No checkpoint specified. Using untrained model.")

    # Set models to evaluation mode
    model.eval()
    vocoder.eval()

    # Perform inference
    print("Running inference...")
    with torch.no_grad():
        # Move audio to device
        noisy_audio = noisy_audio.to(device)

        # Process in chunks if audio is too long
        if cfg.get('chunk_length') and noisy_audio.shape[1] > cfg.chunk_length * sample_rate:
            clean_audio = process_long_audio(model, noisy_audio, cfg, sample_rate)
        else:
            # Direct inference
            clean_audio = model(noisy_audio.unsqueeze(0), use_vocoder=True)
            clean_audio = clean_audio.squeeze(0)

    # Move back to CPU for saving
    clean_audio = clean_audio.cpu()

    # Save output
    output_path = Path(cfg.output_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving clean audio: {output_path}")
    torchaudio.save(output_path, clean_audio, sample_rate)

    # Optional: compute and display metrics
    if cfg.get('compute_metrics', False) and cfg.get('reference_audio'):
        compute_metrics(clean_audio, cfg.reference_audio, sample_rate)

    print("Inference completed!")


def process_long_audio(model, noisy_audio, cfg: DictConfig, sample_rate: int):
    """Process long audio in chunks with overlap."""

    chunk_length = int(cfg.chunk_length * sample_rate)
    overlap_length = int(cfg.get('overlap_length', 0.5) * sample_rate)

    audio_length = noisy_audio.shape[1]
    clean_chunks = []

    print(f"Processing {audio_length / sample_rate:.2f}s audio in chunks of {cfg.chunk_length}s")

    start = 0
    while start < audio_length:
        end = min(start + chunk_length, audio_length)

        # Extract chunk
        chunk = noisy_audio[:, start:end]

        # Process chunk
        clean_chunk = model(chunk.unsqueeze(0), use_vocoder=True)
        clean_chunk = clean_chunk.squeeze(0)

        # Handle overlap
        if start > 0 and overlap_length > 0:
            # Fade in the beginning of current chunk
            fade_samples = min(overlap_length, clean_chunk.shape[1])
            fade_in = torch.linspace(0, 1, fade_samples, device=clean_chunk.device)
            clean_chunk[:, :fade_samples] *= fade_in

            # Fade out the end of previous chunk
            if len(clean_chunks) > 0:
                prev_chunk = clean_chunks[-1]
                fade_out = torch.linspace(1, 0, fade_samples, device=prev_chunk.device)
                prev_chunk[:, -fade_samples:] *= fade_out

                # Add overlapped region
                overlap_start = len(clean_chunks) * chunk_length - overlap_length
                clean_chunk[:, :fade_samples] += prev_chunk[:, -fade_samples:]

        clean_chunks.append(clean_chunk)
        start += chunk_length - overlap_length

    # Concatenate chunks
    return torch.cat(clean_chunks, dim=1)


def compute_metrics(clean_audio, reference_path, sample_rate):
    """Compute and display audio quality metrics."""
    try:
        import pesq
        from pystoi import stoi

        # Load reference audio
        ref_audio, ref_sr = torchaudio.load(reference_path)

        # Resample if necessary
        if ref_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, sample_rate)
            ref_audio = resampler(ref_audio)

        # Convert to mono if necessary
        if ref_audio.shape[0] > 1:
            ref_audio = torch.mean(ref_audio, dim=0)
        if clean_audio.shape[0] > 1:
            clean_audio = torch.mean(clean_audio, dim=0)

        # Align lengths
        min_length = min(clean_audio.shape[0], ref_audio.shape[0])
        clean_audio = clean_audio[:min_length].numpy()
        ref_audio = ref_audio[:min_length].numpy()

        # Compute PESQ
        pesq_score = pesq.pesq(sample_rate, ref_audio, clean_audio, 'wb')
        print(f"PESQ Score: {pesq_score:.3f}")

        # Compute STOI
        stoi_score = stoi(ref_audio, clean_audio, sample_rate, extended=False)
        print(f"STOI Score: {stoi_score:.3f}")

    except ImportError:
        print("PESQ/STOI metrics require 'pesq' and 'pystoi' packages")
    except Exception as e:
        print(f"Metrics computation failed: {e}")


if __name__ == "__main__":
    main()
