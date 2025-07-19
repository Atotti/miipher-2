---
title: Miipher-2 Speech Enhancement Demo
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Miipher-2 Speech Enhancement Demo

Miipher-2 is a speech enhancement system that uses Parallel Adapters inserted into mHuBERT layers to improve audio quality.

## Features

- **Real-time speech enhancement** from noisy or degraded audio
- **Parallel Adapter architecture** for efficient fine-tuning
- **Lightning SSL-Vocoder** for high-quality audio synthesis
- **Easy-to-use Gradio interface**

## Model Architecture

1. **SSL Feature Extractor**: mHuBERT-147 (Layer 6)
2. **Parallel Adapter**: Lightweight feedforward network
3. **Lightning SSL-Vocoder**: HiFi-GAN based vocoder

## Usage

1. Upload an audio file or record using your microphone
2. Click "音声を修復" (Enhance Audio) 
3. Listen to the enhanced audio output

## Models

The demo automatically downloads the unified model from:
- Complete Model: `Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1` (includes both Adapter and Vocoder)

## Technical Details

- **Input**: Audio files (WAV, MP3, FLAC)
- **Output**: Enhanced audio at 22050Hz
- **Supported Languages**: Primarily trained on Japanese but works with other languages
- **Processing**: Real-time inference on CPU/GPU

## License

Apache-2.0

## Citation

If you use Miipher-2 in your research, please cite:

```bibtex
@article{miipher2,
  title={Miipher-2: Speech Enhancement with Parallel Adapters},
  author={Your Name},
  year={2024}
}
```