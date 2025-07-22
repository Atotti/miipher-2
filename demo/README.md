---
title: Miipher-2 Speech Enhancement
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
models:
  - Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1
---

# 🎤 Miipher-2 Speech Enhancement Demo

This is a Gradio demo for **Miipher-2**, a high-quality speech enhancement model that combines HuBERT, Parallel Adapters, and HiFi-GAN vocoder.

## Features

- **Real-time speech enhancement** - Remove noise, reverb, and other degradations
- **Multilingual support** - Built on mHuBERT-147 for 147 languages
- **High-quality output** - 22.05kHz audio output
- **Easy to use** - Simple drag-and-drop or microphone input

## Model Details

- **Paper**: [Miipher-2: High-Quality Speech Enhancement](https://arxiv.org/abs/2505.04457)
- **Model**: [Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1](https://huggingface.co/Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1)
- **GitHub**: [open-miipher-2](https://github.com/your-repo/open-miipher-2)

## How to Use

1. **Upload** an audio file or record using microphone
2. Click **"Enhance Audio"** button
3. **Download** the enhanced result

## Technical Details

The model uses:
- **SSL Backbone**: mHuBERT-147 (multilingual)
- **Adapter**: Parallel adapters inserted at layer 6
- **Vocoder**: HiFi-GAN trained on SSL features
- **Input**: Any sample rate (auto-resampled to 16kHz)
- **Output**: 22.05kHz enhanced audio

## Citation

```bibtex
@article{miipher2024,
  title={Miipher-2: High-Quality Speech Enhancement via Self-Supervised Learning},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:2505.04457},
  year={2024}
}
```