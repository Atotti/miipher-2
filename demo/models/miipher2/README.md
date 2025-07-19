# Miipher-2: Speech Enhancement Model

Complete speech enhancement system consisting of a Parallel Adapter and Lightning SSL-Vocoder.

## Model Components

### 1. Parallel Adapter
- **Architecture**: Lightweight feedforward network inserted into mHuBERT-147
- **Target Layer**: Layer 6  
- **Hidden Dimension**: 768
- **Training Steps**: 199k
- **File**: `checkpoint_199k_fixed.pt`

### 2. Lightning SSL-Vocoder
- **Architecture**: HiFi-GAN based vocoder with PyTorch Lightning
- **Input**: SSL features from enhanced mHuBERT
- **Output**: High-quality audio at 22050Hz
- **Training**: 77 epochs, 137108 steps
- **File**: `epoch=77-step=137108.ckpt`

## Usage

```python
import torch
from omegaconf import DictConfig
from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.lightning_vocoders.lightning_module import HiFiGANLightningModule
from huggingface_hub import hf_hub_download

# Download model files
adapter_path = hf_hub_download(
    repo_id="YOUR_USERNAME/miipher2",
    filename="checkpoint_199k_fixed.pt"
)
vocoder_path = hf_hub_download(
    repo_id="YOUR_USERNAME/miipher2", 
    filename="epoch=77-step=137108.ckpt"
)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature Cleaner (Adapter)
config = DictConfig({
    "hubert_model_name": "utter-project/mHuBERT-147",
    "hubert_layer": 6,
    "adapter_hidden_dim": 768
})

cleaner = FeatureCleaner(config).to(device).eval()
checkpoint = torch.load(adapter_path, map_location=device, weights_only=False)
cleaner.load_state_dict(checkpoint["model_state_dict"])

# Vocoder
vocoder = HiFiGANLightningModule.load_from_checkpoint(
    vocoder_path, map_location=device
).to(device).eval()

# Inference
with torch.inference_mode():
    # Extract and clean features
    enhanced_features = cleaner(input_audio)
    
    # Generate audio
    batch = {"input_feature": enhanced_features.transpose(1, 2)}
    restored_audio = vocoder.generator_forward(batch)
```

## Model Performance

- **Target**: Speech enhancement from noisy/degraded audio
- **Training Data**: Japanese Voice Speech corpus (JVS) and multilingual datasets
- **Evaluation**: Improved speech quality metrics (STOI, PESQ, etc.)

## Files

- `checkpoint_199k_fixed.pt` (442MB) - Parallel Adapter weights
- `epoch=77-step=137108.ckpt` (1.2GB) - Lightning SSL-Vocoder weights
- `config.json` - Model configuration and metadata

## Citation

```bibtex
@article{miipher2,
  title={Miipher-2: Speech Enhancement with Parallel Adapters},
  author={Miipher-2 Team},
  year={2024}
}
```

## License

Apache-2.0