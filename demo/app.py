import gradio as gr
import torch
import torchaudio
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
import sys
import os

# Add parent directory to path to import miipher_2 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miipher_2.model.feature_cleaner import FeatureCleaner
from miipher_2.lightning_vocoders.lightning_module import HiFiGANLightningModule

# Model configuration
MODEL_REPO_ID = "Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1"
ADAPTER_FILENAME = "checkpoint_199k_fixed.pt"
VOCODER_FILENAME = "epoch=77-step=137108.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE_INPUT = 16000
SAMPLE_RATE_OUTPUT = 22050

# Cache for models
models_cache = {}

def download_models():
    """Download models from Hugging Face Hub"""
    print("Downloading models from Hugging Face Hub...")

    adapter_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=ADAPTER_FILENAME,
        cache_dir="./models"
    )

    vocoder_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=VOCODER_FILENAME,
        cache_dir="./models"
    )

    return adapter_path, vocoder_path

def load_models():
    """Load models into memory"""
    if "cleaner" in models_cache and "vocoder" in models_cache:
        return models_cache["cleaner"], models_cache["vocoder"]

    adapter_path, vocoder_path = download_models()

    # Model configuration
    model_config = DictConfig({
        "hubert_model_name": "utter-project/mHuBERT-147",
        "hubert_layer": 6,
        "adapter_hidden_dim": 768
    })

    # Initialize FeatureCleaner
    print("Loading FeatureCleaner...")
    cleaner = FeatureCleaner(model_config).to(DEVICE).eval()

    # Load adapter weights
    adapter_checkpoint = torch.load(adapter_path, map_location=DEVICE, weights_only=False)
    cleaner.load_state_dict(adapter_checkpoint["model_state_dict"])

    # Load vocoder
    print("Loading vocoder...")
    vocoder = HiFiGANLightningModule.load_from_checkpoint(
        vocoder_path, map_location=DEVICE
    ).to(DEVICE).eval()

    # Cache models
    models_cache["cleaner"] = cleaner
    models_cache["vocoder"] = vocoder

    return cleaner, vocoder

@torch.inference_mode()
def enhance_audio(audio_path, progress=gr.Progress()):
    """Enhance audio using Miipher-2 model"""
    try:
        progress(0, desc="Loading models...")
        cleaner, vocoder = load_models()

        progress(0.2, desc="Loading audio...")
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE_INPUT:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE_INPUT)

        # Convert to mono if stereo
        waveform = waveform.mean(0, keepdim=True)

        # Move to device
        waveform = waveform.to(DEVICE)

        progress(0.4, desc="Extracting features...")
        # Extract features using FeatureCleaner
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
            features = cleaner(waveform)

            # Ensure correct shape for vocoder
            if features.dim() == 2:
                features = features.unsqueeze(0)

            progress(0.7, desc="Generating enhanced audio...")
            # Generate audio using vocoder
            # Lightning SSL-Vocoderの入力形式に合わせる (batch, seq_len, input_channels)
            batch = {"input_feature": features.transpose(1, 2)}
            enhanced_audio = vocoder.generator_forward(batch)

            # Convert to numpy
            enhanced_audio = enhanced_audio.squeeze(0).cpu().to(torch.float32).detach().numpy()

        progress(1.0, desc="Enhancement complete!")

        # Save audio using torchaudio to avoid Gradio format issues
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
        enhanced_audio_tensor = torch.from_numpy(enhanced_audio)
        
        # Ensure 2D tensor: (channels, samples)
        if enhanced_audio_tensor.dim() == 1:
            enhanced_audio_tensor = enhanced_audio_tensor.unsqueeze(0)
        
        # Save to temporary file using torchaudio
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, enhanced_audio_tensor, SAMPLE_RATE_OUTPUT)
            return tmp_file.name

    except Exception as e:
        raise gr.Error(f"Error during enhancement: {str(e)}")

# Create Gradio interface
def create_interface():
    title = "🎤 Miipher-2 Speech Enhancement"

    description = """
    <div style="text-align: center;">
        <p>High-quality speech enhancement using <b>Miipher-2</b> (HuBERT + Parallel Adapter + HiFi-GAN)</p>
        <p>📄 <a href="https://arxiv.org/abs/2505.04457">Paper</a> |
           🤗 <a href="https://huggingface.co/Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1">Model</a> |
           💻 <a href="https://github.com/your-repo/open-miipher-2">GitHub</a></p>
    </div>
    """

    article = """
    ## How it works

    1. **Upload** a noisy or degraded audio file
    2. **Process** using Miipher-2 model
    3. **Download** the enhanced audio

    ### Model Details
    - **SSL Backbone**: mHuBERT-147 (Multilingual)
    - **Adapter**: Parallel adapters at layer 6
    - **Vocoder**: HiFi-GAN trained on SSL features
    - **Input**: Any sample rate (automatically resampled to 16kHz)
    - **Output**: 22.05kHz high-quality audio

    ### Tips
    - Works best with speech audio
    - Supports various noise types (background noise, reverb, etc.)
    - Processing time depends on audio length and hardware
    """

    examples = [
        ["examples/noisy_speech_1.wav"],
        ["examples/noisy_speech_2.wav"],
        ["examples/reverb_speech.wav"],
    ]

    with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    label="Input Audio (Noisy/Degraded)",
                    type="filepath",
                    sources=["upload", "microphone"]
                )

                enhance_btn = gr.Button("🚀 Enhance Audio", variant="primary")

            with gr.Column():
                output_audio = gr.Audio(
                    label="Enhanced Audio",
                    type="filepath",
                    interactive=False
                )

        # Add examples if they exist
        examples_dir = Path("examples")
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*.wav")) + list(examples_dir.glob("*.mp3"))
            if example_files:
                gr.Examples(
                    examples=[[str(f)] for f in example_files[:3]],
                    inputs=input_audio,
                    outputs=output_audio,
                    fn=enhance_audio,
                    cache_examples=True
                )

        gr.Markdown(article)

        # Connect the enhancement function
        enhance_btn.click(
            fn=enhance_audio,
            inputs=input_audio,
            outputs=output_audio,
            show_progress=True
        )

    return demo

# Launch the app
if __name__ == "__main__":
    # Pre-load models
    print("Pre-loading models...")
    load_models()
    print("Models loaded successfully!")

    # Create and launch interface
    demo = create_interface()
    demo.launch()
