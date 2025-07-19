import gradio as gr
import torch
import numpy as np
import pathlib
from omegaconf import DictConfig
import librosa
from huggingface_hub import hf_hub_download
import os

# モデルの初期化フラグ
_models_loaded = False
_cleaner = None
_vocoder = None
_device = None

def download_models():
    """Hugging Face Hubからモデルをダウンロード"""
    try:
        print("Downloading models from Hugging Face Hub...")
        
        # Adapter model
        adapter_path = hf_hub_download(
            repo_id="Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1",
            filename="checkpoint_199k_fixed.pt",
            cache_dir="./models"
        )
        
        # Vocoder model  
        vocoder_path = hf_hub_download(
            repo_id="Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1",
            filename="epoch=77-step=137108.ckpt",
            cache_dir="./models"
        )
        
        return adapter_path, vocoder_path
        
    except Exception as e:
        print(f"Model download failed: {e}")
        return None, None

def load_models():
    """モデルを一度だけ読み込む"""
    global _models_loaded, _cleaner, _vocoder, _device
    
    if _models_loaded:
        return
    
    try:
        # デバイス設定
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FeatureCleaner
        from miipher_2.model.feature_cleaner import FeatureCleaner
        from miipher_2.lightning_vocoders.lightning_module import HiFiGANLightningModule
        
        # 設定を辞書として定義
        model_config = DictConfig({
            "hubert_model_name": "utter-project/mHuBERT-147",
            "hubert_layer": 6,
            "adapter_hidden_dim": 768
        })
        
        print("Loading FeatureCleaner model...")
        _cleaner = FeatureCleaner(model_config).to(_device).eval()
        
        # モデルをダウンロード
        adapter_path, vocoder_path = download_models()
        
        # ローカルファイルも確認
        local_adapter = "checkpoint_199k_fixed.pt"
        local_vocoder = "epoch=77-step=137108.ckpt"
        
        adapter_ckpt_path = adapter_path if adapter_path else local_adapter
        vocoder_ckpt_path = vocoder_path if vocoder_path else local_vocoder
        
        if pathlib.Path(adapter_ckpt_path).exists():
            adapter_checkpoint = torch.load(adapter_ckpt_path, map_location=_device, weights_only=False)
            _cleaner.load_state_dict(adapter_checkpoint["model_state_dict"])
            print("FeatureCleaner model loaded.")
        else:
            print("⚠️ Adapter checkpoint not found. Running without trained adapter.")
        
        # Vocoder
        if pathlib.Path(vocoder_ckpt_path).exists():
            print("Loading Lightning SSL-Vocoder...")
            _vocoder = HiFiGANLightningModule.load_from_checkpoint(
                vocoder_ckpt_path, map_location=_device
            ).to(_device).eval()
            print("Lightning SSL-Vocoder loaded.")
        else:
            print("⚠️ Vocoder checkpoint not found. Cannot generate audio.")
            _vocoder = None
        
        _models_loaded = True
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        # デモ用にダミーモデルを設定
        _cleaner = None
        _vocoder = None
        _models_loaded = True

def process_audio(input_audio):
    """音声ファイルを処理して修復された音声を返す"""
    if input_audio is None:
        return None, "音声ファイルがアップロードされていません。"
    
    try:
        # モデルを読み込み
        load_models()
        
        if _cleaner is None or _vocoder is None:
            return None, "⚠️ モデルが利用できません。これはデモ版です。"
        
        # 音声ファイルを読み込み（Gradioは(sample_rate, audio_data)のタプルを返す）
        sample_rate, audio_data = input_audio
        
        # 音声データをfloat32に変換し、正規化
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # ステレオの場合はモノラルに変換
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 22050Hzにリサンプリング
        if sample_rate != 22050:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=22050)
        
        # PyTorchテンソルに変換
        input_wav = torch.from_numpy(audio_data).unsqueeze(0).to(_device)
        
        # 推論実行
        with torch.inference_mode():
            with torch.autocast(device_type=_device.type, dtype=torch.float16, enabled=(_device.type == "cuda")):
                cleaned_features = _cleaner(input_wav)
                batch = {"input_feature": cleaned_features.transpose(1, 2)}
                restored_wav = _vocoder.generator_forward(batch)
        
        # 出力音声を準備
        output_audio = restored_wav.squeeze(0).cpu().to(torch.float32).numpy()
        
        return (22050, output_audio), "✅ 音声修復が完了しました！"
        
    except Exception as e:
        return None, f"❌ エラーが発生しました: {str(e)}"

def create_demo():
    """Gradioデモインターフェースを作成"""
    
    with gr.Blocks(title="Miipher-2 音声修復デモ", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎵 Miipher-2 音声修復デモ
        
        このデモは、Miipher-2を使用して劣化した音声を修復します。
        
        **使用方法:**
        1. マイクから録音するか、音声ファイルをアップロードしてください
        2. 「音声を修復」ボタンをクリックしてください
        3. 修復された音声が下部に表示されます
        
        **モデル情報:**
        - モデルはHugging Face Hubから自動ダウンロードされます
        - Model: [Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1](https://huggingface.co/Atotti/miipher-2-HuBERT-HiFi-GAN-v0.1)
        - 統合モデル（Adapter + Vocoder）を使用
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 入力音声")
                input_audio = gr.Audio(
                    label="音声をアップロードまたは録音",
                    type="numpy",
                    format="wav"
                )
                
                process_btn = gr.Button("🔧 音声を修復", variant="primary", size="lg")
                
            with gr.Column():
                gr.Markdown("### 📥 修復された音声")
                output_audio = gr.Audio(
                    label="修復された音声",
                    type="numpy",
                    format="wav"
                )
                
                status_text = gr.Textbox(
                    label="ステータス",
                    interactive=False,
                    lines=2
                )
        
        # サンプル音声セクション
        gr.Markdown("""
        ### 📚 技術情報
        
        **Miipher-2** は以下の技術を使用しています：
        - **SSL特徴抽出**: mHuBERT-147 (6層目の特徴を使用)
        - **Parallel Adapter**: 軽量なフィードフォワードネットワーク
        - **Lightning SSL-Vocoder**: HiFi-GANベースの音声合成
        
        **対応フォーマット**: WAV, MP3, FLAC (22050Hz推奨)
        """)
        
        # イベントハンドラ
        process_btn.click(
            fn=process_audio,
            inputs=[input_audio],
            outputs=[output_audio, status_text]
        )
        
        # サンプル例の追加
        with gr.Row():
            gr.Examples(
                examples=[],  # サンプルファイルがある場合はここに追加
                inputs=[input_audio],
                outputs=[output_audio, status_text],
                fn=process_audio,
                cache_examples=False
            )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )