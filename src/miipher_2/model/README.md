# Miipher-2: Universal Speech Restoration with SpeechBrain HiFi-GAN

このディレクトリには、論文「Miipher-2」の実装が含まれています。このバージョンでは、以下の主要な改良が加えられています：

## 主な特徴

1. **Universal Speech Model (USM)**: `rinna/hubert-large` を使用した高品質な音声表現
2. **Parallel Adapters**: Conformerベースのfeature cleanerの代替として使用
3. **SpeechBrain HiFi-GAN**: `speechbrain/hifigan-hubert-k1000-LibriTTS` を使用した高品質音声合成
4. **Fine-tuning Support**: 論文の手法に従ったファインチューニング機能

## アーキテクチャ

```
Input Audio → HuBERT (frozen) → Parallel Adapters → SpeechBrain HiFi-GAN → Output Audio
```

### モデル構成要素

- **HuBERT Encoder**: `rinna/hubert-large` による音声特徴抽出（冷凍重み）
- **Parallel Adapters**: 各HuBERTレイヤーに追加される軽量なアダプター
- **SpeechBrain HiFi-GAN**: 高品質な音声合成のためのボコーダー

## インストール

必要なパッケージがインストールされていることを確認してください：

```bash
pip install speechbrain transformers
```

## 基本的な使用方法

### 1. モデルの読み込み

```python
from miipher_2.model import load_usm_model, Miipher2

# USMモデル（rinna/hubert-large）を読み込み
usm_model = load_usm_model()

# Miipher2モデルを作成
model = Miipher2(
    usm_model=usm_model,
    usm_layer_idx=13,  # HuBERTの13層目を使用
    pa_hidden_dim=1024,
    pa_input_output_dim=1024,  # HuBERTの隠れ次元
    freeze_usm=True,
    hifigan_model_id="speechbrain/hifigan-hubert-k1000-LibriTTS",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

### 2. 推論

```python
import torch
import torchaudio

# ノイズのある音声を読み込み
noisy_audio, sr = torchaudio.load("noisy_audio.wav")

# 24kHzにリサンプル
if sr != 24000:
    resampler = torchaudio.transforms.Resample(sr, 24000)
    noisy_audio = resampler(noisy_audio)

# モノラルに変換
if noisy_audio.shape[0] > 1:
    noisy_audio = noisy_audio.mean(dim=0, keepdim=True)

# 推論
model.eval()
with torch.no_grad():
    clean_audio = model.inference(noisy_audio)

# 結果を保存
torchaudio.save("cleaned_audio.wav", clean_audio.cpu(), 24000)
```

## ファインチューニング

論文の手法に従ったファインチューニングを実行できます。

### 1. データセットの準備

```python
from miipher_2.model.example_finetune import AudioDataset
from torch.utils.data import DataLoader

# ノイズのある音声ファイルのパス
train_noisy_files = ["noisy1.wav", "noisy2.wav", ...]
train_clean_files = ["clean1.wav", "clean2.wav", ...]

# データセットを作成
train_dataset = AudioDataset(
    train_noisy_files,
    train_clean_files,
    sample_rate=24000,
    max_length=48000  # 2秒の音声
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```

### 2. トレーナーの作成とファインチューニング

```python
from miipher_2.model.trainer import create_trainer

# トレーナーを作成
trainer = create_trainer(
    model=model,
    learning_rate=1e-4,
    weight_decay=1e-5,
    grad_clip_norm=1.0
)

# ファインチューニングを実行
trainer.finetune(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=20,
    save_dir="./checkpoints",
    save_interval=5
)
```

### 3. 損失関数

ファインチューニングでは、論文に従って以下の損失関数を使用します：

- **Mel-spectrogram Loss** (主要損失): 予測音声と目標音声のメルスペクトログラム間のL1損失
- **Feature Matching Loss**: HuBERT特徴量レベルでの一致度
- **L1 Waveform Loss**: 波形レベルでのL1損失

総損失は以下の重み付き組み合わせです：
```
Total Loss = 1.0 * Mel Loss + 0.5 * Feature Loss + 0.1 * L1 Loss
```

## 高度な使用方法

### チャンク処理による長い音声の処理

```python
# 長い音声を効率的に処理
long_audio, _ = torchaudio.load("long_audio.wav")
clean_audio = model.inference(long_audio, chunk_length=48000)  # 2秒のチャンク
```

### カスタムボコーダーの使用

```python
from miipher_2.model.speechbrain_utils import SpeechBrainHiFiGAN

# カスタムHiFi-GANボコーダー
custom_hifigan = SpeechBrainHiFiGAN(
    model_id="your-custom-hifigan-model",
    device="cuda"
)

model = Miipher2(
    usm_model=usm_model,
    # ... other parameters ...
    hifigan_model_id="your-custom-hifigan-model"
)
```

## 評価メトリクス

モデルの性能評価には以下のメトリクスを使用できます：

- **PESQ**: 知覚的音声品質評価
- **STOI**: 短時間客観的インテリジビリティ
- **SDR**: 信号対雑音比
- **WER**: 音声認識精度による評価

## トラブルシューティング

### よくある問題

1. **GPU メモリ不足**
   - バッチサイズを減らす
   - チャンク処理を使用する

2. **SpeechBrain モデルの読み込みエラー**
   ```bash
   pip install --upgrade speechbrain
   ```

3. **HuggingFace Transformers の問題**
   ```bash
   pip install --upgrade transformers
   ```

### パフォーマンス最適化

1. **Mixed Precision Training**を使用してメモリ使用量を削減
2. **Gradient Checkpointing**で大きなモデルでのメモリ効率を向上
3. **DataLoader**のworker数を調整

## 今後の改善点

- [ ] マルチGPU対応
- [ ] より多様な損失関数の追加
- [ ] リアルタイム処理の最適化
- [ ] 量子化モデルのサポート

## 参考文献

- Miipher-2論文（公開予定）
- SpeechBrain: https://speechbrain.github.io/
- HuggingFace Transformers: https://huggingface.co/transformers/
