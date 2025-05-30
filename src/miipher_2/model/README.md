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
